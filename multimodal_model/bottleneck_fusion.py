"""Simple Multimodal Fusion using MBT - PyTorch Implementation.

This module provides a simplified interface for using MBT to fuse multimodal tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import ml_collections
import time


class SparseMoEFeedForward(nn.Module):
  """Sparse Mixture of Experts Feed-Forward Layer.

  Instead of routing all tokens through a dense MLP, this routes each token
  to only k experts (usually k=1), dramatically reducing computation.

  Args:
    d_model: Model dimension
    expert_dim: Hidden dimension of each expert (can be smaller than d_model * 4)
    num_experts: Number of expert networks (default: 4)
    k: Number of experts to activate per token (default: 1)
    log_activations: Whether to log which experts are activated (default: False)
  """
  def __init__(self, d_model, expert_dim=256, num_experts=4, k=1, log_activations=False):
    super().__init__()
    self.num_experts = num_experts
    self.k = k
    self.log_activations = log_activations
    self.logged_expert_ids = []

    # Create multiple expert networks
    self.experts = nn.ModuleList([
        nn.Sequential(
            nn.Linear(d_model, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, d_model)
        ) for _ in range(num_experts)
    ])

    # Gating network to select experts
    self.gate = nn.Linear(d_model, num_experts)

  def forward(self, x):
    """
    Args:
      x: Input tensor [B, T, D]

    Returns:
      Output tensor [B, T, D] after sparse expert routing
    """
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D)

    # Compute gating scores for each token
    gate_scores = self.gate(x_flat)  # [B*T, num_experts]

    # Select top-k experts for each token
    topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)
    topk_scores = F.softmax(topk_scores, dim=-1)

    if self.log_activations:
      self.logged_expert_ids.append(topk_indices.detach().cpu())

    # Route tokens to selected experts
    output = torch.zeros_like(x_flat)
    for i in range(self.k):
      expert_ids = topk_indices[:, i]  # [B*T]
      one_hot_mask = F.one_hot(expert_ids, self.num_experts).bool()  # [B*T, num_experts]

      for expert_idx in range(self.num_experts):
        expert_mask = one_hot_mask[:, expert_idx]  # [B*T]
        if expert_mask.sum() == 0:
          continue

        # Process tokens assigned to this expert
        selected = x_flat[expert_mask]  # [N, D]
        result = self.experts[expert_idx](selected)  # [N, D]
        score = topk_scores[expert_mask, i].unsqueeze(1)  # [N, 1]
        output[expert_mask] += score * result

    return output.reshape(B, T, D)

  def get_activation_logs(self):
    """Return logged expert activations for analysis."""
    return torch.cat(self.logged_expert_ids, dim=0).numpy() if self.logged_expert_ids else None


class SimpleMBTFusion(nn.Module):
  """Simplified MBT-based multimodal fusion module.

  This class provides an easy-to-use interface for fusing multiple modalities
  using the Multimodal Bottleneck Transformer (MBT) architecture.

  Args:
    input_dims: Dictionary mapping modality names to their input dimensions.
                Example: {'rgb': 768, 'audio': 512}
    hidden_size: Size of the hidden/embedding dimension (default: 512)
    num_layers: Number of transformer encoder layers (default: 6)
    num_heads: Number of attention heads (default: 8)
    mlp_dim: Dimension of the MLP in transformer blocks (default: 2048)
    fusion_layer: At which layer to start fusing modalities (default: 3)
                  0 = early fusion, num_layers = late fusion
    use_bottleneck: Whether to use bottleneck tokens for fusion (default: True)
    n_bottlenecks: Number of bottleneck tokens (default: 4)
    dropout_rate: Dropout rate (default: 0.1)
    output_dim: Output dimension after fusion (default: None, uses hidden_size)

  Example:
    >>> # Create fusion module for RGB and audio
    >>> fusion = SimpleMBTFusion(
    ...     input_dims={'rgb': 768, 'audio': 512},
    ...     hidden_size=512,
    ...     num_layers=6,
    ...     fusion_layer=3
    ... )
    >>>
    >>> # Prepare inputs (batch_size=4, seq_len varies by modality)
    >>> inputs = {
    ...     'rgb': torch.randn(4, 196, 768),      # 4 samples, 196 tokens, 768 dims
    ...     'audio': torch.randn(4, 100, 512)     # 4 samples, 100 tokens, 512 dims
    ... }
    >>>
    >>> # Fuse modalities
    >>> fused = fusion(inputs)  # Output: (4, 512) by default
  """

  def __init__(self,
               input_dims: Dict[str, int],
               hidden_size: int = 512,
               num_layers: int = 6,
               num_heads: int = 8,
               mlp_dim: int = 2048,
               fusion_layer: int = 3,
               use_bottleneck: bool = True,
               n_bottlenecks: int = 4,
               dropout_rate: float = 0.1,
               output_dim: Optional[int] = None,
               use_sparse_moe: bool = False,
               num_experts: int = 4,
               expert_k: int = 1):
    super().__init__()

    self.modalities = list(input_dims.keys())
    self.hidden_size = hidden_size
    self.output_dim = output_dim if output_dim is not None else hidden_size
    self.use_bottleneck = use_bottleneck
    self.n_bottlenecks = n_bottlenecks
    self.use_sparse_moe = use_sparse_moe

    # Input projection layers for each modality
    self.input_projections = nn.ModuleDict()
    for modality, input_dim in input_dims.items():
      if input_dim != hidden_size:
        self.input_projections[modality] = nn.Linear(input_dim, hidden_size)
      else:
        self.input_projections[modality] = nn.Identity()

    # Positional embeddings for each modality (learnable)
    # Initialize with reasonable max length
    max_seq_len = 500
    self.pos_embeds = nn.ParameterDict({
        modality: nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
        for modality in self.modalities
    })

    # Encoder blocks
    self.num_layers = num_layers
    self.fusion_layer = fusion_layer
    self.blocks = nn.ModuleDict()

    for lyr in range(num_layers):
      # Create separate encoder blocks for each modality
      for modality in self.modalities:
        block_name = f'{modality}_layer_{lyr}'
        self.blocks[block_name] = TransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            use_sparse_moe=use_sparse_moe,
            num_experts=num_experts,
            expert_k=expert_k
        )

    # Bottleneck tokens (if used)
    if use_bottleneck:
      self.bottleneck = nn.Parameter(
          torch.randn(1, n_bottlenecks, hidden_size) * 0.02
      )
    else:
      self.bottleneck = None

    # Final normalization
    self.final_norm = nn.LayerNorm(hidden_size)

    # Output projection (optional)
    if output_dim != hidden_size:
      self.output_projection = nn.Linear(hidden_size, output_dim)
    else:
      self.output_projection = nn.Identity()

  def _add_positional_embedding(self, x: torch.Tensor, modality: str) -> torch.Tensor:
    """Add learnable positional embeddings to input tokens."""
    batch_size, seq_len, hidden_dim = x.shape

    # Get positional embedding for this modality
    pos_embed = self.pos_embeds[modality]

    # Handle variable sequence lengths
    if seq_len <= pos_embed.shape[1]:
      return x + pos_embed[:, :seq_len, :]
    else:
      # Interpolate if sequence is longer than initialized
      pos_embed_interp = torch.nn.functional.interpolate(
          pos_embed.permute(0, 2, 1),
          size=seq_len,
          mode='linear',
          align_corners=False
      ).permute(0, 2, 1)
      return x + pos_embed_interp

  def forward(self,
              inputs: Dict[str, torch.Tensor],
              return_tokens: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward pass for multimodal fusion.

    Args:
      inputs: Dictionary of input tensors for each modality
              Shape for each: (batch_size, seq_len, input_dim)
      return_tokens: If True, return all tokens; if False, return pooled output

    Returns:
      If return_tokens=False: Fused representation (batch_size, output_dim)
      If return_tokens=True: All fused tokens (batch_size, total_seq_len, hidden_size)
    """
    batch_size = list(inputs.values())[0].shape[0]

    # Project inputs to hidden dimension
    x = {}
    for modality in self.modalities:
      if modality not in inputs:
        raise ValueError(f"Missing input for modality: {modality}")

      # Project to hidden dimension
      x[modality] = self.input_projections[modality](inputs[modality])

      # Add positional embeddings
      #x[modality] = self._add_positional_embedding(x[modality], modality)

    # Prepare bottleneck tokens if used
    bottleneck = None
    if self.use_bottleneck and self.bottleneck is not None:
      bottleneck = self.bottleneck.expand(batch_size, -1, -1)

    if bottleneck is None:
      time.sleep(1)
      print("!!!!!!!!!!!!!!!!!!!!!!")
      print("!!!!!!!!!!!!!!!!!!!!!!")
      print("!!!!!!!!!!!!!!!!!!!!!!")
      print("!!!!!!!!!!!!!!!!!!!!!!")
      print("!!!!!!!!!!!!!!!!!!!!!!")
    # Process through transformer layers
    for lyr in range(self.num_layers):
      if lyr < self.fusion_layer:
        # Before fusion: process each modality independently
        for modality in self.modalities:
          block = self.blocks[f'{modality}_layer_{lyr}']
          x[modality] = block(x[modality])
      else:
        # After fusion layer: fuse modalities
        if self.use_bottleneck and bottleneck is not None:
          # Bottleneck-based fusion
          new_bottleneck = []
          for modality in self.modalities:
            # Concatenate modality tokens with bottleneck
            combined = torch.cat([x[modality], bottleneck], dim=1)

            # Process through transformer
            block = self.blocks[f'{modality}_layer_{lyr}']
            output = block(combined)

            # Split back
            seq_len = x[modality].shape[1]
            x[modality] = output[:, :seq_len]
            new_bottleneck.append(output[:, seq_len:])

          # Average bottleneck tokens across modalities
          bottleneck = torch.stack(new_bottleneck, dim=0).mean(dim=0)
        else:
          # Cross-attention based fusion: concatenate all modalities
          # Concatenate all modality tokens
          all_tokens = []
          seq_lengths = []
          for modality in self.modalities:
            all_tokens.append(x[modality])
            seq_lengths.append(x[modality].shape[1])

          combined = torch.cat(all_tokens, dim=1)

          # Process through first modality's transformer
          # (In full MBT, different modalities see different views)
          block = self.blocks[f'{self.modalities[0]}_layer_{lyr}']
          combined = block(combined)

          # Split back to individual modalities
          start_idx = 0
          for i, modality in enumerate(self.modalities):
            end_idx = start_idx + seq_lengths[i]
            x[modality] = combined[:, start_idx:end_idx]
            start_idx = end_idx

    # Concatenate all modality tokens
    #fused_tokens = torch.cat([x[modality] for modality in self.modalities], dim=1)

    # Final normalization
    #fused_tokens = self.final_norm(fused_tokens)
    bottleneck = self.final_norm(bottleneck)

    if return_tokens:
      return bottleneck#fused_tokens

    # Global average pooling across all tokens
    fused_representation = bottleneck.mean(dim=1)#fused_tokens.mean(dim=1)

    # Project to output dimension
    output = self.output_projection(fused_representation)

    return output


class TransformerBlock(nn.Module):
  """Transformer encoder block with self-attention and MLP (dense or sparse MoE).

  Args:
    hidden_size: Model dimension
    num_heads: Number of attention heads
    mlp_dim: Hidden dimension for MLP (or expert dimension if using sparse MoE)
    dropout_rate: Dropout rate
    use_sparse_moe: Whether to use Sparse MoE instead of dense MLP (default: False)
    num_experts: Number of experts in MoE (default: 4)
    expert_k: Number of experts to activate per token (default: 1)
  """

  def __init__(self,
               hidden_size: int,
               num_heads: int,
               mlp_dim: int,
               dropout_rate: float = 0.1,
               use_sparse_moe: bool = False,
               num_experts: int = 4,
               expert_k: int = 1):
    super().__init__()

    self.use_sparse_moe = use_sparse_moe

    self.ln1 = nn.LayerNorm(hidden_size)
    self.attn = nn.MultiheadAttention(
        hidden_size,
        num_heads,
        dropout=dropout_rate,
        batch_first=True
    )
    self.dropout1 = nn.Dropout(dropout_rate)

    self.ln2 = nn.LayerNorm(hidden_size)

    # Choose between dense MLP or Sparse MoE
    if use_sparse_moe:
      self.mlp = SparseMoEFeedForward(
          d_model=hidden_size,
          expert_dim=mlp_dim,  # Each expert has mlp_dim hidden size
          num_experts=num_experts,
          k=expert_k,
          log_activations=False
      )
    else:
      self.mlp = nn.Sequential(
          nn.Linear(hidden_size, mlp_dim),
          nn.GELU(),
          nn.Dropout(dropout_rate),
          nn.Linear(mlp_dim, hidden_size),
          nn.Dropout(dropout_rate)
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Self-attention with residual
    residual = x
    x = self.ln1(x)
    x, _ = self.attn(x, x, x, need_weights=False)
    x = self.dropout1(x)
    x = x + residual

    # MLP (dense or sparse) with residual
    residual = x
    x = self.ln2(x)
    x = self.mlp(x)
    x = x + residual

    return x


# Example usage and helper function
def create_simple_fusion(modalities: Dict[str, int],
                        hidden_size: int = 512,
                        **kwargs) -> SimpleMBTFusion:
  """Helper function to create a SimpleMBTFusion module.

  Args:
    modalities: Dict mapping modality names to input dimensions
    hidden_size: Hidden dimension size
    **kwargs: Additional arguments for SimpleMBTFusion

  Returns:
    SimpleMBTFusion module

  Example:
    >>> fusion = create_simple_fusion(
    ...     modalities={'rgb': 768, 'audio': 512, 'text': 768},
    ...     hidden_size=512,
    ...     num_layers=6,
    ...     fusion_layer=3
    ... )
  """
  return SimpleMBTFusion(
      input_dims=modalities,
      hidden_size=hidden_size,
      **kwargs
  )


if __name__ == '__main__':
  # Example usage
  print("Creating SimpleMBTFusion module...")

  # Define modalities and their input dimensions
  modalities = {
      'rgb': 768,      # e.g., from ViT
      'audio': 512,    # e.g., from audio encoder
  }

  # Create fusion module
  fusion = SimpleMBTFusion(
      input_dims=modalities,
      hidden_size=512,
      num_layers=6,
      num_heads=8,
      mlp_dim=2048,
      fusion_layer=3,
      use_bottleneck=True,
      n_bottlenecks=4,
      output_dim=256  # Final output dimension
  )

  # Create sample inputs
  batch_size = 4
  inputs = {
      'rgb': torch.randn(batch_size, 196, 768),    # 196 patches from 14x14 grid
      'audio': torch.randn(batch_size, 100, 512),  # 100 audio tokens
  }

  # Forward pass
  print(f"\nInput shapes:")
  for mod, tensor in inputs.items():
    print(f"  {mod}: {tensor.shape}")

  # Get pooled output
  output = fusion(inputs, return_tokens=False)
  print(f"\nPooled output shape: {output.shape}")  # (4, 256)

  # Get all tokens
  tokens = fusion(inputs, return_tokens=True)
  print(f"All tokens shape: {tokens.shape}")  # (4, 296, 512) - 196+100 tokens

  print("\nFusion module created successfully!")
  print(f"Total parameters: {sum(p.numel() for p in fusion.parameters()):,}")
