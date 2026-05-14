import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.helper_function import AverageMeter, ProgressMeter


def train_one_epoch(train_loader, model, class_loss_criterion, optimizer, epoch, device):
    loss_meter = AverageMeter('Class Loss', ':.4f')
    acc_meter  = AverageMeter('Class Acc', ':.4f')
    model.train()
    model.zero_grad()

    for i, (x, y) in enumerate(train_loader):
        correct = 0
        x, y = x.to(device).float(), y.to(device)
        # plt.plot(x[0].cpu().numpy())
        # plt.savefig('test.png')
        # breakpoint()
        class_output = model(x)
        loss = class_loss_criterion(class_output, y)

        _, predicted = torch.max(class_output.data, 1)
        correct += predicted.eq(y).sum().item()
        acc = correct / x.size(0)

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress = ProgressMeter(len(train_loader), [loss_meter, acc_meter], prefix=f"Epoch: [{epoch}]")
        if (i % 50 == 0) or (i == len(train_loader) - 1):
            progress.display(i)
            if i == len(train_loader) - 1:
                print('End of Epoch', epoch, 'Class loss is', '%.4f' % loss_meter.avg, '    Training accuracy is ', '%.4f' % acc_meter.avg)
    return loss_meter.avg, acc_meter.avg

def evaluate_one_epoch(val_loader, model, class_loss_criterion, epoch, device):
    loss_meter = AverageMeter('Class Loss', ':.4f')
    acc_meter  = AverageMeter('Class Acc', ':.4f')
    model.eval()
    model.zero_grad()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device).float(), y.to(device)
            model = model.to(device)
            class_output = model(x)
            loss = class_loss_criterion(class_output, y)

            _, predicted = torch.max(class_output.data, 1)
            correct = predicted.eq(y).sum().item()
            acc = correct / x.size(0)

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))

            all_preds.append(predicted.cpu())
            all_labels.append(y.cpu())

            if i == len(val_loader) - 1:
                print('End of Epoch', epoch, 
                      '| Validation Class loss:', f'{loss_meter.avg:.4f}', 
                      '| Validation accuracy:', f'{acc_meter.avg:.4f}')

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    return loss_meter.avg, acc_meter.avg, y_true, y_pred


# =============================================================================
# Curriculum Schedules (staggered: clean -> modality drop -> token drop -> hold)
# =============================================================================

def get_modality_curriculum(epoch, num_epochs, max_drop=0.4):
    """Modality dropout ramp: 0 during [0, 20%), linear ramp [20%, 50%), hold after."""
    start = int(num_epochs * 0.2)
    end = int(num_epochs * 0.5)
    if epoch < start:
        return 0.0
    if epoch >= end:
        return max_drop
    return max_drop * (epoch - start) / (end - start)


def get_token_curriculum(epoch, num_epochs, max_drop=0.15):
    """Token dropout ramp: 0 during [0, 50%), linear ramp [50%, 65%), hold after."""
    start = int(num_epochs * 0.5)
    end = int(num_epochs * 0.65)
    if epoch < start:
        return 0.0
    if epoch >= end:
        return max_drop
    return max_drop * (epoch - start) / (end - start)


# =============================================================================
# Modality Gradient Profiler (gradient-based asymmetric dropout)
# =============================================================================

class ModalityGradientProfiler:
    """Tracks per-modality gradient norms for asymmetric dropout scheduling.

    Registers hooks on input_projections to accumulate gradient norms.
    Modalities with higher gradient norms get higher dropout probability.
    """

    def __init__(self, model, modalities):
        self.modalities = modalities
        self.grad_norms = {m: [] for m in modalities}
        self._hooks = []
        # Support both naming conventions: 416 calls it `input_projections`,
        # standalone V5/V6 base calls it `input_projectors`. Pick whichever the
        # given model exposes.
        if hasattr(model, 'input_projections'):
            projection_dict = model.input_projections
        elif hasattr(model, 'input_projectors'):
            projection_dict = model.input_projectors
        else:
            raise AttributeError(
                f"{type(model).__name__} has neither 'input_projections' nor "
                f"'input_projectors'; ModalityGradientProfiler can't hook gradients."
            )
        for m in modalities:
            proj = projection_dict[m]
            hook = proj.weight.register_hook(
                lambda grad, mod=m: self.grad_norms[mod].append(grad.norm().item())
            )
            self._hooks.append(hook)

    def get_asymmetric_dropout_probs(self, base_prob):
        """Scale per-modality dropout by relative gradient norm."""
        mean_norms = {}
        for m in self.modalities:
            recent = self.grad_norms[m][-100:] if self.grad_norms[m] else [0.0]
            mean_norms[m] = float(np.mean(recent))
        total = sum(mean_norms.values())
        if total == 0:
            return {m: base_prob for m in self.modalities}
        N = len(self.modalities)
        ratios = {m: mean_norms[m] / total for m in self.modalities}
        return {m: min(0.8, max(0.0, base_prob * N * ratios[m])) for m in self.modalities}

    def get_norm_snapshot(self, window=100):
        snapshot = {}
        for m in self.modalities:
            recent = self.grad_norms[m][-window:] if self.grad_norms[m] else [0.0]
            snapshot[m] = float(np.mean(recent))
        return snapshot

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# =============================================================================
# Robust Training / Evaluation (for SentinelDropoutWrapper models)
# =============================================================================

def train_one_epoch_robust(train_loader, model, criterion, optimizer, epoch, device,
                           modality_dropout_probs, token_drop_prob):
    """Train one epoch with curriculum modality + token dropout."""
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter  = AverageMeter('Acc',  ':.4f')
    model.train()

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device).float(), y.to(device)
        logits, _ = model(x, modality_dropout_probs=modality_dropout_probs,
                          token_drop_prob=token_drop_prob, training=True)
        loss = criterion(logits, y)
        _, predicted = torch.max(logits, 1)
        acc = predicted.eq(y).sum().item() / x.size(0)

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(acc, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress = ProgressMeter(len(train_loader), [loss_meter, acc_meter], prefix=f"Epoch: [{epoch}]")
        if (i % 50 == 0) or (i == len(train_loader) - 1):
            progress.display(i)
            if i == len(train_loader) - 1:
                avg_mod = np.mean(list(modality_dropout_probs.values())) if isinstance(modality_dropout_probs, dict) else modality_dropout_probs
                print(f'End of Epoch {epoch}  Loss: {loss_meter.avg:.4f}  Acc: {acc_meter.avg:.4f}  '
                      f'ModDrop: {avg_mod:.3f}  TokDrop: {token_drop_prob:.3f}')

    return loss_meter.avg, acc_meter.avg


def evaluate_one_epoch_robust(val_loader, model, criterion, epoch, device):
    """Evaluate one epoch for SentinelDropoutWrapper models."""
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter  = AverageMeter('Acc',  ':.4f')
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device).float(), y.to(device)
            logits, _ = model(x, training=False)
            loss = criterion(logits, y)
            _, predicted = torch.max(logits, 1)
            acc = predicted.eq(y).sum().item() / x.size(0)
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc, x.size(0))
            all_preds.append(predicted.cpu())
            all_labels.append(y.cpu())
            if i == len(val_loader) - 1:
                print(f'End of Epoch {epoch} | Val Loss: {loss_meter.avg:.4f} | Val Acc: {acc_meter.avg:.4f}')

    return loss_meter.avg, acc_meter.avg, torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017). Down-weights well-classified examples."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
