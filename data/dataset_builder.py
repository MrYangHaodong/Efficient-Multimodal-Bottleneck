# ============================ Imports ============================
import os
import sys
import glob
import math
import time
import random
import argparse
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
from torchvision import transforms

from scipy.signal import stft, hilbert
from torch.autograd import Function

from thop import profile

from utils.helper_function import set_seed, count_model_parameters, AverageMeter, ProgressMeter, sax_tokenizer, sax_noisy_tokenizer
# ============================ Dataset ============================
class HARDataset(Dataset):
    def __init__(self, root_dir, modalities, subjects, cfg, transform='sax', sax_params=None, noise_config=None):
        self.root_dir = root_dir
        self.transform = transform
        self.subjects = subjects
        self.modalities = modalities
        self.base_sr = cfg.base_sample_rate
        self.duration = cfg.duration
        self.sampling_rates = cfg.sampling_rates
        self.file_paths = []
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 2}
        self.noise_config = noise_config

        for subject in subjects:
            subject_dir = os.path.join(root_dir, subject)
            if os.path.exists(subject_dir):
                self.file_paths.extend(glob.glob(os.path.join(subject_dir, "*.pt")))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])

        def resample_data(mod_data, orig_sr):
            expected_length = int(self.duration * self.base_sr)
            resample_factor = self.base_sr / orig_sr

            if resample_factor < 1:
                step = int(1 / resample_factor)
                resampled = mod_data[::step]
            else:
                if len(mod_data.shape) == 2:
                    time_dim, feature_dim = mod_data.shape
                    mod_data_reshaped = mod_data.permute(1, 0).unsqueeze(0)
                    target_len = int(time_dim * resample_factor)
                    resampled = F.interpolate(mod_data_reshaped, size=target_len, mode='linear', align_corners=False).squeeze(0).permute(1, 0)
                else:
                    raise ValueError(f"Unexpected tensor shape: {mod_data.shape}")

            current_length = resampled.shape[0]
            if current_length > expected_length:
                resampled = resampled[:expected_length]
            elif current_length < expected_length:
                padding_needed = expected_length - current_length
                last_frame = resampled[-1:].repeat(padding_needed, 1)
                resampled = torch.cat([resampled, last_frame], dim=0)

            assert resampled.shape[0] == expected_length
            return resampled

        # Resample and concatenate modalities
        if len(self.modalities) > 1:
            resampled_data = []
            for modality in self.modalities:
                if modality in data:
                    mod_data = data[modality]
                    orig_sr = self.sampling_rates[modality]
                    resampled = resample_data(mod_data, orig_sr)
                    resampled_data.append(resampled)
            x = torch.cat(resampled_data, dim=1)
        else:
            modality = self.modalities[0]
            mod_data = data[modality]
            orig_sr = self.sampling_rates[modality]
            x = resample_data(mod_data, orig_sr)

        # Apply noise injection if configured (on continuous signal, before SAX)
        if self.noise_config is not None:
            noise_fn = self.noise_config['noise_fn']
            noise_kwargs = self.noise_config.get('noise_kwargs', {})
            x_np = x.numpy()
            noisy_channels = []
            for ch in range(x_np.shape[1]):
                corrupted, _label = noise_fn(x_np[:, ch], **noise_kwargs)
                noisy_channels.append(corrupted)
            x = torch.tensor(np.stack(noisy_channels, axis=1), dtype=x.dtype)

        # Apply SAX if specified
        if self.transform == 'sax':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_tokenizer(
                    x[:, ch].numpy(),
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]
        elif self.transform == 'sax_noisy':
            # 1-based tokens (1..alphabet_size), token 0 reserved for noisy/missing
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_noisy_tokenizer(
                    x[:, ch].numpy(),
                    noisy_mask=None,
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]

        y = data['label']
        return x, y


# ============================ TTN Dataset ============================
class TTNHARDataset(Dataset):
    """
    HARDataset variant for Train-Time Noise (TTN) experiments.

    Always appends a binary mask channel as the last feature column.
    - Training mode (random_mask_ratio > 0): randomly mask positions on clean data.
    - Eval mode (noise_config set): inject real noise and use oracle labels as mask.
    - Neither: mask channel is all-zeros.

    For SAX: corrupted positions get a reserved noise_token = alphabet_size.
    For non-SAX: corrupted positions are zeroed out.
    """
    def __init__(self, root_dir, modalities, subjects, cfg, transform='sax',
                 sax_params=None, noise_config=None, random_mask_ratio=0.0):
        self.root_dir = root_dir
        self.transform = transform
        self.subjects = subjects
        self.modalities = modalities
        self.base_sr = cfg.base_sample_rate
        self.duration = cfg.duration
        self.sampling_rates = cfg.sampling_rates
        self.variates = cfg.variates
        self.file_paths = []
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 1}
        self.noise_config = noise_config
        self.random_mask_ratio = random_mask_ratio

        for subject in subjects:
            subject_dir = os.path.join(root_dir, subject)
            if os.path.exists(subject_dir):
                self.file_paths.extend(glob.glob(os.path.join(subject_dir, "*.pt")))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])

        def resample_data(mod_data, orig_sr):
            expected_length = int(self.duration * self.base_sr)
            resample_factor = self.base_sr / orig_sr

            if resample_factor < 1:
                step = int(1 / resample_factor)
                resampled = mod_data[::step]
            else:
                if len(mod_data.shape) == 2:
                    time_dim, feature_dim = mod_data.shape
                    mod_data_reshaped = mod_data.permute(1, 0).unsqueeze(0)
                    target_len = int(time_dim * resample_factor)
                    resampled = F.interpolate(mod_data_reshaped, size=target_len, mode='linear', align_corners=False).squeeze(0).permute(1, 0)
                else:
                    raise ValueError(f"Unexpected tensor shape: {mod_data.shape}")

            current_length = resampled.shape[0]
            if current_length > expected_length:
                resampled = resampled[:expected_length]
            elif current_length < expected_length:
                padding_needed = expected_length - current_length
                last_frame = resampled[-1:].repeat(padding_needed, 1)
                resampled = torch.cat([resampled, last_frame], dim=0)

            assert resampled.shape[0] == expected_length
            return resampled

        # Resample and concatenate modalities
        if len(self.modalities) > 1:
            resampled_data = []
            for modality in self.modalities:
                if modality in data:
                    mod_data = data[modality]
                    orig_sr = self.sampling_rates[modality]
                    resampled = resample_data(mod_data, orig_sr)
                    resampled_data.append(resampled)
            x = torch.cat(resampled_data, dim=1)
        else:
            modality = self.modalities[0]
            mod_data = data[modality]
            orig_sr = self.sampling_rates[modality]
            x = resample_data(mod_data, orig_sr)

        T_raw, C = x.shape
        raw_labels = None  # [T_raw, C] or None

        # Apply noise injection if configured (on continuous signal, before SAX)
        # Noise is applied per-modality: all variates within a modality share
        # the same corrupted time-point positions. The first variate determines
        # which positions are corrupted; remaining variates are corrupted at
        # those same positions with noise values from their own noise_fn call.
        if self.noise_config is not None:
            noise_fn = self.noise_config['noise_fn']
            noise_kwargs = self.noise_config.get('noise_kwargs', {})
            x_np = x.numpy()
            noisy_channels = [None] * C
            per_channel_labels = [None] * C
            ch_offset = 0
            for mod in self.modalities:
                num_vars = self.variates[mod]
                # First variate: determine corruption positions
                corrupted_ref, label_ref = noise_fn(x_np[:, ch_offset], **noise_kwargs)
                noisy_channels[ch_offset] = corrupted_ref
                per_channel_labels[ch_offset] = label_ref
                # Remaining variates: corrupt at same positions
                noisy_positions = label_ref.astype(bool)
                for v in range(1, num_vars):
                    ch = ch_offset + v
                    corrupted_v, _ = noise_fn(x_np[:, ch], **noise_kwargs)
                    # Blend: keep clean values, replace only at reference positions
                    blended = x_np[:, ch].copy()
                    blended[noisy_positions] = corrupted_v[noisy_positions]
                    noisy_channels[ch] = blended
                    per_channel_labels[ch] = label_ref  # shared label
                ch_offset += num_vars
            x = torch.tensor(np.stack(noisy_channels, axis=1), dtype=x.dtype)
            raw_labels = np.stack(per_channel_labels, axis=1)  # [T_raw, C]

        # Apply SAX if specified
        if self.transform == 'sax':
            alphabet_size = self.sax_params['alphabet_size']
            word_length = self.sax_params['word_length']
            # Token 0 is the reserved noisy/masked symbol; clean tokens are 1..alphabet_size.
            # This is the noisy-SAX convention from sax_noisy_tokenizer.

            sax_features = []
            num_sax_tokens = T_raw // word_length
            sax_mask = np.zeros(num_sax_tokens, dtype=np.float32)

            for ch in range(C):
                if raw_labels is not None:
                    # Eval mode: oracle noise labels available.
                    # Normalize using clean points only; noisy segments get token 0.
                    ch_noisy_mask = raw_labels[:, ch].astype(bool)
                    sax_seq = sax_noisy_tokenizer(
                        x[:, ch].numpy(),
                        noisy_mask=ch_noisy_mask,
                        alphabet_size=alphabet_size,
                        word_length=word_length,
                    )
                    # Union mask: any channel noisy at position t → sax_mask[t] = 1
                    sax_mask = np.maximum(sax_mask, (sax_seq == 0).astype(np.float32))
                else:
                    # Training/clean mode: no oracle labels.
                    # Use noisy-SAX with no noisy mask → 1-based clean tokens (1..alphabet_size).
                    sax_seq = sax_noisy_tokenizer(
                        x[:, ch].numpy(),
                        noisy_mask=None,
                        alphabet_size=alphabet_size,
                        word_length=word_length,
                    )
                    if self.random_mask_ratio > 0:
                        # Random masking: same positions across all channels
                        if ch == 0:
                            num_mask = max(1, int(num_sax_tokens * self.random_mask_ratio))
                            self._mask_positions = np.random.choice(
                                num_sax_tokens, size=num_mask, replace=False
                            )
                        for t in self._mask_positions:
                            sax_seq[t] = 0  # reserved noisy token
                            sax_mask[t] = 1.0

                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))

            x = torch.stack(sax_features, dim=1)  # [T_sax, C]
            mask_ch = torch.tensor(sax_mask, dtype=torch.float32).unsqueeze(1)  # [T_sax, 1]
            x = torch.cat([x.float(), mask_ch], dim=1)  # [T_sax, C+1]

        else:
            # Non-SAX mode
            if raw_labels is not None:
                union_mask = np.any(raw_labels, axis=1).astype(np.float32)
            elif self.random_mask_ratio > 0:
                num_mask = max(1, int(T_raw * self.random_mask_ratio))
                mask_pos = np.random.choice(T_raw, size=num_mask, replace=False)
                union_mask = np.zeros(T_raw, dtype=np.float32)
                union_mask[mask_pos] = 1.0
                x_np = x.numpy()
                x_np[mask_pos, :] = 0.0
                x = torch.tensor(x_np, dtype=x.dtype)
            else:
                union_mask = np.zeros(T_raw, dtype=np.float32)

            mask_ch = torch.tensor(union_mask, dtype=torch.float32).unsqueeze(1)  # [T, 1]
            x = torch.cat([x.float(), mask_ch], dim=1)  # [T, C+1]

        y = data['label']
        return x, y


# ============================ Dataset ============================
class DSADSDataset(Dataset):
    def __init__(self, dataset_dict, modalities, cfg, transform=None, sax_params=None):
        self.transform = transform
        self.modalities = modalities
        self.base_sr = cfg.base_sample_rate
        self.duration = cfg.duration
        self.sampling_rates = cfg.sampling_rates
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 1}

        # Load samples and labels
        self.x_data = dataset_dict["samples"]
        self.y_data = dataset_dict["labels"]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Apply SAX if specified
        if self.transform == 'sax':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_tokenizer(
                    x[:, ch],
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]
        elif self.transform == 'sax_noisy':
            # 1-based tokens (1..alphabet_size), token 0 reserved for noisy/missing
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_noisy_tokenizer(
                    x[:, ch],
                    noisy_mask=None,
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]
        return x, y


# ============================ PAMAP2 Dataset ============================
class PAMAP2Dataset(Dataset):
    """
    PAMAP2 Dataset loader for pre-processed .pt split files.

    Each .pt file is a dict with keys:
        'IMU_hand':   (N, 512, 17)
        'IMU_chest':  (N, 512, 17)
        'IMU_ankle':  (N, 512, 17)
        'heart_rate': (N, 512,  1)
        'labels':     (N, 12) one-hot

    Returns (x, y) where x is [512, C] concatenated features and y is class index.
    """
    def __init__(self, data_dict, modalities, cfg, transform=None, sax_params=None, noise_config=None):
        self.modalities = modalities
        self.transform = transform
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 1}
        self.noise_config = noise_config

        # Store per-modality arrays as float32 numpy
        self.modal_data = {}
        for mod in modalities:
            arr = data_dict[mod]
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy()
            self.modal_data[mod] = np.asarray(arr, dtype=np.float32)

        # Convert one-hot labels to class indices
        labels = data_dict['labels']
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        labels = np.asarray(labels, dtype=np.float32)
        self.labels = labels.argmax(axis=1).astype(np.int64)
        self.n_samples = len(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Concatenate modalities along feature dim: [512, sum(variates)]
        parts = []
        for mod in self.modalities:
            parts.append(torch.tensor(self.modal_data[mod][idx], dtype=torch.float32))
        x = torch.cat(parts, dim=1)  # [T, C]

        # Apply noise if configured
        if self.noise_config is not None:
            noise_fn = self.noise_config['noise_fn']
            noise_kwargs = self.noise_config.get('noise_kwargs', {})
            x_np = x.numpy()
            noisy_channels = []
            for ch in range(x_np.shape[1]):
                corrupted, _label = noise_fn(x_np[:, ch], **noise_kwargs)
                noisy_channels.append(corrupted)
            x = torch.tensor(np.stack(noisy_channels, axis=1), dtype=x.dtype)

        # Apply SAX if specified
        if self.transform == 'sax':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_tokenizer(
                    x[:, ch].numpy() if isinstance(x, torch.Tensor) else x[:, ch],
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]
        elif self.transform == 'sax_noisy':
            # 1-based tokens (1..alphabet_size), token 0 reserved for noisy/missing
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_noisy_tokenizer(
                    x[:, ch].numpy() if isinstance(x, torch.Tensor) else x[:, ch],
                    noisy_mask=None,
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)  # [num_words, channels]

        y = self.labels[idx]
        return x, y


# ============================ DEAP Dataset ============================
class DEAPDataset(Dataset):
    """
    DEAP emotion recognition dataset.

    Loads preprocessed pickles (s01.dat .. s32.dat), strips baseline,
    segments each trial into fixed-length windows, and returns
    per-window samples with binary labels.
    """
    def __init__(self, data_dir, subjects, cfg, transform=None, sax_params=None, noise_config=None):
        self.transform = transform
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 1}
        self.noise_config = noise_config
        self.modalities = cfg.modalities
        self.channel_ranges = cfg.channel_ranges

        window_samples = cfg.window_sec * cfg.base_sample_rate
        stride_samples = cfg.stride_sec * cfg.base_sample_rate

        self.x_data = []
        self.labels = []

        for subj in subjects:
            subj_num = int(subj[1:])
            fpath = os.path.join(data_dir, f's{subj_num:02d}.dat')
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'rb') as f:
                d = pickle.load(f, encoding='latin1')

            data = d['data']       # (40, 40, 8064)
            labels = d['labels']   # (40, 4)

            for trial_idx in range(data.shape[0]):
                trial = data[trial_idx]  # (40, 8064)
                trial = trial[:, cfg.baseline_samples:]  # strip 3s baseline -> (40, 7680)
                trial = trial.T.astype(np.float32)  # -> (7680, 40)

                raw_score = labels[trial_idx, cfg.label_index]
                label = int(raw_score >= cfg.threshold)

                n_samples = trial.shape[0]
                for start in range(0, n_samples - window_samples + 1, stride_samples):
                    self.x_data.append(trial[start:start + window_samples])
                    self.labels.append(label)

        self.x_data = np.array(self.x_data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f'  DEAPDataset: {len(self.labels)} windows ({cfg.window_sec}s, stride {cfg.stride_sec}s), '
              f'{(self.labels == 1).sum()} high / {(self.labels == 0).sum()} low '
              f'({cfg.task} >= {cfg.threshold})')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_full = self.x_data[idx].copy()  # (T, 40)

        parts = []
        for mod in self.modalities:
            start, end = self.channel_ranges[mod]
            parts.append(torch.tensor(x_full[:, start:end], dtype=torch.float32))
        x = torch.cat(parts, dim=1)  # (T, 40)

        if self.noise_config is not None:
            noise_fn = self.noise_config['noise_fn']
            noise_kwargs = self.noise_config.get('noise_kwargs', {})
            x_np = x.numpy()
            noisy_channels = []
            for ch in range(x_np.shape[1]):
                corrupted, _label = noise_fn(x_np[:, ch], **noise_kwargs)
                noisy_channels.append(corrupted)
            x = torch.tensor(np.stack(noisy_channels, axis=1), dtype=x.dtype)

        if self.transform == 'sax':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_tokenizer(
                    x[:, ch].numpy() if isinstance(x, torch.Tensor) else x[:, ch],
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)
        elif self.transform == 'sax_noisy':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_noisy_tokenizer(
                    x[:, ch].numpy() if isinstance(x, torch.Tensor) else x[:, ch],
                    noisy_mask=None,
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)

        y = self.labels[idx]
        return x, y


# ============================ Opportunity Dataset ============================
class OpportunityDataset(Dataset):
    """
    Opportunity Activity Recognition dataset.

    Loads raw .dat files, applies sliding window, handles NaN (forward-fill +
    NaN mask tracking), and concatenates modalities.

    For 'sax_noisy' transform, NaN positions are passed as noisy_mask to
    sax_noisy_tokenizer, producing token 0 at naturally missing positions.
    For 'sax' transform, NaN is forward-filled and tokenized normally.

    Args:
        data_dir:   Path to OpportunityUCIDataset/dataset/ containing .dat files
        subjects:   List of subject IDs, e.g. ['S1', 'S2', 'S3']
        sessions:   List of session names, e.g. ['ADL1', 'ADL2', 'ADL3', 'Drill']
        cfg:        Opportunity config object
        transform:  'sax', 'sax_noisy', or None
        sax_params: dict with 'alphabet_size' and 'word_length'
        noise_config: optional noise injection config (applied after imputation)
    """
    def __init__(self, data_dir, subjects, sessions, cfg, transform=None,
                 sax_params=None, noise_config=None):
        self.transform = transform
        self.sax_params = sax_params or {'alphabet_size': 20, 'word_length': 1}
        self.noise_config = noise_config
        self.label_map = cfg.label_map
        self.modalities = cfg.modalities

        # Build flat list of sensor column indices in modality order
        sensor_cols = []
        for mod in cfg.modalities:
            start, end = cfg.sensor_col_ranges[mod]
            sensor_cols.extend(range(start, end))
        self.sensor_cols = sensor_cols

        window_size = cfg.window_size
        stride = cfg.stride
        label_col = cfg.label_col

        # Load all files and extract windows
        self.x_data = []       # list of np.array [window_size, C]
        self.nan_masks = []    # list of np.array [window_size, C] bool
        self.labels = []       # list of int (0-based class index)

        for subj in subjects:
            for sess in sessions:
                fpath = os.path.join(data_dir, f'{subj}-{sess}.dat')
                if not os.path.exists(fpath):
                    continue
                raw = np.loadtxt(fpath)

                # Extract sensor data and labels
                sensor_data = raw[:, sensor_cols].astype(np.float32)
                lab_arr = raw[:, label_col]
                lab_arr = np.where(np.isnan(lab_arr), 0, lab_arr).astype(int)

                # Track NaN positions before imputation
                nan_mask = np.isnan(sensor_data)

                # Forward-fill NaN column-wise, then zero-fill remaining
                df_sensor = pd.DataFrame(sensor_data)
                df_sensor = df_sensor.ffill().fillna(0)
                sensor_data = df_sensor.values.astype(np.float32)

                # Sliding window with majority-vote labeling
                n = len(sensor_data)
                for start in range(0, n - window_size + 1, stride):
                    end = start + window_size
                    label_seg = lab_arr[start:end]

                    # Majority vote excluding NULL (0)
                    non_null = label_seg[label_seg != 0]
                    if len(non_null) == 0:
                        continue  # skip NULL-only windows

                    from collections import Counter as _Counter
                    majority_label = _Counter(non_null).most_common(1)[0][0]

                    # Only keep windows with known labels
                    if majority_label not in self.label_map:
                        continue

                    self.x_data.append(sensor_data[start:end])
                    self.nan_masks.append(nan_mask[start:end])
                    self.labels.append(self.label_map[majority_label])

        self.x_data = np.array(self.x_data, dtype=np.float32)
        self.nan_masks = np.array(self.nan_masks, dtype=bool)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f'  OpportunityDataset: {len(self.labels)} windows, '
              f'{len(set(self.labels))} classes, '
              f'NaN rate={self.nan_masks.mean()*100:.1f}%')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.x_data[idx].copy()           # [T, C]
        nan_mask = self.nan_masks[idx].copy()  # [T, C] bool

        x = torch.tensor(x, dtype=torch.float32)

        # Apply noise injection if configured (on imputed signal, before SAX)
        if self.noise_config is not None:
            noise_fn = self.noise_config['noise_fn']
            noise_kwargs = self.noise_config.get('noise_kwargs', {})
            x_np = x.numpy()
            noisy_channels = []
            for ch in range(x_np.shape[1]):
                corrupted, _label = noise_fn(x_np[:, ch], **noise_kwargs)
                noisy_channels.append(corrupted)
            x = torch.tensor(np.stack(noisy_channels, axis=1), dtype=x.dtype)

        # Apply SAX tokenization
        if self.transform == 'sax':
            sax_features = []
            for ch in range(x.shape[1]):
                sax_seq = sax_tokenizer(
                    x[:, ch].numpy(),
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)
        elif self.transform == 'sax_noisy':
            # 1-based tokens; token 0 = noisy/missing (includes NaN positions)
            sax_features = []
            for ch in range(x.shape[1]):
                ch_nan_mask = nan_mask[:, ch]
                sax_seq = sax_noisy_tokenizer(
                    x[:, ch].numpy(),
                    noisy_mask=ch_nan_mask,
                    alphabet_size=self.sax_params['alphabet_size'],
                    word_length=self.sax_params['word_length']
                )
                sax_features.append(torch.tensor(sax_seq, dtype=torch.long))
            x = torch.stack(sax_features, dim=1)

        y = self.labels[idx]
        return x, y
