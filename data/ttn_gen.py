'''
Created by Payal Mohapatra on 14 Jan 2026.

Source : https://github.com/datamllab/tods/tree/benchmark/benchmark/synthetic/Generator

Noise Taxonomy
==============
LEVEL I  — Token-Level Shifts (input-space, transient)
  I-1  Additive Gaussian Noise           inject_gaussian
  I-2  Point-Global Spike                inject_point_global
  I-3  Point-Contextual Anomaly          inject_point_contextual
  I-4  Discrete Missing (Dropout)        inject_discrete_missing
  I-5  Quantization / Bit-depth Reduc.   inject_quantization
  I-6  Segment Zero-out (Burst Dropout)  inject_pattern_zero

LEVEL II(a) — Sample-Level Shifts, Input-Space Detectable
  II-a1 Sensor Drift (Baseline Wander)   inject_sensor_drift
  II-a3 Time Axis Warp                   inject_time_warp
  II-a4 Pattern Trend Injection          inject_pattern_trend
  II-a5 Pattern Seasonal Injection       inject_pattern_seasonal
  II-a6 Continuous Missing (Block Drop.) inject_block_missing
  II-a7 Shapelet / Waveform Replacement  inject_pattern_shapelet / inject_waveform_replacement

LEVEL II(b) — Sample-Level Shifts, Latent-Space Detectable
  II-b2 Nonlinear Sensor Degradation     inject_nonlinear_degradation
  II-b3 Freq-Selective Attenuation       inject_freq_selective_attenuation
  II-b4 Nonstationary Noise Floor        inject_nonstationary_noise_floor

Usage:
    1. Generate clean samples using SyntheticDatasetGenerator (synthdata_gen.py)
    2. Use NoiseInjector to read clean samples and inject stochastic noise
    3. Per-channel functions accept 1D array → return (corrupted, label)
'''

import os
import json
import numpy as np
import torch
try:
    from data.synthdata_gen import sine, cosine, square_sine
except ImportError:
    from synthdata_gen import sine, cosine, square_sine


# =============================================================================
# LEVEL I — Token-Level Shifts (per-channel, 1D input → (corrupted, label))
# =============================================================================

def inject_gaussian(data, snr_db=20):
    """
    I-1: Additive Gaussian Noise — thermal / ADC quantization floor.

    Args:
        data: 1D array of shape (seq_length,)
        snr_db: target signal-to-noise ratio in dB

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    sig_power = np.mean(data ** 2) + 1e-12
    noise_std = np.sqrt(sig_power / (10 ** (snr_db / 10)))
    data += np.random.randn(len(data)) * noise_std
    return data, np.ones(len(data), dtype=int)


def inject_discrete_missing(data, ratio=0.05):
    """
    I-4: Discrete Missing (Dropout) — scattered NaN / zero replacements
    simulating packet loss or buffer overflow.

    Args:
        data: 1D array of shape (seq_length,)
        ratio: fraction of individual samples to zero out

    Returns:
        tuple: (corrupted_data, label array)
    """
    data = data.copy()
    length = len(data)
    label = np.zeros(length, dtype=int)
    num_drop = max(1, round(length * ratio))
    positions = np.random.choice(length, size=num_drop, replace=False)
    data[positions] = 0.0
    label[positions] = 1
    return data, label


def inject_point_global(data, ratio=0.05, factor=3.5, radius=5):
    """
    I-2: Point-Global Spike — extreme values exceeding data range.

    Args:
        data: 1D array of shape (seq_length,)
        ratio: fraction of points to corrupt
        factor: outlier magnitude multiplier
        radius: local window for std computation

    Returns:
        tuple: (corrupted_data, label array)
    """
    data = data.copy()
    data_origin = data.copy()
    length = len(data)
    label = np.zeros(length, dtype=int)

    num_outliers = max(1, round(length * ratio))
    position = np.random.choice(length, size=num_outliers, replace=False)
    maximum, minimum = np.max(data), np.min(data)

    for i in position:
        local_std = data_origin[max(0, i - radius):min(i + radius, length)].std()
        data[i] = data_origin[i] * factor * local_std
        if 0 <= data[i] < maximum:
            data[i] = maximum
        if 0 > data[i] > minimum:
            data[i] = minimum
        label[i] = 1

    return data, label


def inject_point_contextual(data, ratio=0.05, factor=2.5, radius=5):
    """
    I-3: Point-Contextual Anomaly — within range but locally abnormal.

    Args:
        data: 1D array of shape (seq_length,)
        ratio: fraction of points to corrupt
        factor: outlier magnitude multiplier
        radius: local window for std computation

    Returns:
        tuple: (corrupted_data, label array)
    """
    data = data.copy()
    data_origin = data.copy()
    length = len(data)
    label = np.zeros(length, dtype=int)

    num_outliers = max(1, round(length * ratio))
    position = np.random.choice(length, size=num_outliers, replace=False)
    maximum, minimum = np.max(data), np.min(data)

    for i in position:
        local_std = data_origin[max(0, i - radius):min(i + radius, length)].std()
        data[i] = data_origin[i] * factor * local_std
        if data[i] > maximum:
            data[i] = maximum * min(0.95, abs(np.random.normal(0, 1)))
        if data[i] < minimum:
            data[i] = minimum * min(0.95, abs(np.random.normal(0, 1)))
        label[i] = 1

    return data, label


def inject_pattern_shapelet(data, ratio=0.05, radius=5, coef=3.0, noise_amp=0.0,
                             level=5, freq=0.04, offset=0.0):
    """
    II-a7: Shapelet / Waveform Replacement — substitute with square wave.

    Args:
        data: 1D array of shape (seq_length,)
        ratio: fraction of data to corrupt
        radius: half-width of corrupted segments
        coef, noise_amp, level, freq, offset: params for square wave

    Returns:
        tuple: (corrupted_data, label array)
    """
    data = data.copy()
    length = len(data)
    label = np.zeros(length, dtype=int)

    num_segments = max(1, round(length * ratio / (2 * radius)))
    position = np.random.choice(length, size=num_segments, replace=False)

    sub_data = square_sine(level=level, length=length, freq=freq,
                           coef=coef, offset=offset, noise_amp=noise_amp)

    for i in position:
        start, end = max(0, i - radius), min(length, i + radius)
        data[start:end] = sub_data[start:end]
        label[start:end] = 1

    return data, label


def inject_pattern_trend(data, ratio=0.05, factor=0.5, radius=5):
    """
    II-a4: Pattern Trend Injection — linear trend shifts.

    Args:
        data: 1D array of shape (seq_length,)
        ratio: fraction of data to corrupt
        factor: slope magnitude
        radius: half-width of corrupted segments

    Returns:
        tuple: (corrupted_data, label array)
    """
    data = data.copy()
    data_origin = data.copy()
    length = len(data)
    label = np.zeros(length, dtype=int)

    num_segments = max(1, round(length * ratio / (2 * radius)))
    position = np.random.choice(length, size=num_segments, replace=False)

    for i in position:
        start, end = max(0, i - radius), min(length, i + radius)
        slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
        data[start:end] = data_origin[start:end] + slope
        label[start:end] = 1
        # The persistent level shift after the ramp is also corrupted
        if end < length:
            data[end:] = data[end:] + slope[-1]
            label[end:] = 1

    return data, label


def inject_pattern_zero(data, ratio=0.05, radius=5):
    """
    I-6: Segment Zero-out (Burst Dropout) — replace subsequences with zeros.

    Args:
        data: 1D array of shape (seq_length,)
        ratio: fraction of data to corrupt
        radius: half-width of corrupted segments

    Returns:
        tuple: (corrupted_data, label array)
    """
    data = data.copy()
    length = len(data)
    label = np.zeros(length, dtype=int)

    num_segments = max(1, round(length * ratio / (2 * radius)))
    position = np.random.choice(length, size=num_segments, replace=False)

    for i in position:
        start, end = max(0, i - radius), min(length, i + radius)
        data[start:end] = 0.0
        label[start:end] = 1

    return data, label


def inject_pattern_seasonal(data, ratio=0.05, factor=3, radius=5, behavior_config=None):
    """
    II-a5: Pattern Seasonal Injection — periodic interference / frequency changes.

    Args:
        data: 1D array of shape (seq_length,)
        ratio: fraction of data to corrupt
        factor: frequency multiplier
        radius: half-width of corrupted segments
        behavior_config: config for signal generation (must have 'freq')

    Returns:
        tuple: (corrupted_data, label array)
    """
    if behavior_config is None:
        behavior_config = {'freq': 0.04, 'coef': 1.5, 'offset': 0.0, 'noise_amp': 0.05}

    data = data.copy()
    length = len(data)
    label = np.zeros(length, dtype=int)

    num_segments = max(1, round(length * ratio / (2 * radius)))
    position = np.random.choice(length, size=num_segments, replace=False)

    seasonal_config = behavior_config.copy()
    seasonal_config['freq'] = factor * behavior_config.get('freq', 0.04)
    seasonal_config['length'] = length

    seasonal_signal = sine(**seasonal_config)

    for i in position:
        start, end = max(0, i - radius), min(length, i + radius)
        data[start:end] = seasonal_signal[start:end]
        label[start:end] = 1

    return data, label


def inject_quantization(data, num_bits=4):
    """
    I-5: Quantization / Bit-depth Reduction — snap values to a uniform grid.

    Args:
        data: 1D array of shape (seq_length,)
        num_bits: effective bit depth (e.g. 4 → 16 levels, 8 → 256 levels)

    Returns:
        tuple: (quantized_data, label array with all ones)
    """
    data = data.copy()
    num_levels = 2 ** num_bits
    d_min, d_max = data.min(), data.max()
    d_range = d_max - d_min
    if d_range < 1e-12:
        return data, np.ones(len(data), dtype=int)
    step = d_range / (num_levels - 1)
    data = np.round((data - d_min) / step) * step + d_min
    return data, np.ones(len(data), dtype=int)


# =============================================================================
# LEVEL II(a) — Sample-Level Shifts, Input-Space Detectable (per-channel)
# =============================================================================

def inject_sensor_drift(data, slope_range=(0.01, 0.05), offset_range=(-0.5, 0.5)):
    """
    II-a1: Sensor Drift (Baseline Wander) — electrode degradation, sweat,
    temperature-induced bias in MEMS.

    Args:
        data: 1D array of shape (seq_length,)
        slope_range: uniform range for drift slope
        offset_range: uniform range for DC offset

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    a = np.random.uniform(*slope_range) * np.random.choice([-1, 1])
    b = np.random.uniform(*offset_range)
    data += a * np.linspace(0, 1, len(data)) + b
    return data, np.ones(len(data), dtype=int)


def inject_time_warp(data, alpha_range=(0.8, 1.2), beta_range=(-0.1, 0.1)):
    """
    II-a3: Time Axis Warp — clock drift, resampling artifacts, firmware
    timing error.

    Args:
        data: 1D array of shape (seq_length,)
        alpha_range: uniform range for time-axis scaling
        beta_range: uniform range for time-axis shift (fraction of length)

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    n = len(data)
    alpha = np.random.uniform(*alpha_range)
    beta = np.random.uniform(*beta_range)
    t_warped = np.clip(alpha * np.arange(n) + beta * n, 0, n - 1)
    return np.interp(t_warped, np.arange(n), data), np.ones(n, dtype=int)


def inject_block_missing(data, missing_fraction=0.3):
    """
    II-a6: Continuous Missing (Block Dropout) — sensor detachment,
    device removal, power failure.

    Args:
        data: 1D array of shape (seq_length,)
        missing_fraction: fraction of signal to zero out as one contiguous block

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    bl = int(missing_fraction * len(data))
    s = np.random.randint(0, len(data) - bl)
    data[s:s + bl] = 0.0
    return data, np.ones(len(data), dtype=int)


def inject_waveform_replacement(data, replacement_fraction=0.2, scale=2.0):
    """
    II-a7: Waveform Replacement — motion artifact overwriting a segment,
    cross-talk from nearby sensor.

    Args:
        data: 1D array of shape (seq_length,)
        replacement_fraction: fraction of signal to replace
        scale: amplitude of injected sinusoid

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    bl = int(replacement_fraction * len(data))
    s = np.random.randint(0, len(data) - bl)
    data[s:s + bl] = scale * np.sin(2 * np.pi * np.linspace(0, 5, bl))
    return data, np.ones(len(data), dtype=int)


# =============================================================================
# LEVEL II(b) — Sample-Level Shifts, Latent-Space Detectable (per-channel)
# =============================================================================

def inject_nonlinear_degradation(data, clip_percentile=80, compression=0.5):
    """
    II-b2: Nonlinear Sensor Degradation — LED aging in PPG, electrode gel
    drying, MEMS spring fatigue.  Clips then compresses amplitude.

    Args:
        data: 1D array of shape (seq_length,)
        clip_percentile: percentile of |data| used as clipping threshold
        compression: exponent for power-law compression (< 1 = compress)

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    thresh = np.percentile(np.abs(data), clip_percentile)
    data = np.clip(data, -thresh, thresh)
    data = np.sign(data) * np.abs(data) ** compression
    return data, np.ones(len(data), dtype=int)


def inject_freq_selective_attenuation(data, cutoff_frac=0.3, attenuation_db=20, fs=32):
    """
    II-b3: Frequency-Selective Attenuation — low-pass filtering from
    skin-electrode interface, anti-aliasing mismatch, cable shielding failure.

    Args:
        data: 1D array of shape (seq_length,)
        cutoff_frac: cutoff as fraction of Nyquist (0-1)
        attenuation_db: suppression above cutoff in dB
        fs: sampling rate (Hz)

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    n = len(data)
    X = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    cutoff = cutoff_frac * fs / 2.0
    mask = np.ones_like(freqs)
    mask[freqs > cutoff] = 10 ** (-attenuation_db / 20.0)
    data = np.fft.irfft(X * mask, n=n)
    return data, np.ones(n, dtype=int)


def inject_nonstationary_noise_floor(data, snr_start_db=30, snr_end_db=5):
    """
    II-b4: Nonstationary Noise Floor — environmental change, sensor aging,
    battery voltage drop.  SNR degrades linearly over the window.

    Args:
        data: 1D array of shape (seq_length,)
        snr_start_db: SNR at beginning of window
        snr_end_db: SNR at end of window

    Returns:
        tuple: (corrupted_data, label array with all ones)
    """
    data = data.copy()
    n = len(data)
    sig_power = np.mean(data ** 2) + 1e-10
    snr = np.linspace(snr_start_db, snr_end_db, n)
    noise_std = np.sqrt(sig_power / (10 ** (snr / 10)))
    data += np.random.randn(n) * noise_std
    return data, np.ones(n, dtype=int)


# =============================================================================
# Noise Function Registry
# =============================================================================

# Taxonomy-aligned keys (preferred for new code)
NOISE_FUNCTIONS = {
    # Level I — Token-Level
    'I1_gaussian':             inject_gaussian,
    'I2_point_global':         inject_point_global,
    'I3_point_contextual':     inject_point_contextual,
    'I4_discrete_missing':     inject_discrete_missing,
    'I5_quantization':         inject_quantization,
    'I6_segment_zero':         inject_pattern_zero,
    # Level II(a) — Sample-Level, Input-Space
    'IIa1_sensor_drift':       inject_sensor_drift,
    'IIa3_time_warp':          inject_time_warp,
    'IIa4_pattern_trend':      inject_pattern_trend,
    'IIa5_pattern_seasonal':   inject_pattern_seasonal,
    'IIa6_block_missing':      inject_block_missing,
    'IIa7_shapelet':           inject_pattern_shapelet,
    'IIa7_waveform_replace':   inject_waveform_replacement,
    # Level II(b) — Sample-Level, Latent-Space
    'IIb2_nonlinear_degrad':   inject_nonlinear_degradation,
    'IIb3_freq_attenuation':   inject_freq_selective_attenuation,
    'IIb4_nonstat_noise':      inject_nonstationary_noise_floor,
}

# Backward-compatible aliases (used by existing experiment scripts)
NOISE_FUNCTIONS.update({
    'point_global':     inject_point_global,
    'point_contextual': inject_point_contextual,
    'pattern_shapelet': inject_pattern_shapelet,
    'pattern_zero':     inject_pattern_zero,
    'pattern_trend':    inject_pattern_trend,
    'pattern_seasonal': inject_pattern_seasonal,
    'quantization':     inject_quantization,
})


# =============================================================================
# Noise Injector Class
# =============================================================================

class NoiseInjector:
    """
    Read clean samples from a directory and inject stochastic noise.

    Reads config.json from input directory to learn behavior settings.
    Injects random noise per sample and saves to output directory.
    """

    def __init__(self, input_dir, seed=None):
        """
        Initialize the noise injector.

        Args:
            input_dir: Directory containing clean samples and config.json
            seed: Random seed for reproducibility
        """
        self.input_dir = input_dir
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Load config from input directory
        config_path = os.path.join(input_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.num_samples = self.config['num_samples']
        self.seq_length = self.config['seq_length']
        self.num_variates = self.config['num_variates']
        self.prefix = self.config.get('prefix', 'sample')
        self.behavior_config = self.config.get('behavior_config', [{}] * self.num_variates)

    def _load_sample(self, idx):
        """Load a single sample by index."""
        filename = f'{self.prefix}_{idx:06d}.npy'
        return np.load(os.path.join(self.input_dir, filename))

    def _inject_noise_to_variate(self, data, variate_idx, noise_type, noise_params):
        """
        Inject noise into a single variate.

        Args:
            data: 1D array of shape (seq_length,)
            variate_idx: index of the variate (for behavior_config lookup)
            noise_type: type of noise to inject
            noise_params: dict of parameters for the noise function

        Returns:
            tuple: (corrupted_data, label array)
        """
        if noise_type not in NOISE_FUNCTIONS:
            raise ValueError(f"Unknown noise type: {noise_type}. "
                           f"Available: {list(NOISE_FUNCTIONS.keys())}")

        kwargs = noise_params.copy()

        # For pattern_seasonal, pass behavior_config
        if noise_type == 'pattern_seasonal':
            kwargs.setdefault('behavior_config', self.behavior_config[variate_idx])

        return NOISE_FUNCTIONS[noise_type](data, **kwargs)

    def inject_and_save(self, output_dir, noise_config, prefix='noisy', correlated=False):
        """
        Inject stochastic noise into all samples and save to output directory.

        Args:
            output_dir: Directory to save noisy samples
            noise_config: Noise configuration. Can be:
                - dict: apply same noise to all variates
                  {'type': 'point_global', 'ratio': 0.05, 'factor': 3.5, ...}
                - list of dicts: one config per variate (use None for clean variates)
                  [{'type': 'point_global', ...}, None, {'type': 'pattern_trend', ...}]
            prefix: Filename prefix for noisy samples (default: 'noisy')
            correlated: If True, inject noise at same positions across all variates.
                        If False, inject noise at different random positions per variate.

        Output structure:
            output_dir/
            ├── noisy_000000.npy     # shape: (V, T)
            ├── noisy_000001.npy
            ├── ...
            ├── labels.npy           # shape: (N, V, T) - per-sample per-variate noise labels
            └── config.json          # includes noise_config
        """
        os.makedirs(output_dir, exist_ok=True)

        # Normalize noise_config to list
        if noise_config is None:
            noise_cfg_list = [None] * self.num_variates
        elif isinstance(noise_config, dict):
            noise_cfg_list = [noise_config.copy() for _ in range(self.num_variates)]
        else:
            noise_cfg_list = [cfg.copy() if cfg else None for cfg in noise_config]

        all_labels = []

        # Process each sample
        for i in range(self.num_samples):
            sample = self._load_sample(i)
            noisy_sample = np.zeros_like(sample)
            sample_label = np.zeros((self.num_variates, self.seq_length), dtype=int)

            if correlated:
                # Generate shared positions for all variates (per sample)
                # Use the first non-None noise config to determine positions
                shared_positions = self._generate_shared_positions(noise_cfg_list)

            # Inject noise per variate
            for v in range(self.num_variates):
                if noise_cfg_list[v] is None:
                    noisy_sample[v] = sample[v]
                else:
                    noise_type = noise_cfg_list[v].get('type', 'point_global')
                    noise_params = {k: val for k, val in noise_cfg_list[v].items() if k != 'type'}

                    if correlated and noise_type in shared_positions:
                        # Use shared positions for correlated noise
                        noisy_data, variate_label = self._inject_noise_with_positions(
                            sample[v], v, noise_type, noise_params, shared_positions[noise_type]
                        )
                    else:
                        # Independent positions per variate
                        noisy_data, variate_label = self._inject_noise_to_variate(
                            sample[v], v, noise_type, noise_params
                        )
                    noisy_sample[v] = noisy_data
                    sample_label[v] = variate_label

            # Save noisy sample
            filename = f'{prefix}_{i:06d}.npy'
            np.save(os.path.join(output_dir, filename), noisy_sample)
            all_labels.append(sample_label)

        # Save labels
        labels_array = np.stack(all_labels, axis=0)
        np.save(os.path.join(output_dir, 'labels.npy'), labels_array)

        # Save config (inherit from input + add noise info)
        out_config = self.config.copy()
        out_config['prefix'] = prefix
        out_config['source_dir'] = self.input_dir
        out_config['noise_seed'] = self.seed
        out_config['correlated'] = correlated
        out_config['noise_config'] = [
            {k: (float(val) if isinstance(val, (int, float)) else val) for k, val in cfg.items()}
            if cfg else None
            for cfg in noise_cfg_list
        ]
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(out_config, f, indent=2)

        print(f"Injected noise into {self.num_samples} samples, saved to {output_dir}/")
        print(f"  Shape per sample: ({self.num_variates}, {self.seq_length})")
        print(f"  Labels shape: {labels_array.shape}")
        print(f"  Correlated noise: {correlated}")

    def _generate_shared_positions(self, noise_cfg_list):
        """
        Generate shared noise positions for correlated injection.

        All variates will use the same positions, regardless of noise type.
        Uses the first non-None config to determine number of positions.
        """
        # Find first non-None config to get ratio/radius
        first_cfg = None
        for cfg in noise_cfg_list:
            if cfg is not None:
                first_cfg = cfg
                break

        if first_cfg is None:
            return None

        ratio = first_cfg.get('ratio', 0.05)
        radius = first_cfg.get('radius', 5)
        noise_type = first_cfg.get('type', 'point_global')

        # Generate single set of positions for all variates
        if noise_type in ['point_global', 'point_contextual']:
            num_positions = max(1, round(self.seq_length * ratio))
        else:  # pattern-based noise
            num_positions = max(1, round(self.seq_length * ratio / (2 * radius)))

        positions = np.random.choice(self.seq_length, size=num_positions, replace=False)
        return positions

    def _inject_noise_with_positions(self, data, variate_idx, noise_type, noise_params, positions):
        """Inject noise at specified positions (for correlated noise)."""
        if noise_type == 'point_global':
            return self._inject_point_global_at_positions(data, positions, noise_params)
        elif noise_type == 'point_contextual':
            return self._inject_point_contextual_at_positions(data, positions, noise_params)
        elif noise_type == 'pattern_shapelet':
            return self._inject_pattern_shapelet_at_positions(data, positions, noise_params)
        elif noise_type == 'pattern_zero':
            return self._inject_pattern_zero_at_positions(data, positions, noise_params)
        elif noise_type == 'pattern_trend':
            return self._inject_pattern_trend_at_positions(data, positions, noise_params)
        elif noise_type == 'pattern_seasonal':
            noise_params.setdefault('behavior_config', self.behavior_config[variate_idx])
            return self._inject_pattern_seasonal_at_positions(data, positions, noise_params)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def _inject_point_global_at_positions(self, data, positions, params):
        """Inject point global outliers at specified positions."""
        data = data.copy()
        data_origin = data.copy()
        length = len(data)
        label = np.zeros(length, dtype=int)

        factor = params.get('factor', 3.5)
        radius = params.get('radius', 5)
        maximum, minimum = np.max(data), np.min(data)

        for i in positions:
            local_std = data_origin[max(0, i - radius):min(i + radius, length)].std()
            data[i] = data_origin[i] * factor * local_std
            if 0 <= data[i] < maximum:
                data[i] = maximum
            if 0 > data[i] > minimum:
                data[i] = minimum
            label[i] = 1

        return data, label

    def _inject_point_contextual_at_positions(self, data, positions, params):
        """Inject point contextual outliers at specified positions."""
        data = data.copy()
        data_origin = data.copy()
        length = len(data)
        label = np.zeros(length, dtype=int)

        factor = params.get('factor', 2.5)
        radius = params.get('radius', 5)
        maximum, minimum = np.max(data), np.min(data)

        for i in positions:
            local_std = data_origin[max(0, i - radius):min(i + radius, length)].std()
            data[i] = data_origin[i] * factor * local_std
            if data[i] > maximum:
                data[i] = maximum * min(0.95, abs(np.random.normal(0, 1)))
            if data[i] < minimum:
                data[i] = minimum * min(0.95, abs(np.random.normal(0, 1)))
            label[i] = 1

        return data, label

    def _inject_pattern_shapelet_at_positions(self, data, positions, params):
        """Inject pattern shapelet at specified positions."""
        data = data.copy()
        length = len(data)
        label = np.zeros(length, dtype=int)

        radius = params.get('radius', 5)
        coef = params.get('coef', 3.0)
        noise_amp = params.get('noise_amp', 0.0)
        level = params.get('level', 5)
        freq = params.get('freq', 0.04)
        offset = params.get('offset', 0.0)

        sub_data = square_sine(level=level, length=length, freq=freq,
                               coef=coef, offset=offset, noise_amp=noise_amp)

        for i in positions:
            start, end = max(0, i - radius), min(length, i + radius)
            data[start:end] = sub_data[start:end]
            label[start:end] = 1

        return data, label

    def _inject_pattern_zero_at_positions(self, data, positions, params):
        """Inject pattern zero at specified positions."""
        data = data.copy()
        length = len(data)
        label = np.zeros(length, dtype=int)

        radius = params.get('radius', 5)

        for i in positions:
            start, end = max(0, i - radius), min(length, i + radius)
            data[start:end] = 0.0
            label[start:end] = 1

        return data, label

    def _inject_pattern_trend_at_positions(self, data, positions, params):
        """Inject pattern trend at specified positions."""
        data = data.copy()
        data_origin = data.copy()
        length = len(data)
        label = np.zeros(length, dtype=int)

        factor = params.get('factor', 0.5)
        radius = params.get('radius', 5)

        for i in positions:
            start, end = max(0, i - radius), min(length, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            data[start:end] = data_origin[start:end] + slope
            if end < length:
                data[end:] = data[end:] + slope[-1]
            label[start:end] = 1
            if end < length and slope[-1] != 0:
                label[end:] = 1

        return data, label

    def _inject_pattern_seasonal_at_positions(self, data, positions, params):
        """Inject pattern seasonal at specified positions."""
        data = data.copy()
        length = len(data)
        label = np.zeros(length, dtype=int)

        factor = params.get('factor', 3)
        radius = params.get('radius', 5)
        behavior_config = params.get('behavior_config',
                                      {'freq': 0.04, 'coef': 1.5, 'offset': 0.0, 'noise_amp': 0.05})

        seasonal_config = behavior_config.copy()
        seasonal_config['freq'] = factor * behavior_config.get('freq', 0.04)
        seasonal_config['length'] = length

        seasonal_signal = sine(**seasonal_config)

        for i in positions:
            start, end = max(0, i - radius), min(length, i + radius)
            data[start:end] = seasonal_signal[start:end]
            label[start:end] = 1

        return data, label

    @staticmethod
    def load_noisy_dataset(input_dir):
        """
        Load noisy dataset from a directory.

        Args:
            input_dir: Directory containing noisy samples, labels.npy, and config.json

        Returns:
            tuple: (dataset array (N, V, T), labels array (N, V, T), config dict)
        """
        with open(os.path.join(input_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        prefix = config.get('prefix', 'noisy')
        samples = []
        for i in range(config['num_samples']):
            filename = f'{prefix}_{i:06d}.npy'
            samples.append(np.load(os.path.join(input_dir, filename)))

        dataset = np.stack(samples, axis=0)
        labels = np.load(os.path.join(input_dir, 'labels.npy'))

        return dataset, labels, config


# =============================================================================
# Multivariate Dataset Generator with Train/Val/Test Splits
# =============================================================================

class MultivariateDatasetGenerator:
    """
    Generate a complete multivariate synthetic dataset with train/val/test splits.

    Test set includes clean samples and noisy versions for all noise types.
    Noisy test samples are generated from the exact same clean test samples.

    All data is saved as single .pt files for fast loading.

    Directory structure:
        output_dir/
        ├── config.json          # Master config
        ├── train/
        │   └── data.pt          # {'data': (N_train, V, T), 'config': {...}}
        ├── val/
        │   └── data.pt          # {'data': (N_val, V, T), 'config': {...}}
        └── test/
            ├── clean/
            │   └── data.pt      # {'data': (N_test, V, T), 'config': {...}}
            ├── noise_point_global/
            │   └── data.pt      # {'data': (N_test, V, T), 'labels': (N_test, V, T), 'config': {...}}
            ├── noise_point_contextual/
            │   └── data.pt
            ├── noise_pattern_shapelet/
            │   └── data.pt
            ├── noise_pattern_trend/
            │   └── data.pt
            └── noise_pattern_seasonal/
                └── data.pt
    """

    DEFAULT_OUTPUT_DIR = '/home/payal/maestro_ttn_robustness/data/synth_data/basic_multivariate'

    def __init__(
        self,
        num_samples,
        seq_length=200,
        num_variates=4,
        behavior=None,
        behavior_config=None,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=42,
        noise_params=None
    ):
        """
        Initialize the multivariate dataset generator.

        Args:
            num_samples: Total number of samples to generate (will be split 70/10/20)
            seq_length: Sequence length (T) of each sample
            num_variates: Number of variates/channels (V) per sample
            behavior: Signal generation function(s). None = alternating sine/cosine
            behavior_config: Config dict(s) for behavior function(s)
            train_ratio: Fraction of samples for training (default: 0.7)
            val_ratio: Fraction of samples for validation (default: 0.1)
            test_ratio: Fraction of samples for testing (default: 0.2)
            seed: Random seed for reproducibility
            noise_params: Dict of noise parameters per noise type. Example:
                {
                    'point_global': {'ratio': 0.05, 'factor': 3.5, 'radius': 5},
                    'point_contextual': {'ratio': 0.05, 'factor': 2.5, 'radius': 5},
                    ...
                }
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"

        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_variates = num_variates
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Calculate split sizes
        self.num_train = int(num_samples * train_ratio)
        self.num_val = int(num_samples * val_ratio)
        self.num_test = num_samples - self.num_train - self.num_val

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Setup behaviors (default: alternating sine/cosine)
        if behavior is None:
            self.behavior = [sine if i % 2 == 0 else cosine for i in range(num_variates)]
        elif callable(behavior):
            self.behavior = [behavior] * num_variates
        else:
            self.behavior = behavior

        # Setup behavior configs
        default_config = {'freq': 0.04, 'coef': 1.5, 'offset': 0.0, 'noise_amp': 0.05}
        if behavior_config is None:
            self.behavior_config = [default_config.copy() for _ in range(num_variates)]
        elif isinstance(behavior_config, dict):
            self.behavior_config = [behavior_config.copy() for _ in range(num_variates)]
        else:
            self.behavior_config = [cfg.copy() for cfg in behavior_config]

        # Default noise parameters for each noise type
        self.noise_params = noise_params or {
            'point_global': {'ratio': 0.05, 'factor': 3.5, 'radius': 5},
            'point_contextual': {'ratio': 0.05, 'factor': 2.5, 'radius': 5},
            'pattern_shapelet': {'ratio': 0.05, 'radius': 5, 'coef': 3.0, 'level': 5, 'freq': 0.04},
            'pattern_zero': {'ratio': 0.05, 'radius': 5},
            'pattern_trend': {'ratio': 0.05, 'factor': 0.5, 'radius': 5},
            'pattern_seasonal': {'ratio': 0.05, 'factor': 3, 'radius': 5},
        }

    def _generate_sample(self):
        """Generate a single clean sample of shape (num_variates, seq_length)."""
        data = np.zeros((self.num_variates, self.seq_length))
        for v in range(self.num_variates):
            config = self.behavior_config[v].copy()
            config['length'] = self.seq_length
            data[v] = self.behavior[v](**config)
        return data

    def _save_samples(self, samples, output_dir):
        """Save list of samples as a single .pt file."""
        os.makedirs(output_dir, exist_ok=True)

        # Stack samples into single tensor: (N, V, T)
        data_array = np.stack(samples, axis=0)
        data_tensor = torch.from_numpy(data_array).float()

        # Save config
        config = {
            'num_samples': len(samples),
            'seq_length': self.seq_length,
            'num_variates': self.num_variates,
            'seed': self.seed,
            'behavior_config': [
                {k: float(v) for k, v in cfg.items()}
                for cfg in self.behavior_config
            ]
        }

        # Save as single .pt file with data and config
        torch.save({
            'data': data_tensor,
            'config': config
        }, os.path.join(output_dir, 'data.pt'))

        return config

    def _inject_noise_to_samples(self, samples, noise_type, noise_params, noise_seed=None):
        """
        Inject noise into a list of samples.

        Args:
            samples: List of numpy arrays, each of shape (V, T)
            noise_type: Type of noise to inject
            noise_params: Parameters for the noise function
            noise_seed: Random seed for reproducibility

        Returns:
            tuple: (list of noisy samples, labels array of shape (N, V, T))
        """
        if noise_seed is not None:
            np.random.seed(noise_seed)

        noisy_samples = []
        all_labels = []

        noise_func = NOISE_FUNCTIONS[noise_type]

        for sample in samples:
            noisy_sample = np.zeros_like(sample)
            sample_labels = np.zeros((self.num_variates, self.seq_length), dtype=int)

            for v in range(self.num_variates):
                kwargs = noise_params.copy()

                # For pattern_seasonal, pass behavior_config
                if noise_type == 'pattern_seasonal':
                    kwargs.setdefault('behavior_config', self.behavior_config[v])

                noisy_data, variate_label = noise_func(sample[v], **kwargs)
                noisy_sample[v] = noisy_data
                sample_labels[v] = variate_label

            noisy_samples.append(noisy_sample)
            all_labels.append(sample_labels)

        labels_array = np.stack(all_labels, axis=0)
        return noisy_samples, labels_array

    def _save_noisy_samples(self, noisy_samples, labels, output_dir, noise_type, noise_params):
        """Save noisy samples and labels as a single .pt file."""
        os.makedirs(output_dir, exist_ok=True)

        # Stack samples into single tensor: (N, V, T)
        data_array = np.stack(noisy_samples, axis=0)
        data_tensor = torch.from_numpy(data_array).float()
        labels_tensor = torch.from_numpy(labels).long()

        # Save config
        config = {
            'num_samples': len(noisy_samples),
            'seq_length': self.seq_length,
            'num_variates': self.num_variates,
            'seed': self.seed,
            'noise_type': noise_type,
            'noise_params': {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in noise_params.items()},
            'behavior_config': [
                {k: float(v) for k, v in cfg.items()}
                for cfg in self.behavior_config
            ]
        }

        # Save as single .pt file with data, labels, and config
        torch.save({
            'data': data_tensor,
            'labels': labels_tensor,
            'config': config
        }, os.path.join(output_dir, 'data.pt'))

    def generate_and_save(self, output_dir=None, noise_types=None):
        """
        Generate complete dataset with train/val/test splits and save to directory.

        Args:
            output_dir: Output directory (default: DEFAULT_OUTPUT_DIR)
            noise_types: List of noise types to generate for test set.
                        None = all available noise types.
                        Example: ['point_global', 'pattern_trend']

        Returns:
            dict: Summary of generated dataset
        """
        if output_dir is None:
            output_dir = self.DEFAULT_OUTPUT_DIR

        if noise_types is None:
            noise_types = list(NOISE_FUNCTIONS.keys())

        print("=" * 70)
        print(f"Generating Multivariate Dataset")
        print(f"  Total samples: {self.num_samples}")
        print(f"  Train: {self.num_train} ({self.train_ratio*100:.0f}%)")
        print(f"  Val: {self.num_val} ({self.val_ratio*100:.0f}%)")
        print(f"  Test: {self.num_test} ({self.test_ratio*100:.0f}%)")
        print(f"  Shape per sample: ({self.num_variates}, {self.seq_length})")
        print(f"  Output: {output_dir}")
        print("=" * 70)

        # Generate all samples first
        print("\nStep 1: Generating all clean samples...")
        np.random.seed(self.seed)  # Reset seed for reproducibility
        all_samples = [self._generate_sample() for _ in range(self.num_samples)]

        # Split samples
        train_samples = all_samples[:self.num_train]
        val_samples = all_samples[self.num_train:self.num_train + self.num_val]
        test_samples = all_samples[self.num_train + self.num_val:]

        # Save train
        print(f"\nStep 2: Saving train set ({self.num_train} samples)...")
        train_dir = os.path.join(output_dir, 'train')
        self._save_samples(train_samples, train_dir)
        print(f"  Saved to: {train_dir}")

        # Save val
        print(f"\nStep 3: Saving val set ({self.num_val} samples)...")
        val_dir = os.path.join(output_dir, 'val')
        self._save_samples(val_samples, val_dir)
        print(f"  Saved to: {val_dir}")

        # Save test/clean
        print(f"\nStep 4: Saving test/clean set ({self.num_test} samples)...")
        test_clean_dir = os.path.join(output_dir, 'test', 'clean')
        self._save_samples(test_samples, test_clean_dir)
        print(f"  Saved to: {test_clean_dir}")

        # Generate and save noisy versions of test set
        print(f"\nStep 5: Generating noisy test sets...")
        for noise_type in noise_types:
            if noise_type not in self.noise_params:
                print(f"  Skipping {noise_type} (no params defined)")
                continue

            noise_params = self.noise_params[noise_type]
            noise_dir = os.path.join(output_dir, 'test', f'noise_{noise_type}')

            # Use a different seed for each noise type but deterministic
            noise_seed = self.seed + hash(noise_type) % 10000 if self.seed else None

            noisy_samples, labels = self._inject_noise_to_samples(
                test_samples, noise_type, noise_params, noise_seed
            )
            self._save_noisy_samples(
                noisy_samples, labels, noise_dir, noise_type, noise_params
            )

            # Calculate corruption stats
            corruption_rate = labels.sum() / labels.size * 100
            print(f"  {noise_type}: {len(noisy_samples)} samples, "
                  f"{corruption_rate:.2f}% corrupted -> {noise_dir}")

        # Save master config
        master_config = {
            'num_samples_total': self.num_samples,
            'num_train': self.num_train,
            'num_val': self.num_val,
            'num_test': self.num_test,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'seq_length': self.seq_length,
            'num_variates': self.num_variates,
            'seed': self.seed,
            'noise_types': noise_types,
            'noise_params': {
                k: {pk: float(pv) if isinstance(pv, (int, float)) else pv
                    for pk, pv in v.items()}
                for k, v in self.noise_params.items()
            },
            'behavior_config': [
                {k: float(v) for k, v in cfg.items()}
                for cfg in self.behavior_config
            ]
        }
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(master_config, f, indent=2)

        print("\n" + "=" * 70)
        print("Dataset generation complete!")
        print(f"Master config saved to: {os.path.join(output_dir, 'config.json')}")
        print("=" * 70)

        return master_config

    @staticmethod
    def load_split(split_dir):
        """
        Load a dataset split (train, val, or test/clean).

        Args:
            split_dir: Directory containing data.pt

        Returns:
            tuple: (data tensor of shape (N, V, T), config dict)
        """
        checkpoint = torch.load(os.path.join(split_dir, 'data.pt'), weights_only=False)
        return checkpoint['data'], checkpoint['config']

    @staticmethod
    def load_noisy_test(noise_dir):
        """
        Load a noisy test split.

        Args:
            noise_dir: Directory containing data.pt with data, labels, and config

        Returns:
            tuple: (data tensor (N, V, T), labels tensor (N, V, T), config dict)
        """
        checkpoint = torch.load(os.path.join(noise_dir, 'data.pt'), weights_only=False)
        return checkpoint['data'], checkpoint['labels'], checkpoint['config']

    @staticmethod
    def load_full_dataset(base_dir):
        """
        Load the complete dataset including all splits.

        Args:
            base_dir: Base directory containing train/, val/, test/ subdirectories

        Returns:
            dict: {
                'train': (data, config),
                'val': (data, config),
                'test_clean': (data, config),
                'test_noisy': {
                    'noise_type': (data, labels, config),
                    ...
                },
                'master_config': config
            }
        """
        result = {}

        # Load master config
        with open(os.path.join(base_dir, 'config.json'), 'r') as f:
            result['master_config'] = json.load(f)

        # Load train and val
        result['train'] = MultivariateDatasetGenerator.load_split(
            os.path.join(base_dir, 'train')
        )
        result['val'] = MultivariateDatasetGenerator.load_split(
            os.path.join(base_dir, 'val')
        )

        # Load test/clean
        result['test_clean'] = MultivariateDatasetGenerator.load_split(
            os.path.join(base_dir, 'test', 'clean')
        )

        # Load all noisy test sets
        result['test_noisy'] = {}
        test_dir = os.path.join(base_dir, 'test')
        for item in os.listdir(test_dir):
            if item.startswith('noise_'):
                noise_type = item[6:]  # Remove 'noise_' prefix
                noise_path = os.path.join(test_dir, item)
                if os.path.isdir(noise_path):
                    result['test_noisy'][noise_type] = \
                        MultivariateDatasetGenerator.load_noisy_test(noise_path)

        return result


# =============================================================================
# Main - Example Usage
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate multivariate synthetic dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Total number of samples (default: 1000)')
    parser.add_argument('--seq_length', type=int, default=200,
                        help='Sequence length per sample (default: 200)')
    parser.add_argument('--num_variates', type=int, default=4,
                        help='Number of variates/channels (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str,
                        default='/home/payal/maestro_ttn_robustness/data/synth_data/basic_multivariate',
                        help='Output directory')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with small dataset')

    args = parser.parse_args()

    if args.demo:
        # Demo mode: small dataset to verify functionality
        print("\n" + "=" * 70)
        print("DEMO MODE: Generating small dataset for verification")
        print("=" * 70 + "\n")

        gen = MultivariateDatasetGenerator(
            num_samples=100,  # 70 train, 10 val, 20 test
            seq_length=200,
            num_variates=4,
            behavior=[sine, cosine, sine, cosine],
            behavior_config=[
                {'freq': 0.04, 'coef': 1.5, 'offset': 0.0, 'noise_amp': 0.05},
                {'freq': 0.04, 'coef': 2.5, 'offset': 0.0, 'noise_amp': 0.05},
                {'freq': 0.06, 'coef': 1.5, 'offset': 1.0, 'noise_amp': 0.03},
                {'freq': 0.03, 'coef': 2.0, 'offset': -1.0, 'noise_amp': 0.08},
            ],
            seed=42
        )
        import tempfile
        demo_dir = tempfile.mkdtemp(prefix='synth_demo_')
        gen.generate_and_save(demo_dir)

        # Verify loading
        print("\n" + "=" * 70)
        print("Verifying dataset loading...")
        print("=" * 70)

        dataset = MultivariateDatasetGenerator.load_full_dataset(demo_dir)

        print(f"\nTrain shape: {dataset['train'][0].shape}")
        print(f"Val shape: {dataset['val'][0].shape}")
        print(f"Test clean shape: {dataset['test_clean'][0].shape}")
        print(f"Noisy test sets: {list(dataset['test_noisy'].keys())}")

        for noise_type, (data, labels, cfg) in dataset['test_noisy'].items():
            corruption = labels.sum() / labels.size * 100
            print(f"  {noise_type}: data={data.shape}, labels={labels.shape}, "
                  f"corruption={corruption:.2f}%")

        print(f"\nDemo directory: {demo_dir}")

    else:
        # Full generation mode
        print("\n" + "=" * 70)
        print("Generating full multivariate dataset")
        print("=" * 70 + "\n")

        gen = MultivariateDatasetGenerator(
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            num_variates=args.num_variates,
            behavior=[sine, cosine, sine, cosine][:args.num_variates],
            behavior_config=[
                {'freq': 0.04, 'coef': 1.5, 'offset': 0.0, 'noise_amp': 0.05},
                {'freq': 0.04, 'coef': 2.5, 'offset': 0.0, 'noise_amp': 0.05},
                {'freq': 0.06, 'coef': 1.5, 'offset': 1.0, 'noise_amp': 0.03},
                {'freq': 0.03, 'coef': 2.0, 'offset': -1.0, 'noise_amp': 0.08},
            ][:args.num_variates],
            seed=args.seed
        )
        gen.generate_and_save(args.output_dir)

        print("\n" + "=" * 70)
        print("Usage examples:")
        print("=" * 70)
        print(f"""
# Load full dataset:
from data.ttn_gen import MultivariateDatasetGenerator
dataset = MultivariateDatasetGenerator.load_full_dataset('{args.output_dir}')

# Access splits:
train_data, train_cfg = dataset['train']           # shape: (N_train, V, T)
val_data, val_cfg = dataset['val']                 # shape: (N_val, V, T)
test_clean, test_cfg = dataset['test_clean']       # shape: (N_test, V, T)

# Access noisy test sets:
for noise_type, (data, labels, cfg) in dataset['test_noisy'].items():
    print(f"{{noise_type}}: data={{data.shape}}, labels={{labels.shape}}")
""")
