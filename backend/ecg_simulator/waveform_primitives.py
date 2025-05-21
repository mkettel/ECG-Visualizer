# // backend/ecg-simulator/waveform_primitives.py
import numpy as np
from .constants import FS, BASELINE_MV

# --- Waveform Primitive & Single Beat Generation ---
def gaussian_wave(t_points, center, amplitude, width_std_dev):
    if width_std_dev <= 1e-9: return np.zeros_like(t_points)
    return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

# --- Fibrillatory Wave Generation for AFib ---
def generate_fibrillatory_waves(duration_sec: float, amplitude_mv: float, fs: int = FS):
    num_samples = int(duration_sec * fs)
    time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)
    f_wave_signal = np.zeros(num_samples)
    if amplitude_mv <= 1e-6: return f_wave_signal
    num_f_waves_per_sec = np.random.uniform(5.8, 10)
    f_wave_component_duration = np.random.uniform(0.05, 0.08)
    current_time = 0.0
    while current_time < duration_sec:
        center = current_time + np.random.uniform(-f_wave_component_duration / 5, f_wave_component_duration / 5)
        amp = amplitude_mv * np.random.uniform(0.6, 1.4) * np.random.choice([-1, 1])
        width_std = f_wave_component_duration / np.random.uniform(3.5, 5.5)
        start_time_comp = center - 3 * width_std
        end_time_comp = center + 3 * width_std
        start_idx = max(0, int(np.floor(start_time_comp * fs)))
        end_idx = min(num_samples, int(np.ceil(end_time_comp * fs)))
        if start_idx < end_idx:
            t_points_fwave_comp = time_axis[start_idx:end_idx]
            if t_points_fwave_comp.size > 0 :
                 f_wave_signal[start_idx:end_idx] += gaussian_wave(t_points_fwave_comp, center, amp, width_std)
        current_time += (1.0 / num_f_waves_per_sec) * np.random.uniform(0.5, 1.5)
    return f_wave_signal