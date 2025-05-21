# // backend/ecg-simulator/beat_generation.py
import numpy as np
from typing import Dict
from .constants import FS, BASELINE_MV
from .waveform_primitives import gaussian_wave

def generate_single_beat_morphology(params: Dict[str, float], fs: int = FS, draw_only_p: bool = False, is_flutter_wave_itself: bool = False):
    if is_flutter_wave_itself:
        wave_duration = params.get('p_duration', 0.10)
        wave_amplitude = params.get('p_amplitude', -0.2)
        num_samples = int(wave_duration * fs)
        if num_samples <= 0: return np.array([]), np.array([]), 0.0
        t_relative = np.linspace(0, wave_duration, num_samples, endpoint=False)
        waveform = np.zeros_like(t_relative)
        peak_time_ratio = 0.7
        nadir_idx = int(num_samples * peak_time_ratio)
        if nadir_idx < num_samples -1 and nadir_idx > 0 :
            waveform[:nadir_idx] = np.linspace(0, wave_amplitude, nadir_idx)
            waveform[nadir_idx:] = np.linspace(wave_amplitude, wave_amplitude * 0.2 , num_samples - nadir_idx)
        elif num_samples > 0 : waveform[:] = wave_amplitude / 2
        return t_relative, waveform, 0.0

    p_wave_total_offset = params.get('pr_interval', 0) if params.get('p_amplitude',0) !=0 else 0
    p_duration = params.get('p_duration', 0) if params.get('p_amplitude',0) !=0 else 0
    qrs_duration = 0.0 if draw_only_p else params.get('qrs_duration', 0.1)
    st_duration = 0.0 if draw_only_p else params.get('st_duration', 0.1)
    t_duration = 0.0 if draw_only_p else params.get('t_duration', 0.1)
    duration_from_p_onset_to_qrs_onset = p_wave_total_offset
    total_complex_duration = duration_from_p_onset_to_qrs_onset + \
                             qrs_duration + st_duration + t_duration + 0.05 # Small buffer
    if draw_only_p: total_complex_duration = p_duration + 0.05 if p_duration > 0 else 0.05

    num_samples = int(total_complex_duration * fs)
    if num_samples <= 0: return np.array([]), np.array([]), 0.0
    t_relative_to_p_onset = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    beat_waveform = np.full(num_samples, BASELINE_MV)

    if params.get('p_amplitude', 0) != 0 and p_duration > 0:
        p_center = p_duration / 2
        p_width_std_dev = p_duration / 4 if p_duration > 0 else 1e-3
        beat_waveform += gaussian_wave(t_relative_to_p_onset, p_center, params['p_amplitude'], p_width_std_dev)

    if not draw_only_p:
        qrs_onset_in_array_time = p_wave_total_offset
        if qrs_duration > 0:
            if params.get('q_amplitude', 0) != 0:
                q_center = qrs_onset_in_array_time + qrs_duration * 0.15
                q_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, q_center, params['q_amplitude'], q_width_std_dev)
            if params.get('r_amplitude', 0) != 0:
                r_center = qrs_onset_in_array_time + qrs_duration * 0.4
                r_width_std_dev = qrs_duration / 6 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, r_center, params['r_amplitude'], r_width_std_dev)
            if params.get('s_amplitude', 0) != 0:
                s_center = qrs_onset_in_array_time + qrs_duration * 0.75
                s_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, s_center, params['s_amplitude'], s_width_std_dev)

        t_onset_in_array_time = qrs_onset_in_array_time + qrs_duration + st_duration
        if params.get('t_amplitude', 0) != 0 and t_duration > 0:
            t_center = t_onset_in_array_time + t_duration / 2
            t_width_std_dev = t_duration / 4 if t_duration > 0 else 1e-3
            beat_waveform += gaussian_wave(t_relative_to_p_onset, t_center, params['t_amplitude'], t_width_std_dev)

    return t_relative_to_p_onset, beat_waveform, p_wave_total_offset