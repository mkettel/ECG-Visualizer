from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List

app = FastAPI()

# CORS: Allow requests from your Next.js dev server (typically localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ECG Generation Logic (Simplified for Sinus Rhythm) ---
FS = 250  # Sampling frequency (Hz)
BASELINE_MV = 0.0

SINUS_PARAMS = {
    "p_duration": 0.09, "pr_interval": 0.16, "qrs_duration": 0.10,
    "st_duration": 0.12, "t_duration": 0.16, "p_amplitude": 0.15,
    "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.3,
}

def gaussian_wave(t_points, center, amplitude, width_std_dev):
    return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

def generate_single_beat_morphology(params, fs=FS):
    total_complex_duration = params['pr_interval'] + params['qrs_duration'] + \
                             params['st_duration'] + params['t_duration'] + 0.1
    num_samples = int(total_complex_duration * fs)
    t = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    beat_waveform = np.full(num_samples, BASELINE_MV)

    p_center = params['p_duration'] / 2
    p_width_std_dev = params['p_duration'] / 4
    beat_waveform += gaussian_wave(t, p_center, params['p_amplitude'], p_width_std_dev)

    qrs_onset_time = params['pr_interval']
    q_center = qrs_onset_time + params['qrs_duration'] * 0.15
    q_width_std_dev = params['qrs_duration'] / 10
    beat_waveform += gaussian_wave(t, q_center, params['q_amplitude'], q_width_std_dev)

    r_center = qrs_onset_time + params['qrs_duration'] * 0.4
    r_width_std_dev = params['qrs_duration'] / 6
    beat_waveform += gaussian_wave(t, r_center, params['r_amplitude'], r_width_std_dev)

    s_center = qrs_onset_time + params['qrs_duration'] * 0.75
    s_width_std_dev = params['qrs_duration'] / 10
    beat_waveform += gaussian_wave(t, s_center, params['s_amplitude'], s_width_std_dev)

    t_onset_time = qrs_onset_time + params['qrs_duration'] + params['st_duration']
    t_center = t_onset_time + params['t_duration'] / 2
    t_width_std_dev = params['t_duration'] / 4
    beat_waveform += gaussian_wave(t, t_center, params['t_amplitude'], t_width_std_dev)
    return t, beat_waveform

def generate_ecg_rhythm_data(heart_rate_bpm, duration_sec, beat_params, fs=FS):
    rr_interval_sec = 60.0 / heart_rate_bpm
    num_total_samples = int(duration_sec * fs)
    
    full_time_axis = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    full_ecg_signal = np.full(num_total_samples, BASELINE_MV)
    
    _, y_beat_shape = generate_single_beat_morphology(beat_params, fs)
    num_samples_beat_shape = len(y_beat_shape)

    current_beat_onset_time_sec = 0.0
    while current_beat_onset_time_sec < duration_sec:
        start_sample_index = int(current_beat_onset_time_sec * fs)
        end_sample_index = start_sample_index + num_samples_beat_shape
        
        if end_sample_index > num_total_samples:
            samples_to_copy = num_total_samples - start_sample_index
            if samples_to_copy <= 0: break
            full_ecg_signal[start_sample_index : start_sample_index + samples_to_copy] += y_beat_shape[:samples_to_copy]
        else:
            full_ecg_signal[start_sample_index : end_sample_index] += y_beat_shape # Use += to allow overlay if beats are very close
        
        current_beat_onset_time_sec += rr_interval_sec
        
    # Add some simple noise
    noise_amplitude = 0.02
    full_ecg_signal += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal))
    
    return full_time_axis.tolist(), full_ecg_signal.tolist()

# --- API Endpoint ---
class ECGRequestParams(BaseModel):
    heart_rate_bpm: float = 75.0
    duration_sec: float = 10.0
    # We'll add more params for arrhythmias later

@app.post("/api/generate_ecg")
async def get_ecg_data(params: ECGRequestParams):
    # For now, always generate sinus rhythm
    time_axis, ecg_signal = generate_ecg_rhythm_data(
        params.heart_rate_bpm,
        params.duration_sec,
        SINUS_PARAMS # Use default sinus parameters
    )
    return {"time_axis": time_axis, "ecg_signal": ecg_signal}

# To run this (from the 'backend' directory):
# 1. python -m venv venv
# 2. source venv/bin/activate  (or venv\Scripts\activate on Windows)
# 3. pip install -r requirements.txt
# 4. uvicorn main:app --reload
# It will usually run on http://localhost:8000