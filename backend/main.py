from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Dict, Any, Optional

app = FastAPI()

# CORS: Allow requests from your Next.js dev server (typically localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"], # Make sure this matches your frontend port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ECG Generation Constants and Parameters ---
FS = 250  # Sampling frequency (Hz)
BASELINE_MV = 0.0

# --- Beat Morphology Definitions ---
SINUS_PARAMS = {
    "p_duration": 0.09, "pr_interval": 0.16, "qrs_duration": 0.10,
    "st_duration": 0.12, "t_duration": 0.16, "p_amplitude": 0.15,
    "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.3,
}

PVC_PARAMS = {
    "p_duration": 0.0,    # No P-wave or dissociated
    "p_amplitude": 0.0,
    "pr_interval": 0.0,   # Not applicable if no P-wave linked to QRS
    "qrs_duration": 0.16, # Typically wider
    "st_duration": 0.10,
    "t_duration": 0.18,
    "q_amplitude": -0.2,  # Can be variable
    "r_amplitude": 0.8,   # Can be variable, often large
    "s_amplitude": -0.6,  # Can be variable, often deep
    "t_amplitude": -0.3,  # Often discordant (opposite to main QRS deflection)
}

PAC_PARAMS = {
    "p_duration": 0.08,  # P-wave might be different shape/earlier
    "p_amplitude": 0.12, # Might be different amplitude
    "pr_interval": 0.14, # Can be shorter or normal
    "qrs_duration": 0.09, # Usually normal if normally conducted
    "st_duration": 0.12,
    "t_duration": 0.16,
    "q_amplitude": -0.1,
    "r_amplitude": 1.0,
    "s_amplitude": -0.25,
    "t_amplitude": 0.3,
}

BEAT_MORPHOLOGIES = {
    "sinus": SINUS_PARAMS,
    "pvc": PVC_PARAMS,
    "pac": PAC_PARAMS,
    # Add more as you define them (e.g., "junctional_escape", "ventricular_escape")
}

# --- Waveform Primitive ---
def gaussian_wave(t_points, center, amplitude, width_std_dev):
    """Generates a Gaussian pulse."""
    # Ensure width_std_dev is not zero to avoid division by zero
    if width_std_dev == 0:
        # If width is zero, effectively no wave or an impulse; returning zeros for smooth handling
        return np.zeros_like(t_points)
    return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

# --- Single Beat Generation ---
def generate_single_beat_morphology(params: Dict[str, float], fs: int = FS):
    """
    Generates the waveform for a single PQRST complex based on given parameters.
    The time axis for this beat starts at 0 (potential P-wave onset).
    """
    # Calculate total duration of the complex for generating time vector
    # Ensure enough time for all components and some padding
    total_complex_duration = params.get('pr_interval', 0) + \
                             params.get('qrs_duration', 0.1) + \
                             params.get('st_duration', 0.1) + \
                             params.get('t_duration', 0.1) + 0.1 # Extra padding

    num_samples = int(total_complex_duration * fs)
    if num_samples == 0: # Avoid creating empty arrays if duration is too short
        return np.array([]), np.array([])
        
    t = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    beat_waveform = np.full(num_samples, BASELINE_MV)

    # 1. P-wave
    if params.get('p_amplitude', 0) != 0 and params.get('p_duration', 0) > 0:
        p_center = params['p_duration'] / 2
        p_width_std_dev = params['p_duration'] / 4 # Heuristic for width
        beat_waveform += gaussian_wave(t, p_center, params['p_amplitude'], p_width_std_dev)

    # 2. QRS Complex
    qrs_onset_time = params.get('pr_interval', 0)
    qrs_duration = params.get('qrs_duration', 0.1)

    if qrs_duration > 0: # Only generate QRS if duration is positive
        # Q-wave (small negative deflection before R)
        if params.get('q_amplitude', 0) != 0:
            q_center = qrs_onset_time + qrs_duration * 0.15
            q_width_std_dev = qrs_duration / 10
            beat_waveform += gaussian_wave(t, q_center, params['q_amplitude'], q_width_std_dev)

        # R-wave (main positive deflection)
        if params.get('r_amplitude', 0) != 0:
            r_center = qrs_onset_time + qrs_duration * 0.4
            r_width_std_dev = qrs_duration / 6 # Sharper than P/T
            beat_waveform += gaussian_wave(t, r_center, params['r_amplitude'], r_width_std_dev)

        # S-wave (small negative deflection after R)
        if params.get('s_amplitude', 0) != 0:
            s_center = qrs_onset_time + qrs_duration * 0.75
            s_width_std_dev = qrs_duration / 10
            beat_waveform += gaussian_wave(t, s_center, params['s_amplitude'], s_width_std_dev)

    # 3. T-wave (after ST segment)
    if params.get('t_amplitude', 0) != 0 and params.get('t_duration', 0) > 0:
        t_onset_time = qrs_onset_time + qrs_duration + params.get('st_duration', 0.1)
        t_center = t_onset_time + params['t_duration'] / 2
        t_width_std_dev = params['t_duration'] / 4
        beat_waveform += gaussian_wave(t, t_center, params['t_amplitude'], t_width_std_dev)
    
    return t, beat_waveform


# --- Rhythm Generation Logic ---
def generate_ecg_rhythm_data(
    heart_rate_bpm: float,
    duration_sec: float,
    base_rhythm_type: str,
    insert_pvc_after_n_beats: Optional[int] = None,
    fs: int = FS
):
    rr_interval_sec = 60.0 / heart_rate_bpm
    num_total_samples = int(duration_sec * fs)
    
    full_time_axis = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    full_ecg_signal_np = np.full(num_total_samples, BASELINE_MV) # Use a different var name

    current_beat_onset_time_sec = 0.0
    beat_count = 0  # 0-indexed count of beats generated

    while current_beat_onset_time_sec < duration_sec:
        beat_type_to_generate = base_rhythm_type
        current_rr_interval = rr_interval_sec  # Default RR for the current beat

        # --- Logic for inserting PVCs (simplified) ---
        # If insert_pvc_after_n_beats is set (e.g., 2), a PVC will be the 3rd beat (index 2),
        # then 6th beat (index 5), etc.
        if insert_pvc_after_n_beats is not None and insert_pvc_after_n_beats >= 0:
            # Check if current beat_count is the one where PVC should be inserted
            # N=0 means PVC is 1st beat, N=1 means PVC is 2nd beat
            if beat_count % (insert_pvc_after_n_beats + 1) == insert_pvc_after_n_beats:
                 beat_type_to_generate = "pvc"
                # Note: This simplified model doesn't yet handle coupling intervals or compensatory pauses.
                # The PVC just replaces the scheduled sinus beat morphology at the expected time.

        # --- Get morphology for the current beat type ---
        selected_beat_params = BEAT_MORPHOLOGIES.get(beat_type_to_generate)
        if not selected_beat_params: # Fallback to sinus if type unknown
            selected_beat_params = SINUS_PARAMS
            print(f"Warning: Unknown beat type '{beat_type_to_generate}', defaulting to sinus.")

        _, y_beat_shape = generate_single_beat_morphology(selected_beat_params, fs)
        num_samples_beat_shape = len(y_beat_shape)

        if num_samples_beat_shape == 0: # Skip if beat shape is empty
            current_beat_onset_time_sec += current_rr_interval
            beat_count +=1 # Still increment beat count even if we skip drawing
            continue

        # --- Place the beat ---
        start_sample_index = int(current_beat_onset_time_sec * fs)
        end_sample_index = start_sample_index + num_samples_beat_shape
        
        # Ensure we don't write past the end of the full_ecg_signal_np array
        # And also correctly copy parts of the beat if it's cut off at the end
        actual_end_sample_index = min(end_sample_index, num_total_samples)
        samples_to_copy_count = actual_end_sample_index - start_sample_index

        if samples_to_copy_count > 0:
            full_ecg_signal_np[start_sample_index : actual_end_sample_index] += y_beat_shape[:samples_to_copy_count]
        
        if start_sample_index >= num_total_samples: # No more space to draw
            break

        current_beat_onset_time_sec += current_rr_interval
        beat_count += 1
            
    # Add some simple noise
    noise_amplitude = 0.02
    full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    
    return full_time_axis.tolist(), full_ecg_signal_np.tolist()


# --- API Endpoint Definition ---
class ECGRequestParams(BaseModel):
    heart_rate_bpm: float = Field(75.0, gt=0, description="Average heart rate in beats per minute.")
    duration_sec: float = Field(10.0, gt=0, description="Total duration of the ECG strip in seconds.")
    base_rhythm: str = Field("sinus", description="The underlying base rhythm type (e.g., 'sinus').")
    insert_pvc_after_n_beats: Optional[int] = Field(None, ge=0, description="Insert a PVC after every N normal beats (0-indexed). E.g., 0 means 1st beat is PVC, 1 means 2nd beat is PVC.")

@app.post("/api/generate_ecg")
async def get_ecg_data(params: ECGRequestParams):
    """
    Generates ECG waveform data based on input parameters.
    Currently supports a base sinus rhythm and optional periodic PVC insertion.
    """
    rhythm_description = f"{params.base_rhythm.capitalize()}"
    if params.insert_pvc_after_n_beats is not None:
        rhythm_description += f" with PVC after every {params.insert_pvc_after_n_beats} sinus beat(s)"
        # If N=0, it's "PVC after every 0 sinus beats", meaning the first beat is a PVC, then a PVC etc.
        # if N=2, it's "PVC after every 2 sinus beats", meaning S-S-PVC-S-S-PVC...

    time_axis, ecg_signal = generate_ecg_rhythm_data(
        heart_rate_bpm=params.heart_rate_bpm,
        duration_sec=params.duration_sec,
        base_rhythm_type=params.base_rhythm,
        insert_pvc_after_n_beats=params.insert_pvc_after_n_beats,
        fs=FS
    )
    return {
        "time_axis": time_axis,
        "ecg_signal": ecg_signal,
        "rhythm_generated": rhythm_description # More descriptive title for the chart
    }

# To run this (from the 'backend' directory in your VS Code terminal):
# 1. (If not already done) Create and activate virtual environment:
#    python -m venv venv
#    source venv/bin/activate  (or venv\Scripts\activate on Windows)
# 2. (If not already done) Install dependencies:
#    pip install -r requirements.txt
# 3. Run Uvicorn:
#    uvicorn main:app --reload
#
# The API will typically be available at http://localhost:8000/api/generate_ecg
# You can test it with tools like Postman or curl, or directly from your frontend.
# Example curl command:
# curl -X POST "http://localhost:8000/api/generate_ecg" -H "Content-Type: application/json" -d '{"heart_rate_bpm": 60, "duration_sec": 5, "base_rhythm": "sinus", "insert_pvc_after_n_beats": 2}'