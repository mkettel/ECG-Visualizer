# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import numpy as np
# from typing import List, Dict, Any, Optional

# app = FastAPI()

# # CORS: Allow requests from your Next.js dev server (typically localhost:3000)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3001"], # Make sure this matches your frontend port
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- ECG Generation Constants and Parameters ---
# FS = 250  # Sampling frequency (Hz)
# BASELINE_MV = 0.0

# # --- Beat Morphology Definitions ---
# SINUS_PARAMS = {
#     "p_duration": 0.09, "pr_interval": 0.16, "qrs_duration": 0.10,
#     "st_duration": 0.12, "t_duration": 0.16, "p_amplitude": 0.15,
#     "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.25,
#     "t_amplitude": 0.3,
# }

# PVC_PARAMS = {
#     "p_duration": 0.0,    # No P-wave or dissociated
#     "p_amplitude": 0.0,
#     "pr_interval": 0.0,   # Not applicable if no P-wave linked to QRS
#     "qrs_duration": 0.16, # Typically wider
#     "st_duration": 0.10,
#     "t_duration": 0.18,
#     "q_amplitude": -0.2,  # Can be variable
#     "r_amplitude": 0.8,   # Can be variable, often large
#     "s_amplitude": -0.6,  # Can be variable, often deep
#     "t_amplitude": -0.3,  # Often discordant (opposite to main QRS deflection)
# }

# PAC_PARAMS = {
#     "p_duration": 0.08,  # P-wave might be different shape/earlier
#     "p_amplitude": 0.12, # Might be different amplitude
#     "pr_interval": 0.14, # Can be shorter or normal
#     "qrs_duration": 0.09, # Usually normal if normally conducted
#     "st_duration": 0.12,
#     "t_duration": 0.16,
#     "q_amplitude": -0.1,
#     "r_amplitude": 1.0,
#     "s_amplitude": -0.25,
#     "t_amplitude": 0.3,
# }

# BEAT_MORPHOLOGIES = {
#     "sinus": SINUS_PARAMS,
#     "pvc": PVC_PARAMS,
#     "pac": PAC_PARAMS,
#     # Add more as you define them (e.g., "junctional_escape", "ventricular_escape")
# }

# # --- Waveform Primitive ---
# def gaussian_wave(t_points, center, amplitude, width_std_dev):
#     """Generates a Gaussian pulse."""
#     # Ensure width_std_dev is not zero to avoid division by zero
#     if width_std_dev == 0:
#         # If width is zero, effectively no wave or an impulse; returning zeros for smooth handling
#         return np.zeros_like(t_points)
#     return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

# # --- Single Beat Generation ---
# def generate_single_beat_morphology(params: Dict[str, float], fs: int = FS):
#     """
#     Generates the waveform for a single PQRST complex based on given parameters.
#     The time axis for this beat starts at 0 (potential P-wave onset).
#     """
#     # Calculate total duration of the complex for generating time vector
#     # Ensure enough time for all components and some padding
#     total_complex_duration = params.get('pr_interval', 0) + \
#                              params.get('qrs_duration', 0.1) + \
#                              params.get('st_duration', 0.1) + \
#                              params.get('t_duration', 0.1) + 0.1 # Extra padding

#     num_samples = int(total_complex_duration * fs)
#     if num_samples == 0: # Avoid creating empty arrays if duration is too short
#         return np.array([]), np.array([])
        
#     t = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
#     beat_waveform = np.full(num_samples, BASELINE_MV)

#     # 1. P-wave
#     if params.get('p_amplitude', 0) != 0 and params.get('p_duration', 0) > 0:
#         p_center = params['p_duration'] / 2
#         p_width_std_dev = params['p_duration'] / 4 # Heuristic for width
#         beat_waveform += gaussian_wave(t, p_center, params['p_amplitude'], p_width_std_dev)

#     # 2. QRS Complex
#     qrs_onset_time = params.get('pr_interval', 0)
#     qrs_duration = params.get('qrs_duration', 0.1)

#     if qrs_duration > 0: # Only generate QRS if duration is positive
#         # Q-wave (small negative deflection before R)
#         if params.get('q_amplitude', 0) != 0:
#             q_center = qrs_onset_time + qrs_duration * 0.15
#             q_width_std_dev = qrs_duration / 10
#             beat_waveform += gaussian_wave(t, q_center, params['q_amplitude'], q_width_std_dev)

#         # R-wave (main positive deflection)
#         if params.get('r_amplitude', 0) != 0:
#             r_center = qrs_onset_time + qrs_duration * 0.4
#             r_width_std_dev = qrs_duration / 6 # Sharper than P/T
#             beat_waveform += gaussian_wave(t, r_center, params['r_amplitude'], r_width_std_dev)

#         # S-wave (small negative deflection after R)
#         if params.get('s_amplitude', 0) != 0:
#             s_center = qrs_onset_time + qrs_duration * 0.75
#             s_width_std_dev = qrs_duration / 10
#             beat_waveform += gaussian_wave(t, s_center, params['s_amplitude'], s_width_std_dev)

#     # 3. T-wave (after ST segment)
#     if params.get('t_amplitude', 0) != 0 and params.get('t_duration', 0) > 0:
#         t_onset_time = qrs_onset_time + qrs_duration + params.get('st_duration', 0.1)
#         t_center = t_onset_time + params['t_duration'] / 2
#         t_width_std_dev = params['t_duration'] / 4
#         beat_waveform += gaussian_wave(t, t_center, params['t_amplitude'], t_width_std_dev)
    
#     return t, beat_waveform

# # --- Rhythm Generation Logic (Refined Simple Version) ---
# def generate_ecg_rhythm_data(
#     heart_rate_bpm: float,
#     duration_sec: float,
#     base_rhythm_type: str, # Currently always "sinus" for the base
#     insert_pvc_after_n_beats: Optional[int] = None,
#     insert_pac_after_n_beats: Optional[int] = None,
#     fs: int = FS
# ):
#     base_rr_interval_sec = 60.0 / heart_rate_bpm
#     num_total_samples = int(duration_sec * fs)
    
#     full_time_axis_np = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
#     full_ecg_signal_np = np.full(num_total_samples, BASELINE_MV)

#     current_beat_placement_time_sec = 0.0
#     # This counter tracks how many "opportunities" for a beat have passed,
#     # which corresponds to the number of base_rr_intervals elapsed.
#     beat_slot_index = 0 

#     while current_beat_placement_time_sec < duration_sec:
#         beat_type_for_this_slot = base_rhythm_type # Default to sinus
        
#         # Determine if an ectopic should replace the sinus beat in this slot
#         # Note: `insert_xxx_after_n_beats = 0` means the 1st beat (slot_index 0) is ectopic.
#         #       `insert_xxx_after_n_beats = 1` means the 2nd beat (slot_index 1) is ectopic after 1 sinus.
        
#         pac_is_due_this_slot = False
#         if insert_pac_after_n_beats is not None and \
#            beat_slot_index % (insert_pac_after_n_beats + 1) == insert_pac_after_n_beats:
#             pac_is_due_this_slot = True

#         pvc_is_due_this_slot = False
#         if insert_pvc_after_n_beats is not None and \
#            beat_slot_index % (insert_pvc_after_n_beats + 1) == insert_pvc_after_n_beats:
#             pvc_is_due_this_slot = True

#         # Precedence: If both are due for the same slot, PAC takes it (arbitrary choice)
#         if pac_is_due_this_slot:
#             beat_type_for_this_slot = "pac"
#         elif pvc_is_due_this_slot:
#             beat_type_for_this_slot = "pvc"
        
#         # Get morphology
#         selected_beat_params = BEAT_MORPHOLOGIES.get(beat_type_for_this_slot, SINUS_PARAMS)
#         _, y_beat_shape = generate_single_beat_morphology(selected_beat_params, fs)
#         num_samples_beat_shape = len(y_beat_shape)

#         if num_samples_beat_shape > 0:
#             start_sample_index = int(current_beat_placement_time_sec * fs)
#             # Ensure we don't write past the end of the full_ecg_signal_np array
#             end_sample_index = min(start_sample_index + num_samples_beat_shape, num_total_samples)
#             samples_to_copy = end_sample_index - start_sample_index

#             if samples_to_copy > 0:
#                 # Ensure y_beat_shape is also sliced if samples_to_copy is less than num_samples_beat_shape
#                 full_ecg_signal_np[start_sample_index : end_sample_index] += y_beat_shape[:samples_to_copy]
        
#         if current_beat_placement_time_sec + base_rr_interval_sec >= duration_sec + base_rr_interval_sec * 0.1 and \
#            start_sample_index >= num_total_samples : # Avoid infinite loop if duration is very short
#             break


#         current_beat_placement_time_sec += base_rr_interval_sec
#         beat_slot_index += 1
            
#     # Add noise
#     noise_amplitude = 0.02
#     full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    
#     return full_time_axis_np.tolist(), full_ecg_signal_np.tolist()


# # --- API Endpoint Definition (ECGRequestParams and get_ecg_data - keep as is from your provided code) ---
# class ECGRequestParams(BaseModel):
#     heart_rate_bpm: float = Field(75.0, gt=0)
#     duration_sec: float = Field(10.0, gt=0)
#     base_rhythm: str = Field("sinus")
#     insert_pvc_after_n_beats: Optional[int] = Field(None, ge=0, description="Insert a PVC after N preceding sinus beats. N=0 means 1st beat is PVC.")
#     insert_pac_after_n_beats: Optional[int] = Field(None, ge=0, description="Insert a PAC after N preceding sinus beats. N=0 means 1st beat is PAC.")

# @app.post("/api/generate_ecg")
# async def get_ecg_data(params: ECGRequestParams):
#     rhythm_description_parts = [params.base_rhythm.capitalize()]
#     ectopic_descriptions = []

#     # Corrected description logic for "after N beats" means N+1th beat overall for that pattern
#     if params.insert_pac_after_n_beats is not None:
#         n_plus_1 = params.insert_pac_after_n_beats + 1
#         suffix = "th"
#         if n_plus_1 % 10 == 1 and n_plus_1 % 100 != 11: suffix = "st"
#         elif n_plus_1 % 10 == 2 and n_plus_1 % 100 != 12: suffix = "nd"
#         elif n_plus_1 % 10 == 3 and n_plus_1 % 100 != 13: suffix = "rd"
#         ectopic_descriptions.append(f"PAC as every {n_plus_1}{suffix} beat")

#     if params.insert_pvc_after_n_beats is not None:
#         n_plus_1 = params.insert_pvc_after_n_beats + 1
#         suffix = "th"
#         if n_plus_1 % 10 == 1 and n_plus_1 % 100 != 11: suffix = "st"
#         elif n_plus_1 % 10 == 2 and n_plus_1 % 100 != 12: suffix = "nd"
#         elif n_plus_1 % 10 == 3 and n_plus_1 % 100 != 13: suffix = "rd"
#         ectopic_descriptions.append(f"PVC as every {n_plus_1}{suffix} beat")

#     if ectopic_descriptions:
#         rhythm_description_parts.append("with " + " and ".join(ectopic_descriptions))
#     rhythm_description = " ".join(rhythm_description_parts)

#     time_axis, ecg_signal = generate_ecg_rhythm_data(
#         heart_rate_bpm=params.heart_rate_bpm,
#         duration_sec=params.duration_sec,
#         base_rhythm_type=params.base_rhythm,
#         insert_pvc_after_n_beats=params.insert_pvc_after_n_beats,
#         insert_pac_after_n_beats=params.insert_pac_after_n_beats,
#         fs=FS
#     )
#     return {
#         "time_axis": time_axis,
#         "ecg_signal": ecg_signal,
#         "rhythm_generated": rhythm_description
#     }


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Dict, Any, Optional
import heapq

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"], # ADJUST TO YOUR FRONTEND PORT
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ECG Generation Constants ---
FS = 250
BASELINE_MV = 0.0
MIN_REFRACTORY_PERIOD_SEC = 0.200 # Ventricular refractory after a QRS

# --- Beat Morphology Definitions ---
SINUS_PARAMS = {
    "p_duration": 0.09, "pr_interval": 0.16, "qrs_duration": 0.10,
    "st_duration": 0.12, "t_duration": 0.16, "p_amplitude": 0.15,
    "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.3,
}
PVC_PARAMS = {
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.16, "q_amplitude": -0.05, "r_amplitude": 1.2,
    "s_amplitude": -0.2, "st_duration": 0.10, "t_duration": 0.22,
    "t_amplitude": -0.6,
}
PAC_PARAMS = {
    "p_duration": 0.08, "p_amplitude": 0.12, "pr_interval": 0.14,
    "qrs_duration": 0.09, "st_duration": 0.12, "t_duration": 0.16,
    "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.3,
}
BEAT_MORPHOLOGIES = {"sinus": SINUS_PARAMS, "pvc": PVC_PARAMS, "pac": PAC_PARAMS}

# --- Ectopic Beat Configuration Constants ---
PVC_COUPLING_FACTOR = 0.60
PAC_COUPLING_FACTOR = 0.70

# --- Waveform Primitive & Single Beat Generation ---
def gaussian_wave(t_points, center, amplitude, width_std_dev):
    if width_std_dev <= 1e-6: return np.zeros_like(t_points)
    return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

def generate_single_beat_morphology(params: Dict[str, float], fs: int = FS, draw_only_p: bool = False):
    p_wave_total_offset = params.get('pr_interval', 0) if params.get('p_amplitude',0) !=0 else 0
    p_duration = params.get('p_duration', 0) if params.get('p_amplitude',0) !=0 else 0
    
    # If only drawing P, other components have zero duration for calculation purposes
    qrs_duration = 0.0 if draw_only_p else params.get('qrs_duration', 0.1)
    st_duration = 0.0 if draw_only_p else params.get('st_duration', 0.1)
    t_duration = 0.0 if draw_only_p else params.get('t_duration', 0.1)
    
    duration_from_p_onset_to_qrs_onset = p_wave_total_offset
    total_complex_duration = duration_from_p_onset_to_qrs_onset + \
                             qrs_duration + st_duration + t_duration + 0.05 # Padding for tail end
    if draw_only_p: # If only P, total duration is just P duration + small padding
        total_complex_duration = p_duration + 0.05 


    num_samples = int(total_complex_duration * fs)
    if num_samples <= 0: return np.array([]), np.array([]), 0.0

    t_relative_to_p_onset = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    beat_waveform = np.full(num_samples, BASELINE_MV)

    if params.get('p_amplitude', 0) != 0 and p_duration > 0:
        p_center = p_duration / 2
        p_width_std_dev = p_duration / 4
        beat_waveform += gaussian_wave(t_relative_to_p_onset, p_center, params['p_amplitude'], p_width_std_dev)

    if not draw_only_p:
        qrs_onset_in_array_time = p_wave_total_offset
        if qrs_duration > 0:
            # QRS components
            if params.get('q_amplitude', 0) != 0:
                q_center = qrs_onset_in_array_time + qrs_duration * 0.15; q_width_std_dev = qrs_duration / 10
                beat_waveform += gaussian_wave(t_relative_to_p_onset, q_center, params['q_amplitude'], q_width_std_dev)
            if params.get('r_amplitude', 0) != 0:
                r_center = qrs_onset_in_array_time + qrs_duration * 0.4; r_width_std_dev = qrs_duration / 6
                beat_waveform += gaussian_wave(t_relative_to_p_onset, r_center, params['r_amplitude'], r_width_std_dev)
            if params.get('s_amplitude', 0) != 0:
                s_center = qrs_onset_in_array_time + qrs_duration * 0.75; s_width_std_dev = qrs_duration / 10
                beat_waveform += gaussian_wave(t_relative_to_p_onset, s_center, params['s_amplitude'], s_width_std_dev)
        
        t_onset_in_array_time = qrs_onset_in_array_time + qrs_duration + st_duration
        if params.get('t_amplitude', 0) != 0 and t_duration > 0:
            # T-wave components
            t_center = t_onset_in_array_time + t_duration / 2; t_width_std_dev = t_duration / 4
            beat_waveform += gaussian_wave(t_relative_to_p_onset, t_center, params['t_amplitude'], t_width_std_dev)
    
    return t_relative_to_p_onset, beat_waveform, p_wave_total_offset


# --- Event-Driven Rhythm Generation ---
class BeatEvent:
    def __init__(self, time: float, beat_type: str, source: str = "sa_node"):
        self.time = time; self.beat_type = beat_type; self.source = source
    def __lt__(self, other): return self.time < other.time
    def __repr__(self): return f"BeatEvent(t={self.time:.3f}, type='{self.beat_type}', src='{self.source}')"

def generate_physiologically_accurate_ecg(
    heart_rate_bpm: float, duration_sec: float,
    enable_pvc: bool, pvc_probability_per_sinus: float,
    enable_pac: bool, pac_probability_per_sinus: float,
    first_degree_av_block_pr_sec: Optional[float],
    enable_mobitz_ii_av_block: bool, # New
    mobitz_ii_p_waves_per_qrs: int,   # New (e.g., 2 for 2:1, 3 for 3:1)
    fs: int = FS
):
    base_rr_interval_sec = 60.0 / heart_rate_bpm
    num_total_samples = int(duration_sec * fs)
    full_time_axis_np = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    full_ecg_signal_np = np.full(num_total_samples, BASELINE_MV)
    event_queue: List[BeatEvent] = []
    
    sa_node_next_fire_time = 0.0
    last_placed_qrs_onset_time = -base_rr_interval_sec
    ventricle_ready_for_next_qrs_at_time = 0.0
    
    # State for Mobitz II
    p_wave_counter_for_mobitz_ii = 0 # Counts P-waves in current Mobitz cycle

    heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

    while event_queue and event_queue[0].time < duration_sec:
        current_event = heapq.heappop(event_queue)
        potential_qrs_onset_time = current_event.time
        is_atrial_event = current_event.source == "sa_node" or current_event.beat_type == "pac" # or current_event.source == "pac_focus"
        
        # --- Prepare current beat parameters (can be overridden) ---
        current_beat_morph_params = BEAT_MORPHOLOGIES[current_event.beat_type].copy()
        if first_degree_av_block_pr_sec is not None and is_atrial_event:
            current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
        
        # --- AV Conduction Logic (Mobitz II) ---
        qrs_is_blocked_by_mobitz_ii = False
        if enable_mobitz_ii_av_block and is_atrial_event:
            p_wave_counter_for_mobitz_ii += 1
            # Conduct if counter is 1 (first P of ratio), block others
            if p_wave_counter_for_mobitz_ii % mobitz_ii_p_waves_per_qrs != 1 : 
                qrs_is_blocked_by_mobitz_ii = True
            
            if p_wave_counter_for_mobitz_ii >= mobitz_ii_p_waves_per_qrs: # Reset counter after full ratio
                p_wave_counter_for_mobitz_ii = 0


        # --- Ventricular Refractory / QRS Blocking Check ---
        if qrs_is_blocked_by_mobitz_ii or \
           potential_qrs_onset_time < ventricle_ready_for_next_qrs_at_time:
            
            # If QRS is blocked but it's an atrial event (SA or PAC), we might still draw its P-wave
            if is_atrial_event and qrs_is_blocked_by_mobitz_ii : # Draw P-wave for Mobitz II blocked beat
                # Use modified params to ensure only P is drawn
                _, y_p_wave_shape, p_wave_offset = generate_single_beat_morphology(current_beat_morph_params, fs, draw_only_p=True)
                if len(y_p_wave_shape) > 0:
                    p_wave_start_time_global = potential_qrs_onset_time - p_wave_offset # P starts before potential QRS time
                    # ... (placement logic for y_p_wave_shape, similar to full beat placement) ...
                    p_start_sample_idx_global = int(p_wave_start_time_global * fs)
                    p_shape_start_idx, p_place_start_idx = 0, p_start_sample_idx_global
                    if p_place_start_idx < 0: p_shape_start_idx = -p_place_start_idx; p_place_start_idx = 0
                    
                    p_samples_in_shape = len(y_p_wave_shape) - p_shape_start_idx
                    p_samples_in_signal = num_total_samples - p_place_start_idx
                    p_samples_to_copy = min(p_samples_in_shape, p_samples_in_signal)

                    if p_samples_to_copy > 0:
                        p_shape_end_idx = p_shape_start_idx + p_samples_to_copy
                        p_place_end_idx = p_place_start_idx + p_samples_to_copy
                        full_ecg_signal_np[p_place_start_idx : p_place_end_idx] += y_p_wave_shape[p_shape_start_idx : p_shape_end_idx]

            # SA node still schedules its next intrinsic firing if its QRS was blocked
            if current_event.source == "sa_node":
                sa_node_next_fire_time = max(sa_node_next_fire_time, potential_qrs_onset_time) + base_rr_interval_sec
                if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                     heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            continue # Skip QRS placement and related state updates

        # --- Place the Full Beat (QRS is conducted) ---
        _, y_beat_shape, qrs_offset_from_shape_start = generate_single_beat_morphology(current_beat_morph_params, fs)
        num_samples_beat_shape = len(y_beat_shape)

        if num_samples_beat_shape > 0: # (Full beat placement logic)
            waveform_start_time_global = potential_qrs_onset_time - qrs_offset_from_shape_start
            start_sample_index_global = int(waveform_start_time_global * fs)
            shape_start_idx, place_start_idx = 0, start_sample_index_global
            if place_start_idx < 0: shape_start_idx = -place_start_idx; place_start_idx = 0
            samples_in_shape_remaining = num_samples_beat_shape - shape_start_idx
            samples_in_signal_remaining = num_total_samples - place_start_idx
            samples_to_copy = min(samples_in_shape_remaining, samples_in_signal_remaining)
            if samples_to_copy > 0:
                shape_end_idx = shape_start_idx + samples_to_copy; place_end_idx = place_start_idx + samples_to_copy
                full_ecg_signal_np[place_start_idx : place_end_idx] += y_beat_shape[shape_start_idx : shape_end_idx]
        
        actual_rr_to_this_beat = potential_qrs_onset_time - last_placed_qrs_onset_time
        last_placed_qrs_onset_time = potential_qrs_onset_time

        # --- Update State and Schedule Next Events ---
        if current_event.source == "sa_node":
            sa_node_next_fire_time = potential_qrs_onset_time + base_rr_interval_sec
            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            ventricle_ready_for_next_qrs_at_time = potential_qrs_onset_time + MIN_REFRACTORY_PERIOD_SEC
            coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
            if enable_pac and np.random.rand() < pac_probability_per_sinus:
                pac_time = potential_qrs_onset_time + (coupling_rr_basis * PAC_COUPLING_FACTOR)
                if pac_time > potential_qrs_onset_time + 0.100: heapq.heappush(event_queue, BeatEvent(pac_time, "pac", "pac_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_time = potential_qrs_onset_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_qrs_onset_time + 0.100: heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif current_event.beat_type == "pac":
            sa_node_next_fire_time = potential_qrs_onset_time + base_rr_interval_sec
            new_event_queue = [e for e in event_queue if not (e.source == "sa_node")]
            heapq.heapify(new_event_queue); event_queue = new_event_queue
            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            ventricle_ready_for_next_qrs_at_time = potential_qrs_onset_time + MIN_REFRACTORY_PERIOD_SEC
        elif current_event.beat_type == "pvc":
            sinus_qrs_before_pvc_cycle = last_placed_qrs_onset_time - actual_rr_to_this_beat
            end_of_compensatory_pause_for_qrs = sinus_qrs_before_pvc_cycle + (2 * base_rr_interval_sec)
            ventricle_ready_for_next_qrs_at_time = end_of_compensatory_pause_for_qrs - 0.01

    noise_amplitude = 0.02
    full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    return full_time_axis_np.tolist(), full_ecg_signal_np.tolist()

# --- API Endpoint Definition ---
class AdvancedECGParams(BaseModel):
    heart_rate_bpm: float = Field(60.0, gt=0) # Default HR to 60 for easier AV block observation
    duration_sec: float = Field(10.0, gt=0)
    enable_pvc: bool = Field(False)
    pvc_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    enable_pac: bool = Field(False)
    pac_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    first_degree_av_block_pr_sec: Optional[float] = Field(None, ge=0.201, le=0.60)
    enable_mobitz_ii_av_block: bool = Field(False) # New
    mobitz_ii_p_waves_per_qrs: int = Field(2, ge=2, description="Ratio of P-waves to QRS for Mobitz II (e.g., 2 for 2:1 block)") # New

@app.post("/api/generate_advanced_ecg")
async def get_advanced_ecg_data(params: AdvancedECGParams):
    description_parts = [f"Sinus {params.heart_rate_bpm}bpm"]
    
    av_block_desc = []
    if params.first_degree_av_block_pr_sec is not None:
        av_block_desc.append(f"1st Degree AVB (PR: {params.first_degree_av_block_pr_sec*1000:.0f}ms)")
    if params.enable_mobitz_ii_av_block:
        av_block_desc.append(f"Mobitz II {params.mobitz_ii_p_waves_per_qrs}:1 AVB")
    
    if av_block_desc:
        description_parts.append("with " + " & ".join(av_block_desc))
    
    ectopic_desc = []
    if params.enable_pac and params.pac_probability_per_sinus > 0:
        ectopic_desc.append(f"PACs ({params.pac_probability_per_sinus*100:.0f}%)")
    if params.enable_pvc and params.pvc_probability_per_sinus > 0:
        ectopic_desc.append(f"PVCs ({params.pvc_probability_per_sinus*100:.0f}%)")
    
    if ectopic_desc:
        # Decide on conjunction based on whether AV block description was already added
        conjunction = " and " if not av_block_desc else ", plus "
        description_parts.append(conjunction + " & ".join(ectopic_desc))
    
    description = "".join(description_parts) # Join without spaces first, then refine
    description = description.replace("with and", "with") # Grammar fix
    description = description.replace("and , plus", ", plus")


    time_axis, ecg_signal = generate_physiologically_accurate_ecg(
        heart_rate_bpm=params.heart_rate_bpm,
        duration_sec=params.duration_sec,
        enable_pvc=params.enable_pvc,
        pvc_probability_per_sinus=params.pvc_probability_per_sinus,
        enable_pac=params.enable_pac,
        pac_probability_per_sinus=params.pac_probability_per_sinus,
        first_degree_av_block_pr_sec=params.first_degree_av_block_pr_sec,
        enable_mobitz_ii_av_block=params.enable_mobitz_ii_av_block, # Pass new
        mobitz_ii_p_waves_per_qrs=params.mobitz_ii_p_waves_per_qrs, # Pass new
        fs=FS
    )
    return {"time_axis": time_axis, "ecg_signal": ecg_signal, "rhythm_generated": description}

# You might want to keep your old endpoint for a while or remove it.
# Example: To keep the old simple one for testing comparison:
# from <your_previous_file_or_code_section> import generate_ecg_rhythm_data as generate_simple_ecg
# @app.post("/api/generate_simple_ecg")
# async def get_simple_ecg_data(params: OldECGRequestParams): ...