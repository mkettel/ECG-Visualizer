from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Dict, Any, Optional
import heapq
import math

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
MIN_REFRACTORY_PERIOD_SEC = 0.200 

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
PAC_PARAMS = { # For the PAC itself, PR might be slightly different before SVT
    "p_duration": 0.08, "p_amplitude": 0.12, "pr_interval": 0.14, # Standard PAC PR
    "qrs_duration": 0.09, "st_duration": 0.12, "t_duration": 0.16,
    "q_amplitude": -0.1, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.3,
}
JUNCTIONAL_ESCAPE_PARAMS = {
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.09, "st_duration": 0.12, "t_duration": 0.16,
    "q_amplitude": -0.05, "r_amplitude": 0.8, "s_amplitude": -0.2, "t_amplitude": 0.25,
}
VENTRICULAR_ESCAPE_PARAMS = {
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.16, "st_duration": 0.10, "t_duration": 0.18,
    "q_amplitude": -0.15, "r_amplitude": 0.7, "s_amplitude": -0.5, "t_amplitude": -0.35,
}
AFIB_CONDUCTED_QRS_PARAMS = SINUS_PARAMS.copy()
AFIB_CONDUCTED_QRS_PARAMS.update({"p_amplitude": 0.0, "p_duration": 0.0, "pr_interval": 0.001})

FLUTTER_WAVE_PARAMS = { 
    "p_duration": 0.10, "p_amplitude": -0.2, 
    "pr_interval": 0.0, "qrs_duration": 0.0, "st_duration": 0.0, "t_duration": 0.0,
    "q_amplitude":0.0, "r_amplitude":0.0, "s_amplitude":0.0, "t_amplitude":0.0,
}
FLUTTER_CONDUCTED_QRS_PARAMS = SINUS_PARAMS.copy() 
FLUTTER_CONDUCTED_QRS_PARAMS.update({"p_amplitude": 0.0, "p_duration": 0.0, "pr_interval": 0.14})

SVT_BEAT_PARAMS = SINUS_PARAMS.copy() # AVNRT-like SVT
SVT_BEAT_PARAMS.update({
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.001, # P-wave hidden
})

BEAT_MORPHOLOGIES = {
    "sinus": SINUS_PARAMS, "pvc": PVC_PARAMS, "pac": PAC_PARAMS,
    "junctional_escape": JUNCTIONAL_ESCAPE_PARAMS,
    "ventricular_escape": VENTRICULAR_ESCAPE_PARAMS,
    "afib_conducted": AFIB_CONDUCTED_QRS_PARAMS,
    "flutter_wave": FLUTTER_WAVE_PARAMS,
    "flutter_conducted_qrs": FLUTTER_CONDUCTED_QRS_PARAMS,
    "svt_beat": SVT_BEAT_PARAMS,
}

# --- Ectopic Beat Configuration Constants ---
PVC_COUPLING_FACTOR = 0.60
PAC_COUPLING_FACTOR = 0.70 # Relative to preceding R-R, for PAC timing

# --- Waveform Primitive & Single Beat Generation (unchanged) ---
def gaussian_wave(t_points, center, amplitude, width_std_dev):
    if width_std_dev <= 1e-9: return np.zeros_like(t_points)
    return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

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
        p_width_std_dev = p_duration / 4 if p_duration > 0 else 1e-3 # Avoid division by zero
        beat_waveform += gaussian_wave(t_relative_to_p_onset, p_center, params['p_amplitude'], p_width_std_dev)

    if not draw_only_p:
        qrs_onset_in_array_time = p_wave_total_offset # QRS starts after PR interval (if P exists)
        if qrs_duration > 0:
            # Q-wave (small, initial negative)
            if params.get('q_amplitude', 0) != 0:
                q_center = qrs_onset_in_array_time + qrs_duration * 0.15 # Early in QRS
                q_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, q_center, params['q_amplitude'], q_width_std_dev)
            # R-wave (main positive)
            if params.get('r_amplitude', 0) != 0:
                r_center = qrs_onset_in_array_time + qrs_duration * 0.4 # Mid QRS
                r_width_std_dev = qrs_duration / 6 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, r_center, params['r_amplitude'], r_width_std_dev)
            # S-wave (final negative)
            if params.get('s_amplitude', 0) != 0:
                s_center = qrs_onset_in_array_time + qrs_duration * 0.75 # Late in QRS
                s_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, s_center, params['s_amplitude'], s_width_std_dev)
        
        # T-wave
        t_onset_in_array_time = qrs_onset_in_array_time + qrs_duration + st_duration
        if params.get('t_amplitude', 0) != 0 and t_duration > 0:
            t_center = t_onset_in_array_time + t_duration / 2
            t_width_std_dev = t_duration / 4 if t_duration > 0 else 1e-3
            beat_waveform += gaussian_wave(t_relative_to_p_onset, t_center, params['t_amplitude'], t_width_std_dev)
            
    return t_relative_to_p_onset, beat_waveform, p_wave_total_offset


# --- Fibrillatory Wave Generation for AFib (unchanged) ---
def generate_fibrillatory_waves(duration_sec: float, amplitude_mv: float, fs: int = FS):
    num_samples = int(duration_sec * fs)
    time_axis = np.linspace(0, duration_sec, num_samples, endpoint=False)
    f_wave_signal = np.zeros(num_samples)
    if amplitude_mv <= 1e-6: return f_wave_signal # No waves if amplitude is negligible
    
    # More chaotic f-wave generation
    num_f_waves_per_sec = np.random.uniform(5.8, 10) # Roughly 350-600 "wavelets" per minute
    f_wave_component_duration = np.random.uniform(0.05, 0.08) # Duration of individual wavelet components
    
    current_time = 0.0
    while current_time < duration_sec:
        center = current_time + np.random.uniform(-f_wave_component_duration / 5, f_wave_component_duration / 5) 
        amp = amplitude_mv * np.random.uniform(0.6, 1.4) * np.random.choice([-1, 1]) # Random amplitude and polarity
        width_std = f_wave_component_duration / np.random.uniform(3.5, 5.5) # Random width
        
        # Calculate bounds for this wavelet component to avoid excessive computation
        start_time_comp = center - 3 * width_std
        end_time_comp = center + 3 * width_std
        
        start_idx = max(0, int(np.floor(start_time_comp * fs)))
        end_idx = min(num_samples, int(np.ceil(end_time_comp * fs)))
        
        if start_idx < end_idx: # Ensure there's a valid range
            t_points_fwave_comp = time_axis[start_idx:end_idx]
            if t_points_fwave_comp.size > 0 : # Ensure t_points is not empty
                 f_wave_signal[start_idx:end_idx] += gaussian_wave(t_points_fwave_comp, center, amp, width_std)
        
        # Advance time for the next wavelet, with some randomness
        current_time += (1.0 / num_f_waves_per_sec) * np.random.uniform(0.5, 1.5)
        
    return f_wave_signal

# --- Event-Driven Rhythm Generation ---
class BeatEvent:
    def __init__(self, time: float, beat_type: str, source: str = "sa_node"):
        self.time = time
        self.beat_type = beat_type
        self.source = source
    def __lt__(self, other): return self.time < other.time
    def __repr__(self): return f"BeatEvent(t={self.time:.3f}, type='{self.beat_type}', src='{self.source}')"

def generate_physiologically_accurate_ecg(
    heart_rate_bpm: float, duration_sec: float,
    enable_pvc: bool, pvc_probability_per_sinus: float,
    enable_pac: bool, pac_probability_per_sinus: float,
    first_degree_av_block_pr_sec: Optional[float],
    enable_mobitz_ii_av_block: bool, mobitz_ii_p_waves_per_qrs: int,
    enable_mobitz_i_wenckebach: bool, wenckebach_initial_pr_sec: float,
    wenckebach_pr_increment_sec: float, wenckebach_max_pr_before_drop_sec: float,
    enable_third_degree_av_block: bool, third_degree_escape_rhythm_origin: str,
    third_degree_escape_rate_bpm: Optional[float],
    enable_atrial_fibrillation: bool, afib_average_ventricular_rate_bpm: int,
    afib_fibrillation_wave_amplitude_mv: float, afib_irregularity_factor: float,
    enable_atrial_flutter: bool, atrial_flutter_rate_bpm: int,
    atrial_flutter_av_block_ratio_qrs_to_f: int, atrial_flutter_wave_amplitude_mv: float,
    allow_svt_initiation_by_pac: bool,
    svt_initiation_probability_after_pac: float,
    svt_duration_sec: float,
    svt_rate_bpm: int,
    fs: int
):
    base_rr_interval_sec = 60.0 / heart_rate_bpm if heart_rate_bpm > 0 else float('inf')
    num_total_samples = int(duration_sec * fs)
    full_time_axis_np = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    full_ecg_signal_np = np.full(num_total_samples, BASELINE_MV)
    event_queue: List[BeatEvent] = []
    
    sa_node_next_fire_time = 0.0
    sa_node_last_actual_fire_time_for_p_wave = -base_rr_interval_sec if base_rr_interval_sec != float('inf') else -1.0
    
    last_placed_qrs_onset_time = -base_rr_interval_sec if base_rr_interval_sec != float('inf') else -1.0
    ventricle_ready_for_next_qrs_at_time = 0.0
    
    p_wave_counter_for_mobitz_ii = 0
    current_wenckebach_pr_sec = wenckebach_initial_pr_sec if enable_mobitz_i_wenckebach else None

    is_svt_currently_active: bool = False
    svt_termination_time: Optional[float] = None
    svt_actual_start_time: Optional[float] = None
    svt_actual_end_time: Optional[float] = None
    
    can_have_dynamic_svt = allow_svt_initiation_by_pac
    is_aflutter_active_base = enable_atrial_flutter and not can_have_dynamic_svt
    is_afib_active_base = enable_atrial_fibrillation and not can_have_dynamic_svt and not is_aflutter_active_base
    is_third_degree_block_active_base = enable_third_degree_av_block and not can_have_dynamic_svt and not is_afib_active_base and not is_aflutter_active_base
    is_mobitz_i_active_base = enable_mobitz_i_wenckebach and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base)
    is_mobitz_ii_active_base = enable_mobitz_ii_av_block and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base)
    is_first_degree_av_block_active_base = (first_degree_av_block_pr_sec is not None) and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base or is_mobitz_ii_active_base)
    
    flutter_wave_rr_interval_sec = 0.0
    flutter_wave_counter_for_av_block = 0

    if is_aflutter_active_base:
        flutter_wave_rr_interval_sec = 60.0 / atrial_flutter_rate_bpm if atrial_flutter_rate_bpm > 0 else float('inf')
        if flutter_wave_rr_interval_sec > 0 and flutter_wave_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(0.0, "flutter_wave", "aflutter_focus"))
    elif is_afib_active_base:
        mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm if afib_average_ventricular_rate_bpm > 0 else float('inf')
        if mean_afib_rr_sec > 0 and mean_afib_rr_sec != float('inf'):
            first_qrs_delay = np.random.uniform(0.1, mean_afib_rr_sec * 0.6) 
            heapq.heappush(event_queue, BeatEvent(first_qrs_delay, "afib_conducted", "afib_av_node"))
    elif is_third_degree_block_active_base:
        if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
             heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
        escape_beat_type = "junctional_escape" if third_degree_escape_rhythm_origin == "junctional" else "ventricular_escape"
        default_escape_rate = 45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0
        actual_escape_rate_bpm = third_degree_escape_rate_bpm or default_escape_rate
        escape_rr_interval_sec = 60.0 / actual_escape_rate_bpm if actual_escape_rate_bpm > 0 else float('inf')
        sinus_pr_for_offset = BEAT_MORPHOLOGIES["sinus"]["pr_interval"]
        if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
             sinus_pr_for_offset = first_degree_av_block_pr_sec
        first_escape_fire_time = max(0.1, sinus_pr_for_offset + np.random.uniform(0.05, 0.15))
        if escape_rr_interval_sec > 0 and escape_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(first_escape_fire_time, escape_beat_type, f"{third_degree_escape_rhythm_origin}_escape"))
    else: 
        if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

    while event_queue and event_queue[0].time < duration_sec:
        current_event = heapq.heappop(event_queue)
        potential_event_time = current_event.time
        
        if is_svt_currently_active and svt_termination_time is not None and potential_event_time >= svt_termination_time:
            print(f"DEBUG: SVT Terminating. Potential Event Time: {potential_event_time:.3f}, SVT Term Time: {svt_termination_time:.3f}")
            svt_actual_end_time = svt_termination_time 
            is_svt_currently_active = False
            svt_termination_time = None

            event_queue = [e for e in event_queue if not (e.beat_type == "svt_beat" and e.time >= svt_actual_end_time - 0.001)]
            heapq.heapify(event_queue)
            print(f"DEBUG: Cleaned event queue: {event_queue}")
            
            if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                print(f"DEBUG: SVT End. Last SA P Wave Time: {sa_node_last_actual_fire_time_for_p_wave:.3f}, SVT Actual End: {svt_actual_end_time:.3f}")
                time_since_last_effective_sa_p = svt_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                num_sa_cycles_to_catch_up = math.floor(time_since_last_effective_sa_p / base_rr_interval_sec)
                resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                physiological_post_svt_pause = 0.1 
                sa_node_next_fire_time_after_svt = max(svt_actual_end_time + physiological_post_svt_pause, resumed_sa_fire_time)
                print(f"DEBUG: Time Since Last SA P: {time_since_last_effective_sa_p:.3f}, Cycles to Catch: {num_sa_cycles_to_catch_up}")
                print(f"DEBUG: Resumed SA Fire (Projected): {resumed_sa_fire_time:.3f}, Final Scheduled SA After SVT: {sa_node_next_fire_time_after_svt:.3f}")

                if sa_node_next_fire_time_after_svt < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_svt) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_svt, "sinus", "sa_node"))
                        print(f"DEBUG: Pushed resumed Sinus beat at {sa_node_next_fire_time_after_svt:.3f}")
                    else:
                        print(f"DEBUG: Duplicate SA node event prevented at {sa_node_next_fire_time_after_svt:.3f}")
                else:
                    print(f"DEBUG: Resumed SA beat {sa_node_next_fire_time_after_svt:.3f} is beyond duration_sec {duration_sec:.3f}")
            else:
                print(f"DEBUG: Base RR interval invalid ({base_rr_interval_sec}), cannot resume SA node rhythmically.")
            
            if current_event.beat_type == "svt_beat" and potential_event_time >= svt_actual_end_time - 0.001 :
                print(f"DEBUG: Current event was SVT beat at termination point, continuing loop.")
                if event_queue and event_queue[0].time < duration_sec: continue 
                else: 
                    print(f"DEBUG: Event queue empty or next event beyond duration after skipping terminating SVT beat.")
                    break 
        
        is_atrial_origin_event = current_event.source == "sa_node" or current_event.beat_type == "pac"
        is_escape_event = current_event.source.endswith("_escape")
        is_afib_qrs_event = current_event.source == "afib_av_node"
        is_flutter_wave_event = current_event.beat_type == "flutter_wave"
        is_flutter_conducted_qrs_event = current_event.beat_type == "flutter_conducted_qrs"
        is_svt_beat_event_type = current_event.beat_type == "svt_beat"

        if not is_svt_currently_active and (is_afib_active_base or is_aflutter_active_base) and is_atrial_origin_event:
            if current_event.source == "sa_node":
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                 if not (is_afib_active_base or is_aflutter_active_base or is_third_degree_block_active_base):
                    if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                        if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            continue

        current_beat_morph_params = BEAT_MORPHOLOGIES[current_event.beat_type].copy()
        qrs_is_blocked_by_av_node = False
        draw_p_wave_only_for_this_atrial_event = False
        
        if is_flutter_wave_event:
            flutter_wave_params_local = BEAT_MORPHOLOGIES["flutter_wave"].copy()
            flutter_wave_params_local["p_amplitude"] = atrial_flutter_wave_amplitude_mv
            flutter_wave_params_local["p_duration"] = flutter_wave_rr_interval_sec
            _, y_flutter_wave_shape, _ = generate_single_beat_morphology(flutter_wave_params_local, fs, is_flutter_wave_itself=True)
            if len(y_flutter_wave_shape) > 0:
                fw_start_time_global = potential_event_time
                fw_start_sample_idx = int(fw_start_time_global * fs)
                fw_end_sample_idx = min(fw_start_sample_idx + len(y_flutter_wave_shape), num_total_samples)
                fw_samples_to_copy = fw_end_sample_idx - fw_start_sample_idx
                if fw_samples_to_copy > 0 and fw_start_sample_idx < num_total_samples:
                    full_ecg_signal_np[fw_start_sample_idx:fw_end_sample_idx] += y_flutter_wave_shape[:fw_samples_to_copy]
            
            flutter_wave_counter_for_av_block += 1
            conducts_this_flutter_wave = (flutter_wave_counter_for_av_block % atrial_flutter_av_block_ratio_qrs_to_f == 0) if atrial_flutter_av_block_ratio_qrs_to_f > 0 else False
            
            if conducts_this_flutter_wave and potential_event_time >= ventricle_ready_for_next_qrs_at_time:
                flutter_qrs_pr = BEAT_MORPHOLOGIES["flutter_conducted_qrs"]["pr_interval"] 
                qrs_time_after_flutter = potential_event_time + flutter_qrs_pr 
                heapq.heappush(event_queue, BeatEvent(qrs_time_after_flutter, "flutter_conducted_qrs", "aflutter_conducted"))
            
            if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_counter_for_av_block >= atrial_flutter_av_block_ratio_qrs_to_f:
                flutter_wave_counter_for_av_block = 0

            if flutter_wave_rr_interval_sec > 0 and flutter_wave_rr_interval_sec != float('inf'):
                next_fw_time = potential_event_time + flutter_wave_rr_interval_sec
                if next_fw_time < duration_sec:
                    heapq.heappush(event_queue, BeatEvent(next_fw_time, "flutter_wave", "aflutter_focus"))
            continue

        if not is_svt_currently_active and \
           not (is_afib_active_base or is_aflutter_active_base) and \
           not is_svt_beat_event_type and \
           not is_afib_qrs_event and \
           not is_flutter_conducted_qrs_event and \
           not is_escape_event:
            
            if is_atrial_origin_event: 
                 sa_node_last_actual_fire_time_for_p_wave = potential_event_time

            if is_third_degree_block_active_base and is_atrial_origin_event:
                qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec :
                    current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_mobitz_i_active_base and is_atrial_origin_event:
                if current_wenckebach_pr_sec is None: current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                current_beat_morph_params["pr_interval"] = current_wenckebach_pr_sec
                if current_wenckebach_pr_sec >= wenckebach_max_pr_before_drop_sec:
                    qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                    current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                else: current_wenckebach_pr_sec += wenckebach_pr_increment_sec
            elif is_mobitz_ii_active_base and is_atrial_origin_event:
                p_wave_counter_for_mobitz_ii += 1
                if mobitz_ii_p_waves_per_qrs > 0 and p_wave_counter_for_mobitz_ii % mobitz_ii_p_waves_per_qrs != 1 and mobitz_ii_p_waves_per_qrs > 1 :
                     qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if mobitz_ii_p_waves_per_qrs > 0 and p_wave_counter_for_mobitz_ii >= mobitz_ii_p_waves_per_qrs : p_wave_counter_for_mobitz_ii = 0
                if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                     current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_first_degree_av_block_active_base and is_atrial_origin_event and first_degree_av_block_pr_sec:
                current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
        
        if not draw_p_wave_only_for_this_atrial_event and \
           not qrs_is_blocked_by_av_node and \
           not is_escape_event and \
           potential_event_time < ventricle_ready_for_next_qrs_at_time:
            qrs_is_blocked_by_av_node = True
            if is_atrial_origin_event: draw_p_wave_only_for_this_atrial_event = True 
        
        if qrs_is_blocked_by_av_node:
            if draw_p_wave_only_for_this_atrial_event:
                _, y_p_wave_shape, p_wave_offset_for_drawing = generate_single_beat_morphology(current_beat_morph_params, fs, draw_only_p=True)
                if len(y_p_wave_shape) > 0:
                    p_wave_start_time_global = potential_event_time - p_wave_offset_for_drawing
                    p_start_sample_idx_global = int(p_wave_start_time_global * fs)
                    p_shape_start_idx, p_place_start_idx = 0, p_start_sample_idx_global
                    if p_place_start_idx < 0: p_shape_start_idx = -p_place_start_idx; p_place_start_idx = 0
                    p_samples_in_shape = len(y_p_wave_shape) - p_shape_start_idx
                    p_samples_in_signal = num_total_samples - p_place_start_idx
                    p_samples_to_copy = min(p_samples_in_shape, p_samples_in_signal)
                    if p_samples_to_copy > 0:
                        p_shape_end_idx = p_shape_start_idx + p_samples_to_copy; p_place_end_idx = p_place_start_idx + p_samples_to_copy
                        full_ecg_signal_np[p_place_start_idx : p_place_end_idx] += y_p_wave_shape[p_shape_start_idx : p_shape_end_idx]

            if current_event.source == "sa_node" and not is_svt_currently_active and not (is_afib_active_base or is_aflutter_active_base):
                sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            continue

        _, y_beat_shape, qrs_offset_from_shape_start = generate_single_beat_morphology(current_beat_morph_params, fs, draw_only_p=False)
        if len(y_beat_shape) > 0:
            waveform_start_time_global = potential_event_time - qrs_offset_from_shape_start
            start_sample_index_global = int(waveform_start_time_global * fs)
            shape_start_idx, place_start_idx = 0, start_sample_index_global
            if place_start_idx < 0: shape_start_idx = -place_start_idx; place_start_idx = 0
            samples_in_shape_remaining = len(y_beat_shape) - shape_start_idx
            samples_in_signal_remaining = num_total_samples - place_start_idx
            samples_to_copy = min(samples_in_shape_remaining, samples_in_signal_remaining)
            if samples_to_copy > 0:
                shape_end_idx = shape_start_idx + samples_to_copy; place_end_idx = place_start_idx + samples_to_copy
                full_ecg_signal_np[place_start_idx : place_end_idx] += y_beat_shape[shape_start_idx : shape_end_idx]

        actual_rr_to_this_beat = potential_event_time - last_placed_qrs_onset_time
        last_placed_qrs_onset_time = potential_event_time
        qrs_duration_this_beat = current_beat_morph_params.get('qrs_duration', 0.10)
        ventricle_ready_for_next_qrs_at_time = potential_event_time + max(MIN_REFRACTORY_PERIOD_SEC, qrs_duration_this_beat * 1.8 if qrs_duration_this_beat else MIN_REFRACTORY_PERIOD_SEC)


        if is_svt_beat_event_type and is_svt_currently_active:
            svt_rr_interval_sec = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
            next_svt_event_time = potential_event_time + svt_rr_interval_sec
            if svt_rr_interval_sec != float('inf') and next_svt_event_time < duration_sec and \
               next_svt_event_time < (svt_termination_time if svt_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_svt_event_time, "svt_beat", "svt_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else svt_rr_interval_sec
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and \
                   pvc_time < next_svt_event_time - 0.100 and \
                   (svt_termination_time is None or pvc_time < svt_termination_time - 0.100) :
                     heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif is_afib_qrs_event: 
            mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm if afib_average_ventricular_rate_bpm > 0 else float('inf')
            std_dev_rr = mean_afib_rr_sec * afib_irregularity_factor
            next_rr_variation = np.random.normal(0, std_dev_rr)
            tentative_next_rr = mean_afib_rr_sec + next_rr_variation
            min_physiological_rr = max(MIN_REFRACTORY_PERIOD_SEC, (qrs_duration_this_beat or 0)) + 0.05 
            next_rr = max(min_physiological_rr, tentative_next_rr)
            next_afib_qrs_event_time = potential_event_time + next_rr
            if mean_afib_rr_sec != float('inf') and next_afib_qrs_event_time < duration_sec:
                heapq.heappush(event_queue, BeatEvent(next_afib_qrs_event_time, "afib_conducted", "afib_av_node"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else mean_afib_rr_sec
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_afib_qrs_event_time - 0.100 : 
                    heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif is_flutter_conducted_qrs_event: 
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                ventricular_rr_in_flutter = (flutter_wave_rr_interval_sec * atrial_flutter_av_block_ratio_qrs_to_f) if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_rr_interval_sec > 0 else float('inf')
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else ventricular_rr_in_flutter
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020:
                    heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        
        elif current_event.source == "sa_node": 
            # sa_node_last_actual_fire_time_for_p_wave was updated when AV conduction logic ran
            sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                 if not is_svt_currently_active: 
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            
            coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
            if enable_pac and np.random.rand() < pac_probability_per_sinus:
                pac_time = potential_event_time + (coupling_rr_basis * PAC_COUPLING_FACTOR)
                if pac_time > potential_event_time + 0.100 and pac_time < sa_node_next_fire_time - 0.100: 
                    heapq.heappush(event_queue, BeatEvent(pac_time, "pac", "pac_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                pr_interval_for_next_sinus = current_beat_morph_params.get('pr_interval', BEAT_MORPHOLOGIES["sinus"]['pr_interval'])
                next_potential_sa_qrs = sa_node_next_fire_time + pr_interval_for_next_sinus
                if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_potential_sa_qrs - 0.100:
                     heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif current_event.beat_type == "pac": 
            # sa_node_last_actual_fire_time_for_p_wave was updated when AV conduction logic ran
            sa_node_next_fire_time = potential_event_time + base_rr_interval_sec 
            
            new_event_queue = [e for e in event_queue if not (e.source == "sa_node")] 
            heapq.heapify(new_event_queue); event_queue = new_event_queue
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                 if not is_svt_currently_active: 
                    heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

            if can_have_dynamic_svt and not is_svt_currently_active and \
               not is_afib_active_base and not is_aflutter_active_base and not is_third_degree_block_active_base:
                if np.random.rand() < svt_initiation_probability_after_pac:
                    is_svt_currently_active = True
                    svt_actual_start_time = potential_event_time 
                    svt_termination_time = svt_actual_start_time + svt_duration_sec
                    physio_pause = 0.4
                    resume_time  = svt_actual_start_time + svt_duration_sec + physio_pause
                    if resume_time < duration_sec:
                        heapq.heappush(
                          event_queue,
                          BeatEvent(resume_time, "sinus", "sa_node")
                        )
                    print(f"DEBUG: SVT Initiated. PAC time: {potential_event_time:.3f}, SVT Start: {svt_actual_start_time:.3f}, SVT Term Time: {svt_termination_time:.3f}")
                    print(f"DEBUG: Value of sa_node_last_actual_fire_time_for_p_wave at SVT initiation: {sa_node_last_actual_fire_time_for_p_wave:.3f}")

                    
                    event_queue = [e for e in event_queue if not (e.source == "sa_node" and e.time >= svt_actual_start_time and e.time < svt_termination_time)]
                    heapq.heapify(event_queue)

                    svt_rr = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
                    if svt_rr != float('inf'):
                        first_svt_beat_time = svt_actual_start_time + svt_rr
                        if first_svt_beat_time < duration_sec and first_svt_beat_time < svt_termination_time:
                            heapq.heappush(event_queue, BeatEvent(first_svt_beat_time, "svt_beat", "svt_focus"))
            
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus: 
                coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
                pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                pr_for_next_sinus_after_pac = BEAT_MORPHOLOGIES["sinus"]["pr_interval"]
                if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                    pr_for_next_sinus_after_pac = first_degree_av_block_pr_sec

                next_potential_sa_qrs_after_pac_reset = sa_node_next_fire_time + pr_for_next_sinus_after_pac
                if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_potential_sa_qrs_after_pac_reset - 0.100:
                    heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif current_event.beat_type == "pvc":
            sinus_qrs_before_pvc_cycle_approx = last_placed_qrs_onset_time - actual_rr_to_this_beat
            if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                end_of_compensatory_pause_for_qrs = sinus_qrs_before_pvc_cycle_approx + (2 * base_rr_interval_sec)
                ventricle_ready_for_next_qrs_at_time = max(ventricle_ready_for_next_qrs_at_time, end_of_compensatory_pause_for_qrs - 0.02)

        elif is_escape_event: 
            escape_rate_used = third_degree_escape_rate_bpm or \
                               (45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0)
            escape_rr_interval_sec = 60.0 / escape_rate_used if escape_rate_used > 0 else float('inf')
            if escape_rr_interval_sec > 0 and escape_rr_interval_sec != float('inf'):
                next_escape_fire_time = potential_event_time + escape_rr_interval_sec
                if next_escape_fire_time < duration_sec:
                    heapq.heappush(event_queue, BeatEvent(next_escape_fire_time, current_event.beat_type, current_event.source))

    if is_afib_active_base and not svt_actual_start_time : 
        f_waves = generate_fibrillatory_waves(duration_sec, afib_fibrillation_wave_amplitude_mv, fs)
        full_ecg_signal_np += f_waves
    elif is_svt_currently_active and svt_actual_start_time is not None and svt_termination_time is None: 
        svt_actual_end_time = duration_sec

    noise_amplitude = 0.02
    full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    
    description_parts = []
    base_desc_set = False

    if svt_actual_start_time is not None and svt_actual_end_time is not None:
        svt_desc = f"SVT ({svt_rate_bpm}bpm) from {svt_actual_start_time:.1f}s to {svt_actual_end_time:.1f}s"
        underlying_rhythm_desc = f"Sinus Rhythm at {heart_rate_bpm}bpm"
        av_block_sub_desc = []
        if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
        elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
        elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
            av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
        if av_block_sub_desc: underlying_rhythm_desc += " with " + " & ".join(av_block_sub_desc)
        
        description_parts.append(f"{underlying_rhythm_desc} with an episode of {svt_desc}")
        base_desc_set = True
    
    if not base_desc_set:
        if is_aflutter_active_base:
            description_parts.append(f"Atrial Flutter ({atrial_flutter_rate_bpm}bpm atrial) with {atrial_flutter_av_block_ratio_qrs_to_f}:1 AV Conduction")
        elif is_afib_active_base:
            description_parts.append(f"Atrial Fibrillation (Avg Ventricular Rate: {afib_average_ventricular_rate_bpm}bpm)")
        elif is_third_degree_block_active_base:
            escape_desc = f"{third_degree_escape_rhythm_origin.capitalize()} Escape ({(third_degree_escape_rate_bpm or (45 if third_degree_escape_rhythm_origin == 'junctional' else 30)):.0f}bpm)"
            description_parts.append(f"3rd Degree AV Block (Atrial Rate {heart_rate_bpm}bpm, Ventricular: {escape_desc})")
        else: 
            description_parts.append(f"Sinus Rhythm at {heart_rate_bpm}bpm")
            av_block_sub_desc = []
            if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
            elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
            elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
            if av_block_sub_desc: description_parts[-1] += " with " + " & ".join(av_block_sub_desc)

    ectopic_desc = []
    if enable_pac and pac_probability_per_sinus > 0 and \
       not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base) and \
       svt_actual_start_time is None : 
        ectopic_desc.append(f"PACs ({pac_probability_per_sinus*100:.0f}%)")
    
    if enable_pvc and pvc_probability_per_sinus > 0:
        ectopic_desc.append(f"PVCs ({pvc_probability_per_sinus*100:.0f}%)")
    
    if ectopic_desc:
        conjunction = " and "
        if description_parts:
            last_part = description_parts[-1]
            if "with" in last_part or svt_actual_start_time is not None:
                conjunction = " and "
            elif not last_part.endswith(")") and not "with" in last_part:
                 conjunction = " with "
        elif not description_parts : 
             conjunction = "" 
             if len(ectopic_desc) > 1:
                 first_ectopic = ectopic_desc.pop(0)
                 description_parts.append(first_ectopic)
                 conjunction = " & " 
             else: 
                 description_parts.append(ectopic_desc[0])
                 ectopic_desc = [] 

        if description_parts and ectopic_desc: 
            description_parts[-1] += conjunction + " & ".join(ectopic_desc)
        elif not description_parts and ectopic_desc: 
             description_parts.append(" & ".join(ectopic_desc))

    final_description = " ".join(description_parts).replace("  ", " ").strip()
    final_description = final_description.replace(" with and ", " with ").replace(" and and ", " and ")
    if not final_description: final_description = "Simulated ECG Data"

    return full_time_axis_np.tolist(), full_ecg_signal_np.tolist(), final_description


# --- API Endpoint Definition ---
class AdvancedECGParams(BaseModel):
    heart_rate_bpm: float = Field(75.0, gt=0, description="Base sinus rate if no other dominant rhythm.")
    duration_sec: float = Field(10.0, gt=0)
    
    # Ectopy
    enable_pvc: bool = Field(False)
    pvc_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    enable_pac: bool = Field(False, description="Enable Premature Atrial Contractions.")
    pac_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    
    # AV Blocks (apply if base is Sinus)
    first_degree_av_block_pr_sec: Optional[float] = Field(None, ge=0.201, le=0.60)
    enable_mobitz_ii_av_block: bool = Field(False)
    mobitz_ii_p_waves_per_qrs: int = Field(2, ge=2)
    enable_mobitz_i_wenckebach: bool = Field(False)
    wenckebach_initial_pr_sec: float = Field(0.16, ge=0.12, le=0.40)
    wenckebach_pr_increment_sec: float = Field(0.04, ge=0.01, le=0.15)
    wenckebach_max_pr_before_drop_sec: float = Field(0.32, ge=0.22, le=0.70)
    
    # Dominant Base Rhythms (mutually exclusive with each other and dynamic SVT initiation)
    enable_third_degree_av_block: bool = Field(False)
    third_degree_escape_rhythm_origin: str = Field("junctional")
    third_degree_escape_rate_bpm: Optional[float] = Field(None, gt=15, lt=65)
    
    enable_atrial_fibrillation: bool = Field(False)
    afib_average_ventricular_rate_bpm: int = Field(100, ge=30, le=220)
    afib_fibrillation_wave_amplitude_mv: float = Field(0.05, ge=0.0, le=0.2)
    afib_irregularity_factor: float = Field(0.20, ge=0.05, le=0.50)
    
    enable_atrial_flutter: bool = Field(False)
    atrial_flutter_rate_bpm: int = Field(300, ge=200, le=400)
    atrial_flutter_av_block_ratio_qrs_to_f: int = Field(2, ge=1)
    atrial_flutter_wave_amplitude_mv: float = Field(0.15, ge=0.05, le=0.5)

    # Dynamic SVT Parameters (initiated by PAC, implies Sinus base)
    allow_svt_initiation_by_pac: bool = Field(False, description="Allow PACs to trigger SVT episodes.")
    svt_initiation_probability_after_pac: float = Field(0.3, ge=0.0, le=1.0, description="Probability a PAC triggers SVT.")
    svt_duration_sec: float = Field(10.0, gt=0, description="Duration of an SVT episode once initiated.")
    svt_rate_bpm: int = Field(180, ge=150, le=250, description="Rate of SVT when active.")


@app.post("/api/generate_advanced_ecg")
async def get_advanced_ecg_data(params: AdvancedECGParams):
    
    # If dynamic SVT is allowed, AFib, AFlutter, and 3rd degree block are implicitly disabled as base rhythms.
    # The generator function handles this hierarchy.
    # The frontend should also enforce this UI-wise.

    time_axis, ecg_signal, rhythm_description = generate_physiologically_accurate_ecg(
        heart_rate_bpm=params.heart_rate_bpm, 
        duration_sec=params.duration_sec,
        enable_pvc=params.enable_pvc, 
        pvc_probability_per_sinus=params.pvc_probability_per_sinus,
        enable_pac=params.enable_pac, 
        pac_probability_per_sinus=params.pac_probability_per_sinus,
        first_degree_av_block_pr_sec=params.first_degree_av_block_pr_sec,
        enable_mobitz_ii_av_block=params.enable_mobitz_ii_av_block, 
        mobitz_ii_p_waves_per_qrs=params.mobitz_ii_p_waves_per_qrs,
        enable_mobitz_i_wenckebach=params.enable_mobitz_i_wenckebach,
        wenckebach_initial_pr_sec=params.wenckebach_initial_pr_sec,
        wenckebach_pr_increment_sec=params.wenckebach_pr_increment_sec,
        wenckebach_max_pr_before_drop_sec=params.wenckebach_max_pr_before_drop_sec,
        enable_third_degree_av_block=params.enable_third_degree_av_block,
        third_degree_escape_rhythm_origin=params.third_degree_escape_rhythm_origin,
        third_degree_escape_rate_bpm=params.third_degree_escape_rate_bpm,
        enable_atrial_fibrillation=params.enable_atrial_fibrillation,
        afib_average_ventricular_rate_bpm=params.afib_average_ventricular_rate_bpm,
        afib_fibrillation_wave_amplitude_mv=params.afib_fibrillation_wave_amplitude_mv,
        afib_irregularity_factor=params.afib_irregularity_factor,
        enable_atrial_flutter=params.enable_atrial_flutter, 
        atrial_flutter_rate_bpm=params.atrial_flutter_rate_bpm,
        atrial_flutter_av_block_ratio_qrs_to_f=params.atrial_flutter_av_block_ratio_qrs_to_f,
        atrial_flutter_wave_amplitude_mv=params.atrial_flutter_wave_amplitude_mv,
        allow_svt_initiation_by_pac=params.allow_svt_initiation_by_pac,
        svt_initiation_probability_after_pac=params.svt_initiation_probability_after_pac,
        svt_duration_sec=params.svt_duration_sec,
        svt_rate_bpm=params.svt_rate_bpm,
        fs=FS
    )
    return {"time_axis": time_axis, "ecg_signal": ecg_signal, "rhythm_generated": rhythm_description}