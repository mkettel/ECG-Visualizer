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
    "p_duration": 0.10, 
    "p_amplitude": -0.2, 
    "pr_interval": 0.0, "qrs_duration": 0.0, "st_duration": 0.0, "t_duration": 0.0,
    "q_amplitude":0.0, "r_amplitude":0.0, "s_amplitude":0.0, "t_amplitude":0.0,
}
FLUTTER_CONDUCTED_QRS_PARAMS = SINUS_PARAMS.copy() 
FLUTTER_CONDUCTED_QRS_PARAMS.update({"p_amplitude": 0.0, "p_duration": 0.0, "pr_interval": 0.14})

# NEW: SVT Beat Morphology (AVNRT-like with hidden P)
SVT_BEAT_PARAMS = SINUS_PARAMS.copy()
SVT_BEAT_PARAMS.update({
    "p_duration": 0.0,  # P-wave hidden
    "p_amplitude": 0.0, # P-wave hidden
    "pr_interval": 0.001, # Effectively no PR interval as P is hidden/simultaneous
    # QRS, ST, T are normal (supraventricular origin)
})


BEAT_MORPHOLOGIES = {
    "sinus": SINUS_PARAMS, "pvc": PVC_PARAMS, "pac": PAC_PARAMS,
    "junctional_escape": JUNCTIONAL_ESCAPE_PARAMS,
    "ventricular_escape": VENTRICULAR_ESCAPE_PARAMS,
    "afib_conducted": AFIB_CONDUCTED_QRS_PARAMS,
    "flutter_wave": FLUTTER_WAVE_PARAMS,
    "flutter_conducted_qrs": FLUTTER_CONDUCTED_QRS_PARAMS,
    "svt_beat": SVT_BEAT_PARAMS, # NEW
}

# --- Ectopic Beat Configuration Constants ---
PVC_COUPLING_FACTOR = 0.60
PAC_COUPLING_FACTOR = 0.70

# --- Waveform Primitive & Single Beat Generation (unchanged from provided code) ---
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
                             qrs_duration + st_duration + t_duration + 0.05
    if draw_only_p: total_complex_duration = p_duration + 0.05 if p_duration > 0 else 0.05
    num_samples = int(total_complex_duration * fs)
    if num_samples <= 0: return np.array([]), np.array([]), 0.0
    t_relative_to_p_onset = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    beat_waveform = np.full(num_samples, BASELINE_MV)
    if params.get('p_amplitude', 0) != 0 and p_duration > 0:
        p_center = p_duration / 2; p_width_std_dev = p_duration / 4 if p_duration > 0 else 1e-3
        beat_waveform += gaussian_wave(t_relative_to_p_onset, p_center, params['p_amplitude'], p_width_std_dev)
    if not draw_only_p:
        qrs_onset_in_array_time = p_wave_total_offset
        if qrs_duration > 0:
            if params.get('q_amplitude', 0) != 0:
                q_center = qrs_onset_in_array_time + qrs_duration * 0.15; q_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, q_center, params['q_amplitude'], q_width_std_dev)
            if params.get('r_amplitude', 0) != 0:
                r_center = qrs_onset_in_array_time + qrs_duration * 0.4; r_width_std_dev = qrs_duration / 6 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, r_center, params['r_amplitude'], r_width_std_dev)
            if params.get('s_amplitude', 0) != 0:
                s_center = qrs_onset_in_array_time + qrs_duration * 0.75; s_width_std_dev = qrs_duration / 10 if qrs_duration > 0 else 1e-3
                beat_waveform += gaussian_wave(t_relative_to_p_onset, s_center, params['s_amplitude'], s_width_std_dev)
        t_onset_in_array_time = qrs_onset_in_array_time + qrs_duration + st_duration
        if params.get('t_amplitude', 0) != 0 and t_duration > 0:
            t_center = t_onset_in_array_time + t_duration / 2; t_width_std_dev = t_duration / 4 if t_duration > 0 else 1e-3
            beat_waveform += gaussian_wave(t_relative_to_p_onset, t_center, params['t_amplitude'], t_width_std_dev)
    return t_relative_to_p_onset, beat_waveform, p_wave_total_offset

# --- Fibrillatory Wave Generation for AFib (unchanged) ---
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
        start_time_comp = center - 3 * width_std; end_time_comp = center + 3 * width_std
        start_idx = max(0, int(np.floor(start_time_comp * fs))); end_idx = min(num_samples, int(np.ceil(end_time_comp * fs)))
        if start_idx < end_idx:
            t_points_fwave_comp = time_axis[start_idx:end_idx]
            if t_points_fwave_comp.size > 0 : f_wave_signal[start_idx:end_idx] += gaussian_wave(t_points_fwave_comp, center, amp, width_std)
        current_time += (1.0 / num_f_waves_per_sec) * np.random.uniform(0.5, 1.5)
    return f_wave_signal

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
    enable_mobitz_ii_av_block: bool, mobitz_ii_p_waves_per_qrs: int,
    enable_mobitz_i_wenckebach: bool, wenckebach_initial_pr_sec: float,
    wenckebach_pr_increment_sec: float, wenckebach_max_pr_before_drop_sec: float,
    enable_third_degree_av_block: bool, third_degree_escape_rhythm_origin: str,
    third_degree_escape_rate_bpm: Optional[float],
    enable_atrial_fibrillation: bool, afib_average_ventricular_rate_bpm: int,
    afib_fibrillation_wave_amplitude_mv: float, afib_irregularity_factor: float,
    enable_atrial_flutter: bool, atrial_flutter_rate_bpm: int,
    atrial_flutter_av_block_ratio_qrs_to_f: int, atrial_flutter_wave_amplitude_mv: float,
    # NEW SVT PARAMS
    enable_svt: bool,
    svt_rate_bpm: int,
    fs: int = FS
):
    base_rr_interval_sec = 60.0 / heart_rate_bpm
    num_total_samples = int(duration_sec * fs)
    full_time_axis_np = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    full_ecg_signal_np = np.full(num_total_samples, BASELINE_MV)
    event_queue: List[BeatEvent] = []
    
    sa_node_next_fire_time = 0.0 # Used if SA node is the pacer
    last_placed_qrs_onset_time = -base_rr_interval_sec # Tracks last ventricular activation
    ventricle_ready_for_next_qrs_at_time = 0.0 # Ventricular refractory period end
    
    p_wave_counter_for_mobitz_ii = 0
    current_wenckebach_pr_sec = wenckebach_initial_pr_sec if enable_mobitz_i_wenckebach else None

    # --- Determine Active Dominant Rhythm & Initialize Pacemakers ---
    is_svt_active = enable_svt
    is_aflutter_active = enable_atrial_flutter and not is_svt_active
    is_afib_active = enable_atrial_fibrillation and not is_aflutter_active and not is_svt_active
    is_third_degree_block_active = enable_third_degree_av_block and not is_afib_active and not is_aflutter_active and not is_svt_active

    # These AV block flags are for non-dominant rhythm scenarios
    is_mobitz_i_active = enable_mobitz_i_wenckebach and not is_svt_active and not is_aflutter_active and not is_afib_active and not is_third_degree_block_active
    is_mobitz_ii_active = enable_mobitz_ii_av_block and not is_svt_active and not is_aflutter_active and not is_afib_active and not is_third_degree_block_active and not is_mobitz_i_active
    is_first_degree_av_block_active = (first_degree_av_block_pr_sec is not None) and not is_svt_active and not is_aflutter_active and not is_afib_active and not is_third_degree_block_active and not is_mobitz_i_active and not is_mobitz_ii_active

    flutter_wave_rr_interval_sec = 0.0
    flutter_wave_counter_for_av_block = 0
    
    if is_svt_active:
        event_queue.clear()
        svt_rr_interval_sec = 60.0 / svt_rate_bpm
        # Start SVT slightly after t=0 to allow for potential initiation visualization later
        initial_svt_delay = np.random.uniform(0.05, svt_rr_interval_sec * 0.3) 
        heapq.heappush(event_queue, BeatEvent(initial_svt_delay, "svt_beat", "svt_focus"))
    elif is_aflutter_active:
        event_queue.clear()
        flutter_wave_rr_interval_sec = 60.0 / atrial_flutter_rate_bpm
        heapq.heappush(event_queue, BeatEvent(0.0, "flutter_wave", "aflutter_focus"))
    elif is_afib_active:
        event_queue.clear()
        mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm
        first_qrs_delay = np.random.uniform(0.1, mean_afib_rr_sec * 0.6)
        heapq.heappush(event_queue, BeatEvent(first_qrs_delay, "afib_conducted", "afib_av_node"))
    elif is_third_degree_block_active:
        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node")) # SA node P-waves
        escape_beat_type = "junctional_escape" if third_degree_escape_rhythm_origin == "junctional" else "ventricular_escape"
        default_escape_rate = 45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0
        actual_escape_rate_bpm = third_degree_escape_rate_bpm or default_escape_rate
        escape_rr_interval_sec = 60.0 / actual_escape_rate_bpm
        
        # Try to make escape beat appear after first P if possible
        sinus_pr_for_offset = SINUS_PARAMS["pr_interval"]
        if is_first_degree_av_block_active: sinus_pr_for_offset = first_degree_av_block_pr_sec

        first_escape_fire_time = max(0.1, sinus_pr_for_offset + np.random.uniform(0.05, 0.15))
        heapq.heappush(event_queue, BeatEvent(first_escape_fire_time, escape_beat_type, f"{third_degree_escape_rhythm_origin}_escape"))
    else: # Sinus rhythm base (potentially with 1st/2nd degree AV blocks)
        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

    # --- Main Event Loop ---
    while event_queue and event_queue[0].time < duration_sec:
        current_event = heapq.heappop(event_queue)
        potential_event_time = current_event.time
        
        is_atrial_origin_event = current_event.source == "sa_node" or current_event.beat_type == "pac"
        is_escape_event = current_event.source.endswith("_escape")
        is_afib_qrs_event = current_event.source == "afib_av_node"
        is_flutter_wave_event = current_event.beat_type == "flutter_wave"
        is_flutter_conducted_qrs_event = current_event.beat_type == "flutter_conducted_qrs"
        is_svt_beat_event = current_event.beat_type == "svt_beat" # NEW

        # If a dominant tachyarrhythmia is active, ignore SA node or PAC events for ECG generation
        if (is_svt_active or is_aflutter_active or is_afib_active) and is_atrial_origin_event:
            if current_event.source == "sa_node": # Keep SA node "ticking" conceptually if it's for base rate
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                 # Only reschedule if no other dominant atrial rhythm is active
                 if not (is_svt_active or is_aflutter_active or is_afib_active or is_third_degree_block_active):
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            continue # Don't draw P or QRS for these if SVT/AFib/AFlutter is dominant

        current_beat_morph_params = BEAT_MORPHOLOGIES[current_event.beat_type].copy()
        qrs_is_blocked_by_av_node = False
        draw_p_wave_only_for_this_atrial_event = False
        
        if is_flutter_wave_event: # Flutter wave handling is separate as it's just an atrial wave
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
            conducts_this_flutter_wave = (flutter_wave_counter_for_av_block % atrial_flutter_av_block_ratio_qrs_to_f == 0)
            
            if conducts_this_flutter_wave and potential_event_time >= ventricle_ready_for_next_qrs_at_time:
                flutter_qrs_pr = FLUTTER_CONDUCTED_QRS_PARAMS["pr_interval"] 
                qrs_time_after_flutter = potential_event_time + flutter_qrs_pr
                heapq.heappush(event_queue, BeatEvent(qrs_time_after_flutter, "flutter_conducted_qrs", "aflutter_conducted"))
            
            if flutter_wave_counter_for_av_block >= atrial_flutter_av_block_ratio_qrs_to_f:
                flutter_wave_counter_for_av_block = 0

            next_fw_time = potential_event_time + flutter_wave_rr_interval_sec
            if next_fw_time < duration_sec:
                heapq.heappush(event_queue, BeatEvent(next_fw_time, "flutter_wave", "aflutter_focus"))
            continue # Flutter wave processed, move to next event in queue

        # --- AV Conduction Logic (for SA node / PACs if no dominant tachyarrhythmia is active) ---
        # This block is skipped if SVT, AFib (QRS event), AFlutter (QRS event), or escape beat is processed.
        if not is_svt_beat_event and not is_afib_qrs_event and not is_flutter_conducted_qrs_event and not is_escape_event:
            if is_third_degree_block_active and is_atrial_origin_event:
                qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                # Apply 1st degree PR to P-wave if present, even if blocked
                if is_first_degree_av_block_active and first_degree_av_block_pr_sec is not None:
                    current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_mobitz_i_active and is_atrial_origin_event: # Wenckebach
                if current_wenckebach_pr_sec is None: current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                current_beat_morph_params["pr_interval"] = current_wenckebach_pr_sec
                if current_wenckebach_pr_sec >= wenckebach_max_pr_before_drop_sec:
                    qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                    current_wenckebach_pr_sec = wenckebach_initial_pr_sec # Reset for next cycle
                else: current_wenckebach_pr_sec += wenckebach_pr_increment_sec
            elif is_mobitz_ii_active and is_atrial_origin_event: # Mobitz II
                p_wave_counter_for_mobitz_ii += 1
                # Conduct only the first P of the ratio (e.g., if 3:1, conduct if counter is 1)
                if p_wave_counter_for_mobitz_ii % mobitz_ii_p_waves_per_qrs != 1 and mobitz_ii_p_waves_per_qrs > 1 : # Blocked
                     qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if p_wave_counter_for_mobitz_ii >= mobitz_ii_p_waves_per_qrs : p_wave_counter_for_mobitz_ii = 0 # Reset counter
                
                # If Mobitz II is active, 1st degree PR can still apply to conducted beats
                if is_first_degree_av_block_active and first_degree_av_block_pr_sec is not None:
                     current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif is_first_degree_av_block_active and is_atrial_origin_event and first_degree_av_block_pr_sec is not None:
                current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
        
        # Final QRS blocking check based on ventricular refractoriness
        # This applies to any event trying to cause a QRS (sinus, pac, svt, afib_conducted, flutter_conducted)
        # unless it's already marked as P-only or AV node blocked.
        # Escape beats are generally not subject to this in the same way (they arise due to lack of other activation).
        if not draw_p_wave_only_for_this_atrial_event and \
           not qrs_is_blocked_by_av_node and \
           not is_escape_event and \
           potential_event_time < ventricle_ready_for_next_qrs_at_time:
            qrs_is_blocked_by_av_node = True # Block QRS due to VRP
            if is_atrial_origin_event: draw_p_wave_only_for_this_atrial_event = True # Still draw its P if atrial

        # --- Process Blocked P-waves or Continue to Place Beat ---
        if qrs_is_blocked_by_av_node:
            if draw_p_wave_only_for_this_atrial_event: # Only for atrial origin events that got blocked
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
            
            # Schedule next SA node beat if current event was SA node origin and it got blocked
            if current_event.source == "sa_node" and not (is_svt_active or is_aflutter_active or is_afib_active) :
                sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                     heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            continue # End processing for this blocked event

        # --- Place the Full Beat (PQRST, QRS-T, or SVT beat) ---
        is_p_only_final = (draw_p_wave_only_for_this_atrial_event and is_atrial_origin_event) # Should be false here
        
        _, y_beat_shape, qrs_offset_from_shape_start = generate_single_beat_morphology(current_beat_morph_params, fs, draw_only_p=is_p_only_final)
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
        # Ventricular refractory period starts from QRS onset, not shape start
        ventricle_ready_for_next_qrs_at_time = potential_event_time + max(MIN_REFRACTORY_PERIOD_SEC, qrs_duration_this_beat * 1.8)


        # --- Update State and Schedule Next Events ---
        if is_svt_beat_event: # NEW
            svt_rr_interval_sec = 60.0 / svt_rate_bpm
            next_svt_event_time = potential_event_time + svt_rr_interval_sec
            if next_svt_event_time < duration_sec:
                heapq.heappush(event_queue, BeatEvent(next_svt_event_time, "svt_beat", "svt_focus"))
            # PVCs can occur during SVT
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else svt_rr_interval_sec
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_event_time + qrs_duration_this_beat + 0.020 and pvc_time < next_svt_event_time - 0.100 : # Ensure PVC is after current QRS and before next SVT beat by a margin
                     heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif is_afib_qrs_event:
            mean_afib_rr_sec = 60.0 / afib_average_ventricular_rate_bpm
            std_dev_rr = mean_afib_rr_sec * afib_irregularity_factor
            next_rr_variation = np.random.normal(0, std_dev_rr)
            tentative_next_rr = mean_afib_rr_sec + next_rr_variation
            min_physiological_rr = max(MIN_REFRACTORY_PERIOD_SEC, qrs_duration_this_beat) + 0.05 
            next_rr = max(min_physiological_rr, tentative_next_rr)
            next_afib_qrs_event_time = potential_event_time + next_rr
            heapq.heappush(event_queue, BeatEvent(next_afib_qrs_event_time, "afib_conducted", "afib_av_node"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else mean_afib_rr_sec
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_event_time + qrs_duration_this_beat + 0.020 and pvc_time < next_afib_qrs_event_time - 0.100 : 
                    heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif is_flutter_conducted_qrs_event:
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                ventricular_rr_in_flutter = flutter_wave_rr_interval_sec * atrial_flutter_av_block_ratio_qrs_to_f
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else ventricular_rr_in_flutter
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                # next_flutter_qrs_approx_time = potential_event_time + ventricular_rr_in_flutter - current_beat_morph_params.get('pr_interval',0.14)
                if pvc_time > potential_event_time + qrs_duration_this_beat + 0.020: # and pvc_time < next_flutter_qrs_approx_time - 0.100:
                    heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif current_event.source == "sa_node": # Conducted Sinus or P-wave in 3rd Degree Block
            sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if not (is_svt_active or is_aflutter_active or is_afib_active): # Schedule next SA if not overridden
                 if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                    heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            
            # Ectopic scheduling (only if SA node is the primary atrial driver and not 3rd deg block for PACs)
            can_schedule_pac = enable_pac and not (is_svt_active or is_aflutter_active or is_afib_active or is_third_degree_block_active)
            if can_schedule_pac and np.random.rand() < pac_probability_per_sinus :
                coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
                pac_time = potential_event_time + (coupling_rr_basis * PAC_COUPLING_FACTOR)
                if pac_time > potential_event_time + 0.100 and pac_time < sa_node_next_fire_time - 0.100: 
                    heapq.heappush(event_queue, BeatEvent(pac_time, "pac", "pac_focus"))
            
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
                pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                next_potential_sa_qrs = sa_node_next_fire_time + current_beat_morph_params.get('pr_interval', SINUS_PARAMS['pr_interval'])
                if pvc_time > potential_event_time + qrs_duration_this_beat + 0.020 and pvc_time < next_potential_sa_qrs - 0.100:
                     heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif current_event.beat_type == "pac": # Conducted PAC
            # PAC resets SA node if SA node is the current pacemaker
            if not (is_svt_active or is_aflutter_active or is_afib_active or is_third_degree_block_active):
                sa_node_next_fire_time = potential_event_time + base_rr_interval_sec 
                new_event_queue = [e for e in event_queue if not (e.source == "sa_node")] # Remove old SA events
                heapq.heapify(new_event_queue); event_queue = new_event_queue
                heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node")) # Add new SA event after reset
            
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus: 
                coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec # Use underlying sinus RR as basis
                pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                next_potential_sa_qrs_after_pac_reset = sa_node_next_fire_time + PAC_PARAMS.get('pr_interval', SINUS_PARAMS['pr_interval'])
                if pvc_time > potential_event_time + qrs_duration_this_beat + 0.020 and pvc_time < next_potential_sa_qrs_after_pac_reset - 0.100:
                    heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        elif current_event.beat_type == "pvc": # PVC
            sinus_qrs_before_pvc_cycle_approx = last_placed_qrs_onset_time - actual_rr_to_this_beat
            # For fully compensatory pause, next QRS (if sinus) should land 2*base_rr after the beat *before* the PVC
            # Ventricle is refractory until this compensatory pause is nearly over
            # The timing of the next SA node beat is NOT reset by PVC usually.
            # However, the ventricle won't respond until it's ready.
            end_of_compensatory_pause_for_qrs = sinus_qrs_before_pvc_cycle_approx + (2 * base_rr_interval_sec)
            # PVC itself makes ventricle refractory. MIN_REFRACTORY_PERIOD_SEC is already set from QRS onset.
            # If a P-wave falls during PVC's refractory period, it might be hidden or non-conducted.
            # For simplicity, the main effect is the pause.
            ventricle_ready_for_next_qrs_at_time = max(ventricle_ready_for_next_qrs_at_time, end_of_compensatory_pause_for_qrs - 0.02)

        elif is_escape_event: 
            escape_rr_interval_sec = 60.0 / (third_degree_escape_rate_bpm or (45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0))
            if escape_rr_interval_sec > 0:
                next_escape_fire_time = potential_event_time + escape_rr_interval_sec
                heapq.heappush(event_queue, BeatEvent(next_escape_fire_time, current_event.beat_type, current_event.source))

    if is_afib_active: # Add f-waves post-process for AFib
        f_waves = generate_fibrillatory_waves(duration_sec, afib_fibrillation_wave_amplitude_mv, fs)
        full_ecg_signal_np += f_waves

    noise_amplitude = 0.02
    full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    return full_time_axis_np.tolist(), full_ecg_signal_np.tolist()

# --- API Endpoint Definition ---
class AdvancedECGParams(BaseModel):
    heart_rate_bpm: float = Field(75.0, gt=0) 
    duration_sec: float = Field(10.0, gt=0)
    enable_pvc: bool = Field(False); pvc_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    enable_pac: bool = Field(False); pac_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    first_degree_av_block_pr_sec: Optional[float] = Field(None, ge=0.201, le=0.60)
    enable_mobitz_ii_av_block: bool = Field(False); mobitz_ii_p_waves_per_qrs: int = Field(2, ge=2)
    enable_mobitz_i_wenckebach: bool = Field(False)
    wenckebach_initial_pr_sec: float = Field(0.16, ge=0.12, le=0.40)
    wenckebach_pr_increment_sec: float = Field(0.04, ge=0.01, le=0.15)
    wenckebach_max_pr_before_drop_sec: float = Field(0.32, ge=0.22, le=0.70)
    enable_third_degree_av_block: bool = Field(False)
    third_degree_escape_rhythm_origin: str = Field("junctional") # "junctional" or "ventricular"
    third_degree_escape_rate_bpm: Optional[float] = Field(None, gt=15, lt=65)
    enable_atrial_fibrillation: bool = Field(False)
    afib_average_ventricular_rate_bpm: int = Field(100, ge=30, le=220) # Renamed for clarity
    afib_fibrillation_wave_amplitude_mv: float = Field(0.05, ge=0.0, le=0.2) # Renamed for clarity
    afib_irregularity_factor: float = Field(0.20, ge=0.05, le=0.50) # Renamed for clarity
    enable_atrial_flutter: bool = Field(False)
    atrial_flutter_rate_bpm: int = Field(300, ge=200, le=400)
    atrial_flutter_av_block_ratio_qrs_to_f: int = Field(2, ge=1) # e.g. 2 means 2:1 block
    atrial_flutter_wave_amplitude_mv: float = Field(0.15, ge=0.05, le=0.5)

    # NEW SVT PARAMS
    enable_svt: bool = Field(False)
    svt_rate_bpm: int = Field(180, ge=150, le=250)


@app.post("/api/generate_advanced_ecg")
async def get_advanced_ecg_data(params: AdvancedECGParams):
    description_parts = []
    # Dominant rhythm description
    if params.enable_svt:
        description_parts.append(f"SVT (AVNRT-like) at {params.svt_rate_bpm}bpm")
    elif params.enable_atrial_flutter:
        description_parts.append(f"Atrial Flutter ({params.atrial_flutter_rate_bpm}bpm atrial) with {params.atrial_flutter_av_block_ratio_qrs_to_f}:1 AV Conduction")
    elif params.enable_atrial_fibrillation:
        description_parts.append(f"Atrial Fibrillation (Avg Ventricular Rate: {params.afib_average_ventricular_rate_bpm}bpm)")
    elif params.enable_third_degree_av_block:
        escape_desc = f"{params.third_degree_escape_rhythm_origin.capitalize()} Escape ({params.third_degree_escape_rate_bpm or (45 if params.third_degree_escape_rhythm_origin == 'junctional' else 30):.0f}bpm)"
        description_parts.append(f"3rd Degree AV Block (Atrial Rate {params.heart_rate_bpm}bpm, Ventricular: {escape_desc})")
    else: # Sinus rhythm or Sinus with other AV blocks
        description_parts.append(f"Sinus Rhythm at {params.heart_rate_bpm}bpm")
        av_block_sub_desc = []
        if params.enable_mobitz_i_wenckebach:
             av_block_sub_desc.append(f"2nd Degree AV Block Type I (Wenckebach)")
        elif params.enable_mobitz_ii_av_block:
            av_block_sub_desc.append(f"2nd Degree AV Block Type II (Mobitz II {params.mobitz_ii_p_waves_per_qrs}:1)")
        elif params.first_degree_av_block_pr_sec is not None: 
            av_block_sub_desc.append(f"1st Degree AV Block (PR {params.first_degree_av_block_pr_sec*1000:.0f}ms)")
        
        if av_block_sub_desc:
            description_parts[-1] += " with " + " & ".join(av_block_sub_desc) # Append to Sinus Rhythm string
            
    # Ectopic beats description
    ectopic_desc = []
    # PACs: not with SVT, AFib, AFlutter, or 3rd degree block (where atrial origin is distinct or chaotic)
    if params.enable_pac and params.pac_probability_per_sinus > 0 and \
       not params.enable_svt and \
       not params.enable_atrial_fibrillation and not params.enable_atrial_flutter and \
       not params.enable_third_degree_av_block :
        ectopic_desc.append(f"Premature Atrial Contractions ({params.pac_probability_per_sinus*100:.0f}%)")
    
    # PVCs can occur with most rhythms
    if params.enable_pvc and params.pvc_probability_per_sinus > 0:
        ectopic_desc.append(f"Premature Ventricular Contractions ({params.pvc_probability_per_sinus*100:.0f}%)")
    
    if ectopic_desc:
        # Check if there was already a "with" or if it's a simple base rhythm
        has_existing_modifiers = "with" in description_parts[-1] or \
                                 any([params.enable_svt, params.enable_atrial_flutter, params.enable_atrial_fibrillation, params.enable_third_degree_av_block,
                                      params.enable_mobitz_i_wenckebach, params.enable_mobitz_ii_av_block, params.first_degree_av_block_pr_sec is not None])
        
        if description_parts[-1].startswith("Sinus Rhythm at") and not has_existing_modifiers :
             conjunction = " with " # Sinus Rhythm at 60bpm with PVCs
        elif not description_parts[-1].endswith(")"): # e.g. "SVT (AVNRT-like) at 180bpm"
            conjunction = ", plus " # SVT at 180bpm, plus PVCs
        else: # e.g. "...(Wenckebach)" or "...(PR 240ms)"
            conjunction = " and " # Sinus Rhythm at 60bpm with Wenckebach and PVCs

        description_parts.append(conjunction + " & ".join(ectopic_desc))
    
    final_description = "".join(description_parts).replace("  ", " ").strip()


    time_axis, ecg_signal = generate_physiologically_accurate_ecg(
        heart_rate_bpm=params.heart_rate_bpm, duration_sec=params.duration_sec,
        enable_pvc=params.enable_pvc, pvc_probability_per_sinus=params.pvc_probability_per_sinus,
        enable_pac=params.enable_pac, pac_probability_per_sinus=params.pac_probability_per_sinus,
        first_degree_av_block_pr_sec=params.first_degree_av_block_pr_sec,
        enable_mobitz_ii_av_block=params.enable_mobitz_ii_av_block, mobitz_ii_p_waves_per_qrs=params.mobitz_ii_p_waves_per_qrs,
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
        # NEW SVT PARAMS
        enable_svt=params.enable_svt,
        svt_rate_bpm=params.svt_rate_bpm,
        fs=FS
    )
    return {"time_axis": time_axis, "ecg_signal": ecg_signal, "rhythm_generated": final_description}

# You might want to keep your old endpoint for a while or remove it.
# Example: To keep the old simple one for testing comparison:
# from <your_previous_file_or_code_section> import generate_ecg_rhythm_data as generate_simple_ecg
# @app.post("/api/generate_simple_ecg")
# async def get_simple_ecg_data(params: OldECGRequestParams): ...