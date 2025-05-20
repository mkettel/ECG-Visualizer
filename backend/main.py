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
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0, # No antegrade P from junctional focus shown
    "qrs_duration": 0.09, "st_duration": 0.12, "t_duration": 0.16,
    "q_amplitude": -0.05, "r_amplitude": 0.8, "s_amplitude": -0.2, "t_amplitude": 0.25,
}
VENTRICULAR_ESCAPE_PARAMS = {
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.16, "st_duration": 0.10, "t_duration": 0.18, # Wide QRS
    "q_amplitude": -0.15, "r_amplitude": 0.7, "s_amplitude": -0.5, "t_amplitude": -0.35, # Bizarre, discordant T
}
BEAT_MORPHOLOGIES = {
    "sinus": SINUS_PARAMS, "pvc": PVC_PARAMS, "pac": PAC_PARAMS,
    "junctional_escape": JUNCTIONAL_ESCAPE_PARAMS,
    "ventricular_escape": VENTRICULAR_ESCAPE_PARAMS
}

# --- Ectopic Beat Configuration Constants ---
PVC_COUPLING_FACTOR = 0.60
PAC_COUPLING_FACTOR = 0.70

# --- Waveform Primitive & Single Beat Generation (no change from previous version) ---
def gaussian_wave(t_points, center, amplitude, width_std_dev):
    if width_std_dev <= 1e-6: return np.zeros_like(t_points)
    return amplitude * np.exp(-((t_points - center)**2) / (2 * width_std_dev**2))

def generate_single_beat_morphology(params: Dict[str, float], fs: int = FS, draw_only_p: bool = False):
    p_wave_total_offset = params.get('pr_interval', 0) if params.get('p_amplitude',0) !=0 else 0
    p_duration = params.get('p_duration', 0) if params.get('p_amplitude',0) !=0 else 0
    qrs_duration = 0.0 if draw_only_p else params.get('qrs_duration', 0.1)
    st_duration = 0.0 if draw_only_p else params.get('st_duration', 0.1)
    t_duration = 0.0 if draw_only_p else params.get('t_duration', 0.1)
    duration_from_p_onset_to_qrs_onset = p_wave_total_offset
    total_complex_duration = duration_from_p_onset_to_qrs_onset + \
                             qrs_duration + st_duration + t_duration + 0.05
    if draw_only_p: total_complex_duration = p_duration + 0.05
    num_samples = int(total_complex_duration * fs)
    if num_samples <= 0: return np.array([]), np.array([]), 0.0
    t_relative_to_p_onset = np.linspace(0, total_complex_duration, num_samples, endpoint=False)
    beat_waveform = np.full(num_samples, BASELINE_MV)
    if params.get('p_amplitude', 0) != 0 and p_duration > 0:
        p_center = p_duration / 2; p_width_std_dev = p_duration / 4
        beat_waveform += gaussian_wave(t_relative_to_p_onset, p_center, params['p_amplitude'], p_width_std_dev)
    if not draw_only_p:
        qrs_onset_in_array_time = p_wave_total_offset
        if qrs_duration > 0:
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
            t_center = t_onset_in_array_time + t_duration / 2; t_width_std_dev = t_duration / 4
            beat_waveform += gaussian_wave(t_relative_to_p_onset, t_center, params['t_amplitude'], t_width_std_dev)
    return t_relative_to_p_onset, beat_waveform, p_wave_total_offset

# --- Event-Driven Rhythm Generation ---
class BeatEvent: # (no change)
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
    enable_third_degree_av_block: bool, # New
    third_degree_escape_rhythm_origin: str, # New
    third_degree_escape_rate_bpm: Optional[float], # New
    fs: int = FS
):
    base_rr_interval_sec = 60.0 / heart_rate_bpm
    num_total_samples = int(duration_sec * fs)
    full_time_axis_np = np.linspace(0, duration_sec, num_total_samples, endpoint=False)
    full_ecg_signal_np = np.full(num_total_samples, BASELINE_MV)
    event_queue: List[BeatEvent] = []
    
    sa_node_next_fire_time = 0.0
    last_placed_qrs_onset_time = -base_rr_interval_sec # For first beat RR calculation
    ventricle_ready_for_next_qrs_at_time = 0.0
    
    p_wave_counter_for_mobitz_ii = 0
    current_wenckebach_pr_sec = wenckebach_initial_pr_sec if enable_mobitz_i_wenckebach else None

    # --- Initialize Pacemakers ---
    # SA Node always attempts to fire (unless a more dominant atrial focus for rhythms like AFlutter/AFib later)
    heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

    # Escape Pacemaker for 3rd Degree Block
    is_third_degree_block_active = enable_third_degree_av_block
    escape_beat_type = None
    escape_rr_interval_sec = None
    if is_third_degree_block_active:
        escape_beat_type = "junctional_escape" if third_degree_escape_rhythm_origin == "junctional" else "ventricular_escape"
        default_escape_rate = 45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0
        actual_escape_rate_bpm = third_degree_escape_rate_bpm or default_escape_rate
        escape_rr_interval_sec = 60.0 / actual_escape_rate_bpm
        # Schedule first escape beat
        # Start escape beat slightly after first P-wave to show dissociation clearly
        first_escape_fire_time = (SINUS_PARAMS["pr_interval"] + 0.05) if not first_degree_av_block_pr_sec else (first_degree_av_block_pr_sec + 0.05)
        first_escape_fire_time = max(first_escape_fire_time, 0.1) # Ensure it's not too early
        heapq.heappush(event_queue, BeatEvent(first_escape_fire_time, escape_beat_type, f"{third_degree_escape_rhythm_origin}_escape"))


    while event_queue and event_queue[0].time < duration_sec:
        current_event = heapq.heappop(event_queue)
        potential_qrs_onset_time = current_event.time
        is_atrial_origin_event = current_event.source == "sa_node" or current_event.beat_type == "pac"
        is_escape_event = current_event.source.endswith("_escape")
        
        current_beat_morph_params = BEAT_MORPHOLOGIES[current_event.beat_type].copy()
        qrs_is_blocked_by_av_node = False
        draw_p_wave_only_for_this_atrial_event = False

        # --- AV Conduction Logic ---
        if is_third_degree_block_active and is_atrial_origin_event:
            qrs_is_blocked_by_av_node = True # All atrial events blocked from conducting QRS
            draw_p_wave_only_for_this_atrial_event = True
            # PR interval for drawing the P-wave (can be affected by 1st degree setting if we want that nuance)
            if first_degree_av_block_pr_sec is not None:
                 current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
        
        elif not is_escape_event: # Only apply other AV blocks if not 3rd degree and not an escape beat
            # Precedence: Wenckebach > Mobitz II > 1st Degree
            if enable_mobitz_i_wenckebach and is_atrial_origin_event:
                if current_wenckebach_pr_sec is None: current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                current_beat_morph_params["pr_interval"] = current_wenckebach_pr_sec
                if current_wenckebach_pr_sec >= wenckebach_max_pr_before_drop_sec:
                    qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                    current_wenckebach_pr_sec = wenckebach_initial_pr_sec
                else:
                    current_wenckebach_pr_sec += wenckebach_pr_increment_sec
            elif enable_mobitz_ii_av_block and is_atrial_origin_event:
                p_wave_counter_for_mobitz_ii += 1
                if p_wave_counter_for_mobitz_ii % mobitz_ii_p_waves_per_qrs != 1:
                    qrs_is_blocked_by_av_node = True; draw_p_wave_only_for_this_atrial_event = True
                if p_wave_counter_for_mobitz_ii >= mobitz_ii_p_waves_per_qrs:
                    p_wave_counter_for_mobitz_ii = 0
                if first_degree_av_block_pr_sec is not None: # Mobitz II can have underlying 1st degree
                     current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
            elif first_degree_av_block_pr_sec is not None and is_atrial_origin_event:
                current_beat_morph_params["pr_interval"] = first_degree_av_block_pr_sec
        
        # --- Ventricular Refractory / Final QRS Blocking Check ---
        # Escape beats also respect ventricular refractoriness.
        # If QRS was already marked as blocked by AV node, this check is redundant for it.
        if not qrs_is_blocked_by_av_node and potential_qrs_onset_time < ventricle_ready_for_next_qrs_at_time:
            qrs_is_blocked_by_av_node = True # Block due to VRP, even if AV node would have conducted
            # If it was an atrial event, we might still draw its P-wave
            if is_atrial_origin_event : draw_p_wave_only_for_this_atrial_event = True


        if qrs_is_blocked_by_av_node:
            if draw_p_wave_only_for_this_atrial_event: # Draw P for AV blocked atrial events
                _, y_p_wave_shape, p_wave_offset_for_drawing = generate_single_beat_morphology(current_beat_morph_params, fs, draw_only_p=True)
                if len(y_p_wave_shape) > 0:
                    p_wave_start_time_global = potential_qrs_onset_time - p_wave_offset_for_drawing
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
            
            if current_event.source == "sa_node":
                sa_node_next_fire_time = max(sa_node_next_fire_time, potential_qrs_onset_time) + base_rr_interval_sec
                if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                     heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            continue

        # --- Place the Full Beat (QRS is conducted) ---
        _, y_beat_shape, qrs_offset_from_shape_start = generate_single_beat_morphology(current_beat_morph_params, fs)
        # ... (Full beat placement logic - no change from previous version) ...
        if len(y_beat_shape) > 0:
            waveform_start_time_global = potential_qrs_onset_time - qrs_offset_from_shape_start
            start_sample_index_global = int(waveform_start_time_global * fs)
            shape_start_idx, place_start_idx = 0, start_sample_index_global
            if place_start_idx < 0: shape_start_idx = -place_start_idx; place_start_idx = 0
            samples_in_shape_remaining = len(y_beat_shape) - shape_start_idx
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
            # Only add SA node event if 3rd degree block is NOT active (handled by escape pacemaker then)
            if not is_third_degree_block_active: # In 3rd degree, SA node fires independently but doesn't drive ventricle
                 heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            
            ventricle_ready_for_next_qrs_at_time = potential_qrs_onset_time + MIN_REFRACTORY_PERIOD_SEC
            coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
            if enable_pac and np.random.rand() < pac_probability_per_sinus:
                pac_time = potential_qrs_onset_time + (coupling_rr_basis * PAC_COUPLING_FACTOR)
                if pac_time > potential_qrs_onset_time + 0.100: heapq.heappush(event_queue, BeatEvent(pac_time, "pac", "pac_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                pvc_time = potential_qrs_onset_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_qrs_onset_time + 0.100: heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        
        elif current_event.beat_type == "pac": # Conducted PAC
            # PAC resets SA node (unless 3rd degree block is active, then SA node is independent)
            if not is_third_degree_block_active:
                sa_node_next_fire_time = potential_qrs_onset_time + base_rr_interval_sec
                new_event_queue = [e for e in event_queue if not (e.source == "sa_node")]
                heapq.heapify(new_event_queue); event_queue = new_event_queue
                heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            ventricle_ready_for_next_qrs_at_time = potential_qrs_onset_time + MIN_REFRACTORY_PERIOD_SEC
        
        elif current_event.beat_type == "pvc": # PVC
            sinus_qrs_before_pvc_cycle = last_placed_qrs_onset_time - actual_rr_to_this_beat
            end_of_compensatory_pause_for_qrs = sinus_qrs_before_pvc_cycle + (2 * base_rr_interval_sec)
            ventricle_ready_for_next_qrs_at_time = end_of_compensatory_pause_for_qrs - 0.01
            # SA node continues independently

        elif is_escape_event: # Junctional or Ventricular Escape beat
            if escape_rr_interval_sec: # Should always be true if is_escape_event
                next_escape_fire_time = potential_qrs_onset_time + escape_rr_interval_sec
                heapq.heappush(event_queue, BeatEvent(next_escape_fire_time, escape_beat_type, current_event.source))
            ventricle_ready_for_next_qrs_at_time = potential_qrs_onset_time + MIN_REFRACTORY_PERIOD_SEC
            # In 3rd degree block, SA node is already firing independently generating P-waves.

    noise_amplitude = 0.02
    full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    return full_time_axis_np.tolist(), full_ecg_signal_np.tolist()

# --- API Endpoint Definition ---
class AdvancedECGParams(BaseModel):
    heart_rate_bpm: float = Field(60.0, gt=0)
    duration_sec: float = Field(10.0, gt=0)
    enable_pvc: bool = Field(False); pvc_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    enable_pac: bool = Field(False); pac_probability_per_sinus: float = Field(0.0, ge=0, le=1.0)
    first_degree_av_block_pr_sec: Optional[float] = Field(None, ge=0.201, le=0.60)
    enable_mobitz_ii_av_block: bool = Field(False)
    mobitz_ii_p_waves_per_qrs: int = Field(2, ge=2)
    enable_mobitz_i_wenckebach: bool = Field(False)
    wenckebach_initial_pr_sec: float = Field(0.16, ge=0.12, le=0.40)
    wenckebach_pr_increment_sec: float = Field(0.04, ge=0.01, le=0.15)
    wenckebach_max_pr_before_drop_sec: float = Field(0.32, ge=0.22, le=0.70)
    enable_third_degree_av_block: bool = Field(False) # New
    third_degree_escape_rhythm_origin: str = Field("junctional", description="'junctional' or 'ventricular'") # New
    third_degree_escape_rate_bpm: Optional[float] = Field(None, gt=15, lt=65) # New

@app.post("/api/generate_advanced_ecg")
async def get_advanced_ecg_data(params: AdvancedECGParams):
    # Description building logic
    description_parts = []
    if params.enable_third_degree_av_block:
        escape_desc = f"{params.third_degree_escape_rhythm_origin.capitalize()} Escape ({params.third_degree_escape_rate_bpm or (45 if params.third_degree_escape_rhythm_origin == 'junctional' else 30):.0f}bpm)"
        description_parts.append(f"3rd Degree AV Block (SA Node @ {params.heart_rate_bpm}bpm, {escape_desc})")
    else:
        description_parts.append(f"Sinus {params.heart_rate_bpm}bpm")
        av_block_sub_desc = []
        if params.enable_mobitz_i_wenckebach:
             av_block_sub_desc.append(f"Wenckebach (Mobitz I) AVB")
        elif params.enable_mobitz_ii_av_block:
            av_block_sub_desc.append(f"Mobitz II {params.mobitz_ii_p_waves_per_qrs}:1 AVB")
        elif params.first_degree_av_block_pr_sec is not None:
            av_block_sub_desc.append(f"1st Degree AVB (PR: {params.first_degree_av_block_pr_sec*1000:.0f}ms)")
        if av_block_sub_desc:
            description_parts.append("with " + " & ".join(av_block_sub_desc))
            
    ectopic_desc = []
    if params.enable_pac and params.pac_probability_per_sinus > 0:
        ectopic_desc.append(f"PACs ({params.pac_probability_per_sinus*100:.0f}%)")
    if params.enable_pvc and params.pvc_probability_per_sinus > 0:
        ectopic_desc.append(f"PVCs ({params.pvc_probability_per_sinus*100:.0f}%)")
    if ectopic_desc:
        conjunction = " and " if len(description_parts) == 1 else ", plus "
        description_parts.append(conjunction + " & ".join(ectopic_desc))
    
    description = " ".join(description_parts).replace("  ", " ").strip()
    description = description.replace("with and", "with").replace("with , plus", "with")


    time_axis, ecg_signal = generate_physiologically_accurate_ecg(
        heart_rate_bpm=params.heart_rate_bpm,
        duration_sec=params.duration_sec,
        enable_pvc=params.enable_pvc, pvc_probability_per_sinus=params.pvc_probability_per_sinus,
        enable_pac=params.enable_pac, pac_probability_per_sinus=params.pac_probability_per_sinus,
        first_degree_av_block_pr_sec=params.first_degree_av_block_pr_sec,
        enable_mobitz_ii_av_block=params.enable_mobitz_ii_av_block,
        mobitz_ii_p_waves_per_qrs=params.mobitz_ii_p_waves_per_qrs,
        enable_mobitz_i_wenckebach=params.enable_mobitz_i_wenckebach,
        wenckebach_initial_pr_sec=params.wenckebach_initial_pr_sec,
        wenckebach_pr_increment_sec=params.wenckebach_pr_increment_sec,
        wenckebach_max_pr_before_drop_sec=params.wenckebach_max_pr_before_drop_sec,
        enable_third_degree_av_block=params.enable_third_degree_av_block, # Pass new
        third_degree_escape_rhythm_origin=params.third_degree_escape_rhythm_origin, # Pass new
        third_degree_escape_rate_bpm=params.third_degree_escape_rate_bpm, # Pass new
        fs=FS
    )
    return {"time_axis": time_axis, "ecg_signal": ecg_signal, "rhythm_generated": description}

# You might want to keep your old endpoint for a while or remove it.
# Example: To keep the old simple one for testing comparison:
# from <your_previous_file_or_code_section> import generate_ecg_rhythm_data as generate_simple_ecg
# @app.post("/api/generate_simple_ecg")
# async def get_simple_ecg_data(params: OldECGRequestParams): ...