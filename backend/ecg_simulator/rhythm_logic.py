# /backend/ecg_simulator/rhythm_logic.py
import numpy as np
import heapq
import math
from typing import List, Dict, Any, Optional

from .constants import (
    BASELINE_MV, MIN_REFRACTORY_PERIOD_SEC, BEAT_MORPHOLOGIES,
    PVC_COUPLING_FACTOR, PAC_COUPLING_FACTOR
)
from .beat_generation import generate_single_beat_morphology
from .waveform_primitives import generate_fibrillatory_waves

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
    # VT Parameters
    enable_vt: bool,
    vt_start_time_sec: Optional[float],
    vt_duration_sec: float,
    vt_rate_bpm: int,
    # TdP Parameters
    enable_torsades: bool = False,
    torsades_start_time_sec: Optional[float] = None,
    torsades_duration_sec: float = 8.0,
    torsades_min_rate_bpm: int = 180,
    torsades_max_rate_bpm: int = 280,
    torsades_amplitude_variation_mv: float = 0.5,
    torsades_cycle_duration_sec: float = 5.0, # Duration for one full amplitude "twist"
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

    # SVT State
    is_svt_currently_active: bool = False
    svt_termination_time: Optional[float] = None
    svt_actual_start_time: Optional[float] = None
    svt_actual_end_time: Optional[float] = None

    # VT State
    is_vt_currently_active: bool = False
    vt_actual_start_time: Optional[float] = None
    vt_calculated_termination_time: Optional[float] = None 
    vt_actual_end_time: Optional[float] = None

    # TdP State Variables
    is_torsades_currently_active: bool = False
    torsades_actual_start_time: Optional[float] = None
    torsades_calculated_termination_time: Optional[float] = None
    torsades_actual_end_time: Optional[float] = None
    torsades_phase_angle_rad: float = 0.0 # For sinusoidal amplitude variation
    
    # Rhythm hierarchy determination
    is_torsades_episode_configured = enable_torsades # New line for TdP

    is_vt_episode_configured = enable_vt and not is_torsades_episode_configured # VT is overridden by TdP
    
    # SVT, AFib, AFlutter, AV blocks are all overridden by TdP or VT
    is_dynamic_svt_episode_configured = allow_svt_initiation_by_pac and not is_torsades_episode_configured and not is_vt_episode_configured
    is_aflutter_active_base = enable_atrial_flutter and not is_dynamic_svt_episode_configured and not is_vt_episode_configured and not is_torsades_episode_configured
    is_afib_active_base = enable_atrial_fibrillation and not is_dynamic_svt_episode_configured and not is_aflutter_active_base and not is_vt_episode_configured and not is_torsades_episode_configured
    is_third_degree_block_active_base = enable_third_degree_av_block and not is_dynamic_svt_episode_configured and not is_afib_active_base and not is_aflutter_active_base and not is_vt_episode_configured and not is_torsades_episode_configured
    
    # Subsequent AV block types are also overridden by TdP or VT, or the atrial tachyarrhythmias
    is_mobitz_i_active_base = enable_mobitz_i_wenckebach and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_vt_episode_configured or is_torsades_episode_configured)
    is_mobitz_ii_active_base = enable_mobitz_ii_av_block and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base or is_vt_episode_configured or is_torsades_episode_configured)
    is_first_degree_av_block_active_base = (first_degree_av_block_pr_sec is not None) and not (is_aflutter_active_base or is_afib_active_base or is_third_degree_block_active_base or is_mobitz_i_active_base or is_mobitz_ii_active_base or is_vt_episode_configured or is_torsades_episode_configured)
    
    flutter_wave_rr_interval_sec = 0.0
    flutter_wave_counter_for_av_block = 0

    # Initial Event Scheduling
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
    elif not is_vt_episode_configured or (vt_start_time_sec is not None and vt_start_time_sec > 0.05): 
        if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))

    if is_vt_episode_configured:
        actual_vt_start_time_for_signal = vt_start_time_sec if vt_start_time_sec is not None else 0.05 
        if actual_vt_start_time_for_signal < duration_sec:
            heapq.heappush(event_queue, BeatEvent(actual_vt_start_time_for_signal, "vt_initiation_signal", "vt_control"))

    if is_torsades_episode_configured:
        actual_torsades_start_time_for_signal = torsades_start_time_sec if torsades_start_time_sec is not None else 0.1 
        if actual_torsades_start_time_for_signal < duration_sec:
            heapq.heappush(event_queue, BeatEvent(actual_torsades_start_time_for_signal, "torsades_initiation_signal", "torsades_control"))


    while event_queue and event_queue[0].time < duration_sec:
        current_event = heapq.heappop(event_queue)
        potential_event_time = current_event.time

        if current_event.beat_type == "torsades_initiation_signal":
            if not is_torsades_currently_active: # Prevent re-initiation if somehow triggered again
                # Terminate any active VT or SVT
                if is_vt_currently_active:
                    # print(f"DEBUG: TdP interrupting active VT at {potential_event_time:.3f}.")
                    is_vt_currently_active = False 
                    vt_calculated_termination_time = None 
                if is_svt_currently_active:
                    # print(f"DEBUG: TdP interrupting active SVT at {potential_event_time:.3f}.")
                    is_svt_currently_active = False 
                    svt_termination_time = None 

                is_torsades_currently_active = True
                torsades_actual_start_time = potential_event_time
                torsades_calculated_termination_time = torsades_actual_start_time + torsades_duration_sec
                torsades_phase_angle_rad = 0.0 

                # Refined event queue cleanup for TdP:
                current_event_queue_copy = list(event_queue) 
                event_queue.clear() 
                for e_val in current_event_queue_copy:
                    # Define types that TdP should interrupt if they are scheduled during its episode
                    conflicting_types = ["sinus", "pac", "pvc", "junctional_escape", "ventricular_escape", 
                                         "afib_conducted", "flutter_conducted_qrs", "svt_beat", "vt_beat",
                                         "flutter_wave", "vt_initiation_signal", "svt_initiation_signal"]
                    
                    is_within_tdp_window = e_val.time >= torsades_actual_start_time and                                       (torsades_calculated_termination_time is None or e_val.time < torsades_calculated_termination_time)
                    
                    is_conflicting = e_val.beat_type in conflicting_types and is_within_tdp_window
                    
                    if not is_conflicting:
                        heapq.heappush(event_queue, e_val)
                    # else:
                        # print(f"DEBUG: TdP removing conflicting event: {e_val}")
                
                # Schedule first TdP beat
                initial_torsades_rr = 60.0 / ((torsades_min_rate_bpm + torsades_max_rate_bpm) / 2) 
                first_torsades_beat_time = torsades_actual_start_time 
                if first_torsades_beat_time < duration_sec and                    first_torsades_beat_time < (torsades_calculated_termination_time if torsades_calculated_termination_time is not None else float('inf')):
                    heapq.heappush(event_queue, BeatEvent(first_torsades_beat_time, "torsades_beat", "torsades_focus"))
            
            if event_queue and event_queue[0].time < duration_sec: heapq.heapify(event_queue); continue 
            else: break
        
        if current_event.beat_type == "vt_initiation_signal":
            if not is_vt_currently_active: 
                if is_svt_currently_active:
                    # print(f"DEBUG: VT interrupting active SVT at {potential_event_time:.3f}.")
                    is_svt_currently_active = False 
                    svt_termination_time = None 
                    event_queue = [e for e in event_queue if not (e.beat_type == "svt_beat" and e.time >= potential_event_time)]
                    heapq.heapify(event_queue)

                # Ensure TdP is not active before initiating VT
                if is_torsades_currently_active:
                    # print(f"DEBUG: VT initiation suppressed by active TdP at {potential_event_time:.3f}")
                    if event_queue and event_queue[0].time < duration_sec: continue
                    else: break
                
                is_vt_currently_active = True
                vt_actual_start_time = potential_event_time
                vt_calculated_termination_time = vt_actual_start_time + vt_duration_sec
                # print(f"DEBUG: VT Initiated. Start: {vt_actual_start_time:.3f}, Scheduled Term Time: {vt_calculated_termination_time:.3f}")

                new_event_queue = []
                for e_val in event_queue:
                    is_sa_or_pac = e_val.source == "sa_node" or e_val.beat_type == "pac"
                    if e_val.time < vt_actual_start_time or \
                       (vt_calculated_termination_time is not None and e_val.time >= vt_calculated_termination_time) or \
                       not is_sa_or_pac: 
                        new_event_queue.append(e_val)
                event_queue = new_event_queue
                heapq.heapify(event_queue)
                print(f"DEBUG: Event queue after VT initiation cleanup: {event_queue}")

                vt_rr = 60.0 / vt_rate_bpm if vt_rate_bpm > 0 else float('inf')
                if vt_rr != float('inf'):
                    first_vt_beat_time = vt_actual_start_time 
                    if first_vt_beat_time < duration_sec and first_vt_beat_time < (vt_calculated_termination_time if vt_calculated_termination_time is not None else float('inf')):
                        heapq.heappush(event_queue, BeatEvent(first_vt_beat_time, "vt_beat", "vt_focus"))
                        physio_pause = 0.15
                        resume_time = vt_actual_start_time + vt_duration_sec + physio_pause
                        heapq.heappush(event_queue, BeatEvent(resume_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : heapq.heapify(event_queue); continue 
            else: break

        if is_torsades_currently_active and torsades_calculated_termination_time is not None and potential_event_time >= torsades_calculated_termination_time:
            # print(f"DEBUG: TdP Terminating. Event Time: {potential_event_time:.3f}, TdP Term Time: {torsades_calculated_termination_time:.3f}")
            torsades_actual_end_time = torsades_calculated_termination_time # Mark actual end time
            is_torsades_currently_active = False
            torsades_calculated_termination_time = None # Clear scheduled termination

            # Remove any TdP beats scheduled at or after this termination time
            event_queue = [e for e in event_queue if not (e.beat_type == "torsades_beat" and e.time >= torsades_actual_end_time - 0.001)]
            
            # Schedule next SA node beat after TdP
            if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                # Use the marked torsades_actual_end_time for calculations
                time_since_last_sa_p_before_episode = torsades_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                num_sa_cycles_to_catch_up = math.floor(time_since_last_sa_p_before_episode / base_rr_interval_sec) if base_rr_interval_sec > 0 else 0
                resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                physiological_post_tachy_pause = 0.20 
                sa_node_next_fire_time_after_episode = max(torsades_actual_end_time + physiological_post_tachy_pause, resumed_sa_fire_time)
                
                if sa_node_next_fire_time_after_episode < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_episode) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_episode, "sinus", "sa_node"))
            
            # If current event is the last TdP beat being processed right at termination time
            if current_event.beat_type == "torsades_beat" and abs(potential_event_time - (torsades_actual_end_time if torsades_actual_end_time is not None else -1)) < 0.001 : # Added None check for safety
                if event_queue and event_queue[0].time < duration_sec: heapq.heapify(event_queue); continue
                else: break
            heapq.heapify(event_queue) # Heapify in other cases before continuing processing or breaking

        if is_vt_currently_active and vt_calculated_termination_time is not None and potential_event_time >= vt_calculated_termination_time:
            # print(f"DEBUG: VT Terminating. Potential Event Time: {potential_event_time:.3f}, VT Scheduled Term Time: {vt_calculated_termination_time:.3f}")
            vt_actual_end_time = vt_calculated_termination_time
            is_vt_currently_active = False
            vt_calculated_termination_time = None 

            event_queue = [e for e in event_queue if not (e.beat_type == "vt_beat" and e.time >= vt_actual_end_time - 0.001)]
            heapq.heapify(event_queue)
            
            # This is the time to schedule the next SA node beat after VT
            if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                time_since_last_sa_p_before_vt_era = vt_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                num_sa_cycles_to_catch_up = math.floor(time_since_last_sa_p_before_vt_era / base_rr_interval_sec) if base_rr_interval_sec > 0 else 0
                resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                physiological_post_vt_pause = 0.15
                sa_node_next_fire_time_after_vt = max(vt_actual_end_time + physiological_post_vt_pause, resumed_sa_fire_time)
                # print(f"DEBUG: Post-VT SA. Last P: {sa_node_last_actual_fire_time_for_p_wave:.3f}, VT End: {vt_actual_end_time:.3f}, "
                #       f"Resumed SA (Proj): {resumed_sa_fire_time:.3f}, Final SA Sched: {sa_node_next_fire_time_after_vt:.3f}")
                
                if sa_node_next_fire_time_after_vt < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_vt) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_vt, "sinus", "sa_node"))
                        # print(f"DEBUG: Pushed resumed Sinus beat post-VT at {sa_node_next_fire_time_after_vt:.3f}")
                # else:
                    # print(f"DEBUG: Resumed SA beat post-VT calculated for {sa_node_next_fire_time_after_vt:.3f} "
                          # f"is NOT scheduled because it is >= duration_sec ({duration_sec:.3f}).")
            
            if current_event.beat_type == "vt_beat" and abs(potential_event_time - (vt_actual_end_time if vt_actual_end_time is not None else -1)) < 0.001 : # This will be the last VT beat, added None check
                if event_queue and event_queue[0].time < duration_sec: heapq.heapify(event_queue); continue
                else: break
            heapq.heapify(event_queue) # Heapify in other cases
        
        if is_svt_currently_active and svt_termination_time is not None and potential_event_time >= svt_termination_time:
            # print(f"DEBUG: SVT Terminating. Potential Event Time: {potential_event_time:.3f}, SVT Term Time: {svt_termination_time:.3f}")
            svt_actual_end_time = svt_termination_time 
            is_svt_currently_active = False 
            svt_termination_time = None

            event_queue = [e for e in event_queue if not (e.beat_type == "svt_beat" and e.time >= svt_actual_end_time - 0.001)]
            heapq.heapify(event_queue)
            
            if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                time_since_last_sa_p_before_svt_era = svt_actual_end_time - sa_node_last_actual_fire_time_for_p_wave
                num_sa_cycles_to_catch_up = math.floor(time_since_last_sa_p_before_svt_era / base_rr_interval_sec) if base_rr_interval_sec > 0 else 0
                resumed_sa_fire_time = sa_node_last_actual_fire_time_for_p_wave + (num_sa_cycles_to_catch_up + 1) * base_rr_interval_sec
                physiological_post_svt_pause = 0.1 
                sa_node_next_fire_time_after_svt = max(svt_actual_end_time + physiological_post_svt_pause, resumed_sa_fire_time)
                # print(f"DEBUG: Post-SVT SA. Last P: {sa_node_last_actual_fire_time_for_p_wave:.3f}, SVT End: {svt_actual_end_time:.3f}, "
                #       f"Resumed SA (Proj): {resumed_sa_fire_time:.3f}, Final SA Sched: {sa_node_next_fire_time_after_svt:.3f}")

                if sa_node_next_fire_time_after_svt < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time_after_svt) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time_after_svt, "sinus", "sa_node"))
                        # print(f"DEBUG: Pushed resumed Sinus beat post-SVT at {sa_node_next_fire_time_after_svt:.3f}")
                # else:
                     # print(f"DEBUG: Resumed SA beat post-SVT calculated for {sa_node_next_fire_time_after_svt:.3f} "
                           # f"is NOT scheduled because it is >= duration_sec ({duration_sec:.3f}).")
            
            if current_event.beat_type == "svt_beat" and abs(potential_event_time - (svt_actual_end_time if svt_actual_end_time is not None else -1)) < 0.001 : # Added None check
                if event_queue and event_queue[0].time < duration_sec: heapq.heapify(event_queue); continue
                else: break
            heapq.heapify(event_queue) # Heapify in other cases
        
        is_atrial_origin_event = current_event.source == "sa_node" or current_event.beat_type == "pac"
        is_escape_event = current_event.source.endswith("_escape")
        is_afib_qrs_event = current_event.source == "afib_av_node"
        is_flutter_wave_event = current_event.beat_type == "flutter_wave"
        is_flutter_conducted_qrs_event = current_event.beat_type == "flutter_conducted_qrs"
        is_svt_beat_event_type = current_event.beat_type == "svt_beat"
        is_vt_beat_event_type = current_event.beat_type == "vt_beat"
        is_torsades_beat_event_type = current_event.beat_type == "torsades_beat"

        if (is_svt_currently_active or is_vt_currently_active or is_torsades_currently_active) and         (is_atrial_origin_event or current_event.source == "sa_node"):
            if current_event.source == "sa_node": 
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if event_queue and event_queue[0].time < duration_sec : heapq.heapify(event_queue); continue # Added heapify
            else: break

        if not (is_svt_currently_active or is_vt_currently_active or is_torsades_currently_active) and         (is_afib_active_base or is_aflutter_active_base) and is_atrial_origin_event: # is_atrial_origin_event includes PACs
            if current_event.source == "sa_node": # Only advance SA node timing if it's an SA node event
                 sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                 # Rescheduling SA node here during Afib/Aflutter is usually not done, as they are the dominant atrial activity.
                 # The original code had a nested check: if not (is_afib_active_base or is_aflutter_active_base or is_third_degree_block_active_base):
                 # This seems to imply that SA node beats are only rescheduled if these aren't active, which is correct.
                 if not (is_afib_active_base or is_aflutter_active_base or is_third_degree_block_active_base): # Keep original rescheduling logic
                    if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                        if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                            heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : heapq.heapify(event_queue); continue # Added heapify
            else: break

        current_beat_morph_params = BEAT_MORPHOLOGIES[current_event.beat_type].copy()
        
        if is_torsades_beat_event_type and is_torsades_currently_active and torsades_actual_start_time is not None:
            time_in_episode = potential_event_time - torsades_actual_start_time
            if torsades_cycle_duration_sec > 0: # Prevent division by zero
                torsades_phase_angle_rad = (time_in_episode / torsades_cycle_duration_sec) * (2 * math.pi)
            else:
                torsades_phase_angle_rad = 0.0
            
            amplitude_mod_factor = math.sin(torsades_phase_angle_rad) # Sinusoidal factor: -1 to 1
            
            current_beat_morph_params["r_amplitude"] += amplitude_mod_factor * torsades_amplitude_variation_mv
            current_beat_morph_params["s_amplitude"] -= amplitude_mod_factor * (torsades_amplitude_variation_mv * 0.7) 
            current_beat_morph_params["q_amplitude"] -= amplitude_mod_factor * (torsades_amplitude_variation_mv * 0.3) 
            current_beat_morph_params["t_amplitude"] -= amplitude_mod_factor * (torsades_amplitude_variation_mv * 0.5)

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
                 if not is_vt_currently_active and not is_svt_currently_active: # AVN conduction suppressed by ventricular/supraventricular tachy
                    flutter_qrs_pr = BEAT_MORPHOLOGIES["flutter_conducted_qrs"]["pr_interval"] 
                    qrs_time_after_flutter = potential_event_time + flutter_qrs_pr 
                    heapq.heappush(event_queue, BeatEvent(qrs_time_after_flutter, "flutter_conducted_qrs", "aflutter_conducted"))
            
            if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_counter_for_av_block >= atrial_flutter_av_block_ratio_qrs_to_f:
                flutter_wave_counter_for_av_block = 0

            if flutter_wave_rr_interval_sec > 0 and flutter_wave_rr_interval_sec != float('inf'):
                next_fw_time = potential_event_time + flutter_wave_rr_interval_sec
                if next_fw_time < duration_sec:
                    if not is_vt_currently_active and not is_svt_currently_active : # Flutter focus suppressed by tachy
                         heapq.heappush(event_queue, BeatEvent(next_fw_time, "flutter_wave", "aflutter_focus"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

        if not is_svt_currently_active and not is_vt_currently_active and \
           not (is_afib_active_base or is_aflutter_active_base) and \
           not is_svt_beat_event_type and not is_vt_beat_event_type and \
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
           not is_escape_event and not is_vt_beat_event_type and \
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

            if current_event.source == "sa_node" and not is_svt_currently_active and not is_vt_currently_active and not (is_afib_active_base or is_aflutter_active_base):
                sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
                if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            if event_queue and event_queue[0].time < duration_sec : continue 
            else: break

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

        if is_vt_beat_event_type and is_vt_currently_active:
            vt_rr_interval_sec = 60.0 / vt_rate_bpm if vt_rate_bpm > 0 else float('inf')
            next_vt_event_time = potential_event_time + vt_rr_interval_sec
            if vt_rr_interval_sec != float('inf') and next_vt_event_time < duration_sec and \
               next_vt_event_time < (vt_calculated_termination_time if vt_calculated_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_vt_event_time, "vt_beat", "vt_focus"))
                print(f"DEBUG: Pushed VT beat at {next_vt_event_time:.3f}")

        elif is_svt_beat_event_type and is_svt_currently_active:
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
                if not is_vt_currently_active and not is_svt_currently_active : # Afib suppressed by other tachy
                    heapq.heappush(event_queue, BeatEvent(next_afib_qrs_event_time, "afib_conducted", "afib_av_node"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active :
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else mean_afib_rr_sec
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_afib_qrs_event_time - 0.100 : 
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif is_flutter_conducted_qrs_event: 
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus:
                if not is_vt_currently_active and not is_svt_currently_active :
                    ventricular_rr_in_flutter = (flutter_wave_rr_interval_sec * atrial_flutter_av_block_ratio_qrs_to_f) if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_rr_interval_sec > 0 else float('inf')
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else ventricular_rr_in_flutter
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020:
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        
        elif is_torsades_beat_event_type and is_torsades_currently_active:
            current_rate_bpm = np.random.uniform(torsades_min_rate_bpm, torsades_max_rate_bpm)
            torsades_rr_interval_sec = 60.0 / current_rate_bpm if current_rate_bpm > 0 else float('inf')
            next_torsades_event_time = potential_event_time + torsades_rr_interval_sec
            if torsades_rr_interval_sec != float('inf') and next_torsades_event_time < duration_sec and                 next_torsades_event_time < (torsades_calculated_termination_time if torsades_calculated_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_torsades_event_time, "torsades_beat", "torsades_focus"))
            # print(f"DEBUG: Pushed TdP beat at {next_torsades_event_time:.3f}, Rate: {current_rate_bpm:.0f}bpm")

        elif is_svt_beat_event_type and is_svt_currently_active:
            svt_rr_interval_sec = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
            next_svt_event_time = potential_event_time + svt_rr_interval_sec
            if svt_rr_interval_sec != float('inf') and next_svt_event_time < duration_sec and                next_svt_event_time < (svt_termination_time if svt_termination_time is not None else float('inf')):
                heapq.heappush(event_queue, BeatEvent(next_svt_event_time, "svt_beat", "svt_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus and                not is_torsades_currently_active and not is_vt_currently_active: # SVT is already active, so no need to check not is_svt_currently_active
                pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else svt_rr_interval_sec
                pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and                    pvc_time < next_svt_event_time - 0.100 and                    (svt_termination_time is None or pvc_time < svt_termination_time - 0.100) :
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
                if not is_vt_currently_active and not is_svt_currently_active and not is_torsades_currently_active : # Afib suppressed by other tachy
                    heapq.heappush(event_queue, BeatEvent(next_afib_qrs_event_time, "afib_conducted", "afib_av_node"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus and                not is_torsades_currently_active and not is_vt_currently_active and not is_svt_currently_active: # Check for all tachycardias
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else mean_afib_rr_sec
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_afib_qrs_event_time - 0.100 : 
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif is_flutter_conducted_qrs_event: 
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus and                not is_torsades_currently_active and not is_vt_currently_active and not is_svt_currently_active: # Check for all tachycardias
                    ventricular_rr_in_flutter = (flutter_wave_rr_interval_sec * atrial_flutter_av_block_ratio_qrs_to_f) if atrial_flutter_av_block_ratio_qrs_to_f > 0 and flutter_wave_rr_interval_sec > 0 else float('inf')
                    pvc_coupling_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else ventricular_rr_in_flutter
                    pvc_time = potential_event_time + (pvc_coupling_basis * PVC_COUPLING_FACTOR)
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020: # Ensure PVC doesn't occur too early
                         # And that it doesn't occur too late (e.g. into the next flutter-conducted QRS, if predictable)
                         # This might need a more robust check if flutter conduction is very regular and fast
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))
        
        elif current_event.source == "sa_node": 
            sa_node_next_fire_time = max(sa_node_next_fire_time, potential_event_time) + base_rr_interval_sec
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                 if not is_svt_currently_active and not is_vt_currently_active and not is_torsades_currently_active: 
                    if not any(e.source == "sa_node" and abs(e.time - sa_node_next_fire_time) < 0.001 for e in event_queue):
                        heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
            
            coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
            if enable_pac and np.random.rand() < pac_probability_per_sinus and                not is_torsades_currently_active and not is_vt_currently_active and not is_svt_currently_active:
                # if not is_vt_currently_active and not is_svt_currently_active: # Original check, now covered
                    pac_time = potential_event_time + (coupling_rr_basis * PAC_COUPLING_FACTOR)
                    if pac_time > potential_event_time + 0.100 and pac_time < sa_node_next_fire_time - 0.100: 
                        heapq.heappush(event_queue, BeatEvent(pac_time, "pac", "pac_focus"))
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus and                not is_torsades_currently_active and not is_vt_currently_active and not is_svt_currently_active:
                # if not is_vt_currently_active and not is_svt_currently_active: # Original check, now covered
                    pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                    pr_interval_for_next_sinus = current_beat_morph_params.get('pr_interval', BEAT_MORPHOLOGIES["sinus"]['pr_interval'])
                    next_potential_sa_qrs = sa_node_next_fire_time + pr_interval_for_next_sinus
                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_potential_sa_qrs - 0.100:
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif current_event.beat_type == "pac": 
                    sa_node_next_fire_time = potential_event_time + base_rr_interval_sec # PAC resets SA node timing
                    
                    # Clear any SA node events that were scheduled before this PAC reset them.
                    # Re-schedule SA node based on the new sa_node_next_fire_time.
                    current_event_queue_copy = list(event_queue)
                    event_queue.clear()
                    sa_rescheduled_post_pac = False
                    for e_val in current_event_queue_copy:
                        if e_val.source == "sa_node": # Remove all previously scheduled SA node events
                            continue 
                        heapq.heappush(event_queue, e_val)
                    
            if base_rr_interval_sec != float('inf') and sa_node_next_fire_time < duration_sec:
                        if not is_svt_currently_active and not is_vt_currently_active and not is_torsades_currently_active: 
                    heapq.heappush(event_queue, BeatEvent(sa_node_next_fire_time, "sinus", "sa_node"))
                            sa_rescheduled_post_pac = True # Mark that it has been rescheduled
                    heapq.heapify(event_queue)


                    if is_dynamic_svt_episode_configured and not is_svt_currently_active and not is_vt_currently_active and not is_torsades_currently_active and                        not is_afib_active_base and not is_aflutter_active_base and not is_third_degree_block_active_base: 
                if np.random.rand() < svt_initiation_probability_after_pac:
                    is_svt_currently_active = True
                    svt_actual_start_time = potential_event_time # This is the PAC's QRS time
                    svt_termination_time = svt_actual_start_time + svt_duration_sec
                            physio_pause = 0.1 
                    resume_time  = svt_actual_start_time + svt_duration_sec + physio_pause
                    
                    # Remove existing SA node events scheduled during SVT, and schedule resumption
                    current_event_queue_copy = list(event_queue)
                    event_queue.clear()
                    sa_rescheduled_post_svt = False
                    for e_val in current_event_queue_copy:
                        is_sa_during_svt = e_val.source == "sa_node" and e_val.time >= svt_actual_start_time and (svt_termination_time is None or e_val.time < svt_termination_time)
                        if not is_sa_during_svt:
                            heapq.heappush(event_queue, e_val)
                    
                    # Schedule SA node resumption if it falls within simulation duration
                    if resume_time < duration_sec:
                        heapq.heappush(event_queue, BeatEvent(resume_time, "sinus", "sa_node"))

                            # print(f"DEBUG: SVT Initiated by PAC. PAC time: {potential_event_time:.3f}, SVT Start: {svt_actual_start_time:.3f}, SVT Term Time: {svt_termination_time:.3f}")
                            # print(f"DEBUG: SA P Wave Time at SVT init: {sa_node_last_actual_fire_time_for_p_wave:.3f}")
                            
                            # Event queue was already rebuilt to remove SA during PAC reset.
                            # Now, ensure SA node is correctly scheduled *after* SVT.
                            current_event_queue_copy_svt = list(event_queue)
                            event_queue.clear()
                            for e_val in current_event_queue_copy_svt:
                                is_sa_during_svt_init = e_val.source == "sa_node" and e_val.time >= svt_actual_start_time and (svt_termination_time is None or e_val.time < svt_termination_time)
                                if not is_sa_during_svt_init:
                                    heapq.heappush(event_queue, e_val)
                            
                            resume_time_svt  = svt_actual_start_time + svt_duration_sec + physio_pause
                            if resume_time_svt < duration_sec:
                                heapq.heappush(event_queue, BeatEvent(resume_time_svt, "sinus", "sa_node"))
                            heapq.heapify(event_queue)


                            svt_rr = 60.0 / svt_rate_bpm if svt_rate_bpm > 0 else float('inf')
                            if svt_rr != float('inf'):
                                first_svt_beat_time = svt_actual_start_time + svt_rr 
                                if first_svt_beat_time < duration_sec and (svt_termination_time is None or first_svt_beat_time < svt_termination_time):
                                    heapq.heappush(event_queue, BeatEvent(first_svt_beat_time, "svt_beat", "svt_focus"))
            
            if enable_pvc and np.random.rand() < pvc_probability_per_sinus and                not is_torsades_currently_active and not is_vt_currently_active and not is_svt_currently_active: # Check all tachy
                    coupling_rr_basis = actual_rr_to_this_beat if actual_rr_to_this_beat > 0.1 else base_rr_interval_sec
                    pvc_time = potential_event_time + (coupling_rr_basis * PVC_COUPLING_FACTOR)
                    
                    # Determine next SA node QRS time *after* PAC has reset the SA node
                    pr_for_next_sinus_after_pac_reset = BEAT_MORPHOLOGIES["sinus"]["pr_interval"]
                    if is_first_degree_av_block_active_base and first_degree_av_block_pr_sec: # This reflects underlying AV conduction if no tachy
                        pr_for_next_sinus_after_pac_reset = first_degree_av_block_pr_sec
                    
                    # sa_node_next_fire_time is already updated by the PAC logic above
                    next_potential_sa_qrs_after_pac = sa_node_next_fire_time + pr_for_next_sinus_after_pac_reset

                    if pvc_time > potential_event_time + (qrs_duration_this_beat or 0) + 0.020 and pvc_time < next_potential_sa_qrs_after_pac - 0.100:
                        heapq.heappush(event_queue, BeatEvent(pvc_time, "pvc", "pvc_focus"))

        elif current_event.beat_type == "pvc":
            if not is_vt_currently_active and not is_svt_currently_active and not is_torsades_currently_active: 
                sinus_qrs_before_pvc_cycle_approx = last_placed_qrs_onset_time - actual_rr_to_this_beat # This is time of QRS preceding PVC's predecessor
                if base_rr_interval_sec > 0 and base_rr_interval_sec != float('inf'):
                    # Compensatory pause: next sinus QRS should be roughly 2*RR from the QRS *before* the PVC
                    # The QRS preceding the PVC's cycle (not the PVC itself)
                    # last_placed_qrs_onset_time is the PVC's onset.
                    # actual_rr_to_this_beat is RR interval ending with PVC.
                    # So, (last_placed_qrs_onset_time - actual_rr_to_this_beat) is the QRS before the PVC.
                    qrs_before_pvc_onset = last_placed_qrs_onset_time - actual_rr_to_this_beat
                    
                    # The next SA node P-wave that *would have occurred* if not for the PVC
                    # is roughly qrs_before_pvc_onset + base_rr_interval_sec (for P-wave time)
                    # Compensatory pause usually means the SA node is not reset by the PVC.
                    # So, the next SA node P-wave occurs on time.
                    # The ventricle, however, is refractory.
                    
                    # Refined compensatory pause logic:
                    # The SA node fires on time after the PVC. The PVC makes the ventricle refractory.
                    # The next expected SA P-wave time if SA node is not reset by PVC:
                    # Find the SA node P-wave that should have followed the cycle where PVC occurred.
                    # sa_node_last_actual_fire_time_for_p_wave is the last P-wave that *actually occurred and was drawn/processed*.
                    
                    # Simpler: extend ventricular refractory period due to PVC.
                    # The next *sinus QRS* is delayed.
                    end_of_compensatory_pause_for_qrs = sinus_qrs_before_pvc_cycle_approx + (2 * base_rr_interval_sec)
                    ventricle_ready_for_next_qrs_at_time = max(ventricle_ready_for_next_qrs_at_time, end_of_compensatory_pause_for_qrs - 0.02)

        elif is_escape_event: 
            if not is_vt_currently_active and not is_svt_currently_active and not is_torsades_currently_active:
                escape_rate_used = third_degree_escape_rate_bpm or \
                                (45.0 if third_degree_escape_rhythm_origin == "junctional" else 30.0)
                escape_rr_interval_sec = 60.0 / escape_rate_used if escape_rate_used > 0 else float('inf')
                if escape_rr_interval_sec > 0 and escape_rr_interval_sec != float('inf'):
                    next_escape_fire_time = potential_event_time + escape_rr_interval_sec
                    if next_escape_fire_time < duration_sec:
                        heapq.heappush(event_queue, BeatEvent(next_escape_fire_time, current_event.beat_type, current_event.source))

    if is_torsades_currently_active and torsades_actual_start_time is not None and torsades_calculated_termination_time is None:
        torsades_actual_end_time = duration_sec # TdP ran till end of simulation
    if is_vt_currently_active and vt_actual_start_time is not None and vt_calculated_termination_time is None: # Must be elif or separate
        vt_actual_end_time = duration_sec # VT ran till end
    if is_svt_currently_active and svt_actual_start_time is not None and svt_termination_time is None:  # Must be elif or separate
        svt_actual_end_time = duration_sec # SVT ran till end

    if is_afib_active_base and not svt_actual_start_time and not vt_actual_start_time and not torsades_actual_start_time: # Check TdP too 
        f_waves = generate_fibrillatory_waves(duration_sec, afib_fibrillation_wave_amplitude_mv, fs)
        full_ecg_signal_np += f_waves


    noise_amplitude = 0.02
    full_ecg_signal_np += noise_amplitude * np.random.normal(0, 1, len(full_ecg_signal_np))
    
    description_parts = []
    base_desc_set = False

    if torsades_actual_start_time is not None and torsades_actual_end_time is not None:
        avg_tdp_rate = (torsades_min_rate_bpm + torsades_max_rate_bpm) // 2
        tdp_desc = f"Torsades de Pointes (~{avg_tdp_rate}bpm, range {torsades_min_rate_bpm}-{torsades_max_rate_bpm}bpm) from {torsades_actual_start_time:.1f}s to {torsades_actual_end_time:.1f}s"
        
        underlying_rhythm_text = ""
        # Check if there was a period before TdP for an underlying rhythm to be stated
        if torsades_actual_start_time > 0.1: # Allow some time for underlying rhythm to be relevant
            if heart_rate_bpm > 0 : # Basic check for sinus
                underlying_rhythm_text = f"Underlying Sinus Rhythm at {heart_rate_bpm}bpm"
                # Check for underlying AV blocks if TdP didn't start right at the beginning
                av_block_sub_desc_underlying = []
                # Use the pre-calculated flags (is_mobitz_i_active_base etc.) which reflect the configuration *without* TdP active
                if enable_mobitz_i_wenckebach and not (enable_atrial_flutter or enable_atrial_fibrillation or enable_third_degree_av_block or enable_vt): av_block_sub_desc_underlying.append("Wenckebach")
                elif enable_mobitz_ii_av_block and not (enable_atrial_flutter or enable_atrial_fibrillation or enable_third_degree_av_block or enable_vt or enable_mobitz_i_wenckebach): av_block_sub_desc_underlying.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
                elif first_degree_av_block_pr_sec and not (enable_atrial_flutter or enable_atrial_fibrillation or enable_third_degree_av_block or enable_vt or enable_mobitz_i_wenckebach or enable_mobitz_ii_av_block):
                    av_block_sub_desc_underlying.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
                
                if av_block_sub_desc_underlying:
                    underlying_rhythm_text += " with " + " & ".join(av_block_sub_desc_underlying)
        
        if underlying_rhythm_text:
          description_parts.append(f"{underlying_rhythm_text} interrupted by {tdp_desc}")
        else:
          description_parts.append(tdp_desc)
        base_desc_set = True 
    
    # Subsequent rhythm descriptions respect base_desc_set
    elif vt_actual_start_time is not None and vt_actual_end_time is not None and not base_desc_set:
        # ... (existing VT description logic, ensure it sets base_desc_set = True)
        # vt_desc = f"Ventricular Tachycardia ({vt_rate_bpm}bpm) from {vt_actual_start_time:.1f}s to {vt_actual_end_time:.1f}s"
        # ... (similar underlying rhythm logic as for TdP might be needed if not already robust)
        # For this subtask, assume existing VT/SVT description logic is mostly okay but ensure they set base_desc_set.
        # The original VT description logic:
        vt_desc = f"Ventricular Tachycardia ({vt_rate_bpm}bpm) from {vt_actual_start_time:.1f}s to {vt_actual_end_time:.1f}s"
        underlying_rhythm_desc_pre_vt = f"Sinus Rhythm at {heart_rate_bpm}bpm" 
        av_block_sub_desc = []
        if not (vt_start_time_sec is not None and vt_start_time_sec < 0.1): 
            if is_mobitz_i_active_base: av_block_sub_desc.append("Wenckebach")
            elif is_mobitz_ii_active_base: av_block_sub_desc.append(f"Mobitz II {mobitz_ii_p_waves_per_qrs}:1")
            elif is_first_degree_av_block_active_base and first_degree_av_block_pr_sec:
                av_block_sub_desc.append(f"1st Deg AVB (PR {first_degree_av_block_pr_sec*1000:.0f}ms)")
        if av_block_sub_desc: underlying_rhythm_desc_pre_vt += " with " + " & ".join(av_block_sub_desc)
        description_parts.append(f"{underlying_rhythm_desc_pre_vt} interrupted by an episode of {vt_desc}")
        base_desc_set = True
    
    elif svt_actual_start_time is not None and svt_actual_end_time is not None:
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
       svt_actual_start_time is None and vt_actual_start_time is None : 
        ectopic_desc.append(f"PACs ({pac_probability_per_sinus*100:.0f}%)")
    
    if enable_pvc and pvc_probability_per_sinus > 0:
        ectopic_desc.append(f"PVCs ({pvc_probability_per_sinus*100:.0f}%)")
    
    if ectopic_desc:
        conjunction = " and "
        if description_parts:
            last_part = description_parts[-1]
            if "with" in last_part or "interrupted by" in last_part or "episode of" in last_part :
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