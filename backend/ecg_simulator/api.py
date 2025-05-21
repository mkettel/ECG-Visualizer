# // backend/ecg-simulator/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api_models import AdvancedECGParams
from .rhythm_logic import generate_physiologically_accurate_ecg
from .constants import FS # Import FS from constants

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"], # ADJUST TO YOUR FRONTEND PORT
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generate_advanced_ecg")
async def get_advanced_ecg_data(params: AdvancedECGParams):

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
        enable_vt=params.enable_vt,
        vt_start_time_sec=params.vt_start_time_sec,
        vt_duration_sec=params.vt_duration_sec,
        vt_rate_bpm=params.vt_rate_bpm,
        fs=FS # Pass the imported FS here
    )
    return {"time_axis": time_axis, "ecg_signal": ecg_signal, "rhythm_generated": rhythm_description}