# --- ECG Generation Constants ---
FS = 250
BASELINE_MV = 0.0
MIN_REFRACTORY_PERIOD_SEC = 0.200

# --- Beat Morphology Definitions ---
SINUS_PARAMS = {
    "p_duration": 0.09, "pr_interval": 0.16, "qrs_duration": 0.10,
    "st_duration": 0.12, "t_duration": 0.16, "p_amplitude": 0.15,
    "q_amplitude": -0.2, "r_amplitude": 1.0, "s_amplitude": -0.25,
    "t_amplitude": 0.25,
}
PVC_PARAMS = { # Also used as a base for VT beats
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.0,
    "qrs_duration": 0.16, "q_amplitude": -0.05, "r_amplitude": 1.2, # Typical PVC/VT morphology
    "s_amplitude": -0.2, "st_duration": 0.10, "t_duration": 0.22,
    "t_amplitude": -0.6, # Often discordant T-wave
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
    "p_duration": 0.10, "p_amplitude": -0.2,
    "pr_interval": 0.0, "qrs_duration": 0.0, "st_duration": 0.0, "t_duration": 0.0,
    "q_amplitude":0.0, "r_amplitude":0.0, "s_amplitude":0.0, "t_amplitude":0.0,
}
FLUTTER_CONDUCTED_QRS_PARAMS = SINUS_PARAMS.copy()
FLUTTER_CONDUCTED_QRS_PARAMS.update({"p_amplitude": 0.0, "p_duration": 0.0, "pr_interval": 0.14})

SVT_BEAT_PARAMS = SINUS_PARAMS.copy()
SVT_BEAT_PARAMS.update({
    "p_duration": 0.0, "p_amplitude": 0.0, "pr_interval": 0.001,
})

VT_BEAT_PARAMS = PVC_PARAMS.copy() # Monomorphic VT beats often resemble PVCs

BEAT_MORPHOLOGIES = {
    "sinus": SINUS_PARAMS, "pvc": PVC_PARAMS, "pac": PAC_PARAMS,
    "junctional_escape": JUNCTIONAL_ESCAPE_PARAMS,
    "ventricular_escape": VENTRICULAR_ESCAPE_PARAMS,
    "afib_conducted": AFIB_CONDUCTED_QRS_PARAMS,
    "flutter_wave": FLUTTER_WAVE_PARAMS,
    "flutter_conducted_qrs": FLUTTER_CONDUCTED_QRS_PARAMS,
    "svt_beat": SVT_BEAT_PARAMS,
    "vt_beat": VT_BEAT_PARAMS,
}

# --- Ectopic Beat Configuration Constants ---
PVC_COUPLING_FACTOR = 0.60
PAC_COUPLING_FACTOR = 0.70