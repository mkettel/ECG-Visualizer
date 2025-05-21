"use client"; // Required for Next.js App Router if using client-side hooks

import React, { useState, useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Decimation,
  ChartOptions,
  ChartData
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Decimation
);

// --- TypeScript Interfaces ---
interface ECGDataState {
  time_axis: number[];
  ecg_signal: number[];
}

interface ECGAPIResponse extends ECGDataState {
  rhythm_generated: string;
}

interface AdvancedRequestBody {
  heart_rate_bpm: number;
  duration_sec: number;
  enable_pvc: boolean;
  pvc_probability_per_sinus: number;
  enable_pac: boolean;
  pac_probability_per_sinus: number;
  first_degree_av_block_pr_sec?: number | null;
  enable_mobitz_ii_av_block?: boolean;
  mobitz_ii_p_waves_per_qrs?: number;
  enable_mobitz_i_wenckebach?: boolean;
  wenckebach_initial_pr_sec?: number;
  wenckebach_pr_increment_sec?: number;
  wenckebach_max_pr_before_drop_sec?: number;
  enable_third_degree_av_block?: boolean;
  third_degree_escape_rhythm_origin?: string;
  third_degree_escape_rate_bpm?: number | null;
  enable_atrial_fibrillation?: boolean;
  afib_average_ventricular_rate_bpm?: number;
  afib_fibrillation_wave_amplitude_mv?: number;
  afib_irregularity_factor?: number;
  enable_atrial_flutter?: boolean;
  atrial_flutter_rate_bpm?: number;
  atrial_flutter_av_block_ratio_qrs_to_f?: number;
  atrial_flutter_wave_amplitude_mv?: number;

  // Dynamic SVT Params
  allow_svt_initiation_by_pac?: boolean;
  svt_initiation_probability_after_pac?: number;
  svt_duration_sec?: number;
  svt_rate_bpm?: number;

  // VT Params
  enable_vt?: boolean;
  vt_start_time_sec?: number | null;
  vt_duration_sec?: number;
  vt_rate_bpm?: number;
}

// Helper function to capitalize strings
const capitalizeFirstLetter = (string: string): string => {
  if (!string) return '';
  return string.charAt(0).toUpperCase() + string.slice(1);
};


const ECGChart: React.FC = () => {
  const [ecgData, setEcgData] = useState<ECGDataState>({ time_axis: [], ecg_signal: [] });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // --- Control State ---
  const [heartRate, setHeartRate] = useState<number>(60);
  const [duration, setDuration] = useState<number>(10);

  const [enablePac, setEnablePac] = useState<boolean>(false);
  const [pacProbability, setPacProbability] = useState<number>(0.1);
  const [enablePvc, setEnablePvc] = useState<boolean>(false);
  const [pvcProbability, setPvcProbability] = useState<number>(0.1);

  const [enableFirstDegreeAVBlock, setEnableFirstDegreeAVBlock] = useState<boolean>(false);
  const [firstDegreePrSec, setFirstDegreePrSec] = useState<number>(0.24);

  const [enableMobitzIIAVBlock, setEnableMobitzIIAVBlock] = useState<boolean>(false);
  const [mobitzIIPWavesPerQRS, setMobitzIIPWavesPerQRS] = useState<number>(3);

  const [enableMobitzIWenckebach, setEnableMobitzIWenckebach] = useState<boolean>(false);
  const [wenckebachInitialPrSec, setWenckebachInitialPrSec] = useState<number>(0.16);
  const [wenckebachPrIncrementSec, setWenckebachPrIncrementSec] = useState<number>(0.04);
  const [wenckebachMaxPrBeforeDropSec, setWenckebachMaxPrBeforeDropSec] = useState<number>(0.32);

  const [enableThirdDegreeAVBlock, setEnableThirdDegreeAVBlock] = useState<boolean>(false);
  const [thirdDegreeEscapeOrigin, setThirdDegreeEscapeOrigin] = useState<string>("junctional");
  const [thirdDegreeEscapeRate, setThirdDegreeEscapeRate] = useState<number>(45);

  const [enableAtrialFibrillation, setEnableAtrialFibrillation] = useState<boolean>(false);
  const [afibVentricularRate, setAfibVentricularRate] = useState<number>(100);
  const [afibAmplitude, setAfibAmplitude] = useState<number>(0.05);
  const [afibIrregularity, setAfibIrregularity] = useState<number>(0.2);

  const [enableAtrialFlutter, setEnableAtrialFlutter] = useState<boolean>(false);
  const [aflutterRate, setAflutterRate] = useState<number>(300);
  const [aflutterConductionRatio, setAflutterConductionRatio] = useState<number>(2);
  const [aflutterAmplitude, setAflutterAmplitude] = useState<number>(0.15);

  // Dynamic SVT State
  const [allowSvtInitiationByPac, setAllowSvtInitiationByPac] = useState<boolean>(false);
  const [svtInitiationProbability, setSvtInitiationProbability] = useState<number>(0.3);
  const [svtDuration, setSvtDuration] = useState<number>(10);
  const [svtRate, setSvtRate] = useState<number>(180);

  // VT State
  const [enableVT, setEnableVT] = useState<boolean>(false);
  const [vtStartTime, setVtStartTime] = useState<number>(0);
  const [vtDuration, setVtDuration] = useState<number>(5);
  const [vtRate, setVtRate] = useState<number>(160);

  const [chartTitle, setChartTitle] = useState<string>('Simulated ECG');
  const chartRef = useRef<ChartJS<'line', number[], string> | null>(null);

  // Helper booleans for disabling controls
  const isAfibActiveBase = enableAtrialFibrillation;
  const isAflutterActiveBase = enableAtrialFlutter;
  const isThirdDegreeBlockActiveBase = enableThirdDegreeAVBlock;
  const isVTActiveBase = enableVT;
  
  const dominantBaseRhythmOverridesPacSvtOrAVBlocks = isAfibActiveBase || isAflutterActiveBase || isThirdDegreeBlockActiveBase || isVTActiveBase;

  const baseHrDisabled = dominantBaseRhythmOverridesPacSvtOrAVBlocks; 
  const avBlocksDisabled = dominantBaseRhythmOverridesPacSvtOrAVBlocks;
  const pacsAndDynamicSvtSettingsDisabled = dominantBaseRhythmOverridesPacSvtOrAVBlocks;


  const fetchEcgData = async () => {
    setIsLoading(true);
    setError(null);

    // --- Validations ---
    if ((enablePac && (pacProbability < 0 || pacProbability > 1)) ||
        (enablePvc && (pvcProbability < 0 || pvcProbability > 1))) {
      setError("Ectopic probabilities must be between 0.0 and 1.0."); setIsLoading(false); return;
    }
    if (enableFirstDegreeAVBlock && (firstDegreePrSec < 0.201 || firstDegreePrSec > 0.60)) {
      setError("1st Degree AV Block PR interval must be between 0.201s and 0.60s."); setIsLoading(false); return;
    }
    if (enableMobitzIIAVBlock && (mobitzIIPWavesPerQRS < 2)) {
      setError("Mobitz II P-waves per QRS must be 2 or greater."); setIsLoading(false); return;
    }
    if (enableMobitzIWenckebach) {
        if (wenckebachInitialPrSec < 0.12 || wenckebachInitialPrSec > 0.40) { setError("Wenckebach Initial PR must be 0.12-0.40s."); setIsLoading(false); return; }
        if (wenckebachPrIncrementSec < 0.01 || wenckebachPrIncrementSec > 0.15) { setError("Wenckebach PR Increment must be 0.01-0.15s."); setIsLoading(false); return; }
        if (wenckebachMaxPrBeforeDropSec < 0.22 || wenckebachMaxPrBeforeDropSec > 0.70 || wenckebachMaxPrBeforeDropSec <= wenckebachInitialPrSec) { setError("Wenckebach Max PR must be 0.22-0.70s & > Initial PR."); setIsLoading(false); return; }
    }
    if (enableThirdDegreeAVBlock && (thirdDegreeEscapeRate < 15 || thirdDegreeEscapeRate > 65)) {
        setError("3rd Degree AV Block Escape Rate must be between 15 and 65 bpm."); setIsLoading(false); return;
    }
    if (enableAtrialFibrillation) {
        if (afibVentricularRate < 30 || afibVentricularRate > 220) {
            setError("Atrial Fibrillation ventricular rate must be between 30 and 220 bpm."); setIsLoading(false); return;
        }
        if (afibAmplitude < 0.0 || afibAmplitude > 0.2) {
            setError("Atrial Fibrillation wave amplitude must be between 0.0 and 0.2mV."); setIsLoading(false); return;
        }
        if (afibIrregularity < 0.05 || afibIrregularity > 0.50) {
            setError("Atrial Fibrillation irregularity factor must be between 0.05 and 0.50."); setIsLoading(false); return;
        }
    }
    if (enableAtrialFlutter) {
        if (aflutterRate < 200 || aflutterRate > 400) {
            setError("Atrial Flutter rate must be between 200 and 400 bpm."); setIsLoading(false); return;
        }
        if (aflutterConductionRatio < 1 ) { 
            setError("Atrial Flutter conduction ratio must be 1 or greater."); setIsLoading(false); return;
        }
        if (aflutterAmplitude < 0.05 || aflutterAmplitude > 0.5) {
          setError("Atrial Flutter wave amplitude must be between 0.05mV and 0.5mV."); setIsLoading(false); return;
        }
    }
    if (allowSvtInitiationByPac) {
        if (svtInitiationProbability < 0.0 || svtInitiationProbability > 1.0) {
            setError("SVT initiation probability must be between 0.0 and 1.0."); setIsLoading(false); return;
        }
        if (svtDuration <= 0 || svtDuration > duration ) {
            setError(`SVT duration must be > 0 and <= total duration (${duration}s).`); setIsLoading(false); return;
        }
        if (svtRate < 150 || svtRate > 250) {
            setError("SVT rate (when active) must be between 150 and 250 bpm."); setIsLoading(false); return;
        }
    }

    // Determine active flags for request body construction
    const sendEnableVT = enableVT;
    const sendEnableAfib = enableAtrialFibrillation && !sendEnableVT;
    const sendEnableAflutter = enableAtrialFlutter && !sendEnableAfib && !sendEnableVT;
    const sendEnableThirdDegreeAVBlock = enableThirdDegreeAVBlock && !sendEnableAfib && !sendEnableAflutter && !sendEnableVT;

    const sendAllowSvtInitiation = allowSvtInitiationByPac && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;
    
    const sendEnableFirstDegreeAVBlock = enableFirstDegreeAVBlock && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;
    const sendEnableMobitzI = enableMobitzIWenckebach && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;
    const sendEnableMobitzII = enableMobitzIIAVBlock && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableMobitzI && !sendEnableVT;
    
    const sendEnablePac = enablePac && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableVT;

    const requestBody: AdvancedRequestBody = {
      heart_rate_bpm: heartRate, 
      duration_sec: duration,
      enable_pac: sendEnablePac, 
      pac_probability_per_sinus: sendEnablePac ? pacProbability : 0,
      enable_pvc: enablePvc, 
      pvc_probability_per_sinus: enablePvc ? pvcProbability : 0,
      
      first_degree_av_block_pr_sec: sendEnableFirstDegreeAVBlock ? firstDegreePrSec : null,
      enable_mobitz_ii_av_block: sendEnableMobitzII,
      mobitz_ii_p_waves_per_qrs: sendEnableMobitzII ? mobitzIIPWavesPerQRS : 2,
      enable_mobitz_i_wenckebach: sendEnableMobitzI,
      wenckebach_initial_pr_sec: sendEnableMobitzI ? wenckebachInitialPrSec : 0.16,
      wenckebach_pr_increment_sec: sendEnableMobitzI ? wenckebachPrIncrementSec : 0.04,
      wenckebach_max_pr_before_drop_sec: sendEnableMobitzI ? wenckebachMaxPrBeforeDropSec : 0.32,
      
      enable_third_degree_av_block: sendEnableThirdDegreeAVBlock,
      third_degree_escape_rhythm_origin: sendEnableThirdDegreeAVBlock ? thirdDegreeEscapeOrigin : "junctional",
      third_degree_escape_rate_bpm: sendEnableThirdDegreeAVBlock ? thirdDegreeEscapeRate : null,
      
      enable_atrial_fibrillation: sendEnableAfib,
      afib_average_ventricular_rate_bpm: sendEnableAfib ? afibVentricularRate : 100,
      afib_fibrillation_wave_amplitude_mv: sendEnableAfib ? afibAmplitude : 0.05,
      afib_irregularity_factor: sendEnableAfib ? afibIrregularity : 0.2,

      enable_atrial_flutter: sendEnableAflutter,
      atrial_flutter_rate_bpm: sendEnableAflutter ? aflutterRate : 300,
      atrial_flutter_av_block_ratio_qrs_to_f: sendEnableAflutter ? aflutterConductionRatio : 2,
      atrial_flutter_wave_amplitude_mv: sendEnableAflutter ? aflutterAmplitude : 0.15,

      allow_svt_initiation_by_pac: sendAllowSvtInitiation,
      svt_initiation_probability_after_pac: sendAllowSvtInitiation ? svtInitiationProbability : 0.3,
      svt_duration_sec: sendAllowSvtInitiation ? svtDuration : 10.0,
      svt_rate_bpm: sendAllowSvtInitiation ? svtRate : 180,
      
      enable_vt: sendEnableVT,
      vt_start_time_sec: sendEnableVT ? vtStartTime : null,
      vt_duration_sec: sendEnableVT ? vtDuration : 5.0,
      vt_rate_bpm: sendEnableVT ? vtRate : 160,
    };

    try {
      const response = await fetch('http://localhost:8000/api/generate_advanced_ecg', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', },
        body: JSON.stringify(requestBody),
      });
      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        try { const errData = await response.json(); errorMessage = errData.detail || errorMessage; }
        catch (jsonError) { errorMessage = response.statusText || "Unknown server error"; }
        throw new Error(errorMessage);
      }
      const data: ECGAPIResponse = await response.json();
      setEcgData({ time_axis: data.time_axis || [], ecg_signal: data.ecg_signal || [] });
      setChartTitle(data.rhythm_generated || `Simulated ECG`);
    } catch (e: any) {
      console.error("Failed to fetch ECG data:", e);
      const message = e instanceof Error ? e.message : "An unknown error occurred";
      setError(message);
      setEcgData({ time_axis: [], ecg_signal: [] });
      setChartTitle('Error generating rhythm');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => { fetchEcgData(); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, []);

  const chartDataConfig: ChartData<'line', number[], string> = {
    labels: ecgData.time_axis.map(t => t.toFixed(2)),
    datasets: [
      {
        label: 'ECG Signal (mV)',
        data: ecgData.ecg_signal,
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.05,
      },
    ],
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: 'Time (s)', color: '#CBD5E1' },
        ticks: {
          maxTicksLimit: Math.max(10, Math.min(20, duration * 2)),
          autoSkipPadding: 20,
          color: '#9CA3AF',
        },
        grid: { color: '#4B5563' }
      },
      y: {
        title: { display: true, text: 'Amplitude (mV)', color: '#CBD5E1' },
        suggestedMin: -1.0,
        suggestedMax: 1.5,
        ticks: { color: '#9CA3AF', stepSize: 0.5 },
        grid: { color: '#4B5563' }
      },
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: { color: '#F3F4F6' }
      },
      title: {
        display: true,
        text: chartTitle,
        color: '#F9FAFB',
        font: { size: 16 }
      },
      decimation: {
        enabled: true,
        algorithm: 'lttb',
        samples: Math.min(1000, ecgData.time_axis.length || 500),
      },
    },
  };
    
  // --- Event Handlers ---
  const handleHeartRateChange = (e: React.ChangeEvent<HTMLInputElement>) => setHeartRate(parseFloat(e.target.value) || 0);
  const handleDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => setDuration(parseFloat(e.target.value) || 0);
  
  const handleEnablePacChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnablePac(isEnabled);
    if (!isEnabled) {
        setAllowSvtInitiationByPac(false);
    }
  };
  const handlePacProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setPacProbability(isNaN(val) ? 0 : Math.max(0, Math.min(1, val)));
  };
  
  const handleEnablePvcChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnablePvc(e.target.checked);
  const handlePvcProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setPvcProbability(isNaN(val) ? 0 : Math.max(0, Math.min(1, val)));
  };

  const handleEnableFirstDegreeAVBChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableFirstDegreeAVBlock(isEnabled);
    if (isEnabled) {
        setEnableMobitzIWenckebach(false); 
    }
  };
  const handleFirstDegreePrChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setFirstDegreePrSec(isNaN(val) ? 0.201 : val);
  };
  const handleFirstDegreePrBlur = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setFirstDegreePrSec(isNaN(val) ? 0.201 : Math.max(0.201, Math.min(0.60, val)));
  };

  const handleEnableMobitzIIAVBChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableMobitzIIAVBlock(isEnabled);
    if (isEnabled) {
        setEnableMobitzIWenckebach(false); 
    }
  };
  const handleMobitzIIRatioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10); setMobitzIIPWavesPerQRS(isNaN(val) ? 2 : Math.max(2, val));
  };

  const handleEnableMobitzIWenckebachChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableMobitzIWenckebach(isEnabled);
    if (isEnabled) {
        setEnableFirstDegreeAVBlock(false); 
        setEnableMobitzIIAVBlock(false); 
    }
  };
  const handleWenckebachInitialPrChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setWenckebachInitialPrSec(isNaN(val) ? 0.12 : val);
  };
  const handleWenckebachInitialPrBlur = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setWenckebachInitialPrSec(isNaN(val) ? 0.12 : Math.max(0.12, Math.min(0.40, val)));
  };
  const handleWenckebachIncrementChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setWenckebachPrIncrementSec(isNaN(val) ? 0.01 : val);
  };
  const handleWenckebachIncrementBlur = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setWenckebachPrIncrementSec(isNaN(val) ? 0.01 : Math.max(0.01, Math.min(0.15, val)));
  };
  const handleWenckebachMaxPrChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setWenckebachMaxPrBeforeDropSec(isNaN(val) ? 0.22 : val);
  };
  const handleWenckebachMaxPrBlur = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setWenckebachMaxPrBeforeDropSec(isNaN(val) ? 0.22 : Math.max(0.22, Math.min(0.70, val)));
  };

  const handleEnableThirdDegreeAVBChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableThirdDegreeAVBlock(isEnabled);
    if (isEnabled) {
      setEnableAtrialFibrillation(false);
      setEnableAtrialFlutter(false);
      setEnableFirstDegreeAVBlock(false);
      setEnableMobitzIIAVBlock(false);
      setEnableMobitzIWenckebach(false);
      setEnablePac(false); 
      setAllowSvtInitiationByPac(false);
      setEnableVT(false);
    }
  };
  const handleThirdDegreeEscapeOriginChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setThirdDegreeEscapeOrigin(e.target.value);
    if (e.target.value === "junctional") setThirdDegreeEscapeRate(45);
    else if (e.target.value === "ventricular") setThirdDegreeEscapeRate(30);
  };
  const handleThirdDegreeEscapeRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setThirdDegreeEscapeRate(isNaN(val) ? 20 : Math.max(15, Math.min(65, val)));
  };

  const handleEnableAtrialFibrillationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableAtrialFibrillation(isEnabled);
    if (isEnabled) {
      setEnableAtrialFlutter(false);
      setEnableThirdDegreeAVBlock(false);
      setEnableFirstDegreeAVBlock(false);
      setEnableMobitzIIAVBlock(false);
      setEnableMobitzIWenckebach(false);
      setEnablePac(false); 
      setAllowSvtInitiationByPac(false);
      setEnableVT(false);
    }
  };
  const handleAfibVentricularRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setAfibVentricularRate(isNaN(val) ? 30 : Math.max(30, Math.min(220, val)));
  };
  const handleAfibAmplitudeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setAfibAmplitude(isNaN(val) ? 0.0 : Math.max(0.0, Math.min(0.2, val)));
  };
  const handleAfibIrregularityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setAfibIrregularity(isNaN(val) ? 0.05 : Math.max(0.05, Math.min(0.5, val)));
  };

  const handleEnableAtrialFlutterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableAtrialFlutter(isEnabled);
    if (isEnabled) {
      setEnableAtrialFibrillation(false);
      setEnableThirdDegreeAVBlock(false);
      setEnableFirstDegreeAVBlock(false);
      setEnableMobitzIIAVBlock(false);
      setEnableMobitzIWenckebach(false);
      setEnablePac(false);
      setAllowSvtInitiationByPac(false);
      setEnableVT(false);
    }
  };
  const handleAflutterRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setAflutterRate(isNaN(val) ? 200 : Math.max(200, Math.min(400, val)));
  };
  const handleAflutterConductionRatioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setAflutterConductionRatio(isNaN(val) ? 1 : Math.max(1, Math.min(8, val)));
  };
   const handleAflutterAmplitudeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setAflutterAmplitude(isNaN(val) ? 0.05 : Math.max(0.05, Math.min(0.5, val)));
  };

  const handleAllowSvtInitiationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setAllowSvtInitiationByPac(isEnabled);
    if (isEnabled) {
      setEnablePac(true); // SVT initiation requires PACs to be enabled
      setEnableAtrialFibrillation(false);
      setEnableAtrialFlutter(false);
      setEnableThirdDegreeAVBlock(false);
    }
  };
  const handleSvtInitiationProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setSvtInitiationProbability(isNaN(val) ? 0 : Math.max(0, Math.min(1, val)));
  };
  const handleSvtDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    // Ensure SVT duration is not longer than total simulation duration
    setSvtDuration(isNaN(val) ? 1 : Math.max(1, Math.min(val, duration > 0 ? duration : 1))); 
  };
  const handleSvtRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setSvtRate(isNaN(val) ? 150 : Math.max(150, Math.min(250, val)));
  };

  // VT Handlers
  const handleEnableVTChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableVT(isEnabled);
    if (isEnabled) {
      setEnableAtrialFibrillation(false);
      setEnableAtrialFlutter(false);
      setEnableThirdDegreeAVBlock(false);
      setEnableFirstDegreeAVBlock(false);
      setEnableMobitzIIAVBlock(false);
      setEnableMobitzIWenckebach(false);
      setEnablePac(false);
      setAllowSvtInitiationByPac(false);
    }
  };
  const handleVtStartTimeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVtStartTime(isNaN(val) ? 0 : Math.max(0, Math.min(duration, val)));
  };
  const handleVtDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVtDuration(isNaN(val) ? 1 : Math.max(1, Math.min(duration, val)));
  };
  const handleVtRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setVtRate(isNaN(val) ? 100 : Math.max(100, Math.min(250, val)));
  };

  const toggleStyles = (isDisabled: boolean) => 
    `block h-5 w-10 cursor-pointer rounded-full ${isDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`;
  const rangeSliderStyles = (isDisabled: boolean) =>
    `h-1 w-full cursor-pointer appearance-none rounded-lg ${isDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 accent-red-500'}`;
  const numberInputStyles = (isDisabled: boolean) =>
    `w-full border ${isDisabled ? 'border-gray-600 bg-neutral-700 text-gray-500 cursor-not-allowed' : 'border-gray-700 bg-[#0e1525] text-neutral-300'} rounded-md px-2 py-1.5 text-sm focus:ring-red-500 focus:border-red-500`;
  const labelTextStyles = (isActive: boolean, isDisabled?: boolean) => 
    `text-sm font-medium ${isDisabled ? 'text-neutral-600' : (isActive ? 'text-neutral-300' : 'text-neutral-500')}`;
  const smallLabelTextStyles = (isDisabled?: boolean) =>
    `text-xs font-medium ${isDisabled ? 'text-neutral-600' : 'text-neutral-400'}`;
  const valueTextStyles = (isDisabled?: boolean) =>
    `text-right text-xs ${isDisabled ? 'text-neutral-600' : 'text-neutral-300'}`;


  return (
    <div className="bg-gray-50 overflow-auto text-neutral-900 rounded-md flex flex-col">
      <div className="px-4 py-4 mx-auto w-full max-w-8xl">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-2">
          <div className="lg:col-span-4 mb-2">
            <h1 className="text-2xl font-bold text-neutral-800 flex items-center">Advanced ECG Simulator</h1>
            <p className='text-neutral-800 text-sm'>Simulate various heart rhythms and episodes. </p>
          </div>
          
          <div className="lg:col-span-1">
            <div className="bg-neutral-900 relative rounded-xl p-6 pb-0 h-full overflow-y-auto max-h-[calc(100vh-150px)]">
              {/* Basic Controls Section */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                <h2 className="flex items-center text-lg font-semibold mb-2 text-gray-50">Basic Settings</h2>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label htmlFor="hrInput" className={`text-sm font-medium ${baseHrDisabled ? 'text-gray-500' : 'text-gray-100'}`}>
                        Underlying Atrial Rate (bpm)
                      </label>
                      <div className={`text-right text-lg font-medium ${baseHrDisabled ? 'text-gray-500' : 'text-gray-100'}`}>{heartRate}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input type="range" value={heartRate} onChange={handleHeartRateChange} min="30" max="250" 
                             disabled={baseHrDisabled}
                             className={rangeSliderStyles(baseHrDisabled)}/>
                      <div className={`text-xs w-12 text-right ${baseHrDisabled ? 'text-gray-500' : 'text-gray-100'}`}>bpm</div>
                    </div>
                     {baseHrDisabled && <p className="text-xs text-gray-500 mt-1">Base atrial rate is overridden if a dominant rhythm (AFib, AFlutter, 3rd Deg Block) is active.</p>}
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label htmlFor="durInput" className="text-sm font-medium text-gray-100">Duration (seconds)</label>
                      <div className="text-right text-gray-100 text-lg font-medium">{duration}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input type="range" value={duration} onChange={handleDurationChange} min="1" max="60" className={rangeSliderStyles(false)}/>
                      <div className="text-gray-100 text-xs w-12 text-right">sec</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* AV Conduction Settings Section (Sinus Base) */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                <h2 className={`flex items-center text-lg font-semibold mb-2 ${avBlocksDisabled ? 'text-gray-600' : 'text-neutral-200'}`}>
                  AV Conduction (Sinus Base) {avBlocksDisabled ? <span className="text-xs ml-2 text-gray-500">(Disabled by Dominant Rhythm)</span> : ""}
                </h2>
                <div className="space-y-5">
                  {/* 1st Degree AV Block */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                      <h3 className={labelTextStyles(enableFirstDegreeAVBlock, avBlocksDisabled)}>1st Degree AV Block</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableFirstDegreeAVBlockCheckbox" checked={enableFirstDegreeAVBlock} onChange={handleEnableFirstDegreeAVBChange} disabled={avBlocksDisabled} className="sr-only peer"/>
                        <label htmlFor="enableFirstDegreeAVBlockCheckbox" className={toggleStyles(avBlocksDisabled)}></label>
                      </div>
                    </div>
                    {enableFirstDegreeAVBlock && !avBlocksDisabled && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="firstDegreePrInput" className={smallLabelTextStyles()}>PR Interval (seconds)</label>
                          <div className={valueTextStyles()}>{firstDegreePrSec.toFixed(3)}s ({(firstDegreePrSec * 1000).toFixed(0)} ms)</div>
                        </div>
                        <input type="range" id="firstDegreePrInput" value={firstDegreePrSec} onChange={handleFirstDegreePrChange} onBlur={handleFirstDegreePrBlur} min="0.201" max="0.60" step="0.001" className={rangeSliderStyles(false)}/>
                      </div>
                    )}
                  </div>
                  {/* Mobitz Type I (Wenckebach) */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                       <h3 className={labelTextStyles(enableMobitzIWenckebach, avBlocksDisabled)}>2nd Degree AV Block Type I (Wenckebach)</h3>
                       <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableMobitzIWenckebachCheckbox" checked={enableMobitzIWenckebach} onChange={handleEnableMobitzIWenckebachChange} disabled={avBlocksDisabled} className="sr-only peer"/>
                        <label htmlFor="enableMobitzIWenckebachCheckbox" className={toggleStyles(avBlocksDisabled)}></label>
                      </div>
                    </div>
                    {enableMobitzIWenckebach && !avBlocksDisabled && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <label htmlFor="wenckebachInitialPrInput" className={smallLabelTextStyles()}>Initial PR (s):</label>
                          <input id="wenckebachInitialPrInput" type="number" value={wenckebachInitialPrSec} onChange={handleWenckebachInitialPrChange} onBlur={handleWenckebachInitialPrBlur} min="0.12" max="0.40" step="0.01" className={numberInputStyles(false)}/>
                        </div>
                        <div>
                          <label htmlFor="wenckebachIncrementInput" className={smallLabelTextStyles()}>PR Increment (s):</label>
                          <input id="wenckebachIncrementInput" type="number" value={wenckebachPrIncrementSec} onChange={handleWenckebachIncrementChange} onBlur={handleWenckebachIncrementBlur} min="0.01" max="0.15" step="0.01" className={numberInputStyles(false)}/>
                        </div>
                        <div>
                          <label htmlFor="wenckebachMaxPrInput" className={smallLabelTextStyles()}>Max PR before Drop (s):</label>
                          <input id="wenckebachMaxPrInput" type="number" value={wenckebachMaxPrBeforeDropSec} onChange={handleWenckebachMaxPrChange} onBlur={handleWenckebachMaxPrBlur} min="0.22" max="0.70" step="0.01" className={numberInputStyles(false)}/>
                        </div>
                      </div>
                    )}
                  </div>
                  {/* Mobitz Type II AV Block */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                       <h3 className={labelTextStyles(enableMobitzIIAVBlock, avBlocksDisabled)}>2nd Degree AV Block Type II (Mobitz II)</h3>
                       <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableMobitzIICheckbox" checked={enableMobitzIIAVBlock} onChange={handleEnableMobitzIIAVBChange} disabled={avBlocksDisabled} className="sr-only peer"/>
                        <label htmlFor="enableMobitzIICheckbox" className={toggleStyles(avBlocksDisabled)}></label>
                      </div>
                    </div>
                    {enableMobitzIIAVBlock && !avBlocksDisabled && (
                      <div className="mt-3">
                        <label htmlFor="mobitzIIRatioInput" className={`${smallLabelTextStyles()} block mb-1`}>P-waves per QRS (e.g., 3 for 3:1 Block):</label>
                        <input id="mobitzIIRatioInput" type="number" value={mobitzIIPWavesPerQRS} onChange={handleMobitzIIRatioChange} min="2" step="1" className={numberInputStyles(false)}/>
                        <p className="text-xs text-gray-500 mt-1">This sets a X:1 block. For {mobitzIIPWavesPerQRS}:1 block, 1 out of {mobitzIIPWavesPerQRS} P-waves conducts.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Dominant Base Rhythms (Overrides Sinus/AV blocks/PAC-SVT) */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                <h2 className="text-lg font-semibold mb-2 text-gray-200">Dominant Base Rhythms</h2>
                <p className="text-xs text-gray-400 mb-3">Enabling one of these will override Sinus base, AV conduction settings, and PAC/Dynamic SVT settings.</p>
                <div className="space-y-5">
                  {/* Ventricular Tachycardia */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={labelTextStyles(enableVT, enableAtrialFibrillation || enableAtrialFlutter || enableThirdDegreeAVBlock)}>Ventricular Tachycardia</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableVTCheckbox" checked={enableVT} onChange={handleEnableVTChange} 
                               disabled={enableAtrialFibrillation || enableAtrialFlutter || enableThirdDegreeAVBlock}
                               className="sr-only peer"/>
                        <label htmlFor="enableVTCheckbox" className={toggleStyles(enableAtrialFibrillation || enableAtrialFlutter || enableThirdDegreeAVBlock)}></label>
                      </div>
                    </div>
                    {enableVT && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="vtStartTimeInput" className={smallLabelTextStyles()}>Start Time (sec):</label>
                            <div className={valueTextStyles()}>{vtStartTime.toFixed(1)}</div>
                          </div>
                          <input id="vtStartTimeInput" type="range" value={vtStartTime} onChange={handleVtStartTimeChange} min="0" max={duration} step="0.1" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="vtDurationInput" className={smallLabelTextStyles()}>Duration (sec):</label>
                            <div className={valueTextStyles()}>{vtDuration.toFixed(1)}</div>
                          </div>
                          <input id="vtDurationInput" type="range" value={vtDuration} onChange={handleVtDurationChange} min="1" max={duration} step="0.5" className={rangeSliderStyles(false)}/>
                          <p className="text-xs text-gray-500 mt-1">VT episode duration</p>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="vtRateInput" className={smallLabelTextStyles()}>VT Rate (bpm):</label>
                            <div className={valueTextStyles()}>{vtRate}</div>
                          </div>
                          <input id="vtRateInput" type="range" value={vtRate} onChange={handleVtRateChange} min="100" max="250" step="5" className={rangeSliderStyles(false)}/>
                        </div>
                      </div>
                    )}
                    {(enableAtrialFibrillation || enableAtrialFlutter || enableThirdDegreeAVBlock) && <p className="text-xs text-gray-500 mt-2">Not available with AFib, AFlutter, or 3rd Deg Block.</p>}
                  </div>
                  {/* Atrial Fibrillation */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={labelTextStyles(enableAtrialFibrillation, enableAtrialFlutter || enableThirdDegreeAVBlock)}>Atrial Fibrillation</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableAtrialFibrillationCheckbox" checked={enableAtrialFibrillation} onChange={handleEnableAtrialFibrillationChange} 
                               disabled={enableAtrialFlutter || enableThirdDegreeAVBlock || enableVT}
                               className="sr-only peer"/>
                        <label htmlFor="enableAtrialFibrillationCheckbox" className={toggleStyles(enableAtrialFlutter || enableThirdDegreeAVBlock || enableVT)}></label>
                      </div>
                    </div>
                    {enableAtrialFibrillation && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="afibVentricularRateInput" className={smallLabelTextStyles()}>Avg. Ventricular Rate (bpm):</label>
                            <div className={valueTextStyles()}>{afibVentricularRate}</div>
                          </div>
                          <input id="afibVentricularRateInput" type="range" value={afibVentricularRate} onChange={handleAfibVentricularRateChange} min="30" max="220" step="5" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="afibAmplitudeInput" className={smallLabelTextStyles()}>f-wave Amplitude (mV):</label>
                            <div className={valueTextStyles()}>{afibAmplitude.toFixed(2)}</div>
                          </div>
                           <input id="afibAmplitudeInput" type="range" value={afibAmplitude} onChange={handleAfibAmplitudeChange} min="0.0" max="0.2" step="0.01" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="afibIrregularityInput" className={smallLabelTextStyles()}>Irregularity Factor:</label>
                            <div className={valueTextStyles()}>{afibIrregularity.toFixed(2)}</div>
                          </div>
                          <input id="afibIrregularityInput" type="range" value={afibIrregularity} onChange={handleAfibIrregularityChange} min="0.05" max="0.5" step="0.01" className={rangeSliderStyles(false)}/>
                          <p className="text-xs text-gray-500 mt-1">Higher values = more irregular R-R</p>
                        </div>
                      </div>
                    )}
                    {(enableAtrialFlutter || enableThirdDegreeAVBlock || enableVT) && <p className="text-xs text-gray-500 mt-2">Not available with AFlutter, 3rd Deg Block, or VT.</p>}
                  </div>
                  {/* Atrial Flutter */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                     <div className="flex justify-between items-center mb-3">
                      <h3 className={labelTextStyles(enableAtrialFlutter, enableAtrialFibrillation || enableThirdDegreeAVBlock)}>Atrial Flutter</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableAtrialFlutterCheckbox" checked={enableAtrialFlutter} onChange={handleEnableAtrialFlutterChange} 
                               disabled={enableAtrialFibrillation || enableThirdDegreeAVBlock || enableVT}
                               className="sr-only peer"/>
                        <label htmlFor="enableAtrialFlutterCheckbox" className={toggleStyles(enableAtrialFibrillation || enableThirdDegreeAVBlock || enableVT)}></label>
                        </div>
                    </div>
                    {enableAtrialFlutter && (
                       <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="aflutterRateInput" className={smallLabelTextStyles()}>Atrial Rate (bpm):</label>
                            <div className={valueTextStyles()}>{aflutterRate}</div>
                          </div>
                          <input id="aflutterRateInput" type="range" value={aflutterRate} onChange={handleAflutterRateChange} min="200" max="400" step="10" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="aflutterConductionRatioInput" className={smallLabelTextStyles()}>AV Conduction Ratio (X:1):</label>
                            <div className={valueTextStyles()}>{aflutterConductionRatio}:1</div>
                          </div>
                          <input id="aflutterConductionRatioInput" type="range" value={aflutterConductionRatio} onChange={handleAflutterConductionRatioChange} min="1" max="8" step="1" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="aflutterAmplitudeInput" className={smallLabelTextStyles()}>Flutter Wave Amplitude (mV):</label>
                            <div className={valueTextStyles()}>{aflutterAmplitude.toFixed(2)}</div>
                          </div>
                          <input id="aflutterAmplitudeInput" type="range" value={aflutterAmplitude} onChange={handleAflutterAmplitudeChange} min="0.05" max="0.5" step="0.01" className={rangeSliderStyles(false)}/>
                        </div>
                      </div>
                    )}
                    {(enableAtrialFibrillation || enableThirdDegreeAVBlock || enableVT) && <p className="text-xs text-gray-500 mt-2">Not available with AFib, 3rd Deg Block, or VT.</p>}
                  </div>
                  {/* 3rd Degree AV Block */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={labelTextStyles(enableThirdDegreeAVBlock, enableAtrialFibrillation || enableAtrialFlutter)}>3rd Degree AV Block</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableThirdDegreeAVBCheckbox" checked={enableThirdDegreeAVBlock} onChange={handleEnableThirdDegreeAVBChange} 
                               disabled={enableAtrialFibrillation || enableAtrialFlutter || enableVT}
                               className="sr-only peer"/>
                        <label htmlFor="enableThirdDegreeAVBCheckbox" className={toggleStyles(enableAtrialFibrillation || enableAtrialFlutter || enableVT)}></label>
                        </div>
                    </div>
                    {enableThirdDegreeAVBlock && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <label htmlFor="thirdDegreeEscapeOriginSelect" className={`${smallLabelTextStyles()} block mb-1`}>Escape Rhythm Origin:</label>
                          <select id="thirdDegreeEscapeOriginSelect" value={thirdDegreeEscapeOrigin} onChange={handleThirdDegreeEscapeOriginChange} className={numberInputStyles(false)}>
                            <option value="junctional">Junctional</option>
                            <option value="ventricular">Ventricular</option>
                          </select>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="thirdDegreeEscapeRateInput" className={smallLabelTextStyles()}>Escape Rate (bpm):</label>
                            <div className={valueTextStyles()}>{thirdDegreeEscapeRate} bpm</div>
                          </div>
                          <input id="thirdDegreeEscapeRateInput" type="range" value={thirdDegreeEscapeRate} onChange={handleThirdDegreeEscapeRateChange} min="15" max="65" step="1" className={rangeSliderStyles(false)}/>
                        </div>
                      </div>
                    )}
                    {(enableAtrialFibrillation || enableAtrialFlutter || enableVT) && <p className="text-xs text-gray-500 mt-2">Not available with AFib, AFlutter, or VT.</p>}
                  </div>
                </div>
              </div>
              
              {/* Ectopy & Dynamic SVT Section */}
              <div className="mb-6">
                <h2 className="text-lg font-semibold mb-2 text-gray-200">Ectopy & Dynamic SVT</h2>
                <div className="space-y-5">
                  {/* PAC Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                      <h3 className={labelTextStyles(enablePac, pacsAndDynamicSvtSettingsDisabled)}>Premature Atrial Contractions (PACs)</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enablePacCheckbox" checked={enablePac} onChange={handleEnablePacChange} 
                               disabled={pacsAndDynamicSvtSettingsDisabled} className="sr-only peer"/>
                        <label htmlFor="enablePacCheckbox" className={toggleStyles(pacsAndDynamicSvtSettingsDisabled)}></label>
                      </div>
                    </div>
                    {enablePac && !pacsAndDynamicSvtSettingsDisabled && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="pacProbInput" className={smallLabelTextStyles()}>Probability per Sinus Beat</label>
                          <div className={valueTextStyles()}>{pacProbability.toFixed(2)}</div>
                        </div>
                        <input type="range" value={pacProbability} onChange={handlePacProbabilityChange} min="0" max="1" step="0.01" className={rangeSliderStyles(false)}/>
                      </div>
                    )}
                    {pacsAndDynamicSvtSettingsDisabled && <p className="text-xs text-gray-500 mt-2">PACs (& Dynamic SVT) disabled if AFib, AFlutter, 3rd Degree Block, or VT is active.</p>}
                  </div>

                  {/* Dynamic SVT (PAC-initiated) Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={labelTextStyles(allowSvtInitiationByPac, !enablePac || pacsAndDynamicSvtSettingsDisabled)}>Dynamic SVT (PAC-initiated)</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="allowSvtInitiationCheckbox" checked={allowSvtInitiationByPac} onChange={handleAllowSvtInitiationChange} 
                               disabled={!enablePac || pacsAndDynamicSvtSettingsDisabled} className="sr-only peer"/>
                        <label htmlFor="allowSvtInitiationCheckbox" className={toggleStyles(!enablePac || pacsAndDynamicSvtSettingsDisabled)}></label>
                        </div>
                    </div>
                    {allowSvtInitiationByPac && enablePac && !pacsAndDynamicSvtSettingsDisabled && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className={smallLabelTextStyles()}>Initiation Probability per PAC:</label>
                            <div className={valueTextStyles()}>{svtInitiationProbability.toFixed(2)}</div>
                          </div>
                          <input type="range" value={svtInitiationProbability} onChange={handleSvtInitiationProbabilityChange} min="0" max="1" step="0.05" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className={smallLabelTextStyles()}>SVT Episode Duration (s):</label>
                            <div className={valueTextStyles()}>{svtDuration}</div>
                          </div>
                          <input type="range" value={svtDuration} onChange={handleSvtDurationChange} min="1" max={Math.max(1, duration > 0 ? duration : 1)} step="1" className={rangeSliderStyles(false)}/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label className={smallLabelTextStyles()}>SVT Rate during Episode (bpm):</label>
                            <div className={valueTextStyles()}>{svtRate}</div>
                          </div>
                          <input type="range" value={svtRate} onChange={handleSvtRateChange} min="150" max="250" step="5" className={rangeSliderStyles(false)}/>
                        </div>
                      </div>
                    )}
                    {(!enablePac || pacsAndDynamicSvtSettingsDisabled) && <p className="text-xs text-gray-500 mt-2">Requires PACs to be enabled and no overriding dominant rhythm (AFib, AFlutter, 3rd Deg Block, VT).</p>}
                  </div>

                  {/* PVC Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                      <h3 className={labelTextStyles(enablePvc, false)}>Premature Ventricular Contractions (PVCs)</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enablePvcCheckbox" checked={enablePvc} onChange={handleEnablePvcChange} className="sr-only peer"/>
                        <label htmlFor="enablePvcCheckbox" className={toggleStyles(false)}></label>
                      </div>
                    </div>
                    {enablePvc && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="pvcProbInput" className={smallLabelTextStyles()}>Probability per Beat</label>
                          <div className={valueTextStyles()}>{pvcProbability.toFixed(2)}</div>
                        </div>
                        <input type="range" value={pvcProbability} onChange={handlePvcProbabilityChange} min="0" max="1" step="0.01" className={rangeSliderStyles(false)}/>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="sticky bottom-0 left-0 right-0 pt-4 pb-4 bg-neutral-900/10 backdrop-blur-md mt-6 -mx-6 px-6 border-t border-gray-800">
                <button onClick={fetchEcgData} disabled={isLoading} className={`w-full px-3 py-3 cursor-pointer rounded-lg text-white font-medium shadow transition-all ${isLoading ? 'bg-gray-700 cursor-not-allowed' : 'bg-red-500 hover:bg-red-600 active:bg-red-700'}`}>
                  {isLoading ? (<span className="flex items-center justify-center"><svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Generating...</span>)
                  : (<span className="flex items-center justify-center"><svg className="mr-1" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2v4a1 1 0 0 0 1 1h4"></path><path d="M18 9v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h7"></path><path d="M3 12h5l2 3 3-6 2 3h6"></path></svg>Generate ECG</span>)}
                </button>
              </div>
            </div>
          </div>
          
          <div className="lg:col-span-3">
            <div className="bg-neutral-900 rounded-xl overflow-hidden h-[calc(100vh-150px)] flex flex-col border border-gray-800">
              <div className="px-6 py-4 border-b border-gray-800 flex justify-between items-center">
                <div className="font-medium text-gray-200">{chartTitle || 'ECG Signal'}</div>
                <div className="text-sm text-gray-500">
                    {isAfibActiveBase ? `Avg ${afibVentricularRate} bpm (AFib)` :
                     isAflutterActiveBase ? `${Math.round(aflutterRate/aflutterConductionRatio)} bpm (AFlutter Vent.)` :
                     isThirdDegreeBlockActiveBase ? `${thirdDegreeEscapeRate} bpm (Escape)` :
                     isVTActiveBase ? `${vtRate} bpm (VT)` :
                     `${heartRate} bpm (Sinus)`}, 
                     {duration}s
                </div>
              </div>
              {error && (<div className="m-4 bg-red-900/30 border border-red-800 text-red-200 px-3 py-2 rounded-md text-sm flex items-center" role="alert"><svg className="mr-2 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg><span>{error}</span></div>)}
              <div className="p-4 flex-grow relative">
                {isLoading && (<div className="absolute inset-0 flex items-center justify-center bg-gray-900/60 backdrop-blur-sm z-10"><div className="text-center"><div className="animate-pulse flex space-x-2 justify-center mb-2"><div className="w-2 h-2 bg-red-400 rounded-full"></div><div className="w-2 h-2 bg-red-500 rounded-full"></div><div className="w-2 h-2 bg-red-600 rounded-full"></div></div><p className="text-gray-400 text-sm">Generating ECG data...</p></div></div>)}
                {!isLoading && ecgData.time_axis.length > 0 && (<Line ref={chartRef} options={chartOptions} data={chartDataConfig} />)}
                {!isLoading && ecgData.time_axis.length === 0 && !error && (<div className="h-full flex items-center justify-center"><div className="text-center text-gray-500"><svg className="mx-auto h-10 w-10 text-gray-600 mb-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg><p className="text-sm">No ECG data to display</p></div></div>)}
              </div>
              <div className="px-4 py-2 border-t border-gray-800 text-xs text-gray-500">* This is a simulated ECG for educational purposes only</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ECGChart;