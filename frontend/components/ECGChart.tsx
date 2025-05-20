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

  // NEW SVT Params
  enable_svt?: boolean;
  svt_rate_bpm?: number;
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

  // NEW SVT State
  const [enableSvt, setEnableSvt] = useState<boolean>(false);
  const [svtRate, setSvtRate] = useState<number>(180);

  const [chartTitle, setChartTitle] = useState<string>('Simulated ECG');
  const chartRef = useRef<ChartJS<'line', number[], string> | null>(null);

  // Helper booleans for disabling controls
  const isSvtActive = enableSvt;
  const isAfibActive = enableAtrialFibrillation && !isSvtActive;
  const isAflutterActive = enableAtrialFlutter && !isSvtActive && !isAfibActive;
  const isThirdDegreeBlockActive = enableThirdDegreeAVBlock && !isSvtActive && !isAfibActive && !isAflutterActive;
  
  // Any rhythm that overrides base HR or SA-driven AV conduction
  const isDominantRhythmActive = isSvtActive || isAfibActive || isAflutterActive || isThirdDegreeBlockActive;
  // Tachyarrhythmias specifically
  const isTachyarrhythmiaActive = isSvtActive || isAfibActive || isAflutterActive;


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
    if (enableSvt && (svtRate < 150 || svtRate > 250)) {
        setError("SVT rate must be between 150 and 250 bpm."); setIsLoading(false); return;
    }

    // Determine active flags for request body construction
    const sendEnableSvt = enableSvt;
    const sendEnableAfib = enableAtrialFibrillation && !sendEnableSvt;
    const sendEnableAflutter = enableAtrialFlutter && !sendEnableSvt && !sendEnableAfib;
    const sendEnableThirdDegreeAVBlock = enableThirdDegreeAVBlock && !sendEnableSvt && !sendEnableAfib && !sendEnableAflutter;

    const sendEnableFirstDegreeAVBlock = enableFirstDegreeAVBlock && !sendEnableSvt && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock;
    const sendEnableMobitzI = enableMobitzIWenckebach && !sendEnableSvt && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock;
    const sendEnableMobitzII = enableMobitzIIAVBlock && !sendEnableSvt && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock && !sendEnableMobitzI;
    
    const sendEnablePac = enablePac && !sendEnableSvt && !sendEnableAfib && !sendEnableAflutter && !sendEnableThirdDegreeAVBlock;


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

      enable_svt: sendEnableSvt,
      svt_rate_bpm: sendEnableSvt ? svtRate : 180,
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
  
  const handleEnablePacChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnablePac(e.target.checked);
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
    if (isEnabled) { // 1st degree can co-exist with Mobitz II if Mobitz I is off
        setEnableMobitzIWenckebach(false); // Typically 1st deg + Wenck is just Wenck.
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
        setEnableMobitzIWenckebach(false); // Mobitz I and II are mutually exclusive
    }
  };
  const handleMobitzIIRatioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10); setMobitzIIPWavesPerQRS(isNaN(val) ? 2 : Math.max(2, val));
  };

  const handleEnableMobitzIWenckebachChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableMobitzIWenckebach(isEnabled);
    if (isEnabled) {
        setEnableFirstDegreeAVBlock(false); // Wenckebach settings define initial PR
        setEnableMobitzIIAVBlock(false); // Mobitz I and II are mutually exclusive
    }
  };
  const handleWenckebachInitialPrChange = (e: React.ChangeEvent<HTMLInputElement>) => { /* ... */ };
  const handleWenckebachInitialPrBlur = (e: React.ChangeEvent<HTMLInputElement>) => { /* ... */ };
  const handleWenckebachIncrementChange = (e: React.ChangeEvent<HTMLInputElement>) => { /* ... */ };
  const handleWenckebachIncrementBlur = (e: React.ChangeEvent<HTMLInputElement>) => { /* ... */ };
  const handleWenckebachMaxPrChange = (e: React.ChangeEvent<HTMLInputElement>) => { /* ... */ };
  const handleWenckebachMaxPrBlur = (e: React.ChangeEvent<HTMLInputElement>) => { /* ... */ };

  const handleEnableThirdDegreeAVBChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableThirdDegreeAVBlock(isEnabled);
    if (isEnabled) {
        setEnableSvt(false);
        setEnableAtrialFibrillation(false);
        setEnableAtrialFlutter(false);
        setEnableFirstDegreeAVBlock(false);
        setEnableMobitzIIAVBlock(false);
        setEnableMobitzIWenckebach(false);
        setEnablePac(false);
    }
  };
  const handleThirdDegreeEscapeOriginChange = (e: React.ChangeEvent<HTMLSelectElement>) => { /* ... */ };
  const handleThirdDegreeEscapeRateChange = (e: React.ChangeEvent<HTMLInputElement>) => { /* ... */ };

  const handleEnableAtrialFibrillationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableAtrialFibrillation(isEnabled);
    if (isEnabled) {
      setEnableSvt(false);
      setEnableAtrialFlutter(false);
      setEnableThirdDegreeAVBlock(false);
      setEnableFirstDegreeAVBlock(false);
      setEnableMobitzIIAVBlock(false);
      setEnableMobitzIWenckebach(false);
      setEnablePac(false); 
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
      setEnableSvt(false);
      setEnableAtrialFibrillation(false);
      setEnableThirdDegreeAVBlock(false);
      setEnableFirstDegreeAVBlock(false);
      setEnableMobitzIIAVBlock(false);
      setEnableMobitzIWenckebach(false);
      setEnablePac(false);
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

  // NEW Event Handlers for SVT
  const handleEnableSvtChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isEnabled = e.target.checked;
    setEnableSvt(isEnabled);
    if (isEnabled) {
      setEnableAtrialFibrillation(false);
      setEnableAtrialFlutter(false);
      setEnableThirdDegreeAVBlock(false);
      setEnableFirstDegreeAVBlock(false);
      setEnableMobitzIWenckebach(false);
      setEnableMobitzIIAVBlock(false);
      setEnablePac(false); 
    }
  };
  const handleSvtRateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10);
    setSvtRate(isNaN(val) ? 150 : Math.max(150, Math.min(250, val)));
  };

  // Dynamic disable flags for UI elements
  const baseHrDisabled = isDominantRhythmActive;
  const avBlocksDisabled = isTachyarrhythmiaActive || isThirdDegreeBlockActive;
  const pacsDisabled = isTachyarrhythmiaActive || isThirdDegreeBlockActive;
  const afibDisabled = isSvtActive || isAflutterActive || isThirdDegreeBlockActive;
  const aflutterDisabled = isSvtActive || isAfibActive || isThirdDegreeBlockActive;
  const svtDisabled = isAfibActive || isAflutterActive || isThirdDegreeBlockActive;
  const thirdDegreeDisabled = isTachyarrhythmiaActive;


  return (
    <div className="bg-gray-50 overflow-auto text-neutral-900 rounded-md flex flex-col">
      {/* Main container */}
      <div className="px-4 py-4 mx-auto w-full max-w-8xl">
        {/* Title & controls layout */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-2">
          {/* Title area */}
          <div className="lg:col-span-4 mb-2">
            <h1 className="text-2xl font-bold text-neutral-800 flex items-center">Advanced ECG Simulator</h1>
            <p className='text-neutral-800 text-sm'>Utilize the different settings to create various heart rhythms. </p>
          </div>
          
          {/* Left side - Controls */}
          <div className="lg:col-span-1">
            <div className="bg-neutral-900 rounded-xl p-6 h-full overflow-y-auto max-h-[calc(100vh-150px)]">
              {/* Basic Controls Section */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                <h2 className="flex items-center text-lg font-semibold mb-2 text-gray-50">Basic Settings</h2>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label htmlFor="hrInput" className={`text-sm font-medium ${baseHrDisabled ? 'text-gray-500' : 'text-gray-100'}`}>
                        {isTachyarrhythmiaActive || isThirdDegreeBlockActive ? 'Underlying Atrial Rate (bpm)' : 'Heart Rate (bpm)'}
                      </label>
                      <div className={`text-right text-lg font-medium ${baseHrDisabled ? 'text-gray-500' : 'text-gray-100'}`}>{heartRate}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input 
                        type="range" value={heartRate} onChange={handleHeartRateChange} min="30" max="250" 
                        disabled={baseHrDisabled}
                        className={`h-1 flex-grow cursor-pointer appearance-none rounded-lg ${baseHrDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 accent-red-500'}`}
                        />
                      <div className={`text-xs w-12 text-right ${baseHrDisabled ? 'text-gray-500' : 'text-gray-100'}`}>bpm</div>
                    </div>
                     {baseHrDisabled && <p className="text-xs text-gray-500 mt-1">Base heart rate is overridden by the active dominant rhythm.</p>}
                  </div>
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label htmlFor="durInput" className="text-sm font-medium text-gray-100">Duration (seconds)</label>
                      <div className="text-right text-gray-100 text-lg font-medium">{duration}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input type="range" value={duration} onChange={handleDurationChange} min="1" max="60" className="h-1 flex-grow cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                      <div className="text-gray-100 text-xs w-12 text-right">sec</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* AV Conduction Settings Section */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                <h2 className={`flex items-center text-lg font-semibold mb-2 ${avBlocksDisabled ? 'text-gray-600' : 'text-neutral-200'}`}>
                  AV Conduction Settings {avBlocksDisabled ? <span className="text-xs ml-2 text-gray-500">(Disabled by Active Rhythm)</span> : ""}
                </h2>
                <div className="space-y-5">
                  {/* 1st Degree AV Block */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                      <h3 className={`text-sm font-medium ${enableFirstDegreeAVBlock && !avBlocksDisabled ? 'text-neutral-300' : 'text-neutral-500'}`}>1st Degree AV Block</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableFirstDegreeAVBlockCheckbox" checked={enableFirstDegreeAVBlock} onChange={handleEnableFirstDegreeAVBChange} disabled={avBlocksDisabled} className="sr-only peer"/>
                        <label htmlFor="enableFirstDegreeAVBlockCheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${avBlocksDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enableFirstDegreeAVBlock && !avBlocksDisabled && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="firstDegreePrInput" className="text-xs font-medium text-neutral-400">PR Interval (seconds)</label>
                          <div className="text-right text-neutral-300 text-xs">{firstDegreePrSec.toFixed(3)}s ({(firstDegreePrSec * 1000).toFixed(0)} ms)</div>
                        </div>
                        <input type="range" id="firstDegreePrInput" value={firstDegreePrSec} onChange={handleFirstDegreePrChange} onBlur={handleFirstDegreePrBlur} min="0.201" max="0.60" step="0.001" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-gray-700 accent-red-500"/>
                      </div>
                    )}
                  </div>
                  {/* Mobitz Type I (Wenckebach) */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                       <h3 className={`text-sm font-medium ${enableMobitzIWenckebach && !avBlocksDisabled ? 'text-neutral-300' : 'text-neutral-500'}`}>2nd Degree AV Block Type I (Wenckebach)</h3>
                       <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableMobitzIWenckebachCheckbox" checked={enableMobitzIWenckebach} onChange={handleEnableMobitzIWenckebachChange} disabled={avBlocksDisabled} className="sr-only peer"/>
                        <label htmlFor="enableMobitzIWenckebachCheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${avBlocksDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enableMobitzIWenckebach && !avBlocksDisabled && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <label htmlFor="wenckebachInitialPrInput" className="text-xs font-medium text-neutral-400 block mb-1">Initial PR (s):</label>
                          <input id="wenckebachInitialPrInput" type="number" value={wenckebachInitialPrSec} onChange={handleWenckebachInitialPrChange} onBlur={handleWenckebachInitialPrBlur} min="0.12" max="0.40" step="0.01" className="w-full border border-gray-700 bg-[#0e1525] rounded-md px-2 py-1.5 text-neutral-300 text-sm focus:ring-red-500 focus:border-red-500"/>
                        </div>
                        <div>
                          <label htmlFor="wenckebachIncrementInput" className="text-xs font-medium text-neutral-400 block mb-1">PR Increment (s):</label>
                          <input id="wenckebachIncrementInput" type="number" value={wenckebachPrIncrementSec} onChange={handleWenckebachIncrementChange} onBlur={handleWenckebachIncrementBlur} min="0.01" max="0.15" step="0.01" className="w-full border border-gray-700 bg-[#0e1525] rounded-md px-2 py-1.5 text-neutral-300 text-sm focus:ring-red-500 focus:border-red-500"/>
                        </div>
                        <div>
                          <label htmlFor="wenckebachMaxPrInput" className="text-xs font-medium text-neutral-400 block mb-1">Max PR before Drop (s):</label>
                          <input id="wenckebachMaxPrInput" type="number" value={wenckebachMaxPrBeforeDropSec} onChange={handleWenckebachMaxPrChange} onBlur={handleWenckebachMaxPrBlur} min="0.22" max="0.70" step="0.01" className="w-full border border-gray-700 bg-[#0e1525] rounded-md px-2 py-1.5 text-neutral-300 text-sm focus:ring-red-500 focus:border-red-500"/>
                        </div>
                      </div>
                    )}
                  </div>
                  {/* Mobitz Type II AV Block */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                       <h3 className={`text-sm font-medium ${enableMobitzIIAVBlock && !avBlocksDisabled ? 'text-neutral-300' : 'text-neutral-500'}`}>2nd Degree AV Block Type II (Mobitz II)</h3>
                       <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableMobitzIICheckbox" checked={enableMobitzIIAVBlock} onChange={handleEnableMobitzIIAVBChange} disabled={avBlocksDisabled} className="sr-only peer"/>
                        <label htmlFor="enableMobitzIICheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${avBlocksDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enableMobitzIIAVBlock && !avBlocksDisabled && (
                      <div className="mt-3">
                        <label htmlFor="mobitzIIRatioInput" className="text-xs font-medium text-neutral-400 block mb-1">P-waves per QRS (e.g., 3 for 3:1 Block):</label>
                        <input id="mobitzIIRatioInput" type="number" value={mobitzIIPWavesPerQRS} onChange={handleMobitzIIRatioChange} min="2" step="1" className="w-full border border-gray-700 bg-[#0e1525] rounded-md px-2 py-1.5 text-neutral-300 text-sm focus:ring-red-500 focus:border-red-500"/>
                        <p className="text-xs text-gray-500 mt-1">This sets a X:1 block, where X is the number entered. For {mobitzIIPWavesPerQRS}:1 block, 1 out of {mobitzIIPWavesPerQRS} P-waves conducts.</p>
                      </div>
                    )}
                  </div>
                  {/* 3rd Degree AV Block Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={`text-sm font-medium ${enableThirdDegreeAVBlock && !thirdDegreeDisabled ? 'text-neutral-300' : 'text-neutral-500'}`}>3rd Degree (Complete) AV Block</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableThirdDegreeAVBCheckbox" checked={enableThirdDegreeAVBlock} onChange={handleEnableThirdDegreeAVBChange} disabled={thirdDegreeDisabled} className="sr-only peer"/>
                        <label htmlFor="enableThirdDegreeAVBCheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${thirdDegreeDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enableThirdDegreeAVBlock && !thirdDegreeDisabled && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <label htmlFor="thirdDegreeEscapeOriginSelect" className="text-xs font-medium text-neutral-400 block mb-1">Escape Rhythm Origin:</label>
                          <select id="thirdDegreeEscapeOriginSelect" value={thirdDegreeEscapeOrigin} onChange={handleThirdDegreeEscapeOriginChange} className="w-full border border-gray-700 bg-[#0e1525] rounded-md px-2 py-1.5 text-neutral-300 text-sm focus:ring-red-500 focus:border-red-500">
                            <option value="junctional">Junctional</option>
                            <option value="ventricular">Ventricular</option>
                          </select>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="thirdDegreeEscapeRateInput" className="text-xs font-medium text-neutral-400">Escape Rate (bpm):</label>
                            <div className="text-right text-neutral-300 text-xs">{thirdDegreeEscapeRate} bpm</div>
                          </div>
                          <input id="thirdDegreeEscapeRateInput" type="range" value={thirdDegreeEscapeRate} onChange={handleThirdDegreeEscapeRateChange} min="15" max="65" step="1" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                        </div>
                      </div>
                    )}
                     {thirdDegreeDisabled && ( <p className="text-xs text-gray-500 mt-2">Not available with SVT, AFib or AFlutter</p> )}
                  </div>
                </div>
              </div>

              {/* Supraventricular Tachyarrhythmias Section */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                <h2 className="flex items-center font-semibold text-lg mb-2 text-gray-200">Supraventricular Tachyarrhythmias</h2>
                <div className="space-y-5">
                  {/* SVT (AVNRT-like) Controls - NEW */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={`text-sm font-medium ${enableSvt && !svtDisabled ? 'text-neutral-300' : 'text-neutral-500'}`}>SVT (AVNRT-like)</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableSvtCheckbox" checked={enableSvt} onChange={handleEnableSvtChange} disabled={svtDisabled} className="sr-only peer" />
                        <label htmlFor="enableSvtCheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${svtDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enableSvt && !svtDisabled && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="svtRateInput" className="text-xs font-medium text-neutral-400">Ventricular Rate (bpm):</label>
                            <div className="text-right text-neutral-300 text-xs">{svtRate}</div>
                          </div>
                          <input id="svtRateInput" type="range" value={svtRate} onChange={handleSvtRateChange} min="150" max="250" step="5"  className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500" />
                          <p className="text-xs text-gray-500 mt-1">Regular, narrow QRS. P-waves usually hidden.</p>
                        </div>
                      </div>
                    )}
                    {svtDisabled && ( <p className="text-xs text-gray-500 mt-2">SVT not available with AFib, AFlutter, or 3rd Degree AV Block</p> )}
                  </div>

                  {/* Atrial Fibrillation Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={`text-sm font-medium ${enableAtrialFibrillation && !afibDisabled ? 'text-neutral-300' : 'text-neutral-500'}`}>Atrial Fibrillation</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableAtrialFibrillationCheckbox" checked={enableAtrialFibrillation} onChange={handleEnableAtrialFibrillationChange} disabled={afibDisabled} className="sr-only peer"/>
                        <label htmlFor="enableAtrialFibrillationCheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${afibDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enableAtrialFibrillation && !afibDisabled && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="afibVentricularRateInput" className="text-xs font-medium text-neutral-400">Avg. Ventricular Rate (bpm):</label>
                            <div className="text-right text-neutral-300 text-xs">{afibVentricularRate}</div>
                          </div>
                          <input id="afibVentricularRateInput" type="range" value={afibVentricularRate} onChange={handleAfibVentricularRateChange} min="30" max="220" step="5" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="afibAmplitudeInput" className="text-xs font-medium text-neutral-400">f-wave Amplitude (mV):</label>
                            <div className="text-right text-neutral-300 text-xs">{afibAmplitude.toFixed(2)}</div>
                          </div>
                           <input id="afibAmplitudeInput" type="range" value={afibAmplitude} onChange={handleAfibAmplitudeChange} min="0.0" max="0.2" step="0.01" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="afibIrregularityInput" className="text-xs font-medium text-neutral-400">Irregularity Factor:</label>
                            <div className="text-right text-neutral-300 text-xs">{afibIrregularity.toFixed(2)}</div>
                          </div>
                          <input id="afibIrregularityInput" type="range" value={afibIrregularity} onChange={handleAfibIrregularityChange} min="0.05" max="0.5" step="0.01" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                          <p className="text-xs text-gray-500 mt-1">Higher values = more irregular R-R intervals</p>
                        </div>
                      </div>
                    )}
                    {afibDisabled && ( <p className="text-xs text-gray-500 mt-2">AFib not available with SVT, AFlutter, or 3rd Degree AV Block</p> )}
                  </div>
                  
                  {/* Atrial Flutter Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className={`text-sm font-medium ${enableAtrialFlutter && !aflutterDisabled ? 'text-neutral-300' : 'text-neutral-500'}`}>Atrial Flutter</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableAtrialFlutterCheckbox" checked={enableAtrialFlutter} onChange={handleEnableAtrialFlutterChange} disabled={aflutterDisabled} className="sr-only peer"/>
                        <label htmlFor="enableAtrialFlutterCheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${aflutterDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enableAtrialFlutter && !aflutterDisabled && (
                      <div className="space-y-3 mt-3">
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="aflutterRateInput" className="text-xs font-medium text-neutral-400">Atrial Rate (bpm):</label>
                            <div className="text-right text-neutral-300 text-xs">{aflutterRate}</div>
                          </div>
                          <input id="aflutterRateInput" type="range" value={aflutterRate} onChange={handleAflutterRateChange} min="200" max="400" step="10" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="aflutterConductionRatioInput" className="text-xs font-medium text-neutral-400">AV Conduction Ratio (X:1):</label>
                            <div className="text-right text-neutral-300 text-xs">{aflutterConductionRatio}:1</div>
                          </div>
                          <input id="aflutterConductionRatioInput" type="range" value={aflutterConductionRatio} onChange={handleAflutterConductionRatioChange} min="1" max="8" step="1" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                        </div>
                        <div>
                          <div className="flex justify-between items-center mb-1">
                            <label htmlFor="aflutterAmplitudeInput" className="text-xs font-medium text-neutral-400">Flutter Wave Amplitude (mV):</label>
                            <div className="text-right text-neutral-300 text-xs">{aflutterAmplitude.toFixed(2)}</div>
                          </div>
                          <input id="aflutterAmplitudeInput" type="range" value={aflutterAmplitude} onChange={handleAflutterAmplitudeChange} min="0.05" max="0.5" step="0.01" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                        </div>
                      </div>
                    )}
                     {aflutterDisabled && ( <p className="text-xs text-gray-500 mt-2">AFlutter not available with SVT, AFib, or 3rd Degree AV Block</p> )}
                  </div>
                </div>
              </div>
              
              {/* Ectopy Section */}
              <div className="mb-6">
                <h2 className="flex items-center font-semibold text-lg mb-2 text-gray-200">Ectopy</h2>
                <div className="space-y-5">
                  {/* PAC (single ectopic) Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                      <h3 className={`text-sm font-medium ${enablePac && !pacsDisabled ? 'text-neutral-100' : 'text-neutral-500'}`}>Premature Atrial Contractions</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enablePacCheckbox" checked={enablePac} onChange={handleEnablePacChange} disabled={pacsDisabled} className="sr-only peer"/>
                        <label htmlFor="enablePacCheckbox" className={`block h-5 w-10 cursor-pointer rounded-full ${pacsDisabled ? 'bg-neutral-600 cursor-not-allowed' : 'bg-neutral-700 peer-checked:bg-red-500'} peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all`}></label>
                      </div>
                    </div>
                    {enablePac && !pacsDisabled && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="pacProbInput" className="text-xs font-medium text-neutral-300">Probability per Sinus Beat</label>
                          <div className="text-right text-neutral-300 text-xs">{pacProbability.toFixed(2)}</div>
                        </div>
                        <input type="range" value={pacProbability} onChange={handlePacProbabilityChange} min="0" max="1" step="0.01" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                      </div>
                    )}
                    {pacsDisabled && ( <p className="text-xs text-gray-500 mt-2">PACs not available with SVT, AFib, AFlutter, or 3rd Degree Block</p> )}
                  </div>
                  {/* PVC Controls */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                      <h3 className="text-sm font-medium text-neutral-300">Premature Ventricular Contractions</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enablePvcCheckbox" checked={enablePvc} onChange={handleEnablePvcChange} className="sr-only peer"/>
                        <label htmlFor="enablePvcCheckbox" className="block h-5 w-10 cursor-pointer rounded-full bg-neutral-700 peer-checked:bg-red-500 peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all"></label>
                      </div>
                    </div>
                    {enablePvc && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="pvcProbInput" className="text-xs font-medium text-neutral-400">Probability per Beat</label>
                          <div className="text-right text-neutral-300 text-xs">{pvcProbability.toFixed(2)}</div>
                        </div>
                        <input type="range" value={pvcProbability} onChange={handlePvcProbabilityChange} min="0" max="1" step="0.01" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Generate Button */}
              <button onClick={fetchEcgData} disabled={isLoading} className={`w-full px-3 py-3 rounded-lg text-white font-medium shadow transition-all ${isLoading ? 'bg-gray-700 cursor-not-allowed' : 'bg-red-500 hover:bg-red-600 active:bg-red-700'}`}>
                {isLoading ? (<span className="flex items-center justify-center"><svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Generating...</span>)
                : (<span className="flex items-center justify-center"><svg className="mr-1" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2v4a1 1 0 0 0 1 1h4"></path><path d="M18 9v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h7"></path><path d="M3 12h5l2 3 3-6 2 3h6"></path></svg>Generate ECG</span>)}
              </button>
            </div>
          </div>
          
          {/* Right side - ECG Visualization */}
          <div className="lg:col-span-3">
            <div className="bg-neutral-900 rounded-xl overflow-hidden h-[calc(100vh-150px)] flex flex-col border border-gray-800">
              <div className="px-6 py-4 border-b border-gray-800 flex justify-between items-center">
                <div className="font-medium text-gray-200">{chartTitle || 'ECG Signal'}</div>
                <div className="text-sm text-gray-500">
                    {isSvtActive ? `${svtRate} bpm (SVT)` : 
                     isAfibActive ? `Avg ${afibVentricularRate} bpm (AFib)` :
                     isAflutterActive ? `${Math.round(aflutterRate/aflutterConductionRatio)} bpm (AFlutter Ventricular)` :
                     isThirdDegreeBlockActive ? `${thirdDegreeEscapeRate} bpm (Escape Rhythm)` :
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