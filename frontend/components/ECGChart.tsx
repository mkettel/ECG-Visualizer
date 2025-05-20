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
// import type { ChartProps } from 'react-chartjs-2'; // Usually not needed with latest versions

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
  enable_mobitz_i_wenckebach?: boolean; // New
  wenckebach_initial_pr_sec?: number;    // New
  wenckebach_pr_increment_sec?: number; // New
  wenckebach_max_pr_before_drop_sec?: number; // New
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

  // New state for Wenckebach (Mobitz I)
  const [enableMobitzIWenckebach, setEnableMobitzIWenckebach] = useState<boolean>(false);
  const [wenckebachInitialPrSec, setWenckebachInitialPrSec] = useState<number>(0.16);
  const [wenckebachPrIncrementSec, setWenckebachPrIncrementSec] = useState<number>(0.04);
  const [wenckebachMaxPrBeforeDropSec, setWenckebachMaxPrBeforeDropSec] = useState<number>(0.32);


  const [chartTitle, setChartTitle] = useState<string>('Simulated ECG');
  const chartRef = useRef<ChartJS<'line', number[], string> | null>(null);

  const fetchEcgData = async () => {
    setIsLoading(true);
    setError(null);

    // --- Validations ---
    if ((enablePac && (pacProbability < 0 || pacProbability > 1)) ||
        (enablePvc && (pvcProbability < 0 || pvcProbability > 1))) {
      setError("Ectopic probabilities must be between 0.0 and 1.0.");
      setIsLoading(false); return;
    }
    if (enableFirstDegreeAVBlock && (firstDegreePrSec < 0.201 || firstDegreePrSec > 0.60)) {
      setError("1st Degree AV Block PR interval must be between 0.201s and 0.60s.");
      setIsLoading(false); return;
    }
    if (enableMobitzIIAVBlock && (mobitzIIPWavesPerQRS < 2)) {
      setError("Mobitz II P-waves per QRS must be 2 or greater.");
      setIsLoading(false); return;
    }
    if (enableMobitzIWenckebach) {
        if (wenckebachInitialPrSec < 0.12 || wenckebachInitialPrSec > 0.40) {
            setError("Wenckebach Initial PR must be between 0.12s and 0.40s.");
            setIsLoading(false); return;
        }
        if (wenckebachPrIncrementSec < 0.01 || wenckebachPrIncrementSec > 0.15) {
            setError("Wenckebach PR Increment must be between 0.01s and 0.15s.");
            setIsLoading(false); return;
        }
        if (wenckebachMaxPrBeforeDropSec < 0.22 || wenckebachMaxPrBeforeDropSec > 0.70 || wenckebachMaxPrBeforeDropSec <= wenckebachInitialPrSec) {
            setError("Wenckebach Max PR must be between 0.22s and 0.70s, and greater than Initial PR.");
            setIsLoading(false); return;
        }
    }

    const requestBody: AdvancedRequestBody = {
      heart_rate_bpm: heartRate,
      duration_sec: duration,
      enable_pac: enablePac,
      pac_probability_per_sinus: enablePac ? pacProbability : 0,
      enable_pvc: enablePvc,
      pvc_probability_per_sinus: enablePvc ? pvcProbability : 0,
      first_degree_av_block_pr_sec: enableFirstDegreeAVBlock ? firstDegreePrSec : null,
      enable_mobitz_ii_av_block: enableMobitzIIAVBlock,
      mobitz_ii_p_waves_per_qrs: enableMobitzIIAVBlock ? mobitzIIPWavesPerQRS : 2,
      enable_mobitz_i_wenckebach: enableMobitzIWenckebach, // New
      wenckebach_initial_pr_sec: enableMobitzIWenckebach ? wenckebachInitialPrSec : 0.16, // New
      wenckebach_pr_increment_sec: enableMobitzIWenckebach ? wenckebachPrIncrementSec : 0.04, // New
      wenckebach_max_pr_before_drop_sec: enableMobitzIWenckebach ? wenckebachMaxPrBeforeDropSec : 0.32, // New
    };

    try {
      // ... (fetch logic remains the same) ...
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
      // ... (error handling remains the same) ...
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

  const handleEnableFirstDegreeAVBChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnableFirstDegreeAVBlock(e.target.checked);
  const handleFirstDegreePrChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setFirstDegreePrSec(isNaN(val) ? 0.201 : val);
  };
  const handleFirstDegreePrBlur = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value); setFirstDegreePrSec(isNaN(val) ? 0.201 : Math.max(0.201, Math.min(0.60, val)));
  };

  const handleEnableMobitzIIAVBChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnableMobitzIIAVBlock(e.target.checked);
  const handleMobitzIIRatioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseInt(e.target.value, 10); setMobitzIIPWavesPerQRS(isNaN(val) ? 2 : Math.max(2, val));
  };

  // Event Handlers for Wenckebach (Mobitz I) - New
  const handleEnableMobitzIWenckebachChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnableMobitzIWenckebach(e.target.checked);
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

  return (
    <div className="bg-gray-50 overflow-auto text-neutral-900 rounded-md flex flex-col">
      {/* Main container */}
      <div className="px-4 py-4 mx-auto w-full max-w-8xl">
        {/* Title & controls layout */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-2">
          {/* Title area */}
          <div className="lg:col-span-4 mb-2">
            {/* ... (Title) */}
            <h1 className="text-2xl font-bold text-neutral-800 flex items-center">Advanced ECG Simulator</h1>
            <p className='text-neutral-800 text-sm'>Utilize the different settings to create various heart rhythms. </p>
          </div>
          
          {/* Left side - Controls */}
          <div className="lg:col-span-1">
            <div className="bg-neutral-900 rounded-xl p-6 h-full overflow-y-auto max-h-[calc(100vh-150px)]">
              {/* Basic Controls Section */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                {/* ... (Basic Settings - no change) ... */}
                <h2 className="flex items-center text-lg font-semibold mb-2 text-gray-50">Basic Settings</h2>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label htmlFor="hrInput" className="text-sm font-medium text-gray-100">Heart Rate (bpm)</label>
                      <div className="text-right text-gray-100 text-lg font-medium">{heartRate}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <input type="range" value={heartRate} onChange={handleHeartRateChange} min="30" max="250" className="h-1 flex-grow cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                      <div className="text-gray-100 text-xs w-12 text-right">bpm</div>
                    </div>
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

              {/* AV Conduction Settings Section - Add Mobitz I */}
              <div className="mb-6 pb-5 border-b border-gray-700">
                <h2 className="flex items-center text-lg font-semibold mb-2 text-neutral-200">
                  AV Conduction Settings
                </h2>
                <div className="space-y-5">
                  {/* 1st Degree AV Block */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    {/* ... (1st Degree UI - no change) ... */}
                    <div className="flex justify-between items-center">
                      <h3 className="text-sm font-medium text-neutral-300">1st Degree AV Block</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableFirstDegreeAVBlockCheckbox" checked={enableFirstDegreeAVBlock} onChange={handleEnableFirstDegreeAVBChange} className="sr-only peer"/>
                        <label htmlFor="enableFirstDegreeAVBlockCheckbox" className="block h-5 w-10 cursor-pointer rounded-full bg-neutral-700 peer-checked:bg-red-500 peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all"></label>
                      </div>
                    </div>
                    {enableFirstDegreeAVBlock && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="firstDegreePrInput" className="text-xs font-medium text-neutral-400">PR Interval (seconds)</label>
                          <div className="text-right text-neutral-300 text-xs">{firstDegreePrSec.toFixed(3)}s ({(firstDegreePrSec * 1000).toFixed(0)} ms)</div>
                        </div>
                        <input type="range" id="firstDegreePrInput" value={firstDegreePrSec} onChange={handleFirstDegreePrChange} min="0.201" max="0.60" step="0.001" disabled={!enableFirstDegreeAVBlock} className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-gray-700 accent-red-500"/>
                      </div>
                    )}
                  </div>

                  {/* Mobitz Type I (Wenckebach) - New */}
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center mb-3">
                      <h3 className="text-sm font-medium text-neutral-300">2nd Degree AV Block Type I (Wenckebach)</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableMobitzIWenckebachCheckbox" checked={enableMobitzIWenckebach} onChange={handleEnableMobitzIWenckebachChange} className="sr-only peer"/>
                        <label htmlFor="enableMobitzIWenckebachCheckbox" className="block h-5 w-10 cursor-pointer rounded-full bg-neutral-700 peer-checked:bg-red-500 peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all"></label>
                      </div>
                    </div>
                    {enableMobitzIWenckebach && (
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
                    {/* ... (Mobitz II UI - no change) ... */}
                    <div className="flex justify-between gap-1 items-center">
                      <h3 className="text-sm font-medium text-neutral-300">2nd Degree AV Block Type II (Mobitz II)</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enableMobitzIICheckbox" checked={enableMobitzIIAVBlock} onChange={handleEnableMobitzIIAVBChange} className="sr-only peer"/>
                        <label htmlFor="enableMobitzIICheckbox" className="block h-5 w-10 cursor-pointer rounded-full bg-neutral-700 peer-checked:bg-red-500 peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all"></label>
                      </div>
                    </div>
                    {enableMobitzIIAVBlock && (
                      <div className="mt-3">
                        <label htmlFor="mobitzIIRatioInput" className="text-xs font-medium text-neutral-400 block mb-1">P-waves per QRS (e.g., 3 for 3:1 Block):</label>
                        <input id="mobitzIIRatioInput" type="number" value={mobitzIIPWavesPerQRS} onChange={handleMobitzIIRatioChange} min="2" step="1" disabled={!enableMobitzIIAVBlock} className="w-full border border-gray-700 bg-[#0e1525] rounded-md px-2 py-1.5 text-neutral-300 text-sm focus:ring-red-500 focus:border-red-500"/>
                        <p className="text-xs text-gray-500 mt-1">This sets a X:1 block, where X is the number entered. For {mobitzIIPWavesPerQRS}:1 block, 1 out of {mobitzIIPWavesPerQRS} P-waves conducts.</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Ectopic Controls Section */}
              <div className="mb-6">
                 {/* ... (Ectopic Settings - no change) ... */}
                 <h2 className="flex items-center font-semibold text-lg mb-2 text-gray-200">Ectopic Beat Settings</h2>
                <div className="space-y-5">
                  <div className="bg-neutral-800 rounded-lg p-4 border border-gray-800">
                    <div className="flex justify-between items-center">
                      <h3 className="text-sm font-medium text-neutral-100">Premature Atrial Contractions</h3>
                      <div className="relative inline-block w-10 align-middle select-none">
                        <input type="checkbox" id="enablePacCheckbox" checked={enablePac} onChange={handleEnablePacChange} className="sr-only peer"/>
                        <label htmlFor="enablePacCheckbox" className="block h-5 w-10 cursor-pointer rounded-full bg-neutral-700 peer-checked:bg-red-500 peer-checked:after:translate-x-full after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:border after:border-gray-700 after:bg-white after:transition-all"></label>
                      </div>
                    </div>
                    {enablePac && (
                      <div className="mt-3">
                        <div className="flex justify-between items-center mb-1">
                          <label htmlFor="pacProbInput" className="text-xs font-medium text-neutral-300">Probability per Sinus Beat</label>
                          <div className="text-right text-neutral-300 text-xs">{pacProbability.toFixed(2)}</div>
                        </div>
                        <input type="range" value={pacProbability} onChange={handlePacProbabilityChange} min="0" max="1" step="0.01" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                      </div>
                    )}
                  </div>
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
                          <label htmlFor="pvcProbInput" className="text-xs font-medium text-neutral-400">Probability per Sinus Beat</label>
                          <div className="text-right text-neutral-300 text-xs">{pvcProbability.toFixed(2)}</div>
                        </div>
                        <input type="range" value={pvcProbability} onChange={handlePvcProbabilityChange} min="0" max="1" step="0.01" className="h-1 w-full cursor-pointer appearance-none rounded-lg bg-neutral-700 accent-red-500"/>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Generate Button */}
              {/* ... (Generate Button - no change) ... */}
              <button onClick={fetchEcgData} disabled={isLoading} className={`w-full px-3 py-3 rounded-lg text-white font-medium shadow transition-all ${isLoading ? 'bg-gray-700 cursor-not-allowed' : 'bg-red-500 hover:bg-red-600 active:bg-red-700'}`}>
                {isLoading ? (<span className="flex items-center justify-center"><svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Generating...</span>)
                : (<span className="flex items-center justify-center"><svg className="mr-1" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2v4a1 1 0 0 0 1 1h4"></path><path d="M18 9v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h7"></path><path d="M3 12h5l2 3 3-6 2 3h6"></path></svg>Generate ECG</span>)}
              </button>
            </div>
          </div>
          
          {/* Right side - ECG Visualization */}
          <div className="lg:col-span-3">
            {/* Chart Display */}
            <div className="bg-neutral-900 rounded-xl overflow-hidden h-[calc(100vh-150px)] flex flex-col border border-gray-800">
              <div className="px-6 py-4 border-b border-gray-800 flex justify-between items-center">
                <div className="font-medium text-gray-200">{chartTitle || 'ECG Signal'}</div>
                <div className="text-sm text-gray-500">{heartRate} bpm, {duration}s</div>
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