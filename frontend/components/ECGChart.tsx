"use client"; // Required for Next.js App Router if using client-side hooks

import React, { useState, useEffect, useRef, CSSProperties } from 'react';
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
  ChartOptions, // For typing chart options
  ChartData    // For typing chart data
} from 'chart.js';
import { ChartJSOrUndefined } from 'react-chartjs-2/dist/types'; // For chartRef type

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

// Updated RequestBody for the advanced endpoint
interface AdvancedRequestBody {
  heart_rate_bpm: number;
  duration_sec: number;
  enable_pvc: boolean;
  pvc_probability_per_sinus: number;
  enable_pac: boolean;
  pac_probability_per_sinus: number;
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
  const [heartRate, setHeartRate] = useState<number>(75);
  const [duration, setDuration] = useState<number>(10);

  // New state for advanced ectopic controls
  const [enablePac, setEnablePac] = useState<boolean>(false);
  const [pacProbability, setPacProbability] = useState<number>(0.1); // Default 10%
  const [enablePvc, setEnablePvc] = useState<boolean>(false);
  const [pvcProbability, setPvcProbability] = useState<number>(0.1); // Default 10%

  const [chartTitle, setChartTitle] = useState<string>('Simulated ECG');
  const chartRef = useRef<ChartJSOrUndefined<'line', number[], string>>(null);

  const fetchEcgData = async () => {
    setIsLoading(true);
    setError(null);

    // Validate probabilities
    if ((enablePac && (pacProbability < 0 || pacProbability > 1)) ||
        (enablePvc && (pvcProbability < 0 || pvcProbability > 1))) {
      setError("Probabilities must be between 0.0 and 1.0.");
      setIsLoading(false);
      return;
    }

    const requestBody: AdvancedRequestBody = {
      heart_rate_bpm: heartRate,
      duration_sec: duration,
      enable_pac: enablePac,
      pac_probability_per_sinus: enablePac ? pacProbability : 0,
      enable_pvc: enablePvc,
      pvc_probability_per_sinus: enablePvc ? pvcProbability : 0,
    };

    try {
      const response = await fetch('http://localhost:8000/api/generate_advanced_ecg', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`;
        try {
          const errData = await response.json();
          errorMessage = errData.detail || errorMessage;
        } catch (jsonError) {
          errorMessage = response.statusText || "Unknown server error";
        }
        throw new Error(errorMessage);
      }

      const data: ECGAPIResponse = await response.json();
      setEcgData({
        time_axis: data.time_axis || [],
        ecg_signal: data.ecg_signal || []
      });
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

  useEffect(() => {
    fetchEcgData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Initial fetch

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
  
  // Event Handlers
  const handleHeartRateChange = (e: React.ChangeEvent<HTMLInputElement>) => setHeartRate(parseFloat(e.target.value) || 0);
  const handleDurationChange = (e: React.ChangeEvent<HTMLInputElement>) => setDuration(parseFloat(e.target.value) || 0);
  
  const handleEnablePacChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnablePac(e.target.checked);
  const handlePacProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setPacProbability(isNaN(val) ? 0 : Math.max(0, Math.min(1, val)));
  };
  
  const handleEnablePvcChange = (e: React.ChangeEvent<HTMLInputElement>) => setEnablePvc(e.target.checked);
  const handlePvcProbabilityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setPvcProbability(isNaN(val) ? 0 : Math.max(0, Math.min(1, val)));
  };

  // --- Styling ---
  const inputStyle: CSSProperties = { marginRight: '10px', padding: '8px', borderRadius: '4px', border: '1px solid #4B5563', backgroundColor: '#374151', color: '#F3F4F6' };
  const checkboxInputStyle: CSSProperties = { ...inputStyle, marginRight: '5px', padding: '0', verticalAlign: 'middle', width: 'auto', height: 'auto' };
  const probabilityInputStyle: CSSProperties = { ...inputStyle, width: '80px' };
  const labelStyle: CSSProperties = { display: 'block', marginBottom: '5px', color: '#D1D5DB', fontSize: '0.9rem' };
  const checkboxLabelStyle: CSSProperties = { ...labelStyle, display: 'inline-block', marginBottom: '0', marginRight: '10px', verticalAlign: 'middle', cursor: 'pointer' };
  const buttonStyle: CSSProperties = { padding: '10px 20px', borderRadius: '4px', border: 'none', backgroundColor: '#EF4444', color: 'white', cursor: 'pointer', fontWeight: 'bold', fontSize: '1rem' };
  const disabledButtonStyle: CSSProperties = { ...buttonStyle, backgroundColor: '#9CA3AF', cursor: 'not-allowed'};
  const controlSectionStyle: CSSProperties = { marginBottom: '20px', paddingBottom: '20px', borderBottom: '1px solid #374151' };
  const ectopicControlGroupStyle: CSSProperties = { display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'center', alignItems: 'center', border: '1px solid #4B5563', padding: '15px', borderRadius: '6px', marginBottom: '10px'};
  const mainContainerStyle: CSSProperties = { padding: '20px 30px', backgroundColor: '#1F2937', color: '#F3F4F6', borderRadius: '8px', maxWidth: '1100px', margin: '30px auto', boxShadow: '0 10px 25px rgba(0,0,0,0.3)' };
  const chartContainerStyle: CSSProperties = { height: '450px', width: '100%', border: '1px solid #4B5563', padding: '10px', borderRadius: '4px', backgroundColor: '#111827', marginTop: '20px' };

  return (
    <div style={mainContainerStyle}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px', color: '#F9FAFB', fontSize: '2.2rem', fontWeight: 'bold' }}>
        Advanced ECG Simulator
      </h1>

      {/* Basic Controls Section */}
      <div style={controlSectionStyle}>
        <h2 style={{marginTop: 0, marginBottom: '15px', fontSize: '1.2rem', color: '#E5E7EB'}}>Basic Settings</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'center', alignItems: 'flex-end' }}>
            <div>
                <label htmlFor="hrInput" style={labelStyle}>Heart Rate (bpm):</label>
                <input id="hrInput" type="number" value={heartRate} onChange={handleHeartRateChange} min="30" max="250" style={inputStyle} />
            </div>
            <div>
                <label htmlFor="durInput" style={labelStyle}>Duration (s):</label>
                <input id="durInput" type="number" value={duration} onChange={handleDurationChange} min="1" max="60" style={inputStyle} />
            </div>
        </div>
      </div>

      {/* Ectopic Controls Section */}
      <div style={controlSectionStyle}>
        <h2 style={{marginTop: 0, marginBottom: '15px', fontSize: '1.2rem', color: '#E5E7EB'}}>Ectopic Beat Settings</h2>
        <div className='flex' style={{display: 'flex', justifyContent: 'space-around', gap: '20px'}}>
            {/* PAC Controls */}
            <div style={ectopicControlGroupStyle}>
                <h3 style={{width: '100%', textAlign:'center', marginTop: '0', marginBottom: '10px', fontSize: '1rem', color: '#E5E7EB'}}>Premature Atrial Contractions (PACs)</h3>
                <div>
                    <input
                        type="checkbox"
                        id="enablePacCheckbox"
                        checked={enablePac}
                        onChange={handleEnablePacChange}
                        style={checkboxInputStyle}
                    />
                    <label htmlFor="enablePacCheckbox" style={checkboxLabelStyle}>Enable PACs</label>
                </div>
                {enablePac && (
                    <div>
                        <label htmlFor="pacProbInput" style={{...labelStyle, display: 'inline-block', marginRight:'5px'}}>Probability (0-1):</label>
                        <input
                            id="pacProbInput"
                            type="number"
                            value={pacProbability}
                            onChange={handlePacProbabilityChange}
                            min="0"
                            max="1"
                            step="0.01"
                            style={probabilityInputStyle}
                            disabled={!enablePac}
                        />
                    </div>
                )}
            </div>

            {/* PVC Controls */}
            <div style={ectopicControlGroupStyle}>
                <h3 style={{width: '100%', textAlign:'center', marginTop: '0', marginBottom: '10px', fontSize: '1rem', color: '#E5E7EB'}}>Premature Ventricular Contractions (PVCs)</h3>
                <div>
                    <input
                        type="checkbox"
                        id="enablePvcCheckbox"
                        checked={enablePvc}
                        onChange={handleEnablePvcChange}
                        style={checkboxInputStyle}
                    />
                    <label htmlFor="enablePvcCheckbox" style={checkboxLabelStyle}>Enable PVCs</label>
                </div>
                {enablePvc && (
                    <div>
                        <label htmlFor="pvcProbInput" style={{...labelStyle, display: 'inline-block', marginRight:'5px'}}>Probability (0-1):</label>
                        <input
                            id="pvcProbInput"
                            type="number"
                            value={pvcProbability}
                            onChange={handlePvcProbabilityChange}
                            min="0"
                            max="1"
                            step="0.01"
                            style={probabilityInputStyle}
                            disabled={!enablePvc}
                        />
                    </div>
                )}
            </div>
        </div>
      </div>
      
      <div style={{textAlign: 'center', marginTop: '30px'}}>
        <button onClick={fetchEcgData} disabled={isLoading} style={isLoading ? disabledButtonStyle : buttonStyle}>
          {isLoading ? 'Generating...' : 'Regenerate ECG'}
        </button>
      </div>

      {error && <p style={{ color: '#F87171', textAlign: 'center', margin: '20px 0', padding: '10px', backgroundColor: '#7f1d1d50', borderRadius: '4px', fontWeight:'bold' }}>Error: {error}</p>}
      
      <div style={chartContainerStyle}>
        {isLoading && <p style={{ textAlign: 'center', marginTop: '20px', color: '#9CA3AF' }}>Loading chart data...</p>}
        {!isLoading && ecgData.time_axis.length > 0 && (
          <Line ref={chartRef} options={chartOptions} data={chartDataConfig} />
        )}
        {!isLoading && ecgData.time_axis.length === 0 && !error && (
          <p style={{ textAlign: 'center', marginTop: '20px', color: '#9CA3AF' }}>No ECG data to display. Adjust parameters and click "Regenerate ECG".</p>
        )}
      </div>
    </div>
  );
};

export default ECGChart;