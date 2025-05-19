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
  Decimation // Important for performance with many data points
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

const ECGChart = () => {
  const [ecgData, setEcgData] = useState({ time_axis: [], ecg_signal: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- Control State ---
  const [heartRate, setHeartRate] = useState(75);
  const [duration, setDuration] = useState(10);
  const [baseRhythm, setBaseRhythm] = useState('sinus');
  const [pvcInterval, setPvcInterval] = useState(''); // Store as string, parse to int for API
  const [chartTitle, setChartTitle] = useState('Simulated ECG'); // Dynamic chart title

  const chartRef = useRef(null);

  const fetchEcgData = async () => {
    setIsLoading(true);
    setError(null);

    const requestBody = {
      heart_rate_bpm: parseFloat(heartRate),
      duration_sec: parseFloat(duration),
      base_rhythm: baseRhythm,
    };

    // Only include insert_pvc_after_n_beats if pvcInterval is a valid number >= 0
    const pvcIntervalNum = parseInt(pvcInterval, 10);
    if (!isNaN(pvcIntervalNum) && pvcIntervalNum >= 0) {
      requestBody.insert_pvc_after_n_beats = pvcIntervalNum;
    } else if (pvcInterval !== '') { // If it's not empty but not a valid number
        setError("PVC interval must be a non-negative number.");
        setIsLoading(false);
        return; // Prevent API call with invalid input
    }


    try {
      const response = await fetch('http://localhost:8000/api/generate_ecg', { // Ensure this URL is correct
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Unknown error from server" }));
        throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setEcgData({ time_axis: data.time_axis || [], ecg_signal: data.ecg_signal || [] });
      setChartTitle(data.rhythm_generated || `Simulated ${baseRhythm.capitalize()}`);

    } catch (e) {
      console.error("Failed to fetch ECG data:", e);
      setError(e.message);
      setEcgData({ time_axis: [], ecg_signal: [] }); // Clear data on error
      setChartTitle('Error generating rhythm');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch data on initial mount
  useEffect(() => {
    fetchEcgData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Add dependencies here if you want auto-refresh on param change, or rely on button

  const chartDataConfig = {
    labels: ecgData.time_axis.map(t => t.toFixed(2)), // X-axis labels (time points)
    datasets: [
      {
        label: 'ECG Signal (mV)',
        data: ecgData.ecg_signal,
        borderColor: 'rgb(239, 68, 68)', // A shade of red (Tailwind red-500)
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.05, // Slightly less tension for a more traditional ECG look
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: 'Time (s)', color: '#CBD5E1' }, // Light gray for dark mode
        ticks: {
          maxTicksLimit: Math.max(10, Math.min(20, duration * 2)), // Dynamic ticks based on duration
          autoSkipPadding: 20,
          color: '#9CA3AF', // Lighter gray for tick labels
        },
        grid: {
            color: '#4B5563', // Darker gray for grid lines
        }
      },
      y: {
        title: { display: true, text: 'Amplitude (mV)', color: '#CBD5E1' },
        suggestedMin: -1.0,
        suggestedMax: 1.5,
        ticks: {
            color: '#9CA3AF',
            stepSize: 0.5, // Standard ECG grid lines every 0.5mV
        },
        grid: {
            color: '#4B5563',
        }
      },
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
            color: '#F3F4F6' // Lightest gray for legend text
        }
      },
      title: {
        display: true,
        text: chartTitle, // Use dynamic title from state
        color: '#F9FAFB', // Off-white for title
        font: {
            size: 16
        }
      },
      decimation: {
        enabled: true,
        algorithm: 'lttb',
        samples: Math.min(1000, ecgData.time_axis.length), // Adjust samples based on data length
      },
    },
  };

  // Helper for consistent styling (example)
  const inputStyle = { marginRight: '10px', padding: '8px', borderRadius: '4px', border: '1px solid #4B5563', backgroundColor: '#374151', color: '#F3F4F6' };
  const labelStyle = { display: 'block', marginBottom: '5px', color: '#D1D5DB' };
  const buttonStyle = { padding: '8px 15px', borderRadius: '4px', border: 'none', backgroundColor: '#EF4444', color: 'white', cursor: 'pointer', fontWeight: 'bold' };
  const disabledButtonStyle = { ...buttonStyle, backgroundColor: '#9CA3AF', cursor: 'not-allowed'};


  return (
    <div style={{ padding: '20px', backgroundColor: '#1F2937', color: '#F3F4F6', borderRadius: '8px' }}> {/* Dark theme container */}
      <h2 style={{ textAlign: 'center', marginBottom: '25px', color: '#F9FAFB', fontSize: '1.8rem' }}>ECG Rhythm Controls</h2>
      <div style={{ marginBottom: '25px', display: 'flex', flexWrap: 'wrap', justifyContent: 'center', alignItems: 'flex-end', gap: '20px' }}>
        <div>
          <label htmlFor="hrInput" style={labelStyle}>Heart Rate (bpm):</label>
          <input
            id="hrInput"
            type="number"
            value={heartRate}
            onChange={(e) => setHeartRate(e.target.value)}
            min="30"
            max="250"
            style={inputStyle}
          />
        </div>
        <div>
          <label htmlFor="durInput" style={labelStyle}>Duration (s):</label>
          <input
            id="durInput"
            type="number"
            value={duration}
            onChange={(e) => setDuration(e.target.value)}
            min="1"
            max="60"
            style={inputStyle}
          />
        </div>
        <div>
          <label htmlFor="rhythmSelect" style={labelStyle}>Base Rhythm:</label>
          <select
            id="rhythmSelect"
            value={baseRhythm}
            onChange={(e) => setBaseRhythm(e.target.value)}
            style={inputStyle}
          >
            <option value="sinus">Sinus</option>
            {/* Add <option value="pac">PAC (as base)</option> when backend supports it as a base rhythm */}
            {/* For now, PAC/PVC are primarily modifiers or specific event types */}
          </select>
        </div>
        <div>
          <label htmlFor="pvcIntervalInput" style={labelStyle}>PVC after N beats (0 for 1st):</label>
          <input
            id="pvcIntervalInput"
            type="number"
            value={pvcInterval}
            onChange={(e) => setPvcInterval(e.target.value)}
            placeholder="e.g., 2 (for S-S-PVC)"
            min="0"
            style={{...inputStyle, width: '150px'}}
          />
        </div>
        <button
            onClick={fetchEcgData}
            disabled={isLoading}
            style={isLoading ? disabledButtonStyle : buttonStyle}
        >
          {isLoading ? 'Generating...' : 'Regenerate ECG'}
        </button>
      </div>

      {error && <p style={{ color: '#F87171', textAlign: 'center', marginBottom: '15px' }}>Error: {error}</p>}
      
      <div style={{ height: '450px', border: '1px solid #4B5563', padding: '10px', borderRadius: '4px', backgroundColor: '#111827' }}> {/* Chart area dark background */}
        {isLoading && <p style={{ textAlign: 'center', marginTop: '20px' }}>Loading chart data...</p>}
        {!isLoading && ecgData.time_axis.length > 0 && (
          <Line ref={chartRef} options={chartOptions} data={chartDataConfig} />
        )}
        {!isLoading && ecgData.time_axis.length === 0 && !error && (
          <p style={{ textAlign: 'center', marginTop: '20px' }}>No ECG data to display. Adjust parameters and click "Regenerate ECG".</p>
        )}
      </div>
    </div>
  );
};

export default ECGChart;

// Helper to capitalize first letter (optional)
String.prototype.capitalize = function() {
    return this.charAt(0).toUpperCase() + this.slice(1);
}