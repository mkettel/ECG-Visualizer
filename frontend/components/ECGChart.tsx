'use client';

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
import type { ChartJSOrUndefined } from 'react-chartjs-2/dist/types';


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

interface EcgResponse {
  time_axis: number[];
  ecg_signal: number[];
}

const ECGChart: React.FC = () => {
  const [ecgData, setEcgData] = useState<EcgResponse>({ time_axis: [], ecg_signal: [] });
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [heartRate, setHeartRate] = useState<number>(75);
  const [duration, setDuration] = useState<number>(10);

  const chartRef = useRef<ChartJSOrUndefined<'line'>>(null);

  const fetchEcgData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/generate_ecg', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          heart_rate_bpm: heartRate,
          duration_sec: duration,
        }),
      });
      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, ${errText}`);
      }
      const data: EcgResponse = await response.json();
      setEcgData(data);
    } catch (e: any) {
      console.error("Failed to fetch ECG data:", e);
      setError(e.message);
      setEcgData({ time_axis: [], ecg_signal: [] });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchEcgData();
  }, []);

  const chartDataConfig: ChartData<'line'> = {
    labels: ecgData.time_axis.map((t) => t.toFixed(2)),
    datasets: [
      {
        label: 'ECG Signal (mV)',
        data: ecgData.ecg_signal,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.1,
      },
    ],
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      x: {
        title: { display: true, text: 'Time (s)' },
        ticks: { maxTicksLimit: 20, autoSkipPadding: 20 },
      },
      y: {
        title: { display: true, text: 'Amplitude (mV)' },
        suggestedMin: -1.0,
        suggestedMax: 1.5,
      },
    },
    plugins: {
      legend: { display: true, position: 'top' },
      title: { display: true, text: 'Simulated Sinus Rhythm' },
      decimation: {
        enabled: true,
        algorithm: 'lttb',
        samples: 500,
      },
    },
  };

  return (
    <div style={{ padding: '20px' }}>
      <div style={{ marginBottom: '20px' }}>
        <label htmlFor="hrInput" style={{ marginRight: '10px' }}>Heart Rate (bpm):</label>
        <input
          id="hrInput"
          type="number"
          value={heartRate}
          onChange={(e) => setHeartRate(parseFloat(e.target.value))}
          min={30}
          max={250}
          style={{ marginRight: '20px', padding: '5px' }}
        />
        <label htmlFor="durInput" style={{ marginRight: '10px' }}>Duration (s):</label>
        <input
          id="durInput"
          type="number"
          value={duration}
          onChange={(e) => setDuration(parseFloat(e.target.value))}
          min={1}
          max={60}
          style={{ marginRight: '20px', padding: '5px' }}
        />
        <button onClick={fetchEcgData} disabled={isLoading} style={{ padding: '5px 10px' }}>
          {isLoading ? 'Generating...' : 'Regenerate ECG'}
        </button>
      </div>

      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      <div style={{ height: '500px', border: '1px solid #ccc', width: '900px' }}>
        {isLoading && <p>Loading chart...</p>}
        {!isLoading && ecgData.time_axis.length > 0 && (
          <Line ref={chartRef} options={chartOptions} data={chartDataConfig} />
        )}
        {!isLoading && ecgData.time_axis.length === 0 && !error && (
          <p>No ECG data to display. Click "Regenerate ECG".</p>
        )}
      </div>
    </div>
  );
};

export default ECGChart;
