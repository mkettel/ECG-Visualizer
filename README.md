# ECG Rhythm Simulator

A web application for generating and visualizing various electrocardiogram (ECG) patterns and cardiac rhythms. This project consists of a Python FastAPI backend for signal generation and a Next.js frontend for interactive visualization.

![ECG Rhythm Simulator Demo](frontend/public/screenshot.png)

## Features

- Interactive ECG visualization with Chart.js
- Configurable ECG parameters (heart rate, duration, amplitude)
- Simulation of various cardiac rhythms:
  - Normal sinus rhythm
  - Atrial fibrillation/flutter
  - AV blocks (first, second, and third degree)
  - Premature contractions (PACs, PVCs)
  - Supraventricular tachycardia (SVT)
  - Ventricular tachycardia (VT)
- Dark/light theme toggle
- Responsive design

## Project Structure

- `/frontend`: Next.js application
  - `/app`: Next.js application routes
  - `/components`: React components including ECGChart.tsx
  - `/public`: Static assets
- `/backend`: FastAPI application
  - `main.py`: Entry point for the backend server
  - `requirements.txt`: Python dependencies

## Prerequisites

- Node.js (v18 or higher)
- Python 3.8+
- npm or yarn

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install the required dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

## Running the Application

### Start the Backend Server

From the backend directory with the virtual environment activated:

```bash
python main.py
or 
uvicorn main:app --reload
```

The backend server will start at http://localhost:8000.

### Start the Frontend Development Server

From the frontend directory:

```bash
npm run dev
# or
yarn dev
```

The frontend development server will start at http://localhost:3000.

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Use the control panel to adjust ECG parameters:
   - Basic settings (heart rate, duration)
   - AV conduction properties
   - Select rhythm types (normal, afib, aflutter, etc.)
   - Configure ectopic beats and arrhythmias
3. Click "Generate ECG" to create and display the simulated ECG

## Development

### Backend API

The backend provides REST endpoints for generating ECG data based on the provided parameters.

- `POST /generate-ecg`: Generates ECG data based on the provided configuration

### Frontend

The frontend is built with Next.js and uses:
- Chart.js and react-chartjs-2 for ECG visualization
- Tailwind CSS for styling
- TypeScript for type safety

## License

[MIT](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request