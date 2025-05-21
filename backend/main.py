import uvicorn
from ecg_simulator.api import app # Import the 'app' instance from your api module

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)