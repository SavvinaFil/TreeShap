import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# --- 1. Path Setup ---
# Adjust parents[3] if your folder depth is different
root_dir = Path(__file__).resolve().parents[3] 
model_dir = root_dir / "source" / "models"
data_dir = root_dir / "source" / "data"

# Create directories
model_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# Define specific file names
data_filename = "rf_regress_dataset.csv"
model_filename = "rf_regress.pkl"

data_save_path = data_dir / data_filename
model_save_path = model_dir / model_filename

# --- 2. Data Generation ---
n_samples = 1000

# Input features
wind_speed = np.random.gamma(shape=3, scale=3.5, size=n_samples)
wind_speed = np.clip(wind_speed, 0, 25)
temperature = np.random.normal(15, 10, n_samples)
temperature = np.clip(temperature, -15, 40)
prev_power = np.random.normal(500, 100, n_samples)
prev_power = np.clip(prev_power, 200, 800)
prev_load = np.random.normal(550, 120, n_samples)
prev_load = np.clip(prev_load, 250, 900)
hour = np.random.randint(0, 24, n_samples)
hour_sin = np.sin(2 * np.pi * hour / 24)

# Output calculation (Physics-based logic)
solar_component = 80 * np.maximum(0, hour_sin)
power_forecast = np.clip(45 * wind_speed + solar_component + 2.5 * temperature + 150 + np.random.normal(0, 15, n_samples), 200, 1200)
load_forecast = np.clip(120 * np.abs(hour_sin) + 3.5 * np.abs(temperature - 18) + 250 + np.random.normal(0, 18, n_samples), 300, 1000)
frequency_deviation = np.clip(0.008 * (load_forecast - power_forecast) + np.random.normal(0, 0.012, n_samples), -0.5, 0.5)
voltage_level = np.clip(1.0 - 0.15 * frequency_deviation - 0.00035 * (load_forecast - 550) + np.random.normal(0, 0.008, n_samples), 0.92, 1.08)
reserve_requirement = np.clip(0.18 * np.abs(load_forecast - power_forecast) + 30 + np.random.normal(0, 8, n_samples), 20, 200)

# Create dataset
df = pd.DataFrame({
    "Wind Speed": wind_speed,
    "Temperature": temperature,
    "Previous Power": prev_power,
    "Previous Load": prev_load,
    "Hour": hour,
    "Power Forecast": power_forecast,
    "Load Forecast": load_forecast,
    "Frequency Deviation": frequency_deviation,
    "Voltage Level": voltage_level,
    "Reserve Requirement": reserve_requirement
})

# --- 3. Save Dataset ---
df.to_csv(data_save_path, index=False)
print(f"Dataset saved to: {data_save_path}")

# --- 4. Train Multi-Output Model ---
features = ["Wind Speed", "Temperature", "Previous Power", "Previous Load", "Hour"]
targets = ["Power Forecast", "Load Forecast", "Frequency Deviation", "Voltage Level", "Reserve Requirement"]

X = df[features]
Y = df[targets]

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42
    )
)
model.fit(X, Y)

# --- 5. Save Model ---
with open(model_save_path, "wb") as f:
    pickle.dump(model, f)

print(f"Multi-Output Model saved to: {model_save_path}")