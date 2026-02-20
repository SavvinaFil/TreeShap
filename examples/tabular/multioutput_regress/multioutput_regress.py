import numpy as np
import pandas as pd
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
n_samples = 1000

# Hybrid system: Wind Power + Solar Power

# Input features
# Wind Speed (m/s), major renewable source
wind_speed = np.random.gamma(shape=3, scale=3.5, size=n_samples)
wind_speed = np.clip(wind_speed, 0, 25)  # Realistic range

# Temperature (°C), affects both demand and generation
temperature = np.random.normal(15, 10, n_samples)
temperature = np.clip(temperature, -15, 40)

# Previous Power Generation (MW)
prev_power = np.random.normal(500, 100, n_samples)
prev_power = np.clip(prev_power, 200, 800)

# Previous Load Demand (MW)
prev_load = np.random.normal(550, 120, n_samples)
prev_load = np.clip(prev_load, 250, 900)

# Hour of Day (0-23)
hour = np.random.randint(0, 24, n_samples)

# Makes Hour 0 and Hour 23 close to each other and not with 23 hours difference)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Output 1: Power Forecast
solar_component = 80 * np.maximum(0, hour_sin)
# hour_sin is positive from 6am to 6pm, max at noon (12:00)

power_forecast = (
    # Wind contribution (major source)
        45 * wind_speed +
        solar_component +
        2.5 * temperature +
        0.25 * prev_power +
        0.08 * prev_load +
        150 +

        np.random.normal(0, 15, n_samples)
)
power_forecast = np.clip(power_forecast, 200, 1200)

# Load Forecast (MW)
load_forecast = (
        # Daily pattern (peak morning and evening, at 6am and 6pm)
        120 * np.abs(hour_sin) +

        # Temperature effects (heating in winter, cooling in summer)
        3.5 * np.abs(temperature - 18) +
        0.45 * prev_load +
        0.12 * prev_power +
        1.2 * wind_speed +
        250 +

        np.random.normal(0, 18, n_samples)
)
load_forecast = np.clip(load_forecast, 300, 1000)

# Output 3: Frequency Deviation (Hz from 50Hz nominal)
frequency_deviation = (
        0.008 * (load_forecast - power_forecast) +
        0.003 * (wind_speed - 10) +
        0.002 * (temperature - 15) +
        0.004 * hour_sin +
        0.001 * (prev_load - prev_power) +

        np.random.normal(0, 0.012, n_samples)
)
frequency_deviation = np.clip(frequency_deviation, -0.5, 0.5)

# Output 4: Voltage Level (per unit, nominal = 1.0)
voltage_level = (
        1.0 +
        -0.15 * frequency_deviation +
        -0.00035 * (load_forecast - 550) +
        0.00025 * (power_forecast - 500) +
        -0.0008 * (temperature - 15) +
        -0.002 * np.abs(wind_speed - 10) +
        -0.008 * np.abs(hour_sin) +
        0.0002 * (prev_power - prev_load) +

        np.random.normal(0, 0.008, n_samples)
)
voltage_level = np.clip(voltage_level, 0.92, 1.08)

# Output 5: Reverse Requirement (MW)
reserve_requirement = (
        0.18 * np.abs(load_forecast - power_forecast) +
        3.5 * np.abs(wind_speed - 10) +
        1.2 * np.abs(temperature - 15) +
        25 * np.abs(hour_sin) +
        0.04 * np.abs(prev_load - prev_power) +
        30 +

        np.random.normal(0, 8, n_samples)
)
reserve_requirement = np.clip(reserve_requirement, 20, 200)

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

# Save dataset
dataset_path = "../../../multioutput_regress_dataset.csv"
df.to_csv(dataset_path, index=False)
print(f"Dataset saved: {dataset_path}")

# Train model
features = [
    "Wind Speed",
    "Temperature",
    "Previous Power",
    "Previous Load",
    "Hour"
]

targets = [
    "Power Forecast",
    "Load Forecast",
    "Frequency Deviation",
    "Voltage Level",
    "Reserve Requirement"
]

X = df[features]
Y = df[targets]

# Train Multi-Output Random Forest
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
)

model.fit(X, Y)

# Save model
model_path = "../../../multioutput_regress_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved: {model_path}")


