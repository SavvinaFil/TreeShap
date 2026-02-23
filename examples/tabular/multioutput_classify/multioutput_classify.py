import numpy as np
import pandas as pd
import pickle
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n_samples = 1500

# Hybrid Energy System: Wind + Solar + Battery + Market Participation

# Wind Speed (m/s)
wind_speed = np.random.gamma(shape=3, scale=3.2, size=n_samples)
wind_speed = np.clip(wind_speed, 0, 25)

# Solar Irradiance (W/m²)
solar_irradiance = np.random.normal(500, 250, n_samples)
solar_irradiance = np.clip(solar_irradiance, 0, 1000)

# Temperature (°C)
temperature = np.random.normal(18, 9, n_samples)
temperature = np.clip(temperature, -10, 40)

# Previous Generation (MW)
prev_generation = np.random.normal(600, 120, n_samples)
prev_generation = np.clip(prev_generation, 250, 1000)

# Previous Load (MW)
prev_load = np.random.normal(650, 150, n_samples)
prev_load = np.clip(prev_load, 300, 1200)

# Day Ahead Price (€/MWh)
day_ahead_price = np.random.normal(95, 30, n_samples)
day_ahead_price = np.clip(day_ahead_price, 20, 250)

# Battery State of Charge (%)
battery_soc = np.random.uniform(15, 95, n_samples)

# Grid Frequency (Hz)
grid_frequency = np.random.normal(50, 0.15, n_samples)

# Congestion Index (0-1)
congestion_index = np.random.uniform(0, 1, n_samples)

# Hour
hour = np.random.randint(0, 24, n_samples)
hour_sin = np.sin(2 * np.pi * hour / 24)

wind_generation = 35 * wind_speed
solar_generation = 0.12 * solar_irradiance * np.maximum(0, hour_sin)
total_generation = wind_generation + solar_generation

load_estimation = (
    150 * np.abs(hour_sin) +
    3.2 * np.abs(temperature - 20) +
    0.5 * prev_load +
    250
)

power_balance = total_generation - load_estimation

# Output 0: Export Mode (when generation > load)
export_mode = (power_balance > 50).astype(int)

# Output 1: Import Mode (when load > generation)
import_mode = (power_balance < -50).astype(int)

# Output 2: Battery Charging (when prices are low and SOC is not full)
battery_charging = (
    (day_ahead_price < 75) &
    (battery_soc < 85) &
    (power_balance > 0)
).astype(int)

# Output 3: Battery Discharging (when prices are high and SOC is sufficient)
battery_discharging = (
    (day_ahead_price > 115) &
    (battery_soc > 30) &
    (power_balance < 0)
).astype(int)

# Output 4: Reserve Activation (when there's significant imbalance)
reserve_activation = (
    np.abs(power_balance) > 200
).astype(int)

# Output 5: Frequency Support Active (when frequency deviates significantly)
frequency_support_active = (
    np.abs(grid_frequency - 50) > 0.12
).astype(int)

# Output 6: Congestion Management (when network is congested)
congestion_management = (
    congestion_index > 0.7
).astype(int)

# Output 7: Curtailment Active (when generation is too high and network is congested)
curtailment_active = (
    (total_generation > 900) &  # Lower threshold for more activations
    (congestion_index > 0.45)   # Lower threshold
).astype(int)

# Output 8: Peak Load Response (during evening peak hours with high load)
peak_load_response = (
    (load_estimation > 800) &   # Lower threshold
    (hour >= 16) &              # Start earlier
    (hour <= 22)                # End later
).astype(int)

# Output 9: High Price Operation (when market prices are very high)
high_price_operation = (
    day_ahead_price > 130
).astype(int)

outputs_dict = {
    "Export Mode": export_mode,
    "Import Mode": import_mode,
    "Battery Charging": battery_charging,
    "Battery Discharging": battery_discharging,
    "Reserve Activation": reserve_activation,
    "Frequency Support Active": frequency_support_active,
    "Congestion Management": congestion_management,
    "Curtailment Active": curtailment_active,
    "Peak Load Response": peak_load_response,
    "High Price Operation": high_price_operation
}

df = pd.DataFrame({
    "Wind Speed": wind_speed,
    "Solar Irradiance": solar_irradiance,
    "Temperature": temperature,
    "Previous Generation": prev_generation,
    "Previous Load": prev_load,
    "Day Ahead Price": day_ahead_price,
    "Battery SOC": battery_soc,
    "Grid Frequency": grid_frequency,
    "Congestion Index": congestion_index,
    "Hour": hour,

    "Export Mode": export_mode,
    "Import Mode": import_mode,
    "Battery Charging": battery_charging,
    "Battery Discharging": battery_discharging,
    "Reserve Activation": reserve_activation,
    "Frequency Support Active": frequency_support_active,
    "Congestion Management": congestion_management,
    "Curtailment Active": curtailment_active,
    "Peak Load Response": peak_load_response,
    "High Price Operation": high_price_operation
})

# Save Dataset
dataset_path = "../../../multioutput_classify_dataset.csv"
df.to_csv(dataset_path, index=False)
print(f"\n\nDataset saved: {dataset_path}")

#Train model
features = [
    "Wind Speed",
    "Solar Irradiance",
    "Temperature",
    "Previous Generation",
    "Previous Load",
    "Day Ahead Price",
    "Battery SOC",
    "Grid Frequency",
    "Congestion Index",
    "Hour"
]

targets = [
    "Export Mode",
    "Import Mode",
    "Battery Charging",
    "Battery Discharging",
    "Reserve Activation",
    "Frequency Support Active",
    "Congestion Management",
    "Curtailment Active",
    "Peak Load Response",
    "High Price Operation"
]

X = df[features]
Y = df[targets]

# Train Model
model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
)

model.fit(X, Y)

# Save Model
model_path = "../../../multioutput_classify_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved: {model_path}")