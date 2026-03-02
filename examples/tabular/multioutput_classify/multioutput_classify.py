import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n_samples = 1500

root_dir = Path(__file__).resolve().parents[3]
model_dir = root_dir / "source" / "models"
data_dir = root_dir / "source" / "data"

model_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

model_filename = "multioutput_classify.pkl"
data_filename = "multioutput_classify.csv"

model_save_path = model_dir / model_filename
data_save_path = data_dir / data_filename

# Input Features
# Wind Speed (m/s)
wind_speed = np.random.gamma(shape=3, scale=3.2, size=n_samples)
wind_speed = np.clip(wind_speed, 0, 25)

# Solar Irradiance (W/m²)
solar_irradiance = np.random.normal(500, 250, n_samples)
solar_irradiance = np.clip(solar_irradiance, 0, 1000)

# Previous Generation (MW)
prev_generation = np.random.normal(600, 120, n_samples)
prev_generation = np.clip(prev_generation, 250, 1000)

# Previous Load (MW)
prev_load = np.random.normal(650, 150, n_samples)
prev_load = np.clip(prev_load, 300, 1200)

# Day Ahead Price (€/MWh)
day_ahead_price = np.random.normal(95, 30, n_samples)
day_ahead_price = np.clip(day_ahead_price, 20, 250)

# Hour of day
hour = np.random.randint(0, 24, n_samples)
hour_sin = np.sin(2 * np.pi * hour / 24)


# Renewable generation estimation
wind_generation = 35 * wind_speed
solar_generation = 0.12 * solar_irradiance * np.maximum(0, hour_sin)
total_renewable = wind_generation + solar_generation

# Load estimation
load_estimation = (
    150 * np.abs(hour_sin) +
    0.5 * prev_load +
    250
)

# Residual load: how much thermal generation is needed
residual_load = load_estimation - total_renewable

# Unit Commitment Outputs (ON=1 / OFF=0)
# Generator 1 (600 MW) — dispatched when residual load is very high
generator_1 = (residual_load > 500).astype(int)

# Generator 2 (400 MW) — dispatched when residual load is high
generator_2 = (residual_load > 300).astype(int)

# Generator 3 (200 MW) — dispatched during peak hours or high residual load
generator_3 = (
    (residual_load > 400) |
    ((hour >= 17) & (hour <= 22) & (load_estimation > 700))
).astype(int)

# Generator 4 (100 MW) — only dispatched under very high price or high residual load
generator_4 = (
    (day_ahead_price > 130) &
    (residual_load > 450)
).astype(int)

# Generator 5 (150 MW) — dispatched when renewable generation is low
generator_5 = (
    (total_renewable < 200) &
    (residual_load > 200)
).astype(int)

# Generator 6 (500 MW) — baseload, always on except very low load periods
generator_6 = (load_estimation > 350).astype(int)

df = pd.DataFrame({
    "Wind Speed": wind_speed,
    "Solar Irradiance": solar_irradiance,
    "Previous Generation": prev_generation,
    "Previous Load": prev_load,
    "Day Ahead Price": day_ahead_price,
    "Hour": hour,

    "Generator 1": generator_1,
    "Generator 2": generator_2,
    "Generator 3": generator_3,
    "Generator 4": generator_4,
    "Generator 5": generator_5,
    "Generator 6": generator_6
})

# Save dataset
df.to_csv(data_save_path, index=False)
print(f"Dataset saved to: {data_save_path}")

# Train model
features = [
    "Wind Speed",
    "Solar Irradiance",
    "Previous Generation",
    "Previous Load",
    "Day Ahead Price",
    "Hour"
]

targets = [
    "Generator 1",
    "Generator 2",
    "Generator 3",
    "Generator 4",
    "Generator 5",
    "Generator 6"
]

X = df[features]
Y = df[targets]

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

# Save model
with open(model_save_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {model_save_path}")