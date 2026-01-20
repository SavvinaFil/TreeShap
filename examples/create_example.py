import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

n_samples = 1000

# First create the ages
ages = np.random.randint(18, 70, size=n_samples)

# Calculate the max years of employment per person - we suppose they started after 18 or later
max_work_years = ages - 18

# Per person years of employment can be from 0 to max_work_years
years_of_employment = np.array([
    np.random.randint(0, max(1, max_years + 1))
    for max_years in max_work_years
])

# Make realistic incomes that increase when do so age and experience
base_income = np.random.randint(15000, 50000, size=n_samples)
experience_bonus = years_of_employment * np.random.randint(500, 2000, size=n_samples)
income_per_year = np.clip(base_income + experience_bonus, 10000, 150000)

data = {
    "Age": ages,
    "Income per Year": income_per_year,
    "Years of Employment": years_of_employment
}

df = pd.DataFrame(data)


df["Loan Approved"] = (
    (df["Income per Year"] > 40000) &
    (df["Age"] >= 30) &
    (df["Years of Employment"] >= 2) &
    (
        # extra criteria for better balance
        ((df["Age"] > 35) & (df["Years of Employment"] > 5)) |
        ((df["Income per Year"] > 60000) & (df["Years of Employment"] > 3))
    )
).astype(int)


# Save dataset
dataset_path = "realistic_dataset.csv"
df.to_csv(dataset_path, index=False)
print(f"\nDataset saved: {dataset_path}")

# Train model
features = [
    "Age",
    "Income per Year",
    "Years of Employment"
]

X = df[features]
y = df["Loan Approved"]

model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X, y)


# Save the model
model_path = "realistic_decision_tree.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Decision tree saved: {model_path}")