import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


np.random.seed(42)

n_samples = 1000

data = {
    "Age": np.random.randint(18, 70, size=n_samples),
    "Income per Year": np.random.randint(10000, 100000, size=n_samples),
    "Years of Employment": np.random.randint(0, 52, size=n_samples)
}

df = pd.DataFrame(data)


df["Loan Approved"] = (
    (df["Income per Year"] > 40000) &
    (df["Age"] > 35) &
    (df["Years of Employment"] > 5)
).astype(int)



dataset_path = "realistic_dataset.csv"
df.to_csv(dataset_path, index=False)
print(f"Dataset saved: {dataset_path}")


features = [
    "Age",
    "Income per Year",
    "Years of Employment"
]

X = df[features]
y = df["Loan Approved"]

model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)


model.fit(X, y)


model_path = "realistic_decision_tree.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Decision tree saved: {model_path}")