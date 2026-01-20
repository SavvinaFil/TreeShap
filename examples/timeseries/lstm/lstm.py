import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------
# 1. Configuration
# -------------------------------
CONFIG = {
    "batch_size": 32,
    "look_back": 6,
    "epochs": 100,
    "hidden_size": 16,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_years": [2022, 2023]
}

# -------------------------------
# 2. Data Engineering Module
# -------------------------------
class DataProcessor:
    def __init__(self, look_back):
        self.look_back = look_back
        self.scaler = None
        self.feature_cols = []

    def load_and_merge(self, load_path, weather_path):
        # Load
        df_load = pd.read_csv(load_path)
        df_weather = pd.read_csv(weather_path)

        # Clean Load
        df_load["HourUTC"] = pd.to_datetime(df_load["TimeUTC"])
        df_load = df_load[df_load["HourUTC"].dt.year.isin(CONFIG["data_years"])]
        df_load = df_load.rename(columns={"Consumption": "Load"})
        df_load = df_load[["HourUTC", "PV", "Load"]]

        # Clean Weather
        df_weather["HourUTC"] = pd.to_datetime(df_weather["HourUTC"])
        df_weather = df_weather[df_weather["HourUTC"].dt.year.isin(CONFIG["data_years"])]
        
        # Merge
        df = pd.merge(df_load, df_weather, on="HourUTC").dropna().reset_index(drop=True)
        return df

    def add_features(self, df):
        # Lags
        for lag in [24, 168]:
            df[f"PV_lag_{lag}"] = df["PV"].shift(lag)

        # Rolling stats
        df["PV_roll_mean_3"] = df["PV"].rolling(window=3).mean()
        df["PV_roll_std_3"] = df["PV"].rolling(window=3).std()

        # Cyclical Time Features
        time_map = {
            "hour": (df["HourUTC"].dt.hour, 24),
            "dow": (df["HourUTC"].dt.dayofweek, 7),
            "month": (df["HourUTC"].dt.month, 12)
        }
        for prefix, (series, max_val) in time_map.items():
            df[f"{prefix}_sin"] = np.sin(2 * np.pi * series / max_val)
            df[f"{prefix}_cos"] = np.cos(2 * np.pi * series / max_val)

        return df.dropna().reset_index(drop=True)

    def prepare_loaders(self, df):
        self.feature_cols = [
            "PV", "ghi", "PV_lag_24", "PV_lag_168", 
            "PV_roll_mean_3", "PV_roll_std_3",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"
        ]
        
        # Define Scaler using column names (much safer than indices)
        self.scaler = ColumnTransformer(
            transformers=[
                ('std_scaler', StandardScaler(), self.feature_cols[:6]),
                ('passthrough', 'passthrough', self.feature_cols[6:])
            ]
        )

        data_scaled = self.scaler.fit_transform(df[self.feature_cols])
        
        X, y = self._create_sequences(data_scaled)
        
        # Split
        split = int(len(X) * 0.8)
        
        train_ds = TensorDataset(torch.tensor(X[:split], dtype=torch.float32), 
                                 torch.tensor(y[:split], dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X[split:], dtype=torch.float32), 
                                torch.tensor(y[split:], dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)
        
        return train_loader, test_loader

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.look_back):
            X.append(data[i : i + self.look_back])
            y.append(data[i + self.look_back, 0]) # Index 0 is PV
        return np.array(X), np.array(y)

# -------------------------------
# 3. Model Architecture
# -------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take the last time step's output
        return self.fc(out[:, -1, :])

# -------------------------------
# 4. Training Engine
# -------------------------------
def run_training(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(CONFIG["device"]), y_batch.to(CONFIG["device"])
        
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

# -------------------------------
# 5. Main Execution
# -------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize Data
    processor = DataProcessor(CONFIG["look_back"])
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up one level and then into the 'data' folder
    data_dir = os.path.join(script_dir, "..", "data")

    df = processor.load_and_merge(
        os.path.join(data_dir, "ProsumerHourly_withUTC.csv"),
        os.path.join(data_dir, "WeatherData.csv")
    )
    df = processor.add_features(df)
    train_loader, test_loader = processor.prepare_loaders(df)

    # Initialize Model
    input_dim = len(processor.feature_cols)
    model = LSTMForecaster(input_dim, CONFIG["hidden_size"]).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.MSELoss()

    # Loop
    history = []
    for epoch in range(CONFIG["epochs"]):
        loss = run_training(model, train_loader, optimizer, criterion)
        history.append(loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{CONFIG["epochs"]} | Loss: {loss:.4f}")
    
    # -------------------------------
    # 6. Save SHAP-Ready Agnostic Tensors
    # -------------------------------
    # Extract tensors from the datasets
    # train_ds.tensors[0] is X (features), [1] is y (labels)
    X_train_tensor = train_loader.dataset.tensors[0]
    X_test_tensor = test_loader.dataset.tensors[0]

    # Define background (representative sample of training data)
    # SHAP usually needs 50-200 samples for the background
    background_samples = X_train_tensor[:100] 
    
    # Define test data to explain (e.g., the first 50 samples of the test set)
    explain_data = X_test_tensor[:50]

    # Save as agnostic .pt files
    torch.save(background_samples, os.path.join(script_dir, "background_data.pt"))
    torch.save(explain_data, os.path.join(script_dir, "test_data_to_explain.pt"))

    # Also save the whole model object (not just state_dict) 
    # for true model-agnostic loading in the explainer
    torch.save(model.state_dict(), os.path.join(script_dir, "lstm_model.pth"))

    print("SHAP artifacts saved: background_data.pt, test_data_to_explain.pt, full_lstm_model.pth")