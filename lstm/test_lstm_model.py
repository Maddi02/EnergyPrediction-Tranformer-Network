import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

input_window = 100
output_window = 24
batch_size = 32

def load_aep_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")
    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    values = df["AEP_MW"].values.reshape(-1, 1)
    return values

def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        seq = input_data[i : i + tw]
        lbl = seq[-output_window:]
        inout_seq.append((seq, lbl))
    return inout_seq

def get_batch(source, i, batch_size, device):
    seq_len = min(batch_size, len(source) - 1 - i)
    data_batch = source[i : i + seq_len]

    inp_list, lbl_list = [], []
    for (inp, lbl) in data_batch:
        inp_list.append(torch.tensor(inp, dtype=torch.float))
        lbl_list.append(torch.tensor(lbl, dtype=torch.float))

    inp_batch = torch.stack(inp_list)
    lbl_batch = torch.stack(lbl_list)

    inp_batch = inp_batch.transpose(0, 1)
    lbl_batch = lbl_batch.transpose(0, 1)

    return inp_batch.to(device), lbl_batch.to(device)

def get_predictions_and_targets(model, data_source, batch_size, device, output_window):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for i in range(0, len(data_source) - 1, batch_size):
            inp, lbl = get_batch(data_source, i, batch_size, device)
            output = model(inp)

            preds = output[-output_window:]

            preds = preds.squeeze(-1).transpose(0, 1).cpu().numpy()
            truth = lbl.squeeze(-1).transpose(0, 1).cpu().numpy()

            all_preds.append(preds)
            all_trues.append(truth)

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    return all_preds, all_trues

def compute_metrics(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    mse = np.mean((y_true_f - y_pred_f)**2)
    mae = np.mean(np.abs(y_true_f - y_pred_f))
    rmse = np.sqrt(mse)

    eps = 1e-9
    mape = np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + eps))) * 100
    return mse, mae, rmse, mape

class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=False
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out)
        return out

def main():
    csv_path = "AEP_hourly.csv"
    model_path = "best_model_lstm.pt"

    values = load_aep_data(csv_path)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    values_scaled = scaler.fit_transform(values)

    n = len(values_scaled)
    split_point = int(n * 0.8)
    test_vals = values_scaled[split_point:]

    test_sequence = create_inout_sequences(test_vals, input_window, output_window)
    if not test_sequence:
        raise ValueError("Zu wenige Daten! Die Test-Sequenz ist leer.")

    model = LSTMForecast(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        dropout=0.1
    ).to(device)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Kein trainiertes LSTM gefunden unter: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("LSTM-Modell erfolgreich geladen.")


    all_preds, all_trues = get_predictions_and_targets(
        model,
        data_source=test_sequence,
        batch_size=batch_size,
        device=device,
        output_window=output_window
    )
    all_preds_f = all_preds.reshape(-1, 1)
    all_trues_f = all_trues.reshape(-1, 1)

    all_preds_inv = scaler.inverse_transform(all_preds_f)
    all_trues_inv = scaler.inverse_transform(all_trues_f)

    mse, mae, rmse, mape = compute_metrics(all_trues_inv, all_preds_inv)
    print(f"=== METRICS (TEST, LSTM) ===")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"MAPE = {mape:.2f}%")

    num_points_to_plot = 500
    y_true_plot = all_trues_inv[-num_points_to_plot:]
    y_pred_plot = all_preds_inv[-num_points_to_plot:]

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_plot, label="True", color="blue")
    plt.plot(y_pred_plot, label="Prediction", color="orange")
    plt.title(f"LSTM: Echte Werte vs. Vorhersagen (letzte {num_points_to_plot} Punkte)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
