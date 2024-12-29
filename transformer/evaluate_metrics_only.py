import os
import math
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
    df = df.sort_values("Datetime").reset_index(drop=True)
    values = df["AEP_MW"].values.reshape(-1, 1)
    datetimes = df["Datetime"].values
    return values, datetimes

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
            data, targets = get_batch(data_source, i, batch_size, device)
            output = model(data)
            preds = output[-output_window:]
            preds = preds.squeeze(-1).transpose(0, 1).cpu().numpy()
            truth = targets.squeeze(-1).transpose(0, 1).cpu().numpy()

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size=1, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(TransAm, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_projection = nn.Linear(feature_size, d_model)
        self.decoder = nn.Linear(d_model, 1)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        return mask

    def forward(self, src):
        seq_len, batch_size, _ = src.size()
        if self.src_mask is None or self.src_mask.size(0) != seq_len:
            self.src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        src = self.input_projection(src)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src, self.src_mask)
        out = self.decoder(encoded)
        return out


def main():
    csv_path = "transformer/AEP_hourly.csv"
    model_path = "transformer/best_model.pt"

    values, _ = load_aep_data(csv_path)

    scaler = MinMaxScaler(feature_range=(-1,1))
    values_scaled = scaler.fit_transform(values)

    train_ratio = 0.8
    n = len(values_scaled)
    split_point = int(n*train_ratio)
    test_vals = values_scaled[split_point:]

    test_sequence = create_inout_sequences(test_vals, input_window, output_window)

    model = TransAm(feature_size=1, d_model=32, nhead=4, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_trues = get_predictions_and_targets(
        model,
        data_source=test_sequence,
        batch_size=batch_size,
        device=device,
        output_window=output_window
    )

    all_preds_f = all_preds.reshape(-1,1)
    all_trues_f = all_trues.reshape(-1,1)
    all_preds_inv = scaler.inverse_transform(all_preds_f)
    all_trues_inv = scaler.inverse_transform(all_trues_f)

    mse, mae, rmse, mape = compute_metrics(all_trues_inv, all_preds_inv)
    print(f"=== METRICS (TEST) ===")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"MAPE = {mape:.2f}%")

    num_points_to_plot = 500
    y_true_plot = all_trues_inv[-num_points_to_plot:]
    y_pred_plot = all_preds_inv[-num_points_to_plot:]

    plt.figure(figsize=(10,5))
    plt.plot(y_true_plot, label="True", color="blue")
    plt.plot(y_pred_plot, label="Prediction", color="orange")
    plt.title(f"Vergleich: Echte Werte vs. Vorhersagen (letzte {num_points_to_plot} Punkte)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
