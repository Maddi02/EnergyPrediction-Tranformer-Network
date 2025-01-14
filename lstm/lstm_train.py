import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


INPUT_WINDOW = 100
OUTPUT_WINDOW = 24
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3

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


# -------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------
def load_aep_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.sort_values("Datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    values = df["AEP_MW"].values.reshape(-1, 1)
    return values


def train_val_split(values, train_ratio=0.8):
    n = len(values)
    split_point = int(n * train_ratio)
    train_data = values[:split_point]
    val_data = values[split_point:]
    return train_data, val_data


def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        seq = input_data[i : i + tw]
        lbl = seq[-output_window:]
        inout_seq.append((seq, lbl))
    return inout_seq


def get_batch(source, i, batch_size):
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


def train_one_epoch(model, train_data, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_batches = (len(train_data) - 1) // BATCH_SIZE

    for batch_idx, i in enumerate(range(0, len(train_data) - 1, BATCH_SIZE)):
        inp, lbl = get_batch(train_data, i, BATCH_SIZE)
        optimizer.zero_grad()
        output = model(inp)

        pred = output[-OUTPUT_WINDOW:]
        true = lbl

        loss = criterion(pred, true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
    return avg_loss


def evaluate(model, val_data, criterion):
    model.eval()
    total_loss = 0.0
    n_batches = (len(val_data) - 1) // BATCH_SIZE

    with torch.no_grad():
        for i in range(0, len(val_data) - 1, BATCH_SIZE):
            inp, lbl = get_batch(val_data, i, BATCH_SIZE)
            output = model(inp)

            pred = output[-OUTPUT_WINDOW:]
            true = lbl

            loss = criterion(pred, true)
            total_loss += loss.item()

    avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
    return avg_loss

def main():
    csv_path = "AEP_hourly.csv"
    model_path = "best_model_lstm.pt"

    values = load_aep_data(csv_path)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    values_scaled = scaler.fit_transform(values)

    train_vals, val_vals = train_val_split(values_scaled, train_ratio=0.8)
    print(f"Train-Datensätze: {len(train_vals)}, Val-Datensätze: {len(val_vals)}")

    train_sequence = create_inout_sequences(train_vals, INPUT_WINDOW, OUTPUT_WINDOW)
    val_sequence = create_inout_sequences(val_vals, INPUT_WINDOW, OUTPUT_WINDOW)

    model = LSTMForecast(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_sequence, optimizer, criterion)
        val_loss = evaluate(model, val_sequence, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"| Train Loss = {train_loss:.6f} "
              f"| Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Train & Val Loss (LSTM, 80/20)")
    plt.legend()
    plt.show()

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler
    }, model_path)
    print(f"Trainiertes LSTM-Modell wurde gespeichert unter: {model_path}")



if __name__ == "__main__":
    main()
