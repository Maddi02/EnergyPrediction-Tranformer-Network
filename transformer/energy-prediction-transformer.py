import os
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

input_window = 100
output_window = 24
batch_size = 32
epochs = 5
lr = 1e-3
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
def load_aep_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

    df = pd.read_csv(csv_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    values = df["AEP_MW"].values.reshape(-1, 1)
    return values, df["Datetime"].values


def train_test_split(values, train_ratio=0.8):
    n = len(values)
    split_point = int(n * train_ratio)
    train_data = values[:split_point]
    test_data = values[split_point:]
    return train_data, test_data


def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i : i + tw]
        train_label = train_seq[-output_window:]
        inout_seq.append((train_seq, train_label))
    return inout_seq


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=1, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.input_projection = nn.Linear(feature_size, d_model)
        self.decoder = nn.Linear(d_model, 1)

        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
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

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i : i + seq_len]
    input_batch = []
    label_batch = []
    for (inp, lbl) in data:
        inp_torch = torch.tensor(inp, dtype=torch.float)
        lbl_torch = torch.tensor(lbl, dtype=torch.float)
        input_batch.append(inp_torch)
        label_batch.append(lbl_torch)

    input_batch = torch.stack(input_batch)
    label_batch = torch.stack(label_batch)

    input_batch = input_batch.transpose(0, 1)
    label_batch = label_batch.transpose(0, 1)

    return input_batch.to(device), label_batch.to(device)


def train_one_epoch(model, train_data, optimizer, criterion):
    model.train()
    total_loss = 0.
    start_time = time.time()

    n_batches = (len(train_data) - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()

        output = model(data)
        pred = output[-output_window:]
        true = targets

        loss = criterion(pred, true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
    return avg_loss


def evaluate(model, val_data, criterion):
    model.eval()
    total_loss = 0.
    n_batches = (len(val_data) - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(val_data) - 1, batch_size):
            data, targets = get_batch(val_data, i, batch_size)
            output = model(data)
            pred = output[-output_window:]
            true = targets
            loss = criterion(pred, true)
            total_loss += loss.item()
    avg_loss = total_loss / n_batches if n_batches > 0 else total_loss
    return avg_loss


def predict_future(model, data_source, steps=24):
    model.eval()
    last_seq = data_source[-1][0]
    last_seq = last_seq.unsqueeze(1).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(steps):
            out = model(last_seq)
            next_val = out[-1].item()
            preds.append(next_val)
            new_seq = torch.cat([last_seq[1:], out[-1:]], dim=0)
            last_seq = new_seq

    return preds


def main():
    csv_path = "transformer/AEP_hourly.csv"
    values, datetimes = load_aep_data(csv_path)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    values_scaled = scaler.fit_transform(values)

    train_vals, test_vals = train_test_split(values_scaled, 0.8)

    train_sequence = create_inout_sequences(train_vals, input_window, output_window)
    test_sequence = create_inout_sequences(test_vals, input_window, output_window)

    feature_size = 1
    d_model = 32
    nhead = 4
    num_layers = 2

    model = TransAm(
        feature_size=feature_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model = None

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_sequence, optimizer, criterion)
        print(f"Epoch [{epoch}/{epochs}] | Train Loss = {train_loss:.4f}")
        val_loss = evaluate(model, test_sequence, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch}/{epochs}] | Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    if best_model is not None:
        model.load_state_dict(best_model)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss-Verlauf")
    plt.show()
    torch.save(model.state_dict(), "best_model.pt")
    future_steps = 24
    preds = predict_future(model, test_sequence, steps=future_steps)
    preds = np.array(preds).reshape(-1, 1)

    preds_rescaled = scaler.inverse_transform(preds)

    print(f"Vorhersage (nächste {future_steps} Stunden):")
    print(preds_rescaled.squeeze())

    plt.figure()
    plt.plot(preds_rescaled, label="Prediction (Future)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
