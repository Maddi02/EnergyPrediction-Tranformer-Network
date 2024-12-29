# Transformer-Based Energy Forecasting

This repository provides an example of using a Transformer-based architecture for forecasting hourly energy consumption. The code leverages **PyTorch** (including MPS support on macOS if available), **scikit-learn** for data scaling, and **matplotlib** for visualization.

## Contents

- **`main.py`**: Main script for training and evaluating the Transformer model.
- **`AEP_hourly.csv`**: Hourly energy consumption data (date/time and MW values).
- **`best_model.pt`**: Model checkpoint saved after training (created automatically).

## Requirements

- Python 3.8+
- PyTorch (with optional MPS support on macOS)
- NumPy
- Pandas
- scikit-learn
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```
## How It Works

### Data Loading
- Reads `AEP_hourly.csv` containing datetime and power consumption (MW).

### Preprocessing
- Sorts records by datetime.
- Scales values into a range of `(-1, 1)` using `MinMaxScaler`.

### Sequence Creation
- Slices the data into sequences of length `input_window` (default: 100).
- Each sequence is paired with an `output_window` (default: 24) for forecasting.

### Model Architecture
- Uses a Transformer-based encoder (`TransAm` class).
- Includes a positional encoding layer to retain time-step information.
- Outputs one value per time step (forecast in MW).

### Training & Validation
- Splits data into training and testing sets (default ratio: 80:20).
- Optimizes with `AdamW` and `MSELoss`.
- Tracks both training and validation losses, saving the best model as `best_model.pt`.

### Prediction
- After training, `predict_future` forecasts `future_steps` (default: 24) time steps.
- Predictions are inverse-transformed back to the original MW scale for interpretability.

## Quick Start

### Clone the Repository
```bash
git clone https://github.com/<YourUsername>/<RepositoryName>.git
cd <RepositoryName>
```


## Execute script
- Carries out the training of the model.  
- Displays the plots for training and validation loss.  
- Outputs forecasts for future values.

## Adjust hyperparameters

The following hyperparameters can be modified directly in **main.py**:
- **input_window**
- **output_window**
- **epochs**
- **Transformer-Parameter** wie:
  - **d_model**
  - **nhead**

---

## Check results

- The generated plots show the loss curves (training and validation loss).
- The best model is saved under the file name **best_model.pt** (based on the smallest validation loss).
- Further plots and console outputs show the predictions for future data.

---

## Notes on MPS (macOS)

- The script automatically detects whether MPS is available.
- If MPS is not available, the CPU is used by default.


