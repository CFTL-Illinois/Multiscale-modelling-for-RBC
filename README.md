# Multi-scale DManD Modeling for RBC

This repository implements a **multi-scale reduced-order modeling** approach for **Rayleigh–Bénard convection (RBC)** based on **Dynamic Mode Decomposition with Manifold Learning (DManD)**.  
The workflow consists of three main steps:

1. **POD** reduction of high-dimensional flow fields to obtain modal coefficients.  
2. **Autoencoder** to separate the coefficients into **large-scale** and **small-scale** components.  
3. **Neural ODE** to learn the temporal evolution of latent variables for both scales.

## Files

| File | Description |
|------|-------------|
| `1_Autoencoder_RB.py` | Trains autoencoders for large and small scales, evaluates reconstruction error for different latent dimensions (`dh`), and selects the optimal `dh`. |
| `1_NODE.py` | Loads the pretrained autoencoders, projects data into latent space, trains two Neural ODE networks to model the dynamics of large- and small-scale latent variables, and generates prediction plots. |

## Data Preparation

> **Note:** Due to storage limitations, the required data files are **not** included in this repository. You need to generate or obtain them separately.

The code expects the following `.npy` files:

- `POD_Results/modes.npy` – POD mode matrix  
- `POD_Results/eigenvalues.npy` – POD eigenvalues  
- `a_large_scale.npy` – large‑scale POD coefficients (shape `[time_steps, k]`)  
- `a_small_scale.npy` – small‑scale POD coefficients  
- `a_std.npy` and `a_mean.npy` (used in `Autoencoder_RB.py`)

Default split: 80% training, 20% testing.

## Usage

### 1. Train autoencoders

  python 1_Autoencoder_RB.py

- Automatically determines the number of POD modes `k` that retain 99.95% of the energy.  
- Tests latent dimensions `dh = [2,4,6,8,12,16]` and trains separate autoencoders for large and small scales.  
- Saves best models as `best_model_large_dh{dh}.pt` and `best_model_small_dh{dh}.pt`.  
- Generates reconstruction error plot: `MSE_vs_dh.png`.

### 2. Train Neural ODEs

  python 1_NODE.py

- Requires the pretrained autoencoders from step 1 (example uses `dh_large=6`, `dh_small=14`).  
- Encodes large‑ and small‑scale coefficients into latent variables and normalizes them.  
- Trains two `ODEFunc` networks (6 hidden layers + time encoder).  
- Logs losses every 20 iterations and generates prediction comparison plots every 100 iterations (saved in `test_predictions/`).  
- Checkpoints are saved in the `checkpoints/` folder.

## Configuration

### 1_Autoencoder_RB.py

| Parameter | Description |
|-----------|-------------|
| `dh_list` | List of latent dimensions to test |
| `iters` | Maximum training epochs (default 500) |
| `batch_size` | Initial batch size (automatically reduced if GPU OOM) |
| `patience` | Early stopping patience (default 150) |

### 1_NODE.py

| Argument | Description |
|----------|-------------|
| `--data_size` | Length of training data (automatically inferred) |
| `--batch_time` | ODE integration window (100 for large scale, 1 for small scale) |
| `--method` | ODE solver (`dopri5` or `adams`) |
| `--niters` | Total training iterations (default 50000; script uses additional 40000 steps) |
| `--gpu` | GPU device index |
| `--adjoint` | Enable adjoint method for gradient computation |

## Output Files

### Autoencoder training

- `Out_large_dh{dh}.txt`, `Out_small_dh{dh}.txt` – training logs  
- `training_curve.png` – loss curve (updated every 10 epochs)  
- `best_model_*dh*.pt` – best model weights  

### Neural ODE training

- `NODE_multiscale_analysis.png` – semi‑log plot of total, large‑scale, and small‑scale losses  
- `test_predictions/test_a_predictions_iter_{itr}.png` – prediction comparison plots (first three modes) every 100 iterations  
- `checkpoints/` – model checkpoints (`.pt` files)