import os
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)

def output(text, filename='Out.txt'):
    with open(filename, 'a+') as newfile:
        newfile.write(text + '\n')

def plot_loss_curve(train_loss, val_loss, lr, filename='training_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_loss, '.-', label='Train Loss')
    plt.semilogy(val_loss, '.-', label='Validation Loss')
    plt.title(f'Loss Curve (Current LR: {lr:.2e})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, trunc, k, dropout_rate=0.2):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(k, 3000),
            nn.GELU(),
            nn.Linear(3000, 500),
            nn.GELU(),
            nn.Linear(500, 250),
            nn.GELU(),
            nn.Linear(250, trunc)
        )

        self.decode = nn.Sequential(
            nn.Linear(trunc, 250),
            nn.GELU(),
            nn.Linear(250, 500),
            nn.GELU(),
            nn.Linear(500, 3000),
            nn.GELU(),
            nn.Linear(3000, k)
        )

    def forward(self, y):
        return self.decode(self.encode(y))

def Linear(N,a):
    A = np.diag(-a * np.ones(N))
    return A

class Autoencoder_small(nn.Module):
    def __init__(self, trunc, k, dropout_rate=0.2):
        super(Autoencoder_small, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(k, 3000),
            nn.GELU(),
            nn.Linear(3000, 1500),
            nn.GELU(),
            nn.Linear(1500, 500),
            nn.GELU(),
            nn.Linear(500, 100),
            nn.GELU(),
            nn.Linear(100, trunc),
        )
        self.decode = nn.Sequential(
            nn.Linear(trunc, 100),
            nn.GELU(),
            nn.Linear(100, 500),
            nn.GELU(),
            nn.Linear(500, 1500),
            nn.GELU(),
            nn.Linear(1500, 3000),
            nn.GELU(),
            nn.Linear(3000, k),
        )

        self.lin=nn.Sequential(nn.Linear(trunc, trunc,bias=False),
                               nn.Linear(trunc, trunc,bias=False),
                               nn.Linear(trunc, trunc,bias=False),
                               nn.Linear(trunc, trunc,bias=False),)

    def forward(self, y):
        return self.decode(self.encode(y))


def train_model(a_train, a_test, k, trunc, out_name, model_name, iters=500, batch_size=128, patience=150):
    a_train_tensor = torch.tensor(a_train, dtype=torch.float64).to(device)
    a_test_tensor = torch.tensor(a_test, dtype=torch.float64).to(device)

    def auto_batch_size(initial_bs):
        current_bs = initial_bs
        while True:
            try:
                test_tensor = torch.randn((current_bs, k), dtype=torch.float64).to(device)
                _ = model(test_tensor)
                del test_tensor
                torch.cuda.empty_cache()
                return current_bs
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    current_bs = max(current_bs//2, 32)
                    print(f"自动减小batch_size到 {current_bs}")
                    torch.cuda.empty_cache()
                else:
                    raise e

    model = Autoencoder(trunc, k, dropout_rate=0.3).double().to(device)
    optimized_batch_size = auto_batch_size(batch_size)
    
    train_loader = DataLoader(
        a_train_tensor.cpu().numpy(),
        batch_size=optimized_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True  
    )
    optimizer = optim.AdamW([
            {'params': model.encode.parameters(),'weight_decay': 1e-6},
            {'params': model.decode.parameters(),'weight_decay': 1e-6},
        ],lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(iters/2), gamma=0.1)
    train_loss_history, val_loss_history = [], []
    best_loss = float('inf')
    patience_counter = 0
    last_log_time = time.time()
    total_start_time = time.time()

    for itr in range(1, iters + 1):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = torch.tensor(batch, dtype=torch.float64).to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = torch.mean((pred - batch) ** 2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            del batch, pred
            torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(a_test_tensor)
            val_loss = torch.mean((val_pred - a_test_tensor) ** 2).item()
            del val_pred
            torch.cuda.empty_cache()
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_name)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        if itr % 10 == 0:
            current_time = time.time()
            elapsed = current_time - last_log_time
            last_log_time = current_time
            total_elapsed = current_time - total_start_time

            output_msg = (
                f'Iter {itr:04d} | '
                f'Time {elapsed:.1f}s (Total {total_elapsed/60:.1f}min) | '
                f'Train Loss {train_loss:.3e} | '
                f'Val Loss {val_loss:.3e} | '
                f'LR {optimizer.param_groups[0]["lr"]:.2e}'
            )
            output(output_msg, filename = out_name)
            plot_loss_curve(train_loss_history, val_loss_history, optimizer.param_groups[0]["lr"])

            gc.collect()

    return model

from scipy.ndimage import gaussian_filter1d

def load_data(file_path, nx=128, nz=64, frac=0.8, sigma=5):

    modes = np.load('POD_Results/modes.npy')
    eigenvalues = np.load('POD_Results/eigenvalues.npy')

    cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k = np.argmax(cumulative_energy >= 0.9995)
    modes_k = modes[:, :k]

    a_large_scale = np.load('a_large_scale.npy')
    a_small_scale = np.load('a_small_scale.npy')
    a_std = np.load('a_std.npy')
    a_mean = np.load('a_mean.npy')

    M = len(a_large_scale)
    indices = np.arange(M)
    np.random.shuffle(indices)

    train_size = round(M * frac)
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    a_large_train, a_large_test = a_large_scale[train_idx], a_large_scale[test_idx]
    a_small_train, a_small_test = a_small_scale[train_idx], a_small_scale[test_idx]

    return a_large_train, a_large_test, a_small_train, a_small_test, modes_k, a_mean, a_std, k

if __name__ == '__main__':
    torch.cuda.empty_cache()
    a_large_train, a_large_test, a_small_train, a_small_test, modes_k, a_mean, a_std, k = load_data('RB.npy')
    print(f"Select first {k:04d} POD modes.")

    dh_list = [2,4,6,8,12,16]
    mse_list = []

    for dh in dh_list:
        print(f"Training Autoencoder with latent dim dh = {dh}")
        try:
            model_large = train_model(a_large_train, a_large_test, k, dh, f'Out_large_dh{dh}.txt', f'best_model_large_dh{dh}.pt')
            model_small = train_model(a_small_train, a_small_test, k, dh, f'Out_small_dh{dh}.txt', f'best_model_small_dh{dh}.pt')

            model_large.eval()
            with torch.no_grad():
                test_tensor_large = torch.tensor(a_large_test, dtype=torch.float64).to(device)
                test_tensor_small = torch.tensor(a_small_test, dtype=torch.float64).to(device)
                pred_large = model_large(test_tensor_large).cpu().numpy()
                pred_small = model_small(test_tensor_small).cpu().numpy()

            mse = np.mean((pred_large - a_large_test) ** 2)
            mse_list.append(mse)
            print(f"dh = {dh}, Test MSE = {mse:.4e}")

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"Out of memory when training dh={dh}, skipping.")
                mse_list.append(np.nan)
                torch.cuda.empty_cache()
            else:
                raise e

    plt.figure(figsize=(8, 6))
    plt.plot(dh_list, mse_list, 'o-', linewidth=2)
    plt.xlabel('Latent Dimension (dh)')
    plt.ylabel('Test MSE')
    plt.title('Reconstruction Error vs Latent Dimension')
    plt.grid(True)
    plt.savefig('MSE_vs_dh.png', dpi=150)
    plt.show()


