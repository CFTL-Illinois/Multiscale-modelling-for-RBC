import os
import sys
import math
import numpy as np
import pickle
import matplotlib as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from numpy import genfromtxt
import torch.nn.functional as F

import argparse
import time

from torchdiffeq import odeint, odeint_adjoint

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--data_size', type=int, default=100)  
parser.add_argument('--step',type=float,default=1) 
parser.add_argument('--batch_time', type=int, default=100)  # 1 for fast scale model
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='adams')    
parser.add_argument('--niters', type=int, default=50000)       
parser.add_argument('--test_freq', type=int, default=20)    
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)  
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--n_scales', type=int, default=3)  
args = parser.parse_args()

args.batch_time += 1 

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

use_parallel = torch.cuda.device_count() > 1
if use_parallel:
    print(f"Using {torch.cuda.device_count()} GPUs.")
else:
    print("Using single GPU or CPU.")


class Autoencoder(nn.Module):
    def __init__(self, trunc, k, dropout_rate=0.2):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encode = nn.Sequential(
            nn.Linear(k, 3000),
            nn.GELU(),
            nn.Linear(3000, 500),
            nn.GELU(),
            nn.Linear(500, 250),
            nn.GELU(),
            nn.Linear(250, trunc)
        )

        # Decoder
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

class ODEFunc(nn.Module):
    def __init__(self, trunc, a):
        super(ODEFunc, self).__init__()
        self.trunc = trunc
        self.net = nn.Sequential(
            nn.Linear(trunc, 600),
            nn.GELU(),
            nn.Linear(600,600),
            nn.GELU(),
            nn.Linear(600,600),
            nn.GELU(),
            nn.Linear(600,600),
            nn.GELU(),
            nn.Linear(600,600),
            nn.GELU(),
            nn.Linear(600,600),
            nn.GELU(),
            nn.Linear(600,600),
            nn.GELU(),
            nn.Linear(600, trunc),
        )

        self.time_net = nn.Sequential(
            nn.Linear(1, 20),
            nn.GELU(),
            nn.Linear(20,20),
            nn.GELU(),
            nn.Linear(20, 2),
            nn.Tanh()  
        )

        self.lin = nn.Sequential(nn.Linear(trunc, trunc, bias=False))

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.time_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.Parameter(torch.from_numpy(Linear(trunc, a)).float())
                m.weight.requires_grad = False

    def forward(self, t, y):
        t_min = 0
        t_max = args.batch_time*0.1
        t_norm = (t - t_min) / (t_max - t_min)
        time_input = torch.tensor([[t_norm]], dtype=y.dtype, device=y.device)
        t_emb = self.time_net(time_input) 
        amp_factor = 1.0 + 0.1 * t_emb[:, 0]  
        freq_factor = torch.exp(0.1 * t_emb[:, 1]) 
        return amp_factor * freq_factor * (self.lin(y) + self.net(y))

# Functions
def get_batch(t, true_y, batch_time):
    rand = np.random.choice(np.arange(np.floor(args.data_size / args.step) - batch_time, dtype=np.int64), args.batch_size, replace=False)
    s = torch.from_numpy(rand).to(device)  # Move to device
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    #batch_t = (t[s:s + args.batch_time]-t[s]).to(device)  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0).to(device)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def output(text, filename='Out_2.txt'):
    with open(filename, 'a+') as newfile:
        newfile.write(text + '\n')

def split_scales(h, sigma=60):
    h_np = h.detach().cpu().numpy()  
    h_large = gaussian_filter1d(h_np, sigma=sigma, axis=0)  
    h_small = h_np - h_large  

    return h_large, h_small

def compute_energy_loss(h_pred, h_true, autoencoder, std, mean):
    with torch.no_grad():
        std = torch.tensor(std, dtype=h_pred.dtype, device=h_pred.device)
        mean = torch.tensor(mean, dtype=h_pred.dtype, device=h_pred.device)

        h_pred_denorm = h_pred * std + mean
        h_true_denorm = h_true * std + mean

        a_pred = autoencoder.decode(h_pred_denorm)
        a_true = autoencoder.decode(h_true_denorm)

        E_pred = torch.sum(a_pred ** 2, dim=-1)
        E_true = torch.sum(a_true ** 2, dim=-1)

        return F.mse_loss(E_pred, E_true)


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("test_predictions", exist_ok=True)


if __name__ == '__main__':
    modes = np.load('POD_Results/modes.npy')
    eigenvalues = np.load('POD_Results/eigenvalues.npy')

    cumulative_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k = np.argmax(cumulative_energy >= 0.9995)
    print(f"Selected k = {k} based on 99.9% energy retention.")
    modes_k = modes[:, :k]

    a_large_scale = np.load('a_large_scale.npy')
    a_small_scale = np.load('a_small_scale.npy')

    [M,N] = a_large_scale.shape

    frac = .8
    train_size = round(M * frac)
    a_large_train, a_large_test = a_large_scale[:train_size], a_large_scale[train_size:]
    a_small_train, a_small_test = a_small_scale[:train_size], a_small_scale[train_size:]

    dh_large = 6
    dh_small = 14
    auto_large = Autoencoder(dh_large, a_large_scale.shape[1]).double().to(device)
    auto_large.load_state_dict(torch.load('best_model_large_dh6.pt', map_location=device))

    auto_small = Autoencoder_small(dh_small, a_small_scale.shape[1]).double().to(device)
    auto_small.load_state_dict(torch.load('best_model_small_dh14.pt', map_location=device))

    a_large_tensor = torch.tensor(a_large_scale, dtype=torch.float64).to(device)
    a_small_tensor = torch.tensor(a_small_scale, dtype=torch.float64).to(device)

    with torch.no_grad():
        h_large = auto_large.encode(a_large_tensor).cpu().numpy()
        h_small = auto_small.encode(a_small_tensor).cpu().numpy()

    h_large_mean, h_large_std = np.mean(h_large, axis=0), np.max(np.std(h_large, axis=0))
    h_small_mean, h_small_std = np.mean(h_small, axis=0), np.max(np.std(h_small, axis=0))

    h_large_norm = (h_large - h_large_mean) / h_large_std
    h_small_norm = (h_small - h_small_mean) / h_small_std

    h_large_train = h_large_norm[:round(M * frac), :]
    h_large_test = h_large_norm[round(M * frac):M, :]
    h_small_train = h_small_norm[:round(M * frac), :]
    h_small_test = h_small_norm[round(M * frac):M, :]

    true_y_large = torch.tensor(h_large_norm[:train_size, np.newaxis, :], dtype=torch.float64).to(device)
    h_large_test_tensor = torch.tensor(h_large_norm[train_size:, np.newaxis, :], dtype=torch.float64).to(device)

    true_y_small = torch.tensor(h_small_norm[:train_size, np.newaxis, :], dtype=torch.float64).to(device)
    h_small_test_tensor = torch.tensor(h_small_norm[train_size:, np.newaxis, :], dtype=torch.float64).to(device)

    t = np.arange(round(M * frac)) * 0.1
    args.data_size = round(M * frac)
    t = torch.tensor(t, dtype=torch.float64).to(device)

    A = .1
    func_large = ODEFunc(dh_large, A).double().to(device)  # Move ODE function to GPU with multi-scale
    func_small = ODEFunc(dh_small, A).double().to(device)  # Move ODE function to GPU with multi-scale
    optimizer_large = optim.Adam(func_large.parameters(), lr=1e-4)
    scheduler_large = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_large, T_max=args.niters)
    optimizer_small = optim.Adam(func_small.parameters(), lr=1e-4)
    scheduler_small = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_small, T_max=args.niters)

    checkpoint_itr = 16000 
    resume_training = False  

    if resume_training:
        func_small = torch.load(f'checkpoints/model_small_NODE_full_{checkpoint_itr}.pt').to(device)
        func_large = torch.load(f'checkpoints/model_large_NODE_full_{checkpoint_itr}.pt').to(device)
    else:
        func_large = ODEFunc(dh_large, A).double().to(device)
        func_small = ODEFunc(dh_small, A).double().to(device)

    optimizer = optim.Adam(
        list(func_large.parameters()) + list(func_small.parameters()), 
        lr=2.5e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.niters)
    end = time.time()

    err_large = []
    err_small = []
    err_total = []
    ii = 0

    batch_time_large = 500
    batch_time_small = 50

    start_itr = checkpoint_itr + 1 if resume_training else 1
    total_additional_iters = 40000

    for itr in range(start_itr, start_itr + total_additional_iters):

        optimizer.zero_grad()

        batch_y0_large, batch_t_large, batch_y_large = get_batch(t, true_y_large, batch_time_large)
        batch_y0_small, batch_t_small, batch_y_small = get_batch(t, true_y_small, batch_time_small)

        pred_y_large = odeint(func_large, batch_y0_large, batch_t_large, rtol=1e-6, atol=1e-8, method='dopri5')
        pred_y_small = odeint(func_small, batch_y0_small, batch_t_small, rtol=1e-6, atol=1e-8, method='dopri5')

        loss_small = torch.nn.functional.mse_loss(pred_y_small, batch_y_small)
        loss_large = torch.nn.functional.mse_loss(pred_y_large, batch_y_large)
        loss_energy = compute_energy_loss(pred_y_large, batch_y_large, auto_large, h_large_std, h_large_mean) + \
        compute_energy_loss(pred_y_small, batch_y_small, auto_small, h_small_std, h_small_mean)
        loss_total = loss_large + loss_small

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if itr % 20 == 0:
                        
            err_large.append(loss_large.item())
            err_small.append(loss_small.item())
            err_total.append(loss_total.item())

            with torch.no_grad():
                output(f'Iter {itr:04d} | Total Loss {loss_total.item():.8f} | Large Loss {loss_large.item():.8f} | Small Loss {loss_small.item():.8f} | Energy Loss {loss_energy.item():.8f} | Time {time.time() - end:.6f}')

                plt.figure(figsize=(12, 8))

                y_data = np.asarray(err_large)
                y_data_2 = np.asarray(err_small)
                y_data_3 = np.asarray(err_total)
                plt.semilogy(y_data, label='Large Loss')
                plt.semilogy(y_data_2, label='Small Loss')
                plt.semilogy(y_data_3, label='Total Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title(f'Training and Testing Loss (Iteration {itr})')
                plt.grid(True)
                plt.legend()
                plt.savefig('NODE_multiscale_analysis.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                if itr % 100 == 0:
                    
                    test_subset = min(10000, h_small_test.shape[0])
                    full_test_pred_small = odeint(func_small, h_small_test_tensor[0], torch.arange(test_subset) * 0.1, rtol=1e-6, atol=1e-8)
                    full_test_pred_large = odeint(func_large, h_large_test_tensor[0], torch.arange(test_subset) * 0.1, rtol=1e-6, atol=1e-8)
                    full_test_pred_small = full_test_pred_small.detach().cpu().numpy()
                    full_test_pred_large = full_test_pred_large.detach().cpu().numpy()

                    with torch.no_grad():

                        h_large_std_tensor = torch.tensor(h_large_std, dtype=torch.float64, device=device)
                        h_large_mean_tensor = torch.tensor(h_large_mean, dtype=torch.float64, device=device)

                        full_test_pred_large_tensor = torch.tensor(full_test_pred_large.squeeze(), dtype=torch.float64, device=device)
                        pred_a_large = auto_large.decode(full_test_pred_large_tensor * h_large_std_tensor + h_large_mean_tensor).cpu().numpy()
                        
                        h_small_std_tensor = torch.tensor(h_small_std, dtype=torch.float64, device=device)
                        h_small_mean_tensor = torch.tensor(h_small_mean, dtype=torch.float64, device=device)

                        full_test_pred_small_tensor = torch.tensor(full_test_pred_small.squeeze(), dtype=torch.float64, device=device)
                        pred_a_small = auto_small.decode(full_test_pred_small_tensor * h_small_std_tensor + h_small_mean_tensor).cpu().numpy()
                        
                    pred_a_total = pred_a_large + pred_a_small

                    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
                    for i in range(3):
                        axes[i, 0].plot(pred_a_total[:, i], 'r-', label='Pred Total')
                        axes[i, 0].plot((a_large_test + a_small_test)[:test_subset, i], 'b--', label='True Total')
                        axes[i, 0].set_title(f'Total a[{i}]'); axes[i, 0].legend(); axes[i, 0].grid(True)

                        axes[i, 1].plot(pred_a_large[:, i], 'r-', label='Pred Large')
                        axes[i, 1].plot(a_large_test[:test_subset, i], 'b--', label='True Large')
                        axes[i, 1].set_title(f'Large a[{i}]'); axes[i, 1].legend(); axes[i, 1].grid(True)

                        axes[i, 2].plot(pred_a_small[:, i], 'r-', label='Pred Small')
                        axes[i, 2].plot(a_small_test[:test_subset, i], 'b--', label='True Small')
                        axes[i, 2].set_title(f'Small a[{i}]'); axes[i, 2].legend(); axes[i, 2].grid(True)

                    plt.tight_layout()
                    plt.savefig(f'test_predictions/test_a_predictions_iter_{itr}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                            
                ii += 1
            
            torch.save(func_small.state_dict(), f'checkpoints/model_small_NODE_checkpoint_{itr}.pt')
            torch.save(func_large.state_dict(), f'checkpoints/model_large_NODE_checkpoint_{itr}.pt')

            if itr % 500 == 0:
                torch.save(func_small, f'checkpoints/model_small_NODE_full_{itr}.pt')
                torch.save(func_large, f'checkpoints/model_large_NODE_full_{itr}.pt')

        end = time.time()
