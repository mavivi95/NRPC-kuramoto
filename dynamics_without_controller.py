###############################################################
#                                                             #
#  Kuramoto Network Dynamics (Uncontrolled)                   #
#                                                             #
#  This script simulates the uncontrolled dynamics of a       #
#  Kuramoto oscillator network under a Frequency Drift Noise  #
#  perturbation affecting a subset of nodes. It records the   #
#  evolution of the synchronization index r(t) and the phase  #
#  trajectories theta(t) to analyze the impact of the         #
#  perturbation on collective oscillatory behavior.           #
#                                                             #
#  Intended use: sanity check for oscillatory dynamics.       #
#                                                             #
###############################################################

import torch
import numpy as np
import matplotlib.pyplot as plt

from kuramoto.perturbations import FreqDriftNoise
from kuramoto.simulate import simulate_trajectory_grid
from kuramoto.topology import build_topology



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROWS, COLS = 10, 10
N_AGENTS = ROWS * COLS
T_MAX = 60.0
DT = 0.1
STEPS = int(T_MAX / DT)
K_COORD = 0.7
TARGET_IDX = 12

torch.manual_seed(42)
np.random.seed(42)

omegas = torch.tensor(np.random.normal(1.0, 0.01, size=(N_AGENTS)), dtype=torch.float32).to(device)
theta0 = torch.tensor(np.random.uniform(0, 2*np.pi, size=(N_AGENTS)), dtype=torch.float32).to(device)


min_w = torch.min(omegas)
max_w = torch.max(omegas)
topo_grid = build_topology("grid", rows=ROWS, cols=COLS, periodic=True)
lambda_2 = topo_grid.lambda2
K_critical = (max_w - min_w) / lambda_2
print(f'[info] Critical K value: {K_critical}')

diagonal_idxs = [i * 11 for i in range(10)]

perturbations_list = [
    FreqDriftNoise(
        t0=30.0,
        tf=35.0,
        source_idxs=diagonal_idxs,
        drift=0.9,
        sigma=1.5
    )
]

with torch.no_grad():
    # Create a controller that returns zeros
    t_axis = np.arange(STEPS) * DT
    dummy_ctrl = lambda x: torch.zeros_like(x)
    r_hist_base, _, theta_hist_base = simulate_trajectory_grid(dummy_ctrl, theta0, omegas, perturbations_list,False, STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD)
    r_base = torch.stack(r_hist_base).cpu().numpy().flatten()

# --- PLOTS ---
plt.figure(figsize=(12, 10))

# 1. Synchronization index
plt.subplot(2, 1, 1)
plt.plot(t_axis, r_base, 'k--')
plt.axvline(30, color='r', linestyle=':', label='Perturbation')
plt.ylabel("Synchronization (r)")
plt.title("Resulting r(t) Without Controller Grid Topology")
plt.legend()
plt.grid(True)

# 2. Phases
plt.subplot(2, 1, 2)
th_np = torch.stack(theta_hist_base).cpu().numpy().squeeze()
plt.plot(t_axis, np.sin(th_np), alpha=0.3)
plt.title("Resulting Phase Dynamics Without Controller Grid Topology")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig("../results/baseline_no_controller.png", dpi=200)
plt.close()
print('[done] Save figures in results')