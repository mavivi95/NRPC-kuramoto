###############################################################
#                                                             #
#  NRPC Training Script                                       #
#                                                             #
#  This script builds and trains the Neural Resilient Pinning #
#  Controller (NRPC). The trained controller is saved at the  #
#  end of the training process and later used in the          #
#  evaluation script.                                         #
#                                                             #
#  Training is performed under randomly generated             #
#  perturbations at each epoch. A differentiable loss is      #
#  optimized, combining resilience, energy consumption, and   #
#  mask regularization objectives.                            #
#                                                             #
###############################################################

import torch
import torch.nn as nn
import numpy as np
import random

from kuramoto.simulate import simulate_trajectory_grid
from kuramoto.topology import build_topology
from kuramoto.perturbations import FreqDriftNoise
from kuramoto.controllers import NRPConsensusControllerTrainableNode, RNMaskControllerNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROWS, COLS = 10, 10
N_AGENTS = ROWS * COLS
T_MAX = 60.0
DT = 0.1
STEPS = int(T_MAX / DT)

K_COORD = 0.5
k_random = 10
target_mask = k_random / N_AGENTS

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def sample_source_indices(n_agents, k):
    return sorted(random.sample(range(n_agents), k))


def build_random_perturbations(n_agents):
    """
    Genera una perturbación aleatoria por época.
    Puedes luego extender esto para escoger también el tipo al azar.
    """
    source_idxs = sample_source_indices(n_agents, k=8)

    perturbation = FreqDriftNoise(
        t0=38,
        tf=48,
        source_idxs=source_idxs,
        drift=1.2,
        sigma=1.5
    )

    return [perturbation]


def spatial_entropy_loss(m_tensor, eps=1e-8):
    # m_tensor: [T, N] o [T, B, N]
    p = m_tensor / (m_tensor.sum(dim=-1, keepdim=True) + eps)
    ent = -(p * torch.log(p + eps)).sum(dim=-1).mean()
    return ent


def temporal_smoothness_loss(m_tensor):
    if m_tensor.shape[0] < 2:
        return torch.tensor(0.0, device=m_tensor.device)
    return torch.mean((m_tensor[1:] - m_tensor[:-1]) ** 2)


def differentiable_resilience_metrics(r_tensor, t0, tf, dt):
    impact_idx = int(t0 / dt)
    recovery_idx = int(tf / dt)

    r_during_hit = r_tensor[impact_idx:recovery_idx]
    r_post_hit = r_tensor[recovery_idx:]

    # Protección por si una ventana queda vacía
    if r_during_hit.numel() == 0 or r_post_hit.numel() == 0:
        return torch.tensor(0.0, device=r_tensor.device)

    absorption = 0.5 * torch.min(r_during_hit) + 0.5 * torch.mean(r_during_hit)
    recovery_error = torch.mean((1.0 - r_post_hit) ** 2)

    res_score = absorption * (1.0 - recovery_error)
    return res_score


def calculate_differentiable_loss(
    r_history,
    u_history,
    m_history,
    perturbations_list,
    dt,
    target_mask
):
    r_tensor = torch.stack(r_history).squeeze()   # [T]
    u_tensor = torch.stack(u_history).squeeze()   # [T, N]
    m_tensor = torch.stack(m_history).squeeze()   # [T, N]

    res_list = []
    for per in perturbations_list:
        res = differentiable_resilience_metrics(r_tensor, per.t0, per.tf, dt)
        res_list.append(res)

    if len(res_list) == 0:
        res_score = torch.tensor(0.0, device=r_tensor.device)
    else:
        res_score = torch.stack(res_list).mean()

    control_energy = torch.mean(u_tensor ** 2)
    mask_usage = torch.mean(m_tensor)

    sparsity_penalty = (mask_usage - target_mask) ** 2
    smooth_penalty = temporal_smoothness_loss(m_tensor)
    entropy_penalty = spatial_entropy_loss(m_tensor)

    loss_total = (
        -4.0 * res_score
        + 0.01 * control_energy
        + 2.0 * sparsity_penalty
        + 0.05 * smooth_penalty
        + 0.01 * entropy_penalty
    )

    return loss_total, res_score, control_energy, mask_usage


# Topology
topo_grid = build_topology("grid", rows=ROWS, cols=COLS, periodic=True)
lambda_2 = topo_grid.lambda2

print(f'[info] Training model per node with K: {K_COORD:0.2f}')

# --- Training Loop ---
controller = RNMaskControllerNode(in_features=8, hidden_dim=64).to(device)
optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)
loss_log = []

nrp_consensus_ctrl = NRPConsensusControllerTrainableNode(
    neural_model=controller,
    beta=1.0,
    gating=True,
    alpha=10.0,
    k_select=k_random,
    soft_selection=True,
    adjacency = torch.tensor(topo_grid.A, dtype=torch.float32, device=device),
    u_clip=2.0
)

for epoch in range(500):
    torch.manual_seed(1000 + epoch)
    np.random.seed(1000 + epoch)
    random.seed(1000 + epoch)

    optimizer.zero_grad()

    omegas = torch.tensor(
        np.random.normal(1.0, 0.01, size=(N_AGENTS)),
        dtype=torch.float32,
        device=device
    )

    theta0 = torch.tensor(
        np.random.uniform(0, 2 * np.pi, size=(N_AGENTS)),
        dtype=torch.float32,
        device=device
    )

    # Random disruption for this epoch
    perturbations_list = build_random_perturbations(N_AGENTS)

    r_hist, u_hist, _, m_his = simulate_trajectory_grid(
        nrp_consensus_ctrl,
        theta0,
        omegas,
        perturbations_list,
        False,
        STEPS,
        DT,
        ROWS,
        COLS,
        N_AGENTS,
        K_COORD
    )

    loss, res_score, control_energy, mask_usage = calculate_differentiable_loss(
        r_hist,
        u_hist,
        m_his,
        perturbations_list,
        dt=DT,
        target_mask=target_mask
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
    optimizer.step()

    loss_log.append(loss.item())

    if epoch % 10 == 0:
        r_tensor = torch.stack(r_hist).squeeze()
        sync_error = torch.mean((1.0 - r_tensor) ** 2)
        r_mean = r_tensor.mean().item()
        m_tensor = torch.stack(m_his).squeeze()
        mask_var_nodes = m_tensor.var(dim=1).mean().item()
        mask_delta_time = torch.mean(torch.abs(m_tensor[1:] - m_tensor[:-1])).item()
        top_nodes = torch.topk(m_tensor.mean(dim=0), k=10).indices.detach().cpu().tolist()

        print(
            f"[metrics] Epoch {epoch:03d}: "
            f"Loss={loss.item():.5f} | "
            f"Res={res_score.item():.4f} | "
            f"Sync={r_mean:.4f} | "
            f"SyncErr={sync_error.item():.4f} | "
            f"Mask={mask_usage.item():.4f} | "
            f"Energy={control_energy.item():.6f} | "
            f"MaskVar={mask_var_nodes:.6f} | "
            f"DeltaMask={mask_delta_time:.6f} | "
            f"TopNodes={np.sort(top_nodes)} | "
            f"Src={np.sort(perturbations_list[0].source_idxs)}"
        )

print()

data_train = {
    "state_dict": controller.state_dict(),
    "seed": 42,
    "dt": DT,
    "k_coord": K_COORD,
    "omega": omegas.detach().cpu(),
    "theta_0": theta0.detach().cpu(),
    "loss": loss_log
}

out_path = "results/data_train.pt"
torch.save(data_train, out_path)

print("[info] Train model save in:")
print(out_path)
print("[done] Simulation OK")