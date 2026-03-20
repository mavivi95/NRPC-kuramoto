###############################################################
#                                                             #
#  Controller Mask Analysis under Structural Perturbations    #
#                                                             #
#  This script evaluates the activation patterns of controlled#
#  nodes under different structural perturbation scenarios. It#
#  focuses on recording the temporal evolution of control     #
#  masks for further analysis.                                #
#                                                             #
#  Two perturbation types are considered:                     #
#    - Diagonal disconnection (structural split of the grid)  #
#    - Concentric ring perturbation centered on the grid,     #
#      with intensity decreasing with distance from the center#
#                                                             #
#  Multiple controllers are evaluated:                        #
#    - Neural Resilient Pinning (NRP)                         #
#    - Full consensus controller                              #
#    - Random k-mask consensus controller                     #
#    - Heuristic k-mask consensus controller                  #
#                                                             #
#  For each scenario, the script records resilience, energy   #
#  consumption, and the temporal evolution of control masks.  #
#  Results are stored in a pickle file for further analysis.  #
#                                                             #
###############################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from kuramoto.perturbations import FreqDriftNoise, DisconnectionTotal, CancelDynamics, GaussianPulse, Disconnection
from kuramoto.simulate import simulate_trajectory_grid_disconected, simulate_trajectory_topology, simulate_trajectory_grid
from kuramoto.controllers import NRPConsensusControllerTrainableNode, RNMaskControllerNode, NRPConsensusController, MeanPhaseFeedbackController, RandomKMaskModel, FixedKMaskModel
from kuramoto.topology import build_graph, build_topology

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROWS, COLS = 11, 11
N_AGENTS = ROWS * COLS
T_MAX = 100.0
DT = 1
STEPS = int(T_MAX / DT)
K_COORD = 0.5
topo_grid = build_topology("grid", rows=ROWS, cols=COLS, periodic=True)
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 3. CONTROLLERS
# ==========================================

ckpt = torch.load(
    "results/data_train.pt",
    map_location=device,
    weights_only=False
)

mask = RNMaskControllerNode(in_features=8, hidden_dim=64).to(device)
mask.load_state_dict(ckpt["state_dict"])
mask.eval()

rho_target = 0.3 # user defined target for the fraction of controlled nodes
k_random = int(np.round(rho_target * N_AGENTS))
random_model = RandomKMaskModel(N_AGENTS, k_random).to(device)

# Heuristic: select the k nodes closest to the center of the grid
def center_k_indices(rows, cols, k):
    coords = []
    center = np.array([(rows - 1) / 2.0, (cols - 1) / 2.0])

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            dist = np.linalg.norm(np.array([i, j]) - center)
            coords.append((idx, dist))

    coords = sorted(coords, key=lambda x: x[1])
    return [idx for idx, _ in coords[:k]]


# ==========================================
# 5. METRICS FOR EVALUATION
# ==========================================


def differentiable_resilience_metrics(r_tensor, t0, tf, dt):
    impact_idx = int(t0 / dt)
    recovery_idx = int(tf / dt)

    r_during_hit = r_tensor[impact_idx:recovery_idx]
    r_post_hit = r_tensor[recovery_idx:]

    absorption = 0.5 * torch.min(r_during_hit) + 0.5 * torch.mean(r_during_hit)
    recovery_error = torch.mean((1.0 - r_post_hit)**2)

    res_score = absorption * (1.0 - recovery_error)
    return res_score




# Betas value to evaluate 
betas = [2.5] # If you cant to prove other performance change for other beta value [0.1, 0.7, 1.0, 2.0, 5.0, 10.0]

data_NRPC = {} # Neural Resilience Pinning Controller
data_full_consensus = {} # Full state feedback consensus controller
data_dummy = {} # Dummy controller with zero input
data_krandom_consensus = {} # Random k-mask consensus controller
data_heuristic_consensus = {} # Heuristic k-mask consensus controller

# =================================================
# Evaluation loop for each controller and beta
# =================================================

for beta in betas:

    full_consensus_ctrl = MeanPhaseFeedbackController(beta=beta, gating=True, alpha=10.0)
    selected_indices = center_k_indices(ROWS, COLS, k_random)
    heuristic_model = FixedKMaskModel(N_AGENTS, selected_indices).to(device)

    selected_indices = np.random.choice(np.arange(100), size=k_random, replace=False)
    random_model = RandomKMaskModel(N_AGENTS, k_random).to(device)

    random_consensus_ctrl = NRPConsensusController(
        neural_model=random_model,
        beta=beta,
        gating=True,
        use_sigmoid=True,
        k_select=k_random
    )
    
    
    heuristic_consensus_ctrl = NRPConsensusController(
        neural_model=heuristic_model,
        beta=beta,
        gating=True,
        alpha=10.0,
    )

    nrp_consensus_ctrl = NRPConsensusControllerTrainableNode(
        neural_model=mask,
        beta=beta,
        gating=True,
        alpha=10.0,
        k_select=k_random,
        soft_selection=False,
        adjacency = torch.tensor(topo_grid.A, dtype=torch.float32, device=device),
        u_clip=2.0
    )
    
    E_mean_full_consensus = []
    r_mean_full_consensus = []
    m_mean_full_consensus = []
    m_std_full_consensus = []
    resilience_full_consensus = []

    E_mean_nrp_consensus = []
    r_mean_nrp_consensus = []
    m_mean_nrp_consensus = []
    m_std_nrp_consensus = []
    resilience_nrp_consensus = []

    E_mean_krandom_consensus = []
    r_mean_krandom_consensus = []
    m_mean_krandom_consensus = []
    m_std_krandom_consensus = []
    resilience_krandom_consensus = []

    E_mean_heuristic_consensus = []
    r_mean_heuristic_consensus = []
    m_mean_heuristic_consensus = []
    m_std_heuristic_consensus = []
    resilience_heuristic_consensus = []

    m_hist_krandom_consensus = []
    m_hist_heuristic_consensus = []
    m_hist_full_consensus = []
    m_hist_nrp_consensus = []
    print(f'[info] Evaluating for beta: {beta:0.2f}')
    
   
    # =========================================================
    # Diagonal disconnection patterns
    # =========================================================

    # Case 1: Disconnection of upper-right and lower-left regions
    # (diagonal split: top-right vs bottom-left)
    # source_idxs = []

    # for i in range(ROWS):
    #     for j in range(COLS):
    #         idx = i * COLS + j

    #         in_upper_right = (i < ROWS // 2) and (j >= COLS // 2)
    #         in_lower_left = (i >= ROWS // 2) and (j < COLS // 2)

    #         if in_upper_right or in_lower_left:
    #             source_idxs.append(idx)


    # Case 2: Disconnection of upper-left and lower-right regions
    # (diagonal split: top-left vs bottom-right)
    # source_idxs = []

    # for i in range(ROWS):
    #     for j in range(COLS):
    #         idx = i * COLS + j

    #         in_upper_left = (i < ROWS // 2) and (j < COLS // 2)
    #         in_lower_right = (i >= ROWS // 2) and (j >= COLS // 2)

    #         if in_upper_left or in_lower_right:
    #             source_idxs.append(idx)

    # Convert selected indices to array
    # source_idxs = np.array(source_idxs)


    # =========================================================
    # Ring-shaped perturbation (Manhattan distance from center)
    # =========================================================

    center_i = ROWS // 2
    center_j = COLS // 2

    ring0 = []   # Strong perturbation (center region)
    ring1 = []   # Medium perturbation (intermediate ring)
    ring2 = []   # Weak perturbation (outer ring)

    for i in range(ROWS):
        for j in range(COLS):

            idx = i * COLS + j

            # Manhattan distance from grid center
            dist = abs(i - center_i) + abs(j - center_j)

            if dist <= 1:
                ring0.append(idx)

            elif dist <= 3:
                ring1.append(idx)

            elif dist <= 5:
                ring2.append(idx)
    


    for trial in range(100):

        base_seed = 1000 + trial

        # =========================================================
        # Structural perturbations (select one scenario)
        # =========================================================

        # Square / diagonal disconnection perturbation
        # (activate this block to simulate total disconnection between
        # selected node subgroups, e.g., upper-right vs lower-left or
        # upper-left vs lower-right depending on source_idxs definition)
        # perturbations_list = [
        #     DisconnectionTotal(t0=20, tf=50, source_idxs=source_idxs)
        # ]


        # Circular / ring-based perturbation
        # (activate this scenario to apply spatially structured perturbations
        # based on distance from the center using predefined rings)

        perturbations_list = [

            FreqDriftNoise(
                t0=20,
                tf=50,
                source_idxs=ring0,
                drift=2.0,
                sigma=2.0
            ),

            FreqDriftNoise(
                t0=20,
                tf=50,
                source_idxs=ring1,
                drift=1.2,
                sigma=1.2
            ),

            FreqDriftNoise(
                t0=20,
                tf=50,
                source_idxs=ring2,
                drift=0.6,
                sigma=0.6
            ),

        ]

        torch.manual_seed(base_seed)
        np.random.seed(base_seed)

        omegas = torch.tensor(np.random.normal(1.0, 0.01, size=(N_AGENTS)), dtype=torch.float32).to(device)
        theta0 = torch.tensor(np.random.uniform(0, 2*np.pi, size=(N_AGENTS)), dtype=torch.float32).to(device)

        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        with torch.no_grad():

            t_axis = np.arange(STEPS) * DT
            # Use `simulate_trajectory_grid_disconnected` to simulate the full disconnection
            # of the affected nodes. Compare it with `simulate_trajectory_grid` to assess
            # the impact of the perturbation in the uncontrolled case.
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid(nrp_consensus_ctrl,
                                                                theta0,
                                                                omegas,
                                                                perturbations_list,
                                                                False,
                                                                STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD)
            r_eval = torch.stack(r_hist_eval).cpu().numpy().flatten()
            u_hist_np = torch.stack(u_hist).cpu().numpy().flatten() 
            r_tensor = torch.stack(r_hist_eval).squeeze()
            resilience = np.mean([differentiable_resilience_metrics(r_tensor, per.t0, per.tf, DT) for per in perturbations_list])

            E_mean_nrp_consensus.append(np.mean(u_hist_np**2))
            r_mean_nrp_consensus.append(np.mean(r_eval))
            m_mean_nrp_consensus.append(np.mean(m_his))
            m_hist_nrp_consensus.append(m_his)
            m_std_nrp_consensus.append(np.std(m_his))
            resilience_nrp_consensus.append(resilience)

        torch.manual_seed(base_seed)
        np.random.seed(base_seed)    
        with torch.no_grad():

            t_axis = np.arange(STEPS) * DT
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid(full_consensus_ctrl,
                                                                theta0,
                                                                omegas,
                                                                perturbations_list,
                                                                False,
                                                                STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD)
            r_eval = torch.stack(r_hist_eval).cpu().numpy().flatten()
            u_hist_np = torch.stack(u_hist).cpu().numpy().flatten()
            r_tensor = torch.stack(r_hist_eval).squeeze()
            resilience = np.mean([differentiable_resilience_metrics(r_tensor, per.t0, per.tf, DT) for per in perturbations_list])

            E_mean_full_consensus.append(np.mean(u_hist_np**2))
            r_mean_full_consensus.append(np.mean(r_eval))
            m_mean_full_consensus.append(np.mean(m_his))
            m_hist_full_consensus.append(m_his)
            m_std_full_consensus.append(np.std(m_his))
            resilience_full_consensus.append(resilience)
        

        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        with torch.no_grad():

            t_axis = np.arange(STEPS) * DT
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid(random_consensus_ctrl,
                                                                theta0,
                                                                omegas,
                                                                perturbations_list,
                                                                False,
                                                                STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD)
            r_eval = torch.stack(r_hist_eval).cpu().numpy().flatten()
            u_hist_np = torch.stack(u_hist).cpu().numpy().flatten()
            r_tensor = torch.stack(r_hist_eval).squeeze()
            resilience = np.mean([differentiable_resilience_metrics(r_tensor, per.t0, per.tf, DT) for per in perturbations_list])

            E_mean_krandom_consensus.append(np.mean(u_hist_np**2))
            r_mean_krandom_consensus.append(np.mean(r_eval))
            m_mean_krandom_consensus.append(np.mean(m_his))
            m_std_krandom_consensus.append(np.std(m_his))
            m_hist_krandom_consensus.append(m_his)
            resilience_krandom_consensus.append(resilience)
        
        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        with torch.no_grad():

            t_axis = np.arange(STEPS) * DT
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid(heuristic_consensus_ctrl,
                                                                theta0,
                                                                omegas,
                                                                perturbations_list,
                                                                False,
                                                                STEPS, DT, ROWS, COLS, N_AGENTS, K_COORD)
            r_eval = torch.stack(r_hist_eval).cpu().numpy().flatten()
            u_hist_np = torch.stack(u_hist).cpu().numpy().flatten()
            r_tensor = torch.stack(r_hist_eval).squeeze()
            resilience = np.mean([differentiable_resilience_metrics(r_tensor, per.t0, per.tf, DT) for per in perturbations_list])

            E_mean_heuristic_consensus.append(np.mean(u_hist_np**2))
            r_mean_heuristic_consensus.append(np.mean(r_eval))
            m_mean_heuristic_consensus.append(np.mean(m_his))
            m_std_heuristic_consensus.append(np.std(m_his))
            resilience_heuristic_consensus.append(resilience)
        
    data_NRPC[f'nrp_beta_{beta}'] = {"energy": E_mean_nrp_consensus, "r_mean": r_mean_nrp_consensus, "m_mean": m_mean_nrp_consensus, "m_std": m_std_nrp_consensus, "resilience": resilience_nrp_consensus, 'm_hist': m_hist_nrp_consensus}
    data_full_consensus[f'consensus_beta_{beta}'] = {"energy": E_mean_full_consensus, "r_mean": r_mean_full_consensus, "m_mean": m_mean_full_consensus, "m_std": m_std_full_consensus, "resilience": resilience_full_consensus}
    data_heuristic_consensus[f'heuristic_beta_{beta}'] = {"energy": E_mean_heuristic_consensus, "r_mean": r_mean_heuristic_consensus, "m_mean": m_mean_heuristic_consensus, "m_std": m_std_heuristic_consensus, "resilience": resilience_heuristic_consensus} 
    data_krandom_consensus[f'krandom_beta_{beta}'] = {"energy": E_mean_krandom_consensus, "r_mean": r_mean_krandom_consensus, "m_mean": m_mean_krandom_consensus, "m_std": m_std_krandom_consensus, "resilience": resilience_krandom_consensus, 'm_hist': m_hist_krandom_consensus} 
data_total = {"nrp": data_NRPC, "full": data_full_consensus, "heuristic": data_heuristic_consensus, "krandom": data_krandom_consensus}    
with open("results/eval_data_hist_mask.pkl", "wb") as f:
    pickle.dump(data_total, f)

# Note save like eval_data_hist_mask_a for diagonal upper right and lower left disconnection
# Note save like eval_data_hist_mask_b for diagonal upper left and lower right disconnection
# Note save like eval_data_hist_mask_c for ring-based perturbation