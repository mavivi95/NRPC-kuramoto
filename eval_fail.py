###############################################################
#                                                             #
#  NRPC Evaluation under Structural Disconnection Scenario    #
#                                                             #
#  This script evaluates the trained controller in a scenario #
#  with total disconnection between two diagonal subgroups    #
#  of nodes (upper-right and lower-left), affecting 50% of    #
#  the network.                                               #
#                                                             #
#  The performance is compared against:                       #
#    - Full consensus feedback controller                     #
#    - Random mask consensus controller                       #
#    - Heuristic mask controller (nodes closest to the grid   #
#      center)                                                #
#                                                             #
#  Energy, synchronization, and resilience metrics are        #
#  computed for each controller and beta value. Results are   #
#  stored in a pickle file for further analysis and           #
#  visualization.                                             #
#                                                             #
###############################################################


import torch
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from kuramoto.perturbations import FreqDriftNoise, DisconnectionTotal, CancelDynamics, GaussianPulse, Disconnection
from kuramoto.simulate import simulate_trajectory_grid_disconected
from kuramoto.controllers import NRPConsensusControllerTrainableNode, RNMaskControllerNode, NRPConsensusController, MeanPhaseFeedbackController, RandomKMaskModel, FixedKMaskModel
from kuramoto.topology import build_graph, build_topology

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROWS, COLS = 10, 10
N_AGENTS = ROWS * COLS
T_MAX = 100.0
DT = 1
STEPS = int(T_MAX / DT)
K_COORD = 0.5

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. TOPOLOGY
# ==========================================
topo_grid = build_topology("grid", rows=ROWS, cols=COLS, periodic=True)



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

rho_target = 0.2 # user defined target for the fraction of controlled nodes
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



# Betas value to evaluate
betas = [0.5, 1.0, 1.5 ,2.0, 2.5]  # If you cant to prove other performance change for other beta value [0.1, 0.7, 1.0, 2.0, 5.0, 10.0]

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
    random_model = FixedKMaskModel(N_AGENTS, selected_indices).to(device)

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


    # Creation of a total disconnection perturbation between two diagonal subgroups
    # of nodes (upper-right and lower-left), affecting 50% of the network.
    source_idxs = []

    for i in range(ROWS):
        for j in range(COLS):
            idx = i * COLS + j

            in_upper_right = (i < ROWS // 2) and (j >= COLS // 2)
            in_lower_left = (i >= ROWS // 2) and (j < COLS // 2)

            if in_upper_right or in_lower_left:
                source_idxs.append(idx)

    source_idxs = np.array(source_idxs)

    for trial in range(100):

        base_seed = 1000 + trial

        perturbations_list = [
            DisconnectionTotal(t0=20, tf=50, source_idxs=source_idxs)
        ]

        torch.manual_seed(base_seed)
        np.random.seed(base_seed)

        omegas = torch.tensor(np.random.normal(1.0, 0.01, size=(N_AGENTS)), dtype=torch.float32).to(device)
        theta0 = torch.tensor(np.random.uniform(0, 2*np.pi, size=(N_AGENTS)), dtype=torch.float32).to(device)

        torch.manual_seed(base_seed)
        np.random.seed(base_seed)
        with torch.no_grad():

            t_axis = np.arange(STEPS) * DT
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid_disconected(nrp_consensus_ctrl,
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
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid_disconected(full_consensus_ctrl,
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
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid_disconected(random_consensus_ctrl,
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
            r_hist_eval, u_hist, theta_hist_eval, m_his = simulate_trajectory_grid_disconected(heuristic_consensus_ctrl,
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

with open("results/eval_data_fail.pkl", "wb") as f:
    pickle.dump(data_total, f)