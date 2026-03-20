import matplotlib.pyplot as plt
import numpy as np
import pickle

from pathlib import Path

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

file_path = RESULTS / "eval_data_fail.pkl"
with open(file_path, 'rb') as f:
    data_total = pickle.load(f)


betas = [0.5, 1.0, 1.5 ,2.0, 2.5] 
betas_string =  [str(beta) for beta in betas]  

# -------------------------
# NRPC
# -------------------------

hybrid_energy_mean = [np.mean(data_total['nrp'][f'nrp_beta_{beta}']['energy']) for beta in betas_string]
hybrid_res_mean = [np.mean(data_total['nrp'][f'nrp_beta_{beta}']['resilience']) for beta in betas_string]
hybrid_res_std = [np.std(data_total['nrp'][f'nrp_beta_{beta}']['resilience']) for beta in betas_string]

# -------------------------
# Full consensus
# -------------------------
consensus_energy_mean = [np.mean(data_total['full'][f'consensus_beta_{beta}']['energy']) for beta in betas_string]
consensus_res_mean = [np.mean(data_total['full'][f'consensus_beta_{beta}']['resilience']) for beta in betas_string]
consensus_res_std = [np.std(data_total['full'][f'consensus_beta_{beta}']['resilience']) for beta in betas_string]

# -------------------------
# Random
# -------------------------
random_energy_mean = [np.mean(data_total['krandom'][f'krandom_beta_{beta}']['energy']) for beta in betas_string]
random_res_mean = [np.mean(data_total['krandom'][f'krandom_beta_{beta}']['resilience']) for beta in betas_string]
random_res_std = [np.std(data_total['krandom'][f'krandom_beta_{beta}']['resilience']) for beta in betas_string]


# -------------------------
# Center
# -------------------------
heuristic_energy_mean = [np.mean(data_total['heuristic'][f'heuristic_beta_{beta}']['energy']) for beta in betas_string]
heuristic_res_mean = [np.mean(data_total['heuristic'][f'heuristic_beta_{beta}']['resilience']) for beta in betas_string]
heuristic_res_std = [np.std(data_total['heuristic'][f'heuristic_beta_{beta}']['resilience']) for beta in betas_string]

plt.figure(figsize=(10, 6))

# Plot Pareto front
plt.plot(hybrid_energy_mean, hybrid_res_mean, 'o--',
         color='#2ecc71', markersize=10, linewidth=2.5,
         label='Neural Resilience Pinning Consensus', zorder=4,
         markeredgecolor="black",
    markeredgewidth=0.5)

plt.plot(random_energy_mean, random_res_mean, '*--',
         color='magenta', markersize=10, linewidth=2.5,
         label=r'Random-$k$ Pinning Consensus', zorder=2,
         markeredgecolor="black",
    markeredgewidth=0.5)

plt.plot(heuristic_energy_mean, heuristic_res_mean, 'd--',
         color='blue', markersize=10, linewidth=2.5,
         label=r'Center-$k$ Pinning Consensus', zorder=3,
         markeredgecolor="black",
    markeredgewidth=0.5)

plt.plot(consensus_energy_mean, consensus_res_mean, 's--',
         color='#e74c3c', markersize=10, linewidth=2.5,
         label='Full Consensus', zorder=1,
         markeredgecolor="black",
    markeredgewidth=0.5)

plt.fill_between(
    hybrid_energy_mean,
    np.array(hybrid_res_mean) - np.array(hybrid_res_std),
    np.array(hybrid_res_mean) + np.array(hybrid_res_std),
    color='#2ecc71',
    alpha=0.1
)

plt.fill_between(
    consensus_energy_mean,
    np.array(consensus_res_mean) - np.array(consensus_res_std),
    np.array(consensus_res_mean) + np.array(consensus_res_std),
    color='#e74c3c',
    alpha=0.1
)

plt.fill_between(
    random_energy_mean,
    np.array(random_res_mean) - np.array(random_res_std),
    np.array(random_res_mean) + np.array(random_res_std),
    color='magenta',
    alpha=0.1
)

plt.fill_between(
    heuristic_energy_mean,
    np.array(heuristic_res_mean) - np.array(heuristic_res_std),
    np.array(heuristic_res_mean) + np.array(heuristic_res_std),
    color='blue',
    alpha=0.1
)

plt.xlim(0, 10)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xscale('symlog', linthresh=0.3)
plt.xlabel('Total Control Energy $\|u\|_2$', fontsize=20, fontweight='bold')
plt.ylabel(r'Resilience $\rho(\tau)$', fontsize=20, fontweight='bold')
plt.legend(fontsize=16, loc='best')
plt.text(
    0.35, 0.89,
    "Pinning controllers: 20% active nodes\nStructural disruption: 50% nodes disconnected",
    transform=plt.gca().transAxes,
    fontsize=16,
    bbox=dict(boxstyle="round", facecolor="cyan", alpha=0.1)
)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.ylim(0.0, 1.1)
plt.savefig(FIGURES / "pareto_eval_fail.pdf", dpi=300, bbox_inches='tight')
plt.show()