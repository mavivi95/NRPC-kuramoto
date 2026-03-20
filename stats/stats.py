from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "stats"

file_path = RESULTS / "eval_data.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

p_values = []
betas = [0.5, 1.0, 1.5 ,2.0, 2.5, 3.0, 3.5, 4.0]
print("Wilcoxon signed-rank test results (NRPC vs Full Consensus):")
print("30% nodes perturbed:")
print("\n")
print("Results for resilience metric:")
for beta in betas:

    key_nrp = f'nrp_beta_{beta}'
    key_full = f'consensus_beta_{beta}'
    res_nrp = np.array(data['nrp'][key_nrp]['resilience'])
    res_full = np.array(data['full'][key_full]['resilience'])
    stat, p = wilcoxon(res_nrp, res_full, alternative='greater')
    p_values.append(p)
    print(f'beta={beta}: W={stat:.2f}, p-value={p:.4e}')

reject, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

print("\nCorrected p-values:")
for b, p in zip(betas, pvals_corrected):
    print(f'beta={b}: corrected p={p:.4e}')

print("#######")
print("\n")
print("Results for energy metric:")
p_values = []
for beta in betas:

    key_nrp = f'nrp_beta_{beta}'
    key_full = f'consensus_beta_{beta}'

    res_nrp = np.array(data['nrp'][key_nrp]['energy'])
    res_full = np.array(data['full'][key_full]['energy'])

    stat, p = wilcoxon(res_nrp, res_full, alternative='less')

    p_values.append(p)

    print(f'beta={beta}: W={stat:.2f}, p-value={p:.4e}')

reject, pvals_corrected, _, _ = multipletests(p_values, method='bonferroni')

print("\nCorrected p-values:")
for b, p in zip(betas, pvals_corrected):
    print(f'beta={b}: corrected p={p:.4e}')

########################################
# Graphical representation improvement #
########################################

resilience_gain_mean = []
resilience_gain_std = []

for beta in betas:

    key_nrp = f'nrp_beta_{beta}'
    key_full = f'consensus_beta_{beta}'

    res_nrp = np.array(data['nrp'][key_nrp]['resilience'])
    res_full = np.array(data['full'][key_full]['resilience'])

    gain = (res_nrp - res_full)

    resilience_gain_mean.append(np.mean(gain))
    resilience_gain_std.append(np.std(gain))

energy_reduction = []
for beta in betas:

    key_h = f'nrp_beta_{beta}'
    key_c = f'consensus_beta_{beta}'

    # Calcular la media de energía para cada método
    mean_energy_hybrid = np.mean(data['nrp'][key_h]['energy'])
    mean_energy_consensus = np.mean(data['full'][key_c]['energy'])

    # Calcular el porcentaje de reducción (ahorro)
    # ((Consenso - NRPC) / Consenso) * 100
    reduction = ((mean_energy_consensus - mean_energy_hybrid) / mean_energy_consensus) * 100
    # Guardar en la lista redondeando a 2 decimales
    energy_reduction.append(round(reduction, 2))

x = np.arange(len(betas))
width = 0.4

# resilience_gain = (rho_controller - rho_full)*100
resilience_gain_percent = [r*100 for r in resilience_gain_mean]

plt.figure(figsize=(11,7))

bars1 = plt.bar(
    x - width/2,
    energy_reduction,
    width,
    color='#F39C12',
    edgecolor='black',
    hatch='//',        # rayas
    alpha=0.85,
    label='Energy Reduction'
)

bars2 = plt.bar(
    x + width/2,
    resilience_gain_percent,
    width,
    color='#2ECC71',
    edgecolor='black',
    alpha=0.85,
    hatch='..',        # puntos
    label='Resilience Gain'
)


for bars in [bars1, bars2]:
    for bar in bars:

        height = bar.get_height()
        if height > 0:
          plt.text(
              bar.get_x() + bar.get_width()/2 +0.1,
              height + 1,
              f'{height:.1f}%',
              ha='center',
              va='bottom',
              fontsize=13,
              fontweight='bold'
          )
        else:
          plt.text(
              bar.get_x() + bar.get_width()/2,
              height - 4.5,
              f'{height:.1f}%',
              ha='center',
              va='bottom',
              fontsize=13,
              fontweight='bold'
          )

plt.xlabel(r'Parameter $\beta$', fontsize=20, fontweight='bold')
plt.ylabel('Improvement vs Full Consensus (%)', fontsize=20, fontweight='bold')
plt.xticks(x, betas, fontsize=18)
plt.yticks(fontsize=18)
plt.axhline(0, color='black', linewidth=1)
plt.ylim(-10, 105)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend(fontsize=16)
plt.text(
    0.02, 0.75,
    r'Energy reduction significant ($p<0.001$)'
    '\n'
    r'No significant resilience loss for $\beta \leq 1.5$'
    '\n'
    r'Significant resilience gains for $\beta \geq 2$',
    transform=plt.gca().transAxes,
    ha='left',
    va='top',
    fontsize=14,
    bbox=dict(boxstyle="round", facecolor="cyan", alpha=0.1)
)
plt.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES / 'energy_resilience_gain.pdf', dpi=300, bbox_inches='tight')
plt.show()