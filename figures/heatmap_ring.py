import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

file_path = RESULTS / "eval_data_hist_mask_c.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)


###############################################################
# --- Extract data for heatmap case circular ---              #
###############################################################

# -----------------------------
# Data
# -----------------------------

betas = ['2.5']
for beta in betas:
    key_h = f'nrp_beta_{beta}'
    m_hist = np.array(data['nrp'][key_h]['m_hist'])
    m_avg_nrp = m_hist.mean(axis=(0,1))   


ROWS, COLS = 11, 11
m_grid = m_avg_nrp.reshape(ROWS, COLS)

# -----------------------------
# Style
# -----------------------------
sns.set_theme(style="white")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

fig, ax = plt.subplots(figsize=(6.8, 6.2), dpi=300)

# Optional: tighter color range for better contrast
vmin = np.min(m_grid)
vmax = np.max(m_grid)

hm = sns.heatmap(
    m_grid,
    ax=ax,
    cmap="viridis",
    square=True,
    linewidths=0.6,
    linecolor="white",
    vmin=vmin,
    vmax=vmax,
    cbar=True,
    cbar_kws={
        "label": "Activation frequency",
        "shrink": 0.9,
        "pad": 0.02
    }
)

# -----------------------------
# Overlay X on perturbed/disconnected nodes
# grid[i, j] == False means perturbed/disconnected
# -----------------------------

center_i = ROWS // 2
center_j = COLS // 2

ring0 = []   # centro fuerte
ring1 = []   # medio
ring2 = []   # débil

for i in range(ROWS):
    for j in range(COLS):

        idx = i * COLS + j

        dist = abs(i - center_i) + abs(j - center_j)   # distancia Manhattan

        if dist <= 1:
            ring0.append(idx)

        elif dist <= 3:
            ring1.append(idx)

        elif dist <= 5:
            ring2.append(idx)

for idx in ring0:
    i = idx // COLS
    j = idx % COLS
    ax.text(j+0.3, i+0.6, "×", color="black", fontsize=20, fontweight="bold")

for idx in ring1:
    i = idx // COLS
    j = idx % COLS
    ax.text(j+0.3, i+0.6, "×", color="black", fontsize=16)

for idx in ring2:
    i = idx // COLS
    j = idx % COLS
    ax.text(j+0.3, i+0.6, "×", color="black", fontsize=12)

# -----------------------------
# Labels and title
# -----------------------------
#ax.set_title("Learned pinning activation frequency", pad=12)
ax.set_xlabel("Column index")
ax.set_ylabel("Row index")

# Cleaner ticks centered on cells
ax.set_xticks(np.arange(COLS) + 0.5)
ax.set_yticks(np.arange(ROWS) + 0.5)
ax.set_xticklabels(np.arange(COLS))
ax.set_yticklabels(np.arange(ROWS), rotation=0)

# -----------------------------
# Colorbar formatting
# -----------------------------
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Activation frequency", fontsize=13)

# -----------------------------
# Legend for perturbed nodes
# -----------------------------
legend_elements = [
    Line2D(
        [0],[0],
        marker=r'$\times$',
        color='black',
        label='Perturbed nodes',
        markersize=12,
        linestyle='None'
    )
]

leg = ax.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.8,0.98),
    frameon=True,
    fontsize=11
)

plt.tight_layout()
plt.savefig(FIGURES / "heatmap_circular.pdf", dpi=300, bbox_inches="tight")
plt.show()