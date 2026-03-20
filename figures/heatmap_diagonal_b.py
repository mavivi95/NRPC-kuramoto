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

file_path = RESULTS / "eval_data_hist_mask_b.pkl"
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


ROWS, COLS = 10, 10
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

source_idxs = []

for i in range(ROWS):
    for j in range(COLS):
        idx = i * COLS + j

        in_upper_right = (i < ROWS // 2) and (j >= COLS // 2)
        in_lower_left = (i >= ROWS // 2) and (j < COLS // 2)

        if in_upper_right or in_lower_left:
            source_idxs.append(idx)


source_idxs = np.array(source_idxs)

grid = np.zeros((ROWS, COLS))

for idx in source_idxs:
    i = idx // COLS
    j = idx % COLS
    grid[i, j] = 1

# -----------------------------
# Overlay X on perturbed/disconnected nodes
# grid[i, j] == False means perturbed/disconnected
# -----------------------------
for i in range(ROWS):
    for j in range(COLS):
        if not grid[i, j]:
            ax.text(
                j + 0.5, i + 0.5, "×",
                ha="center", va="center",
                color="black",
                fontsize=16,
                fontweight="bold"
            )

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
plt.savefig(FIGURES / "heatmap_diagional_b.pdf", dpi=300, bbox_inches="tight")
plt.show()