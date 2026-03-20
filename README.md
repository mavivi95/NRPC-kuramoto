# Neural Resilient Pinning Control for Kuramoto Networks

---
## 🌐 Overview

This repository implements a Neural Resilient Pinning Controller (NRPC) for synchronization in Kuramoto oscillator networks under structural and dynamical perturbations.
The goal is to learn which nodes to control in order to achieve a better resilience–energy trade-off, compared to classical full-actuation and heuristic strategies.
---
## ✨ Key Features

🎯 Learned node selection via neural networks
⚡ Energy-efficient control (sparse actuation)
🛡️ Resilience to structural disruptions
📊 Full evaluation pipeline
🔁 Reproducible experiments
---
## 📁 Repository Structure

NRPC-kuramoto/
├── kuramoto/
├── results/
├── figures/
├── stats/
├── train.py
├── eval.py
├── eval_fail.py
├── eval_hist_mask.py
├── dynamics_without_controller.py
├── requirements.txt
└── README.md
---
## ⚙️ Installation

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m pip install -r requirements.txt
```
---
## 🚀 Usage

Train:
```bash
python train.py
```
Evaluate:
```bash
python eval.py
```
Additional evaluation scenarios:

``` 
python eval_fail.py
python eval_hist_mask.py
```

---
📊 Outputs
`.pt` trained models
`.pkl` evaluation results
`.pdf` figures
---
## 📈 Generate figures and statistics
Generates heatmaps, Pareto fronts, and statistical summaries.
```
python figures/pareto_eval.py
python figures/heatmap_ring.py
python figures/heatmap_diagonal_a.py
python figures/heatmap_diagonal_b.py
python stats/stats.py
python stats/stats_fail.py
```
---
## 🔁 Reproducibility
All results can be reproduced using the provided scripts.
---
## 📜 License
See LICENSE file.


### Note

This repository provides the **additional and reproducible material** for the paper:

**"Resilient Synchronization of Kuramoto Networks via Sparse Energy-Efficient Control"**,  

by Manuela Chacon-Chamorro, María José Charria-Macías,  
Luis Felipe Giraldo (Member, IEEE), and  
Nicanor Quijano (Senior Member, IEEE),  

submitted to the **IEEE Conference on Decision and Control (CDC), 2026**.

---
