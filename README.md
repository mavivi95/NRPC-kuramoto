# Neural Resilient Pinning Control for Kuramoto Networks

Research code and numerical results for **sparse neural pinning control and resilient synchronization in Kuramoto oscillator networks**. The repository evaluates learned local (NRPC-L) and global–local (NRPC-GL) node selectors, comparing synchronization resilience and control effort against benchmark actuation strategies.

## Contents

- `kuramoto/`: network dynamics, perturbations, controllers, and topology utilities.
- `train.py`: training the local and global–local selectors.
- `eval_experiment1.py`, `eval_experiment1_theory.py`: resilience–energy evaluation and directional recovery analysis.
- `eval_experiment2.py`: multi-event perturbations.
- `eval_experiment3.py`: structural disruptions and actuator selection.
- `eval_experiment4_generate_topologies.py`, `eval_experiment4.py`: unseen-topology generation and zero-shot evaluation.
- `plot_experiment*.py`: manuscript figure generation from saved results.
- `results/`: saved evaluation data, topology bank, and trained checkpoints.
- `figures/`: rendered figure outputs.
- `stats/`: statistical analysis scripts.

## Environment

Run from the repository root. Python dependencies are listed in `requirements.txt` (saved as UTF-8).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The package versions originate from the research environment; check Python and platform compatibility before installing.

## Models and evaluation

Saved checkpoints:

- `results/data_train_v4_modular_local.pt`
- `results/data_train_v4_modular_global_local.pt`

To train a model anew (computationally costly):

```bash
python train.py --mode local --updates 800
python train.py --mode global_local --updates 800
```

The results and trained checkpoints bundled with this release avoid needing to retrain just to inspect the figures.

### Experiment I — resilience–energy trade-off and theory

```bash
python eval_experiment1.py --trials 100 --tag final
python eval_experiment1_theory.py --trials 100 --tag theory
python plot_experiment1_a.py
python plot_experiment1_b.py
```

### Experiment II — multiple perturbation events

```bash
python eval_experiment2.py --trials 100 --tag final
python plot_experiment2.py
```

### Experiment III — structural disturbances

```bash
python eval_experiment3.py --trials 100 --tag final
python plot_experiment3.py
```

### Experiment IV — unseen topologies

The release includes a saved topology bank at `results/experiment4/unseen_topology_bank.npz`. Use the same bank when evaluating to preserve the original unseen-topology sample.

```bash
python eval_experiment4.py \
  --bank results/experiment4/unseen_topology_bank.npz \
  --outdir results/experiment4 --tag final
python plot_experiment4_a.py
python plot_experiment4_b.py
```

To generate a **new** topology bank instead (this changes the benchmark graphs):

```bash
python eval_experiment4_generate_topologies.py --outdir results/experiment4/new_bank
```

For a quick, non-definitive check of Experiment IV, the evaluation script provides `--smoke`. This is **not** equivalent to reproducing its full reported results.

## Reproducibility notes

Evaluation scripts expose settings including random seeds, perturbation settings, control gains and trial counts through command-line arguments. Saved configurations and outputs under `results/` should be considered the source of record for the numerical results. Running evaluations with default arguments may not reproduce all the exact output filenames found in this archive. Avoid overwriting definitive data during validation: use a separate `--tag` or `--outdir` where offered.


## License

See `LICENSE`.
