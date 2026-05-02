# QM9 HOMO-LUMO Gap Prediction
### Geometry Matters: 2D MPNNs vs. E(3)-Equivariant Models
**CMU 24-788 Introduction to Deep Learning - Mini-Project**

---

## Results summary

| Model | Test MAE (meV) | Test RMSE (meV) | R² |
|-------|:-:|:-:|:-:|
| MPNN Baseline (2D only) | 139.6 | 213.1 | 0.9728 |
| **EGNN (3D equivariant)** | **75.7** | **116.0** | **0.9919** |

EGNN achieves **45.8% lower MAE** than the MPNN with *fewer* parameters (482k vs 871k).
A learning-curve analysis reveals a crossover: EGNN is initially worse below ~10k training
molecules but pulls decisively ahead above 25k, reaching 40.5% improvement at full scale.

---

## Repository structure

```
QM9-HOMO-LUMO-Gap-Prediction/
├── models.py                     # MPNNBaseline + EGNNModel class definitions
├── train.py                      # Training script (produces checkpoints + logs)
├── reproduce.py                  # Loads checkpoints and regenerates all figures + Table 1
├── Main_deep_learning_qm9.ipynb  # My main notebook
├── checkpoints/
│   ├── mpnn_best.pt  
│   └── egnn_best.pt  
├── outputs/
│   ├── mpnn_log.csv  
│   └── egnn_log.csv
└── README.md
```

---

## Environment setup

### Option A - Google Colab / Kaggle (recommended, GPU included)

Open the training notebook [`deep-learning-qm9.ipynb`](deep-learning-qm9.ipynb) and
run Cell 1. It auto-detects your PyTorch+CUDA version and installs the correct wheels:

```python
# Cell 1 does this automatically:
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-<ver>+<cu>.html
pip install scikit-learn matplotlib pandas
```

### Option B - Local (conda)

```bash
# 1. Create environment
conda create -n qm9 python=3.10 -y
conda activate qm9

# 2. Install PyTorch (adjust cuda version to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch_geometric
pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 4. Install remaining dependencies
pip install scikit-learn matplotlib pandas
```

**Data download:** QM9 downloads automatically on first run via PyTorch Geometric.
No manual download needed. It will be cached at `./data/` (~600 MB).

---

## Reproducing key results (no retraining needed)

### Step 1 - Download checkpoints and training logs

Download `checkpoints.zip` from:
> **[Google Drive link - paste your link here]**

Unzip into the repo root so the paths are:
```
checkpoints/mpnn_best.pt
checkpoints/egnn_best.pt
outputs/mpnn_log.csv
outputs/egnn_log.csv
```

### Step 2 - Run the reproduction script

```bash
python reproduce.py
```

This will:
1. Load both checkpoints
2. Evaluate on the 10k test set and print **Table 1**
3. Save all 4 report figures to `outputs/`

Expected console output:
```
============================================================
  TABLE 1 - Test Set Results
============================================================
  Model                        MAE      RMSE       R²
  -------------------------------------------------------
  MPNN Baseline (2D only)    139.6 meV  213.1 meV  0.9728
  EGNN (3D equivariant)       75.7 meV  116.0 meV  0.9919
  Improvement                 +45.8%
============================================================
```

To skip the t-SNE figure (slow, ~2 min):
```bash
python reproduce.py --skip_tsne
```

### Step 3 (optional) - Use the Jupyter notebook instead

Open `deep-learning-qm9.ipynb` and run cells 1–4 (install + config + data + models),
then jump directly to the figure cells (cells 8–10) after placing the checkpoints.

---

## Retraining from scratch

```bash
# Train MPNN baseline (~60 min on T4 GPU)
python train.py --model mpnn

# Train EGNN variant (~90 min on T4 GPU)
python train.py --model egnn

# Smoke test (5 epochs, 5k molecules, ~5 min)
python train.py --model mpnn --quick
python train.py --model egnn --quick
```

Checkpoints are saved to `checkpoints/<model>_best.pt`.
Training logs are saved to `outputs/<model>_log.csv`.

---

## Model details

### Baseline: MPNN (Gilmer et al. 2017)
- **Input:** 2D molecular bond graph (atom features + bond type)
- **Architecture:** NNConv edge-conditioned message passing × 3 layers + Set2Set readout
- **Parameters:** 870,913
- **No 3D coordinates used**

### Contribution 1: EGNN (Satorras et al. 2021)
- **Input:** 2D graph + 3D atomic coordinates
- **Architecture:** E(3)-equivariant distance-based message passing × 4 layers + sum pooling
- **Parameters:** 482,561 (fewer than baseline)
- **Key property:** invariant to rotations and translations by construction

### Contribution 2: Learning curves (Category B analysis)
- Both models trained on 6 dataset sizes: 1k / 5k / 10k / 25k / 50k / 110k
- 50 epochs per size; val MAE reported at best epoch
- Reveals crossover: MPNN wins below ~10k molecules, EGNN dominates above 25k

---

## Training hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 5e-4 |
| Weight decay | 1e-8 |
| LR schedule | CosineAnnealingLR (eta_min=1e-6) |
| Batch size | 32 |
| Gradient clip | max norm 10 |
| Early stopping patience | 20 epochs |
| Target normalisation | z-score (mean=6858 meV, std=1283 meV) |
| Train/val/test split | 110k / 10k / 10k (SchNet paper standard) |
| Random seed | 42 |

---

## References

1. Ramakrishnan et al. (2014). *Quantum chemistry structures and properties of 134 kilo molecules.* Scientific Data.
2. Gilmer et al. (2017). *Neural Message Passing for Quantum Chemistry.* ICML.
3. Satorras, Hoogeboom & Welling (2021). *E(n) Equivariant Graph Neural Networks.* ICML. [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)
4. Schütt et al. (2017). *SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.* NeurIPS.
5. Fey & Lenssen (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop.


