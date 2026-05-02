import os, argparse, warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 120})

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from models import MPNNBaseline, EGNNModel

TARGET_IDX = 4
EV_TO_MEV  = 1000.0
SEED       = 42
N_TRAIN    = 110_000
N_VAL      =  10_000
N_TEST     =  10_000
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LC_RESULTS = {
    'MPNN': {
        'sizes':   [1_000, 5_000, 10_000, 25_000,  50_000, 110_000],
        'val_mae': [346.2,  255.2,  221.8,  194.5,   165.6,   142.2],
    },
    'EGNN': {
        'sizes':   [1_000, 5_000, 10_000, 25_000,  50_000, 110_000],
        'val_mae': [388.1,  278.6,  230.3,  151.2,   117.3,    84.6],
    },
}



class SelectTarget:
    def __init__(self, idx=TARGET_IDX):
        self.idx = idx
    def __call__(self, data):
        data.y = data.y[:, self.idx].unsqueeze(1)
        return data


def load_splits(data_root):
    print(f'Loading QM9 from {data_root} ...')
    dataset = QM9(root=data_root, transform=T.Compose([SelectTarget()]))
    torch.manual_seed(SEED)
    dataset   = dataset.shuffle()
    train_ds  = dataset[:N_TRAIN]
    val_ds    = dataset[N_TRAIN : N_TRAIN + N_VAL]
    test_ds   = dataset[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]
    ys        = torch.cat([d.y for d in train_ds], dim=0)
    mean, std = ys.mean().item(), ys.std().item()
    print(f'  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}')
    print(f'  mean={mean*EV_TO_MEV:.1f} meV  std={std*EV_TO_MEV:.1f} meV')
    return val_ds, test_ds, mean, std


def evaluate(model, loader, mean, std):
    model.eval()
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out   = model(batch) * std + mean
            preds_all.append(out.cpu())
            tgts_all.append(batch.y.cpu())
    p = torch.cat(preds_all).squeeze()
    t = torch.cat(tgts_all).squeeze()
    mae  = (p - t).abs().mean().item() * EV_TO_MEV
    rmse = ((p - t) ** 2).mean().sqrt().item() * EV_TO_MEV
    r2   = (1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()
    return mae, rmse, r2, p, t


def load_checkpoint(model, name):
    path = f'checkpoints/{name}_best.pt'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Checkpoint not found: {path}\n'
            'Download checkpoints from the link in README.md'
        )
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    return model.to(DEVICE), ckpt['mean'], ckpt['std']



def plot_loss_curves(out_dir):
    log_m = pd.read_csv('outputs/mpnn_log.csv')
    log_e = pd.read_csv('outputs/egnn_log.csv')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(log_m['epoch'], log_m['val_mae'],
            label='MPNN Baseline', color='#4878CF', linewidth=2)
    ax.plot(log_e['epoch'], log_e['val_mae'],
            label='EGNN (3D equivariant)', color='#6ACC65', linewidth=2)
    ax.axhline(log_m['val_mae'].min(), color='#4878CF', linestyle=':', alpha=0.5)
    ax.axhline(log_e['val_mae'].min(), color='#6ACC65', linestyle=':', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation MAE (meV)', fontsize=12)
    ax.set_title('Figure 1: Training Dynamics — Validation MAE vs. Epoch', fontsize=13)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{out_dir}/fig1_loss_curves.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig1_loss_curves.pdf + .png')



def plot_parity(preds_m, preds_e, targets, mae_m, mae_e, r2_m, r2_e, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, preds, mae, r2, title, color in zip(
        axes,
        [preds_m, preds_e],
        [mae_m, mae_e],
        [r2_m, r2_e],
        ['MPNN Baseline (2D only)', 'EGNN (3D Equivariant)'],
        ['#4878CF', '#6ACC65'],
    ):
        p   = preds.numpy()   * EV_TO_MEV
        t   = targets.numpy() * EV_TO_MEV
        lim = [min(t.min(), p.min()) - 80, max(t.max(), p.max()) + 80]
        ax.scatter(t, p, s=4, alpha=0.2, color=color, rasterized=True)
        ax.plot(lim, lim, 'r--', linewidth=1.5, label='Ideal (y=x)')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('True \u0394\u03b5 (meV)', fontsize=11)
        ax.set_ylabel('Predicted \u0394\u03b5 (meV)', fontsize=11)
        ax.set_title(f'{title}\nMAE={mae:.1f} meV,  R\u00b2={r2:.3f}', fontsize=11)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
    plt.suptitle('Figure 2: Parity Plots — Predicted vs. True HOMO-LUMO Gap',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{out_dir}/fig2_parity_plots.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig2_parity_plots.pdf + .png')



def plot_learning_curves(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: MAE vs training size
    ax = axes[0]
    for model_name, color, marker in [('MPNN', '#4878CF', 'o'), ('EGNN', '#6ACC65', 's')]:
        r = LC_RESULTS[model_name]
        ax.plot(r['sizes'], r['val_mae'], color=color, marker=marker,
                linewidth=2, markersize=7, label=model_name)
        ax.annotate(f"{r['val_mae'][-1]:.0f}",
                    xy=(r['sizes'][-1], r['val_mae'][-1]),
                    xytext=(10, -15), textcoords='offset points',
                    color=color, fontsize=9, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlabel('Training set size (molecules)', fontsize=12)
    ax.set_ylabel('Best Validation MAE (meV)', fontsize=12)
    ax.set_title('Sample Efficiency: MAE vs. Training Size', fontsize=12)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: relative improvement
    ax2 = axes[1]
    sizes    = LC_RESULTS['MPNN']['sizes']
    mpnn_m   = LC_RESULTS['MPNN']['val_mae']
    egnn_m   = LC_RESULTS['EGNN']['val_mae']
    rel_imp  = [(m - e) / m * 100 for m, e in zip(mpnn_m, egnn_m)]
    labels   = [str(s) if s < 1000 else f'{s//1000}k' for s in sizes]
    bars = ax2.bar(labels, rel_imp,
                   color=['#D65F5F' if v < 0 else '#6ACC65' for v in rel_imp],
                   edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, rel_imp):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.5 if val >= 0 else -2),
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel('Training set size', fontsize=12)
    ax2.set_ylabel('EGNN improvement over MPNN (%)', fontsize=12)
    ax2.set_title('Relative Improvement: EGNN vs. MPNN\nby Training Set Size', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('Figure 3: Learning Curves (Sample Efficiency Analysis)',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{out_dir}/fig3_learning_curves.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig3_learning_curves.pdf + .png')



def extract_embeddings(model, loader, n_samples=2000):
    model.eval()
    embs, ys = [], []
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= n_samples:
                break
            batch = batch.to(DEVICE)
            embs.append(model.embed(batch).cpu())
            ys.append(batch.y.cpu())
            count += batch.num_graphs
    embs = torch.cat(embs)[:n_samples].numpy()
    ys   = torch.cat(ys)[:n_samples].squeeze().numpy() * EV_TO_MEV
    return embs, ys


def plot_tsne(mpnn_model, egnn_model, val_loader, out_dir):
    print('  Extracting embeddings (2000 val molecules)...')
    embs_m, ys_val = extract_embeddings(mpnn_model, val_loader)
    embs_e, _      = extract_embeddings(egnn_model,  val_loader)

    print('  Running t-SNE (1-2 min)...')
    vmin = np.percentile(ys_val, 2)
    vmax = np.percentile(ys_val, 98)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, embs, title in zip(
        axes,
        [embs_m, embs_e],
        ['MPNN Baseline', 'EGNN (3D Equivariant)'],
    ):
        n_comp = min(50, embs.shape[1])
        pre    = PCA(n_components=n_comp, random_state=SEED).fit_transform(embs)
        proj   = TSNE(n_components=2, perplexity=30, random_state=SEED,
                      n_iter=1000).fit_transform(pre)
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=ys_val,
                        cmap='RdYlBu_r', s=10, alpha=0.75,
                        vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label='HOMO-LUMO gap (meV)')
        ax.set_xlabel('t-SNE dim 1', fontsize=10)
        ax.set_ylabel('t-SNE dim 2', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle('Figure 4: t-SNE of Molecular Embeddings (Supporting Analysis)',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'{out_dir}/fig4_tsne.{ext}', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig4_tsne.pdf + .png')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--out_dir',   default='./outputs')
    parser.add_argument('--skip_tsne', action='store_true',
                        help='Skip t-SNE for a faster run')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Device: {DEVICE}\n')

    val_ds, test_ds, mean, std = load_splits(args.data_root)
    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    print('\nLoading checkpoints...')
    mpnn_model, m_mean, m_std = load_checkpoint(MPNNBaseline(hidden=64, n_layers=3), 'mpnn')
    egnn_model, e_mean, e_std = load_checkpoint(EGNNModel(hidden=128, n_layers=4, use_pos=True), 'egnn')

    print('\nEvaluating on test set (10,000 molecules)...')
    mae_m, rmse_m, r2_m, preds_m, targets = evaluate(mpnn_model, test_loader, m_mean, m_std)
    mae_e, rmse_e, r2_e, preds_e, _       = evaluate(egnn_model, test_loader, e_mean, e_std)

    print('\n' + '=' * 60)
    print('  TABLE 1 — Test Set Results')
    print('=' * 60)
    print(f'  {"Model":<28} {"MAE":>8} {"RMSE":>9} {"R²":>8}')
    print(f'  {"-"*55}')
    print(f'  {"MPNN Baseline (2D only)":<28} {mae_m:>6.1f} meV {rmse_m:>7.1f} meV {r2_m:>7.4f}')
    print(f'  {"EGNN (3D equivariant)":<28} {mae_e:>6.1f} meV {rmse_e:>7.1f} meV {r2_e:>7.4f}')
    print(f'  {"Improvement":<28} {(mae_m-mae_e)/mae_m*100:>+6.1f}%')
    print('=' * 60)

    print('\nGenerating figures...')

    if os.path.exists('outputs/mpnn_log.csv') and os.path.exists('outputs/egnn_log.csv'):
        plot_loss_curves(args.out_dir)
    else:
        print('  Skipping Fig 1 (training logs not found, run train.py first)')

    plot_parity(preds_m, preds_e, targets, mae_m, mae_e, r2_m, r2_e, args.out_dir)
    plot_learning_curves(args.out_dir)

    if not args.skip_tsne:
        plot_tsne(mpnn_model, egnn_model, val_loader, args.out_dir)
    else:
        print('  Skipping Fig 4 (--skip_tsne set)')

    print(f'\nAll figures saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
