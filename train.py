import os, copy, time, argparse, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
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



class SelectTarget:
    def __init__(self, idx=TARGET_IDX):
        self.idx = idx
    def __call__(self, data):
        data.y = data.y[:, self.idx].unsqueeze(1)
        return data


def load_qm9(data_root='./data', n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST):
    dataset = QM9(root=data_root, transform=T.Compose([SelectTarget()]))
    torch.manual_seed(SEED)
    dataset    = dataset.shuffle()
    train_ds   = dataset[:n_train]
    val_ds     = dataset[n_train : n_train + n_val]
    test_ds    = dataset[n_train + n_val : n_train + n_val + n_test]
    ys         = torch.cat([d.y for d in train_ds], dim=0)
    mean, std  = ys.mean().item(), ys.std().item()
    return train_ds, val_ds, test_ds, mean, std



def evaluate(model, loader, mean, std, device=DEVICE):
    model.eval()
    preds_all, tgts_all = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch) * std + mean   # denormalise → eV
            preds_all.append(out.cpu())
            tgts_all.append(batch.y.cpu())
    p = torch.cat(preds_all).squeeze()
    t = torch.cat(tgts_all).squeeze()
    mae  = (p - t).abs().mean().item() * EV_TO_MEV
    rmse = ((p - t) ** 2).mean().sqrt().item() * EV_TO_MEV
    r2   = (1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()
    return mae, rmse, r2, p, t


def train_model(model, name, train_ds, val_loader, mean, std,
                epochs=100, lr=5e-4, patience=20,
                batch_size=32, num_workers=2,
                save_ckpt=True, verbose=True):
    model   = model.to(DEVICE)
    mean_t  = torch.tensor(mean, device=DEVICE)
    std_t   = torch.tensor(std,  device=DEVICE)
    opt     = Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    sched   = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit    = nn.L1Loss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    best_val, patience_ctr = float('inf'), 0
    log_rows, best_state   = [], None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()

        for batch in train_loader:
            batch  = batch.to(DEVICE)
            y_norm = (batch.y - mean_t) / std_t
            opt.zero_grad()
            loss = crit(model(batch), y_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        sched.step()
        train_loss = total_loss / len(train_ds)
        val_mae, val_rmse, val_r2, _, _ = evaluate(model, val_loader, mean, std)
        elapsed = time.time() - t0

        log_rows.append(dict(epoch=epoch, train_loss=train_loss,
                             val_mae=val_mae, val_rmse=val_rmse,
                             val_r2=val_r2, lr=sched.get_last_lr()[0],
                             time_s=elapsed))

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f'  [{name}] ep {epoch:3d} | loss={train_loss:.4f} '
                  f'| val MAE={val_mae:.1f} meV | R²={val_r2:.3f} | {elapsed:.0f}s')

        if val_mae < best_val:
            best_val, patience_ctr = val_mae, 0
            best_state = copy.deepcopy(model.state_dict())
            if save_ckpt:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({'model_state': best_state, 'epoch': epoch,
                            'mean': mean, 'std': std, 'val_mae': val_mae},
                           f'checkpoints/{name}_best.pt')
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f'  Early stopping at epoch {epoch}')
                break

    model.load_state_dict(best_state)
    if verbose:
        print(f'  → Best val MAE: {best_val:.1f} meV')
    return best_val, pd.DataFrame(log_rows)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      choices=['mpnn', 'egnn'], required=True)
    parser.add_argument('--quick',      action='store_true',
                        help='Smoke test: 5 epochs on 5k molecules (~5 min)')
    parser.add_argument('--data_root',  default='./data')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=5e-4)
    parser.add_argument('--patience',   type=int,   default=20)
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs',     exist_ok=True)

    # Data
    n_train = 5_000 if args.quick else N_TRAIN
    epochs  = 5     if args.quick else args.epochs
    patience = 5    if args.quick else args.patience

    print(f'Loading QM9 (data_root={args.data_root})...')
    train_ds, val_ds, test_ds, mean, std = load_qm9(args.data_root,
                                                      n_train=n_train)
    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    print(f'Train: {len(train_ds):,}  |  '
          f'mean={mean*EV_TO_MEV:.1f} meV  std={std*EV_TO_MEV:.1f} meV')

    # Model
    if args.model == 'mpnn':
        model = MPNNBaseline(hidden=64, n_layers=3)
    else:
        model = EGNNModel(hidden=128, n_layers=4, use_pos=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.model.upper()}  |  {n_params:,} parameters  |  device: {DEVICE}')
    if args.quick:
        print('QUICK MODE: 5 epochs, 5k molecules')
    print('=' * 60)

    # Train
    _, log_df = train_model(model, args.model, train_ds, val_loader, mean, std,
                            epochs=epochs, lr=args.lr, patience=patience,
                            batch_size=args.batch_size)
    log_df.to_csv(f'outputs/{args.model}_log.csv', index=False)

    # Test evaluation with best checkpoint
    ckpt = torch.load(f'checkpoints/{args.model}_best.pt', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    test_mae, test_rmse, test_r2, _, _ = evaluate(model, test_loader, mean, std)

    print(f'\n[{args.model.upper()} TEST]  '
          f'MAE={test_mae:.1f} meV  |  RMSE={test_rmse:.1f} meV  |  R²={test_r2:.4f}')
    print(f'Checkpoint saved in checkpoints/{args.model}_best.pt')
    print(f'Log saved        in outputs/{args.model}_log.csv')


if __name__ == '__main__':
    main()
