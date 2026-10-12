"""Train DenseNet121-TrustXAI for one seed with a *fully annealed* OneCycleLR schedule (fixes the epoch-37 truncation).
Checkpoint selection uses val_sel (half of validation); temperature scaling later uses the disjoint val_cal half."""
import argparse, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, balanced_accuracy_score
from common import *

train_tf = A.Compose([
    A.RandomResizedCrop(size=(IMG, IMG), scale=(0.7, 1.0), p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.7), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 20), hole_width_range=(1, 20), fill=0, p=0.3),
    ])
val_tf = A.Compose([A.NoOp()])
MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1); STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def gpu_norm(xb):   # uint8 NHWC (on device) -> normalized NCHW float (same arithmetic as albumentations.Normalize)
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    return ((x - MEAN_T.to(x.device)) / STD_T.to(x.device)).contiguous()



class DS(Dataset):
    def __init__(self, path, idx, y, tf):
        self.path, self.idx, self.y, self.tf, self.X = path, idx, y, tf, None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.X is None:
            self.X = np.load(self.path, mmap_mode='r')
        return torch.from_numpy(np.ascontiguousarray(self.tf(image=np.ascontiguousarray(self.X[self.idx[i]]))['image'])), int(self.y[i])


def wseed(_):
    s = torch.initial_seed() % 2**32; np.random.seed(s); random.seed(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--tag', default=''); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--arch', default='densenet121', choices=list(ARCHES))
    a = ap.parse_args()
    arch_tag = '' if a.arch == 'densenet121' else f'_{a.arch}'
    name = f'seed{a.seed}{a.tag}{arch_tag}'; odir = OUT / 'models'; odir.mkdir(exist_ok=True, parents=True)
    ham = pd.read_csv(CACHE / 'ham_meta.csv'); xp = str(CACHE / 'ham_x.npy')
    tr = ham[ham.split == 'train']; vs = ham[(ham.split == 'val') & (ham.val_role == 'sel')]
    set_seed(a.seed); g = torch.Generator(); g.manual_seed(a.seed)
    cnt = tr.label_idx.value_counts().sort_index(); w = [1.0 / cnt[l] for l in tr.label_idx]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=g)
    tl = DataLoader(DS(xp, tr.cache_idx.values, tr.label_idx.values, train_tf), batch_size=a.bs, sampler=sampler,
                    num_workers=a.workers, worker_init_fn=wseed, generator=g, persistent_workers=True)
    vl = DataLoader(DS(xp, vs.cache_idx.values, vs.label_idx.values, val_tf), batch_size=64, shuffle=False, num_workers=2)
    model = ARCHES[a.arch]().to(DEVICE)
    crit = CombinedLoss(make_alpha(tr)).to(DEVICE)
    opt = torch.optim.AdamW(model.param_groups(3e-4), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[pg['lr'] for pg in opt.param_groups],
                                                steps_per_epoch=len(tl), epochs=a.epochs, pct_start=0.3)
    scaler = torch.amp.GradScaler('cuda'); hist = []; best = (-1, 1e9); best_ep = 0
    t00 = time.time()
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); tot = cor = n = 0
        for x, y in tl:
            x, y = gpu_norm(x.to(DEVICE, non_blocking=True)), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                lg = model(x)
            loss = crit(lg.float(), y)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); cor += (lg.argmax(1) == y).sum().item(); n += len(y)
        model.eval(); P = []; L = []; vloss = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = gpu_norm(x.to(DEVICE)), y.to(DEVICE); lg = model(x).float(); vloss += crit(lg, y).item() * len(y)
                P.append(lg.argmax(1).cpu()); L.append(y.cpu())
        P = torch.cat(P).numpy(); L = torch.cat(L).numpy()
        f1 = f1_score(L, P, average='macro', labels=list(range(7)), zero_division=0); ba = balanced_accuracy_score(L, P)
        hist.append(dict(epoch=ep, train_loss=tot / n, train_acc=cor / n, val_loss=vloss / len(vs), val_f1=f1, val_bacc=ba,
                         lr_head=opt.param_groups[-1]['lr'], secs=time.time() - t0))
        if f1 > best[0] or (f1 == best[0] and hist[-1]['val_loss'] < best[1]):
            best = (f1, hist[-1]['val_loss']); best_ep = ep; torch.save(model.state_dict(), odir / f'{name}_best.pt')
        pd.DataFrame(hist).to_csv(odir / f'{name}_history.csv', index=False)
        print(f'[{name}] ep{ep:03d} tr_loss={tot/n:.4f} tr_acc={cor/n:.3f} val_loss={vloss/len(vs):.4f} val_f1={f1:.4f} '
              f'bacc={ba:.4f} best_ep={best_ep} lr={opt.param_groups[-1]["lr"]:.2e} {time.time()-t0:.0f}s', flush=True)
    torch.save(model.state_dict(), odir / f'{name}_last.pt')
    json.dump(dict(seed=a.seed, epochs=a.epochs, best_epoch=best_ep, best_val_f1=best[0], total_secs=time.time() - t00),
              open(odir / f'{name}_summary.json', 'w'))
