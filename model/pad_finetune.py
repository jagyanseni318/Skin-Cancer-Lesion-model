"""Target-domain adaptation by fine-tuning on PAD-UFES-20 (reviewer request: is target-domain adaptation needed, and how much does it recover?).

Same protocol as the adaptation ladder of domain_gap.py (Table 17): the five mappable classes of PAD-UFES-20, patient-level splits
with GroupShuffleSplit(train_size=0.3, random_state=11); the models are adapted on 30% of the patients and evaluated on the other
70%. Here the adaptation is partial fine-tuning: dense block 4 (the block holding the Grad-CAM++ target layer), the final batch norm and the classifier head are trained, with the study's own layer-wise learning rates, while the earlier layers stay frozen (full fine-tuning was not run).

Everything is fixed in advance and no evaluation patient is used for any choice: 8 epochs, AdamW, the study's parameter-group learning rates
(head 3e-4, dense block 4 and final norm 5e-5), cosine decay to zero, class-weighted cross-entropy on the five mappable logits with
weights from the adaptation split only, no augmentation (frozen-layer features are cached), batch size 32. No validation set is carved out of the adaptation patients, so
there is no early stopping or checkpoint choice. Zero-shot results on the same evaluation patients are recorded before adaptation for a paired comparison.

    python pad_finetune.py [--splits 10] [--seeds 42 43 44 45 46]  ->  out/results/pad_finetune.json"""
import argparse
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from common import *

SH5 = [C2I[c] for c in ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV']]
EPOCHS, BS = 8, 32
torch.manual_seed(0)


def metrics(pred, y):
    return [float((pred == y).mean()), float(balanced_accuracy_score(y, pred)), float(f1_score(y, pred, labels=list(range(5)), average='macro', zero_division=0))]


class Tail(nn.Module):
    """The part of the network that is adapted: dense block 4, the final batch norm, global pooling and the head. Everything before it is frozen
    (eval mode) and its output is computed once per image, which is exact because nothing before dense block 4 changes."""
    def __init__(self, m):
        super().__init__()
        f = m.backbone.features
        self.block4, self.norm5, self.fc1, self.bn1, self.classifier = f.denseblock4, f.norm5, m.fc1, m.bn1, m.classifier

    def forward(self, z):
        z = torch.relu(self.norm5(self.block4(z))); z = torch.flatten(nn.functional.adaptive_avg_pool2d(z, 1), 1)
        return self.classifier(torch.relu(self.bn1(self.fc1(z))))[:, SH5]


@torch.no_grad()
def frozen_features(m, X, idx, bs=64):
    """Output of transition3 (input of dense block 4) for every image; frozen layers in eval mode."""
    m.eval(); front = nn.Sequential(*list(m.backbone.features.children())[:10]); out = []
    for i in range(0, len(idx), bs):
        out.append(front(norm_batch(X[idx[i:i + bs]]).to(DEVICE)).cpu())
    return torch.cat(out)


@torch.no_grad()
def predict(tail, Z):
    tail.eval(); return torch.cat([tail(Z[i:i + 128].to(DEVICE)).cpu() for i in range(0, len(Z), 128)]).numpy()


def finetune(tail, Z, y, seed):
    rng = np.random.RandomState(seed)
    cnt = np.bincount(y, minlength=5); w = torch.tensor(len(y) / (5.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    groups = [{'params': tail.block4.parameters(), 'lr': 5e-5}, {'params': tail.norm5.parameters(), 'lr': 5e-5},
              {'params': list(tail.fc1.parameters()) + list(tail.bn1.parameters()) + list(tail.classifier.parameters()), 'lr': 3e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(y) / BS)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps); ce = nn.CrossEntropyLoss(weight=w)
    for ep in range(EPOCHS):
        tail.train(); perm = rng.permutation(len(y))
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            if len(b) < 2: continue
            opt.zero_grad(set_to_none=True); loss = ce(tail(Z[b].to(DEVICE)), torch.as_tensor(y[b], device=DEVICE)); loss.backward(); opt.step(); sched.step()
    return tail


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--splits', type=int, default=20, help='use valid splits [start, splits)'); ap.add_argument('--start', type=int, default=0, help='first split to run; later splits are appended to an existing result file'); ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS)
    a = ap.parse_args()
    pad = pd.read_csv(CACHE / 'pad_meta.csv'); mapped = pad.label.notna().values; pm = pad[mapped].reset_index(drop=True); yp = pm.label_idx.astype(int).values
    XP = np.load(CACHE / 'pad_x.npy', mmap_mode='r'); pad_idx = np.where(mapped)[0]; y5 = np.array([SH5.index(v) for v in yp])
    splits = []
    for tr, ev in GroupShuffleSplit(n_splits=20, train_size=0.3, random_state=11).split(np.zeros(len(y5)), y5, groups=pm.patient_id.values):
        if len(set(y5[tr])) < 5 or len(set(y5[ev])) < 5: continue
        splits.append((tr, ev))
    splits = splits[:a.splits]; k0 = a.start
    res = {'protocol': {'epochs': EPOCHS, 'batch_size': BS, 'optimizer': 'AdamW', 'adapted_layers': 'dense block 4, final batch norm, classifier head (earlier layers frozen)', 'lr_head': 3e-4, 'lr_block4': 5e-5, 'weight_decay': 1e-4, 'schedule': 'cosine to zero',
                        'split': 'GroupShuffleSplit(n_splits=20, train_size=0.3, random_state=11), %d valid splits (the splits of the adaptation ladder), patient-level' % len(splits),
                        'classes': ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV'], 'n_images': int(len(y5)), 'n_patients': int(pm.patient_id.nunique())}, 'seeds': {}}
    out = OUT / 'results' / 'pad_finetune.json'
    if k0 > 0 and out.exists(): old = json.load(open(out)); res['seeds'] = old['seeds']
    for s in a.seeds:
        base = DenseNet121_TrustXAI().to(DEVICE); base.load_state_dict(torch.load(OUT / 'models' / f'seed{s}_best.pt', map_location=DEVICE)); base.eval()
        t0 = time.time(); Z = frozen_features(base, XP, pad_idx); print(f'seed {s}: frozen features {tuple(Z.shape)} in {time.time() - t0:.0f}s', flush=True)
        zs_all = metrics(predict(Tail(base), Z).argmax(1), y5); print(f'seed {s}: zero-shot accuracy on all mapped images {zs_all[0]:.3f} (cached-feature path)', flush=True)
        prev = res['seeds'].get(str(s)) if k0 > 0 else None
        zs, ft, n_tr, n_ev = (list(prev['zero_shot']), list(prev['fine_tuned']), list(prev['n_adapt']), list(prev['n_eval'])) if prev else ([], [], [], [])
        assert len(ft) == k0, (len(ft), k0)
        for k, (tr, ev) in enumerate(splits):
            if k < k0: continue
            t0 = time.time(); tail = Tail(copy.deepcopy(base))
            zs.append(metrics(predict(tail, Z[ev]).argmax(1), y5[ev]))
            finetune(tail, Z[tr], y5[tr], seed=1000 * s + k)
            ft.append(metrics(predict(tail, Z[ev]).argmax(1), y5[ev])); n_tr.append(int(len(tr))); n_ev.append(int(len(ev)))
            print(f'seed {s} split {k}: zero-shot acc {zs[-1][0]:.3f} -> adapted acc {ft[-1][0]:.3f} (bacc {ft[-1][1]:.3f}, F1 {ft[-1][2]:.3f})  {time.time() - t0:.0f}s', flush=True)
        res['seeds'][str(s)] = {'zero_shot_all_mapped_images': zs_all, 'zero_shot': zs, 'fine_tuned': ft, 'n_adapt': n_tr, 'n_eval': n_ev}
        json.dump(res, open(out, 'w'), indent=1)
    print('done', out)


if __name__ == '__main__':
    main()
