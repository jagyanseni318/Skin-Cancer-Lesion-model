"""Classification / calibration / multi-seed / convergence analysis (all from saved logits + training histories)."""
import json, sys
import numpy as np, pandas as pd
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
                             cohen_kappa_score, brier_score_loss, confusion_matrix, precision_recall_fscore_support)
from common import *

RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
ham = pd.read_csv(CACHE / 'ham_meta.csv'); te = ham[ham.split == 'test'].reset_index(drop=True)
y = te.label_idx.values; groups = te.lesion_id.values
SEEDS_AVAIL = [s for s in SEEDS if (OUT / f'seed{s}' / 'logits.npz').exists()]


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece_top(P, y, nb=15):
    conf = P.max(1); pred = P.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, nb + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def metrics(P, y):
    P = P / P.sum(1, keepdims=True); pred = P.argmax(1); n = len(y); yb = np.eye(7)[y]
    present = [c for c in range(7) if (y == c).any()]
    return dict(acc=float((pred == y).mean()), bacc=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred, average='macro', labels=list(range(7)), zero_division=0),
                auroc=roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(7))),
                prauc=float(np.mean([average_precision_score(yb[:, c], P[:, c]) for c in range(7)])),
                mcc=matthews_corrcoef(y, pred), kappa=cohen_kappa_score(y, pred),
                brier=float(np.mean([brier_score_loss(yb[:, c], P[:, c]) for c in range(7)])),
                ece=ece_top(P, y), nll=float(-np.log(P[np.arange(n), y] + 1e-12).mean()))


def cluster_boot(P, y, groups, B=1000, seed=0):
    rng = np.random.RandomState(seed); ug = np.unique(groups); idx_by = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(B):
        gs = rng.choice(ug, len(ug), replace=True); ii = np.concatenate([idx_by[g] for g in gs])
        try: out.append(metrics(P[ii], y[ii]))
        except Exception: pass
    d = pd.DataFrame(out); return {k: (d[k].quantile(0.025), d[k].quantile(0.975)) for k in d.columns}


rows = []; probs = {}; cal = {}; boots = {}
for s in SEEDS_AVAIL:
    z = np.load(OUT / f'seed{s}' / 'logits.npz'); T = float(z['T']); lg = z['test']
    P = softmax(lg / T); P0 = softmax(lg); probs[s] = P
    m = metrics(P, y); m0 = metrics(P0, y)
    rows.append(dict(seed=s, T=T, **m, ece_uncal=m0['ece'], nll_uncal=m0['nll'], brier_uncal=m0['brier']))
    boots[s] = cluster_boot(P, y, groups, B=500, seed=s)
    print('seed', s, {k: round(v, 4) for k, v in m.items()}, flush=True)
df = pd.DataFrame(rows); df.to_csv(RES / 'cls_per_seed.csv', index=False)
Pens = np.mean([probs[s] for s in SEEDS_AVAIL], 0); mens = metrics(Pens, y)
agg = {k: dict(mean=float(df[k].mean()), sd=float(df[k].std(ddof=1)) if len(df) > 1 else np.nan, min=float(df[k].min()), max=float(df[k].max()))
       for k in df.columns if k != 'seed'}
# t-based 95% CI of the seed mean (n seeds) - describes training-stochasticity, not test sampling
from scipy import stats as st
for k in agg:
    n = len(df)
    if n > 1: agg[k]['ci_seed_mean'] = [agg[k]['mean'] - st.t.ppf(.975, n - 1) * agg[k]['sd'] / np.sqrt(n),
                                        agg[k]['mean'] + st.t.ppf(.975, n - 1) * agg[k]['sd'] / np.sqrt(n)]
# test-sampling CI: average of per-seed cluster-bootstrap half-widths (reported per metric)
agg['boot_halfwidth_mean'] = {k: float(np.mean([(boots[s][k][1] - boots[s][k][0]) / 2 for s in SEEDS_AVAIL])) for k in boots[SEEDS_AVAIL[0]]}
json.dump(dict(agg=agg, ensemble=mens, seeds=SEEDS_AVAIL, boots={str(s): {k: list(v) for k, v in boots[s].items()} for s in boots}),
          open(RES / 'cls_summary.json', 'w'), indent=1, default=float)

# per-class precision/recall/F1/AUROC per seed
pc = []
for s in SEEDS_AVAIL:
    P = probs[s]; pred = P.argmax(1); p, r, f, sup = precision_recall_fscore_support(y, pred, labels=list(range(7)), zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(7)))
    for c in range(7):
        tn = cm.sum() - cm[c].sum() - cm[:, c].sum() + cm[c, c]; fp = cm[:, c].sum() - cm[c, c]
        pc.append(dict(seed=s, cls=CLASSES[c], precision=p[c], recall=r[c], f1=f[c], specificity=tn / (tn + fp), support=int(sup[c]),
                       auroc=roc_auc_score(y == c, P[:, c]), prauc=average_precision_score(y == c, P[:, c])))
pcd = pd.DataFrame(pc); pcd.to_csv(RES / 'cls_per_class_per_seed.csv', index=False)
pcd.groupby('cls').agg(['mean', 'std']).drop(columns='seed').to_csv(RES / 'cls_per_class_agg.csv')
cms = {s: confusion_matrix(y, probs[s].argmax(1), labels=list(range(7))).tolist() for s in SEEDS_AVAIL}
json.dump(cms, open(RES / 'confusion_matrices.json', 'w'))

# reliability bins (seed 42, pre/post temperature scaling)
if 42 in SEEDS_AVAIL:
    z = np.load(OUT / 'seed42' / 'logits.npz'); rel = {}
    for nm, P in [('uncal', softmax(z['test'])), ('cal', probs[42])]:
        conf = P.max(1); acc = (P.argmax(1) == y).astype(float); edges = np.linspace(0, 1, 11)
        rel[nm] = [dict(lo=lo, hi=hi, n=int(((conf > lo) & (conf <= hi)).sum()),
                        acc=float(acc[(conf > lo) & (conf <= hi)].mean()) if ((conf > lo) & (conf <= hi)).any() else None,
                        conf=float(conf[(conf > lo) & (conf <= hi)].mean()) if ((conf > lo) & (conf <= hi)).any() else None)
                   for lo, hi in zip(edges[:-1], edges[1:])]
    json.dump(rel, open(RES / 'reliability_seed42.json', 'w'))

# ---- convergence audit ----------------------------------------------------------------------------
conv = []
for s in SEEDS_AVAIL + ([('42_ep100')] if (OUT / 'models' / 'seed42_ep100_history.csv').exists() else []):
    nm = f'seed{s}' if not isinstance(s, str) else 'seed42_ep100'
    h = pd.read_csv(OUT / 'models' / f'{nm}_history.csv'); E = len(h)
    bi = int(h.val_f1.idxmax()); last = h.iloc[-1]; tail = h.iloc[-15:]
    ns = 0; bestf = -1; stop_ep = None
    for _, r in h.iterrows():
        if r.val_f1 > bestf: bestf = r.val_f1; ns = 0
        else:
            ns += 1
            if ns >= 15 and stop_ep is None: stop_ep = int(r.epoch)
    sl_loss = np.polyfit(tail.epoch, tail.val_loss, 1)[0]; sl_f1 = np.polyfit(tail.epoch, tail.val_f1, 1)[0]
    conv.append(dict(run=nm, epochs=E, best_epoch=int(h.epoch[bi]), best_val_f1=float(h.val_f1[bi]), last_val_f1=float(last.val_f1),
                     last10_val_f1_mean=float(h.val_f1.iloc[-10:].mean()), min_val_loss=float(h.val_loss.min()),
                     min_val_loss_epoch=int(h.epoch[h.val_loss.idxmin()]), final_train_loss=float(last.train_loss), final_train_acc=float(last.train_acc),
                     final_lr_head=float(last.lr_head), val_loss_slope_last15=float(sl_loss), val_f1_slope_last15=float(sl_f1),
                     patience15_would_stop_at=stop_ep, epoch37_val_f1=float(h.val_f1.iloc[36]) if E >= 37 else None,
                     epoch37_lr_head=float(h.lr_head.iloc[36]) if E >= 37 else None, secs=float(h.secs.sum())))
pd.DataFrame(conv).to_csv(RES / 'convergence.csv', index=False); print(pd.DataFrame(conv).round(4).to_string())

# checkpoint sensitivity: test metrics of the last-epoch checkpoint vs selected (best-val) checkpoint  (reporting only)
if False:
    pass
print(df.round(4).to_string()); print('ensemble', {k: round(v, 4) for k, v in mens.items()})
