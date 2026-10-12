"""Uniform comparison of all model configurations from their saved logits (identical metric code for every configuration).
Configurations: dn_fp32 (DenseNet121), rn_fp32 (ResNet50), dn_int8, rn_int8 (INT8 versions of the same checkpoints).
Outputs out/results/model_comparison.json + model_comparison.csv.  Nothing is retrained here."""
import json, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats as st
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from common import *
warnings.filterwarnings('ignore')
RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
ham = pd.read_csv(CACHE / 'ham_meta.csv'); te = ham[ham.split == 'test'].reset_index(drop=True); yh = te.label_idx.values
pad = pd.read_csv(CACHE / 'pad_meta.csv'); mapped = pad.label.notna().values; yp = pad[mapped].label_idx.astype(int).values
SH5 = [C2I[c] for c in ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV']]
CONF = {'dn_fp32': ('DenseNet121 (FP32)', lambda s: OUT / f'seed{s}'), 'rn_fp32': ('ResNet50 (FP32)', lambda s: OUT / f'seed{s}_resnet50'),
        'dn_int8': ('DenseNet121 (INT8)', lambda s: OUT / f'seed{s}_int8ptq'), 'rn_int8': ('ResNet50 (INT8)', lambda s: OUT / f'seed{s}_resnet50_int8ptq'),
        'dn_mixed': ('DenseNet121 (INT8, dense block 1 in FP32)', lambda s: OUT / f'seed{s}_int8mixed')}


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece_top(P, y, nb=15):
    conf = P.max(1); acc = (P.argmax(1) == y).astype(float); e = 0.0; edges = np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def metrics(lg, y, T):
    P = softmax(lg / T); pred = P.argmax(1); P1 = softmax(lg)
    yb = np.eye(7)[y]
    return dict(acc=float((pred == y).mean()), bacc=float(balanced_accuracy_score(y, pred)),
                f1=float(f1_score(y, pred, average='macro', labels=list(range(7)), zero_division=0)),
                auroc=float(roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(7)))),
                mcc=float(matthews_corrcoef(y, pred)), ece_ts=ece_top(P, y), ece_uncal=ece_top(P1, y),
                nll_ts=float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()),
                brier_ts=float(np.mean((P - yb) ** 2)), T=float(T))


def pad_metrics(lg, T):
    P = softmax(lg[:, SH5] / T); pred = np.array(SH5)[P.argmax(1)]; present = sorted(set(yp.tolist()))
    yi = np.array([SH5.index(v) for v in yp])
    return dict(pad_acc=float((pred == yp).mean()), pad_bacc=float(balanced_accuracy_score(yp, pred)),
                pad_f1=float(f1_score(yp, pred, labels=present, average='macro', zero_division=0)), pad_ece=ece_top(P, yi))


def mcnemar_p(a, b, y):
    ca, cb = a == y, b == y; n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    return float(st.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0, n01, n10


def boot_diff(pa, pb, y, lesion, B=1000, seed=0):
    rng = np.random.RandomState(seed); by = pd.Series(np.arange(len(y))).groupby(lesion).apply(lambda s: s.values).tolist(); K = len(by)
    da, df = [], []
    for _ in range(B):
        ii = np.concatenate([by[k] for k in rng.randint(0, K, K)])
        da.append((pa[ii] == y[ii]).mean() - (pb[ii] == y[ii]).mean())
        df.append(f1_score(y[ii], pa[ii], average='macro', labels=list(range(7)), zero_division=0) - f1_score(y[ii], pb[ii], average='macro', labels=list(range(7)), zero_division=0))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(acc=q(da), f1=q(df))


def perm_p(x, y):
    x = np.asarray(x); y = np.asarray(y); z = np.concatenate([x, y]); n = len(x); obs = abs(x.mean() - y.mean()); cnt = tot = 0
    for idx in itertools.combinations(range(len(z)), n):
        m = np.zeros(len(z), bool); m[list(idx)] = True; tot += 1; cnt += abs(z[m].mean() - z[~m].mean()) >= obs - 1e-12
    return cnt / tot


out = dict(configs={}, tests={}, per_seed={}); preds = {}
for key, (label, d) in CONF.items():
    rows = []; preds[key] = {}
    for s in SEEDS:
        f = d(s) / 'logits.npz'
        if not f.exists(): continue
        z = np.load(f); T = float(z['T']); r = dict(seed=s, **metrics(z['test'], yh, T), **pad_metrics(z['pad'][mapped], T))
        rows.append(r); preds[key][s] = z['test'].argmax(1)
    if rows:
        df = pd.DataFrame(rows); out['per_seed'][key] = rows
        out['configs'][key] = dict(label=label, n_seeds=len(rows), seeds=[int(x) for x in df.seed],
                                   **{m: dict(mean=float(df[m].mean()), sd=float(df[m].std(ddof=1)) if len(df) > 1 else float('nan'), min=float(df[m].min()), max=float(df[m].max()))
                                      for m in df.columns if m != 'seed'})

lesion = te.lesion_id.values
# ---- architecture comparison (FP32 vs FP32; independent training runs -> Welch + exact permutation on the 5-vs-5 seed metrics)
if 'dn_fp32' in out['per_seed'] and 'rn_fp32' in out['per_seed']:
    A = pd.DataFrame(out['per_seed']['dn_fp32']); Bm = pd.DataFrame(out['per_seed']['rn_fp32']); arch = {}
    for m in ['acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts', 'pad_acc', 'pad_bacc', 'pad_f1']:
        a, b = A[m].values, Bm[m].values; sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        arch[m] = dict(diff_rn_minus_dn=float(b.mean() - a.mean()), welch_p=float(st.ttest_ind(b, a, equal_var=False).pvalue), perm_p=perm_p(b, a),
                       cohen_d=float((b.mean() - a.mean()) / sp) if sp > 0 else float('nan'), n_dn=len(a), n_rn=len(b))
    per = []
    for s in SEEDS:
        if s in preds['dn_fp32'] and s in preds['rn_fp32']:
            p, n01, n10 = mcnemar_p(preds['dn_fp32'][s], preds['rn_fp32'][s], yh)
            per.append(dict(seed=s, mcnemar_p=p, dn_only_correct=n01, rn_only_correct=n10, boot_ci_dn_minus_rn=boot_diff(preds['dn_fp32'][s], preds['rn_fp32'][s], yh, lesion, seed=s)))
    arch['per_seed_pairwise'] = per; out['tests']['arch_dn_vs_rn'] = arch
# ---- precision comparison (INT8 vs its own FP32 parent: seed-matched, paired)
for ar, fp, q in [('densenet121', 'dn_fp32', 'dn_int8'), ('resnet50', 'rn_fp32', 'rn_int8'), ('densenet121', 'dn_fp32', 'dn_mixed')]:
    if fp in out['per_seed'] and q in out['per_seed']:
        F = pd.DataFrame(out['per_seed'][fp]).set_index('seed'); Q = pd.DataFrame(out['per_seed'][q]).set_index('seed'); common = F.index.intersection(Q.index); pr = {}
        for m in ['acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts', 'pad_acc', 'pad_bacc', 'pad_f1']:
            d = (Q.loc[common, m] - F.loc[common, m]).values; n = len(d)
            ci = st.t.interval(0.95, n - 1, loc=d.mean(), scale=st.sem(d)) if n > 1 and d.std(ddof=1) > 0 else (float('nan'), float('nan'))
            pr[m] = dict(mean_diff_int8_minus_fp32=float(d.mean()), sd=float(d.std(ddof=1)) if n > 1 else float('nan'), ci95=[float(ci[0]), float(ci[1])],
                         seeds_worse=int((d < 0).sum()), n=n)
        per = []
        for s in common:
            p, n01, n10 = mcnemar_p(preds[fp][s], preds[q][s], yh)
            per.append(dict(seed=int(s), mcnemar_p=p, fp32_only_correct=n01, int8_only_correct=n10, agreement=float((preds[fp][s] == preds[q][s]).mean()),
                            boot_ci_fp32_minus_int8=boot_diff(preds[fp][s], preds[q][s], yh, lesion, seed=int(s))))
        pr['per_seed_pairwise'] = per; out['tests'][f'precision_{ar}_{q.split("_")[1]}'] = pr
json.dump(out, open(RES / 'model_comparison.json', 'w'), indent=1)
rows = []
for k, c in out['configs'].items():
    rows.append(dict(config=c['label'], n_seeds=c['n_seeds'], **{f'{m}_mean': c[m]['mean'] for m in ['acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts', 'pad_acc', 'pad_bacc']},
                     **{f'{m}_sd': c[m]['sd'] for m in ['acc', 'bacc', 'f1', 'auroc']}))
pd.DataFrame(rows).to_csv(RES / 'model_comparison.csv', index=False)
print(pd.DataFrame(rows).round(4).T.to_string())
print('DONE')
