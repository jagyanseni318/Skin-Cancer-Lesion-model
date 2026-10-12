"""Independent one-time verification of the descriptive statistics in the manuscript.

Re-derives every per-seed metric from the saved raw logits with code written separately from compare_models.py
(confusion-matrix macro-F1, rank-based one-vs-rest AUROC, own ECE), then summarizes the five seeds with mean, median, mode, SD
(sample, n - 1), SE, min, max, range and a t-based 95% interval, and compares everything with (a) out/results/model_comparison.json
and (b) the numbers printed in the final manuscript tables. Writes out/results/statistical_verification.json and .md.
Nothing is retrained and no number is copied from the manuscript into the computation."""
import json, re, sys
from collections import Counter
import numpy as np, pandas as pd
from scipy import stats as st
from common import OUT, CACHE, CLASSES

RES = OUT / 'results'
ham = pd.read_csv(CACHE / 'ham_meta.csv'); te = ham[ham.split == 'test'].reset_index(drop=True); y = te.label_idx.values.astype(int)
K = 7
CONF = {'dn_fp32': lambda s: OUT / f'seed{s}', 'rn_fp32': lambda s: OUT / f'seed{s}_resnet50', 'dn_int8': lambda s: OUT / f'seed{s}_int8ptq',
        'rn_int8': lambda s: OUT / f'seed{s}_resnet50_int8ptq', 'dn_mixed': lambda s: OUT / f'seed{s}_int8mixed'}
LABEL = {'dn_fp32': 'DenseNet121 FP32', 'rn_fp32': 'ResNet50 FP32', 'dn_int8': 'DenseNet121 INT8', 'rn_int8': 'ResNet50 INT8', 'dn_mixed': 'DenseNet121 INT8 (dense block 1 FP32)'}
SEEDS = [42, 43, 44, 45, 46]


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def conf_matrix(pred):
    m = np.zeros((K, K), int)
    np.add.at(m, (y, pred), 1); return m


def macro_f1(cm):
    f = []
    for c in range(K):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        f.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def bal_acc(cm):
    return float(np.mean([cm[c, c] / cm[c, :].sum() for c in range(K) if cm[c, :].sum() > 0]))


def auroc_ovr(P):
    out = []
    for c in range(K):
        pos = y == c
        if pos.sum() == 0 or (~pos).sum() == 0: continue
        r = st.rankdata(P[:, c]); n1 = pos.sum(); n0 = (~pos).sum()
        out.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out))


def mcc(cm):
    t = cm.sum(1).astype(float); p = cm.sum(0).astype(float); c = np.trace(cm); n = cm.sum()
    return float((c * n - (t * p).sum()) / np.sqrt((n ** 2 - (p * p).sum()) * (n ** 2 - (t * t).sum())))


def ece(P, nb=15):
    conf = P.max(1); ok = (P.argmax(1) == y); tot = 0.0
    for b in range(nb):
        lo, hi = b / nb, (b + 1) / nb
        m = (conf > lo) & (conf <= hi)
        if m.any(): tot += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(tot)


def summarize(v):
    v = np.asarray(v, float); n = len(v); sd = v.std(ddof=1); se = sd / np.sqrt(n); t = st.t.ppf(0.975, n - 1)
    rounded = [round(x, 3) for x in v]; cnt = Counter(rounded).most_common(1)[0]
    return dict(n=n, mean=float(v.mean()), median=float(np.median(v)), mode=(cnt[0] if cnt[1] > 1 else None), sd=float(sd), se=float(se),
                min=float(v.min()), max=float(v.max()), range=float(v.max() - v.min()), ci95_low=float(v.mean() - t * se), ci95_high=float(v.mean() + t * se),
                cv_percent=float(100 * sd / v.mean()) if v.mean() else None)


rows = []; per = {}
for k, path in CONF.items():
    per[k] = {m: [] for m in ('acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts')}
    for s in SEEDS:
        z = np.load(path(s) / 'logits.npz'); T = float(z['T']); P = softmax(z['test'].astype(np.float64) / T); pred = P.argmax(1); cm = conf_matrix(pred)
        per[k]['acc'].append(float((pred == y).mean())); per[k]['bacc'].append(bal_acc(cm)); per[k]['f1'].append(macro_f1(cm))
        per[k]['auroc'].append(auroc_ovr(P)); per[k]['mcc'].append(mcc(cm)); per[k]['ece_ts'].append(ece(P))

stats = {k: {m: summarize(v) for m, v in d.items()} for k, d in per.items()}
for k in stats:
    for m in stats[k]: stats[k][m]['per_seed'] = per[k][m]

# ---- (a) compare with the pipeline's own aggregate file
MC = json.load(open(RES / 'model_comparison.json'))['configs']
cmp_rows = []; worst = 0.0
for k in CONF:
    for m in ('acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts'):
        a, b = stats[k][m], MC[k][m]
        dm, ds = abs(a['mean'] - b['mean']), abs(a['sd'] - b['sd']); worst = max(worst, dm, ds)
        cmp_rows.append((k, m, a['mean'], b['mean'], a['sd'], b['sd'], dm, ds))

# ---- test-set composition and the majority-class (mode) baseline
cnt = np.bincount(y, minlength=K); modal = int(cnt.argmax())
baseline = dict(n_test=int(len(y)), class_counts={CLASSES[i]: int(cnt[i]) for i in range(K)}, modal_class=CLASSES[modal],
                modal_class_share=float(cnt[modal] / len(y)), majority_baseline_accuracy=float(cnt[modal] / len(y)), majority_baseline_macro_f1=float(
                    (2 * cnt[modal] / (cnt[modal] + len(y))) / K), chance_macro_auroc=0.5)
pred_mode = {k: Counter(np.load(CONF[k](42) / 'logits.npz')['test'].argmax(1).tolist()).most_common(1)[0][0] for k in CONF}
baseline['modal_predicted_class_seed42'] = {k: CLASSES[v] for k, v in pred_mode.items()}

# ---- (b) manuscript numbers: every "mean (SD x)" pair and the Table 26/28 rows must reproduce
md = (OUT / 'Revised_Manuscript_SkinCancer_ICCSDI2026.md').read_text(encoding='utf-8')
pair = re.findall(r'(\d\.\d{3}) \(SD (\d\.\d{3,4})\)', md)
def present(mean, sd):
    return (f'{mean:.3f} (SD {sd:.3f})' in md) or (f'{mean:.3f} (SD {sd:.4f})' in md)
doc_checks = []
names = {'acc': 'Accuracy', 'bacc': 'Balanced accuracy', 'f1': 'Macro-F1', 'auroc': 'Macro-AUROC', 'mcc': 'MCC', 'ece_ts': 'ECE'}
for k in ('dn_fp32', 'rn_fp32', 'dn_int8', 'rn_int8'):
    for m in ('acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts'):
        s = stats[k][m]
        doc_checks.append((LABEL[k], names[m], f"{s['mean']:.3f} (SD {s['sd']:.3f})", present(s['mean'], s['sd'])))
n_ok = sum(1 for c in doc_checks if c[3])

out = dict(method='independent recomputation from saved logits; sample SD (n-1); t-based 95% interval across five seeds', seeds=SEEDS,
           per_config=stats, baseline=baseline, max_abs_difference_vs_pipeline=worst, manuscript_pairs_found=len(pair),
           manuscript_table_checks=[dict(model=a, metric=b, value=c, found_in_manuscript=d) for a, b, c, d in doc_checks])
json.dump(out, open(RES / 'statistical_verification.json', 'w'), indent=1)

L = ['# One-time statistical verification (recomputed from raw logits)', '',
     f'Test set: {baseline["n_test"]} HAM10000 images; most frequent class (mode) {baseline["modal_class"]} = {100 * baseline["modal_class_share"]:.1f}% of images, so a classifier that always predicts it would score accuracy {baseline["majority_baseline_accuracy"]:.3f}, macro-F1 {baseline["majority_baseline_macro_f1"]:.3f}, macro-AUROC 0.500.', '',
     'Five seeds per configuration; SD is the sample SD (n - 1); mode is reported only when a value repeats among the five seeds (otherwise "none": a mode is not defined for five distinct continuous values).', '']
for m in ('acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts'):
    L += [f'## {names[m]}', '', '| Configuration | Mean | Median | Mode | SD | SE | Min | Max | 95% interval (t, df 4) | CV % |', '|---|---|---|---|---|---|---|---|---|---|']
    for k in CONF:
        s = stats[k][m]
        L.append(f"| {LABEL[k]} | {s['mean']:.4f} | {s['median']:.4f} | {s['mode'] if s['mode'] is not None else 'none'} | {s['sd']:.4f} | {s['se']:.4f} | {s['min']:.4f} | {s['max']:.4f} | {s['ci95_low']:.4f} to {s['ci95_high']:.4f} | {s['cv_percent']:.1f} |")
    L.append('')
L += [f'Largest absolute difference between this recomputation and the pipeline (model_comparison.json), over every mean and SD: {worst:.2e}.',
      f'Manuscript check: {n_ok} of {len(doc_checks)} mean-and-SD pairs for the four headline configurations reproduce exactly (three decimals) in the manuscript text.']
bad = [c for c in doc_checks if not c[3]]
if bad: L += ['Not found verbatim: ' + '; '.join(f'{a} {b} {c}' for a, b, c, _ in bad)]
open(RES / 'statistical_verification.md', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L[:6])); print('max diff vs pipeline', worst); print('manuscript pairs reproduced', n_ok, '/', len(doc_checks)); print('not found:', bad)
