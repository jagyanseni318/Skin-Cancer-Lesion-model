"""Explanation-audit statistics computed with identical code for both architectures and every available seed
(DenseNet121: out/seed{S}; ResNet50: out/seed{S}_resnet50).  Reuses the exact definitions of analysis_xai.py
(DG/IG, excess LSAS, permutation null) but is computed here directly because importing analysis_xai.py would re-run its full analysis."""
import json
import numpy as np, pandas as pd
from scipy import stats as st
from common import *

RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
ARCH_DIR = {'densenet121': lambda s: OUT / f'seed{s}', 'resnet50': lambda s: OUT / f'seed{s}_resnet50'}


def spear(x, y):
    m = ~(np.isnan(x) | np.isnan(y)); r, p = st.spearmanr(x[m], y[m]); return dict(rho=float(r), p=float(p))


def analyze(arch, seed):
    od = ARCH_DIR[arch](seed); d = pd.read_csv(od / 'per_image.csv'); n = len(d)
    d['e_shuf'] = d.lsas - d.null_shuf_mean; d['DG'] = d.mean_del_rand - d.mean_del_cam; d['IG'] = d.mean_ins_cam - d.mean_ins_rand
    S = np.load(od / 'S_full.npy'); rho_obs = st.spearmanr(d.lsas, d.area224)[0]
    rng = np.random.RandomState(500 + seed); B = 2000; rho_null = np.zeros(B)
    for b in range(B):
        pi = rng.permutation(n); fixed = pi == np.arange(n)
        while fixed.any(): pi[fixed] = rng.randint(0, n, fixed.sum()); fixed = pi == np.arange(n)
        rho_null[b] = st.spearmanr(S[pi, np.arange(n)], d.area224)[0]
    sts = d.sts.dropna()
    rec = dict(arch=arch, seed=seed, n=n, acc=float(d.correct.mean()),
               rho_lsas_area=float(rho_obs), rho_null_mean=float(rho_null.mean()),
               p_null_greater=float((np.sum(rho_null >= rho_obs) + 1) / (B + 1)), p_null_less=float((np.sum(rho_null <= rho_obs) + 1) / (B + 1)),
               e_shuf_median=float(d.e_shuf.median()), e_shuf_wilcoxon_p=float(st.wilcoxon(d.e_shuf).pvalue), e_shuf_frac_pos=float((d.e_shuf > 0).mean()),
               rho_eshuf_area=spear(d.e_shuf.values, d.area224.values),
               DG_median=float(d.DG.median()), DG_frac_pos=float((d.DG > 0).mean()), DG_wilcoxon_p=float(st.wilcoxon(d.DG).pvalue),
               IG_median=float(d.IG.median()), IG_frac_pos=float((d.IG > 0).mean()), IG_wilcoxon_p=float(st.wilcoxon(d.IG).pvalue),
               rho_lsas_IG=spear(d.lsas.values, d.IG.values), rho_eshuf_IG=spear(d.e_shuf.values, d.IG.values),
               sts_median=float(sts.median()), sts_frac_below_05=float((sts < 0.5).mean()),
               ins_own_median=float(d.mean_ins_cam.median()), ins_rand_median=float(d.mean_ins_rand.median()))
    f = od / 'null_faithfulness.csv'
    if f.exists():
        nf = pd.read_csv(f)
        rec.update(EIG_median=float(nf.mean_EIG.median()), EIG_frac_pos=float((nf.mean_EIG > 0).mean()), EIG_wilcoxon_p=float(st.wilcoxon(nf.mean_EIG).pvalue),
                   EDG_median=float(nf.mean_EDG.median()), EDG_frac_pos=float((nf.mean_EDG > 0).mean()), EDG_wilcoxon_p=float(st.wilcoxon(nf.mean_EDG).pvalue),
                   ins_null_median=float(nf.mean_ins_nullcam.median()))
    return rec


rows = []
for arch in ARCH_DIR:
    for s in SEEDS:
        if (ARCH_DIR[arch](s) / 'per_image.csv').exists() and (ARCH_DIR[arch](s) / 'S_full.npy').exists():
            rows.append(analyze(arch, s)); r = rows[-1]
            print(f"{arch} seed {s}: rho={r['rho_lsas_area']:.3f} null={r['rho_null_mean']:.3f} e_med={r['e_shuf_median']:+.4f} IG_med={r['IG_median']:+.4f}" +
                  (f" EIG={r['EIG_median']:+.4f} EDG={r['EDG_median']:+.4f}" if 'EIG_median' in r else ''), flush=True)
json.dump(rows, open(RES / 'arch_robustness.json', 'w'), indent=1, default=float)
print('DONE', len(rows))
