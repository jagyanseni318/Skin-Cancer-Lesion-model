"""Aggregates the null-controlled faithfulness metric (null_faithfulness.py) across all five seeds:
descriptive statistics, Wilcoxon tests against zero, and correlation with excess LSAS and with the
existing insertion/deletion gains, per seed and across seeds."""
import json
import numpy as np, pandas as pd
from scipy import stats as st
from common import *

RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
rows = []
for s in SEEDS:
    f = OUT / f'seed{s}' / 'null_faithfulness.csv'
    if not f.exists(): continue
    nf = pd.read_csv(f); pi = pd.read_csv(OUT / f'seed{s}' / 'per_image.csv')
    d = nf.merge(pi[['image_id', 'lsas', 'null_shuf_mean', 'correct', 'area224', 'mean_ins_cam', 'mean_del_cam', 'mean_ins_rand', 'mean_del_rand']], on='image_id')
    d['e_shuf'] = d.lsas - d.null_shuf_mean
    r = {}
    r['seed'] = s; r['n'] = len(d)
    for col, lab in [('mean_EIG', 'EIG'), ('mean_EDG', 'EDG')]:
        v = d[col].values
        r[f'{lab}_median'] = float(np.median(v)); r[f'{lab}_q1'] = float(np.percentile(v, 25)); r[f'{lab}_q3'] = float(np.percentile(v, 75))
        r[f'{lab}_frac_pos'] = float((v > 0).mean())
        r[f'{lab}_wilcoxon_p'] = float(st.wilcoxon(v).pvalue)
        rr, pp = st.spearmanr(d.e_shuf.values, v); r[f'{lab}_rho_eshuf'] = float(rr); r[f'{lab}_rho_eshuf_p'] = float(pp)
        ra, pa = st.spearmanr(d.area224.values, v); r[f'{lab}_rho_area'] = float(ra); r[f'{lab}_rho_area_p'] = float(pa)
    # sanity ordering: null-cam order should sit strictly between the model's own CAM order and pure random order
    r['ins_cam_med'] = float(d.mean_ins_cam.median()); r['ins_nullcam_med'] = float(d.mean_ins_nullcam.median()); r['ins_rand_med'] = float(d.mean_ins_rand.median())
    r['del_cam_med'] = float(d.mean_del_cam.median()); r['del_nullcam_med'] = float(d.mean_del_nullcam.median()); r['del_rand_med'] = float(d.mean_del_rand.median())
    # correct vs misclassified
    c = d[d.correct]; w = d[~d.correct]
    for col, lab in [('mean_EIG', 'EIG'), ('mean_EDG', 'EDG')]:
        if len(w) > 5:
            u = st.mannwhitneyu(c[col], w[col], alternative='two-sided')
            r[f'{lab}_correct_median'] = float(c[col].median()); r[f'{lab}_incorrect_median'] = float(w[col].median()); r[f'{lab}_corr_vs_incorr_p'] = float(u.pvalue)
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(RES / 'null_faith_per_seed.csv', index=False)
summary = {}
for lab in ['EIG', 'EDG']:
    summary[lab] = dict(median_mean=float(df[f'{lab}_median'].mean()), median_sd=float(df[f'{lab}_median'].std(ddof=1)) if len(df) > 1 else 0.0,
                        median_range=[float(df[f'{lab}_median'].min()), float(df[f'{lab}_median'].max())],
                        frac_pos_range=[float(df[f'{lab}_frac_pos'].min()), float(df[f'{lab}_frac_pos'].max())],
                        n_seeds_wilcoxon_sig=int((df[f'{lab}_wilcoxon_p'] < 0.05).sum()), n_seeds=len(df),
                        rho_eshuf_range=[float(df[f'{lab}_rho_eshuf'].min()), float(df[f'{lab}_rho_eshuf'].max())],
                        n_seeds_rho_eshuf_sig=int((df[f'{lab}_rho_eshuf_p'] < 0.05).sum()),
                        rho_area_range=[float(df[f'{lab}_rho_area'].min()), float(df[f'{lab}_rho_area'].max())])
json.dump(summary, open(RES / 'null_faith_summary.json', 'w'), indent=1)
print(df.round(4).to_string()); print(json.dumps(summary, indent=1))
