"""Explainability analyses (per seed + reference seed detail): correct-vs-misclassified, lesion-size/LSAS mathematical audit,
faithfulness, stability, ITA / skin-tone proxy, class-wise LSAS.  Reference seed = 42 (first pre-specified seed, not chosen by result)."""
import json, warnings, sys
import numpy as np, pandas as pd, cv2
from scipy import stats as st
import statsmodels.formula.api as smf
from common import *
warnings.filterwarnings('ignore')
RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
REF = 42
ham = pd.read_csv(CACHE / 'ham_meta.csv'); te = ham[ham.split == 'test'].reset_index(drop=True)
MK = np.load(CACHE / 'ham_mask.npy', mmap_mode='r'); masks = np.stack([MK[i] for i in te.cache_idx.values]).astype(np.float32)
ita = pd.read_csv(OUT / 'ita_all.csv')
SEEDS_AVAIL = [s for s in SEEDS if (OUT / f'seed{s}' / 'per_image.csv').exists()]
STRATA = ['small', 'medium', 'large']
size_of = lambda a: np.where(a < 0.05, 'small', np.where(a < 0.15, 'medium', 'large'))


# ---------------------------------------------------------------- helpers
def rbc(x, y):
    """rank-biserial correlation (Kerby 2014): >0 => x tends to be larger than y."""
    x = np.asarray(x); y = np.asarray(y)
    if len(x) == 0 or len(y) == 0: return np.nan
    u = st.mannwhitneyu(x, y, alternative='two-sided').statistic; return 2 * u / (len(x) * len(y)) - 1


def mwu(x, y):
    x = np.asarray(x)[~np.isnan(x)]; y = np.asarray(y)[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2: return dict(n1=len(x), n2=len(y), p=np.nan, rbc=np.nan)
    return dict(n1=len(x), n2=len(y), p=float(st.mannwhitneyu(x, y, alternative='two-sided').pvalue), rbc=float(rbc(x, y)))


def kw(groups):
    g = [np.asarray(v)[~np.isnan(v)] for v in groups if len(v) >= 2]
    if len(g) < 2: return dict(H=np.nan, p=np.nan, eps2=np.nan, k=len(g))
    H, p = st.kruskal(*g); n = sum(len(v) for v in g); return dict(H=float(H), p=float(p), eps2=float(H / (n - 1)), k=len(g), n=n)


def spear(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5: return dict(rho=np.nan, p=np.nan, n=int(m.sum()))
    r, p = st.spearmanr(x[m], y[m]); return dict(rho=float(r), p=float(p), n=int(m.sum()))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z: correlate the residuals of the rank-regressions on rank(z)."""
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z)); rx, ry, rz = [st.rankdata(v[m]) for v in (x, y, z)]
    def res(a, b): return a - np.polyval(np.polyfit(b, a, 1), b)
    r, p = st.pearsonr(res(rx, rz), res(ry, rz)); return dict(rho=float(r), p=float(p), n=int(m.sum()))


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


def cluster_boot_stat(df, fn, B=1000, seed=0, key='lesion_id'):
    rng = np.random.RandomState(seed); df = df.reset_index(drop=True); by = df.groupby(key).indices; ks = list(by); out = []
    for _ in range(B):
        ii = np.concatenate([by[k] for k in rng.choice(len(ks), len(ks), replace=True).tolist() and [ks[t] for t in rng.choice(len(ks), len(ks), replace=True)]])
        try: out.append(fn(df.iloc[ii]))
        except Exception: pass
    out = np.array(out, float); return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def med_iqr(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return dict(n=len(v), median=float(np.median(v)) if len(v) else np.nan, q1=float(np.percentile(v, 25)) if len(v) else np.nan,
                q3=float(np.percentile(v, 75)) if len(v) else np.nan, mean=float(v.mean()) if len(v) else np.nan)


def bca_median_ci(v, B=5000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2: return [np.nan, np.nan]
    r = st.bootstrap((v,), np.median, n_resamples=B, confidence_level=0.95, method='BCa', random_state=seed); return [float(r.confidence_interval.low), float(r.confidence_interval.high)]


# ---------------------------------------------------------------- model-independent: oracle ceiling + saliency-scenario simulation
def minmax(h): return (h - h.min()) / (h.max() - h.min() + 1e-7)
oracle = np.zeros(len(te)); area224 = masks.mean((1, 2))
for i in range(len(te)):
    c7 = cv2.resize(masks[i], (7, 7), interpolation=cv2.INTER_AREA); h = minmax(cv2.resize(c7, (IMG, IMG), interpolation=cv2.INTER_LINEAR))
    oracle[i] = (h * masks[i]).sum() / (h.sum() + 1e-7)
yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32); rng = np.random.RandomState(7)
def blob(cy, cx, s): return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
sim = {}
for nm, fn in {'uniform': lambda i: np.ones((IMG, IMG), np.float32),
               'diffuse_random_location(sigma=60)': lambda i: blob(rng.uniform(40, 184), rng.uniform(40, 184), 60),
               'concentrated_random_location(sigma=12)': lambda i: blob(rng.uniform(20, 204), rng.uniform(20, 204), 12),
               'lesion_centred_concentrated(sigma=12)': lambda i: blob(*(np.argwhere(masks[i] > 0).mean(0)), 12),
               'CAM-resolution_lesion_oracle': None}.items():
    if fn is None: v = oracle
    else: v = np.array([(fn(i) * masks[i]).sum() / (fn(i).sum() + 1e-7) if nm == 'uniform' else None for i in range(len(te))]) if False else None
    if fn is not None:
        v = np.zeros(len(te))
        for i in range(len(te)):
            h = fn(i); v[i] = (h * masks[i]).sum() / (h.sum() + 1e-7)
    sim[nm] = dict(rho_area=float(st.spearmanr(v, area224)[0]),
                   **{f'mean_{s}': float(v[size_of(area224) == s].mean()) for s in STRATA})
sim['n_by_stratum'] = {s: int((size_of(area224) == s).sum()) for s in STRATA}
json.dump(sim, open(RES / 'lsas_scenarios_simulation.json', 'w'), indent=1)
print('scenario simulation', json.dumps(sim, indent=1))


# ---------------------------------------------------------------- per-seed analysis
def analyze(seed, detail=False):
    od = OUT / f'seed{seed}'; d = pd.read_csv(od / 'per_image.csv'); d['oracle'] = oracle
    d = d.merge(ita[['image_id', 'ita_v1', 'ita_v2', 'fst_v1', 'fst_v2']], on='image_id', how='left')
    d['size'] = size_of(d.area224.values); d['pred_cls'] = [CLASSES[i] for i in d.pred]; d['logA'] = np.log(d.area224)
    d['e_shuf'] = d.lsas - d.null_shuf_mean; d['e_shift'] = d.lsas - d.null_shift_mean; d['e_unif'] = d.lsas - d.area224
    d['lift'] = d.lsas / d.area224; d['frac_att'] = d.lsas / d.oracle
    d['z_shift'] = (d.lsas - d.null_shift_mean) / (d.null_shift_sd + 1e-9)
    R = {}; c = d[d.correct]; w = d[~d.correct]
    R['n'] = len(d); R['n_correct'] = len(c); R['n_incorrect'] = len(w)
    # ---- LSAS descriptive
    R['lsas_all'] = med_iqr(d.lsas); R['lsas_correct'] = med_iqr(c.lsas); R['lsas_incorrect'] = med_iqr(w.lsas)
    # ---- (1) correct vs misclassified
    cm = {}
    for v in ['lsas', 'e_shuf', 'e_shift', 'loc_auc', 'frac_att', 'area224', 'conf', 'sts']:
        cm[v] = dict(correct=med_iqr(c[v]), incorrect=med_iqr(w[v]), **mwu(c[v].values, w[v].values))
    R['correct_vs_incorrect'] = cm
    R['spearman_lsas_area'] = dict(all=spear(d.lsas.values, d.area224.values), correct=spear(c.lsas.values, c.area224.values), incorrect=spear(w.lsas.values, w.area224.values))
    R['spearman_conf_lsas'] = dict(all=spear(d.conf.values, d.lsas.values), correct=spear(c.conf.values, c.lsas.values), incorrect=spear(w.conf.values, w.lsas.values),
                                   correct_partial_area=partial_spearman(c.conf.values, c.lsas.values, c.area224.values),
                                   correct_e_shuf=spear(c.conf.values, c.e_shuf.values), all_e_shuf=spear(d.conf.values, d.e_shuf.values))
    try:
        m1 = smf.ols('lsas ~ logA * C(correct)', data=d.assign(correct=d.correct.astype(int))).fit(cov_type='cluster', cov_kwds={'groups': d.lesion_id.astype('category').cat.codes})
        R['interaction_lsas_logA_x_correct'] = dict(coef=float(m1.params.get('logA:C(correct)[T.1]', np.nan)), p=float(m1.pvalues.get('logA:C(correct)[T.1]', np.nan)))
        m2 = smf.logit('correct ~ conf + logA + lsas', data=d.assign(correct=d.correct.astype(int))).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.lesion_id.astype('category').cat.codes})
        R['logit_correct'] = {k: dict(coef=float(m2.params[k]), p=float(m2.pvalues[k])) for k in ['conf', 'logA', 'lsas']}
        m3 = smf.logit('correct ~ conf + logA + e_shuf', data=d.assign(correct=d.correct.astype(int))).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.lesion_id.astype('category').cat.codes})
        R['logit_correct_excess'] = {k: dict(coef=float(m3.params[k]), p=float(m3.pvalues[k])) for k in ['conf', 'logA', 'e_shuf']}
    except Exception as e: R['logit_error'] = str(e)
    # class-wise (predicted class = class explained)
    R['kw_class_correct_only'] = kw([c[c.pred_cls == k].lsas.values for k in CLASSES]); R['kw_class_all'] = kw([d[d.pred_cls == k].lsas.values for k in CLASSES])
    R['kw_class_all_e_shuf'] = kw([d[d.pred_cls == k].e_shuf.values for k in CLASSES]); R['kw_class_correct_e_shuf'] = kw([c[c.pred_cls == k].e_shuf.values for k in CLASSES])
    # ---- (5) lesion-size audit
    S = np.load(od / 'S_full.npy'); n = len(d); rng = np.random.RandomState(500 + seed); B = 2000
    rho_obs = st.spearmanr(d.lsas, d.area224)[0]; Hobs = kw([d[d['size'] == s].lsas.values for s in STRATA])['H']
    rho_null = np.zeros(B); H_null = np.zeros(B)
    for b in range(B):
        pi = rng.permutation(n); fixed = pi == np.arange(n)
        while fixed.any(): pi[fixed] = rng.randint(0, n, fixed.sum()); fixed = pi == np.arange(n)
        v = S[pi, np.arange(n)]; rho_null[b] = st.spearmanr(v, d.area224)[0]; H_null[b] = kw([v[d['size'].values == s] for s in STRATA])['H']
    R['perm_null'] = dict(rho_obs=float(rho_obs), rho_null_mean=float(rho_null.mean()), rho_null_sd=float(rho_null.std()), rho_null_q025=float(np.percentile(rho_null, 2.5)),
                          rho_null_q975=float(np.percentile(rho_null, 97.5)), p_greater=float((np.sum(rho_null >= rho_obs) + 1) / (B + 1)), H_obs=float(Hobs),
                          H_null_mean=float(H_null.mean()), H_null_q975=float(np.percentile(H_null, 97.5)), p_H=float((np.sum(H_null >= Hobs) + 1) / (B + 1)), B=B)
    R['spearman_null_vs_area'] = dict(shuffle=spear(d.null_shuf_mean.values, d.area224.values), shift=spear(d.null_shift_mean.values, d.area224.values), uniform_rho=1.0)
    size = {}
    for v in ['lsas', 'null_shuf_mean', 'null_shift_mean', 'oracle', 'e_shuf', 'e_shift', 'e_unif', 'lift', 'frac_att', 'loc_auc', 'z_shift', 'null_shuf_rank']:
        size[v] = dict(spearman_area=spear(d[v].values, d.area224.values), kw=kw([d[d['size'] == s][v].values for s in STRATA]),
                       kw_correct_only=kw([c[c['size'] == s][v].values for s in STRATA]),
                       **{f'med_{s}': med_iqr(d[d['size'] == s][v])['median'] for s in STRATA})
    R['size_measures'] = size; R['size_n'] = {s: int((d['size'] == s).sum()) for s in STRATA}; R['size_n_correct'] = {s: int((c['size'] == s).sum()) for s in STRATA}
    # excess LSAS: fraction of variance / adjusted models with cluster-robust SEs
    g = d.lesion_id.astype('category').cat.codes
    adj = {}
    for outc in ['lsas', 'e_shuf', 'e_shift', 'loc_auc']:
        try:
            f = smf.ols(f'{outc} ~ logA + C(pred_cls)', data=d).fit(cov_type='cluster', cov_kwds={'groups': g})
            f0 = smf.ols(f'{outc} ~ C(pred_cls)', data=d).fit()
            adj[outc] = dict(coef_logA=float(f.params['logA']), ci=[float(v) for v in f.conf_int().loc['logA']], p=float(f.pvalues['logA']),
                             r2_full=float(f.rsquared), r2_class_only=float(f0.rsquared), partial_r2_logA=float(f.rsquared - f0.rsquared))
        except Exception as e: adj[outc] = dict(error=str(e))
    R['ols_area_adj_class'] = adj
    # nonparametric residualisation: rank-regression of e_shuf on area with LOWESS from the null
    import statsmodels.api as sm
    lo = sm.nonparametric.lowess(d.null_shuf_mean.values, d.area224.values, frac=0.5, return_sorted=False)
    d['resid_lowess_null'] = d.lsas - lo
    R['resid_lowess_null'] = dict(spearman_area=spear(d.resid_lowess_null.values, d.area224.values), kw=kw([d[d['size'] == s].resid_lowess_null.values for s in STRATA]),
                                  **{f'med_{s}': med_iqr(d[d['size'] == s].resid_lowess_null)['median'] for s in STRATA}, mean_all=float(d.resid_lowess_null.mean()))
    # within-stratum enrichment above chance (Wilcoxon signed-rank on e_shuf, per stratum) : is saliency more lesion-aligned than chance?
    R['enrichment_vs_null'] = {s: dict(n=int((d['size'] == s).sum()), median_e_shuf=float(d[d['size'] == s].e_shuf.median()),
                                       p_wilcoxon=float(st.wilcoxon(d[d['size'] == s].e_shuf).pvalue) if (d['size'] == s).sum() > 5 else np.nan) for s in STRATA}
    R['enrichment_vs_null']['all'] = dict(n=n, median_e_shuf=float(d.e_shuf.median()), p_wilcoxon=float(st.wilcoxon(d.e_shuf).pvalue), frac_above_chance=float((d.e_shuf > 0).mean()))
    R['frac_beyond_shift_null_p05'] = float((d.null_shift_p < 0.05).mean())
    # ---- (4) faithfulness
    fa = {}
    for bk in ['mean', 'blur']:
        if f'{bk}_del_cam' not in d: continue
        dd = d.copy(); dd['DG'] = dd[f'{bk}_del_rand'] - dd[f'{bk}_del_cam']; dd['IG'] = dd[f'{bk}_ins_cam'] - dd[f'{bk}_ins_rand']
        dd['DG_les'] = dd[f'{bk}_del_bg_first'] - dd[f'{bk}_del_lesion_first']; dd['IG_les'] = dd[f'{bk}_ins_lesion_first'] - dd[f'{bk}_ins_bg_first']
        dd['DG_lecam'] = dd[f'{bk}_del_lecam'] - dd[f'{bk}_del_cam']
        cc = dd[dd.correct]; ww = dd[~dd.correct]
        fa[bk] = dict(n=len(dd),
                      del_cam=med_iqr(dd[f'{bk}_del_cam']), del_rand=med_iqr(dd[f'{bk}_del_rand']), del_lecam=med_iqr(dd[f'{bk}_del_lecam']),
                      ins_cam=med_iqr(dd[f'{bk}_ins_cam']), ins_rand=med_iqr(dd[f'{bk}_ins_rand']), ins_lecam=med_iqr(dd[f'{bk}_ins_lecam']),
                      del_lesion_first=med_iqr(dd[f'{bk}_del_lesion_first']), del_bg_first=med_iqr(dd[f'{bk}_del_bg_first']),
                      ins_lesion_first=med_iqr(dd[f'{bk}_ins_lesion_first']), ins_bg_first=med_iqr(dd[f'{bk}_ins_bg_first']),
                      DG=med_iqr(dd.DG), IG=med_iqr(dd.IG), DG_les=med_iqr(dd.DG_les), IG_les=med_iqr(dd.IG_les),
                      frac_DG_pos=float((dd.DG > 0).mean()), frac_IG_pos=float((dd.IG > 0).mean()),
                      wilcoxon_DG_p=float(st.wilcoxon(dd.DG).pvalue), wilcoxon_IG_p=float(st.wilcoxon(dd.IG).pvalue),
                      wilcoxon_lecam_p=float(st.wilcoxon(dd.DG_lecam).pvalue), wilcoxon_DG_les_p=float(st.wilcoxon(dd.DG_les).pvalue),
                      DG_correct=med_iqr(cc.DG), DG_incorrect=med_iqr(ww.DG), DG_corr_vs_incorr=mwu(cc.DG.values, ww.DG.values),
                      IG_correct=med_iqr(cc.IG), IG_incorrect=med_iqr(ww.IG), IG_corr_vs_incorr=mwu(cc.IG.values, ww.IG.values),
                      rho_lsas_DG=spear(dd.lsas.values, dd.DG.values), rho_lsas_IG=spear(dd.lsas.values, dd.IG.values),
                      rho_eshuf_DG=spear(dd.e_shuf.values, dd.DG.values), rho_eshuf_IG=spear(dd.e_shuf.values, dd.IG.values),
                      rho_locauc_DG=spear(dd.loc_auc.values, dd.DG.values), rho_locauc_IG=spear(dd.loc_auc.values, dd.IG.values),
                      partial_lsas_DG_area=partial_spearman(dd.lsas.values, dd.DG.values, dd.area224.values),
                      partial_lsas_IG_area=partial_spearman(dd.lsas.values, dd.IG.values, dd.area224.values),
                      rho_conf_DG=spear(dd.conf.values, dd.DG.values), rho_area_DG=spear(dd.area224.values, dd.DG.values), rho_area_IG=spear(dd.area224.values, dd.IG.values))
        q = pd.qcut(dd.e_shuf, 3, labels=['low', 'mid', 'high'])
        fa[bk]['DG_by_eshuf_tertile'] = {k: med_iqr(dd[q == k].DG) for k in ['low', 'mid', 'high']}
        fa[bk]['IG_by_eshuf_tertile'] = {k: med_iqr(dd[q == k].IG) for k in ['low', 'mid', 'high']}
        fa[bk]['kw_DG_tertile'] = kw([dd[q == k].DG.values for k in ['low', 'mid', 'high']])
        fa[bk]['DG_by_size'] = {s: med_iqr(dd[dd['size'] == s].DG) for s in STRATA}; fa[bk]['kw_DG_size'] = kw([dd[dd['size'] == s].DG.values for s in STRATA])
        for k in ['DG', 'IG', 'DG_les', 'IG_les']: d[f'{bk}_{k}'] = dd[k].values
    R['faithfulness'] = fa
    # ---- stability
    s_ = d.dropna(subset=['sts'])
    R['sts'] = dict(**med_iqr(s_.sts), frac_below_05=float((s_.sts < 0.5).mean()), n_below_05=int((s_.sts < 0.5).sum()),
                    rho_lsas=spear(s_.sts.values, s_.lsas.values), rho_eshuf=spear(s_.sts.values, s_.e_shuf.values),
                    rho_DG=spear(s_.sts.values, s_['mean_DG'].values) if 'mean_DG' in s_ else None, rho_conf=spear(s_.sts.values, s_.conf.values),
                    rho_area=spear(s_.sts.values, s_.area224.values), corr_vs_incorr=mwu(s_[s_.correct].sts.values, s_[~s_.correct].sts.values),
                    mc_agree_mean=float(s_.mc_pred_agree.mean()))
    # ---- skin tone
    stn = {}
    for ver in ['v1', 'v2']:
        gcol = f'fst_{ver}'; icol = f'ita_{ver}'; G = ['I-II', 'III-IV', 'V-VI']
        for subset_name, dd in [('correct_only', c), ('all', d)]:
            dd = dd.dropna(subset=[gcol])
            r = dict(n={g: int((dd[gcol] == g).sum()) for g in G}, med={g: med_iqr(dd[dd[gcol] == g].lsas)['median'] for g in G},
                     med_e_shuf={g: med_iqr(dd[dd[gcol] == g].e_shuf)['median'] for g in G},
                     kw_lsas=kw([dd[dd[gcol] == g].lsas.values for g in G]), kw_e_shuf=kw([dd[dd[gcol] == g].e_shuf.values for g in G]),
                     spearman_ita_lsas=spear(dd[icol].values, dd.lsas.values), spearman_ita_eshuf=spear(dd[icol].values, dd.e_shuf.values),
                     spearman_ita_area=spear(dd[icol].values, dd.area224.values))
            pw = []
            for a_, b_ in [('I-II', 'III-IV'), ('I-II', 'V-VI'), ('III-IV', 'V-VI')]:
                pw.append(dict(pair=f'{a_} vs {b_}', **mwu(dd[dd[gcol] == a_].lsas.values, dd[dd[gcol] == b_].lsas.values)))
            adjp = holm([x['p'] for x in pw])
            for x, ap_ in zip(pw, adjp): x['p_holm'] = float(ap_)
            r['pairwise_lsas'] = pw
            try:
                f = smf.ols(f'e_shuf ~ {icol} + logA + C(pred_cls)', data=dd).fit(cov_type='cluster', cov_kwds={'groups': dd.lesion_id.astype('category').cat.codes})
                r['ols_eshuf_adj'] = dict(coef_ita=float(f.params[icol]), ci=[float(v) for v in f.conf_int().loc[icol]], p=float(f.pvalues[icol]))
                f2 = smf.ols(f'lsas ~ {icol}', data=dd).fit(); r['ols_lsas_raw'] = dict(coef_ita=float(f2.params[icol]), r2=float(f2.rsquared))
            except Exception as e: r['ols_error'] = str(e)
            if detail and subset_name == 'correct_only' or detail and subset_name == 'all':
                r['ols_slope_boot_ci'] = cluster_boot_stat(dd, lambda b: np.polyfit(b[icol], b.lsas, 1)[0], B=1000, seed=3)
                r['ols_eshuf_slope_boot_ci'] = cluster_boot_stat(dd, lambda b: np.polyfit(b[icol], b.e_shuf, 1)[0], B=1000, seed=4)
                if subset_name == 'all':
                    r['perf_by_group'] = {}
                    for g in G:
                        q_ = dd[dd[gcol] == g]; k_ = int(q_.correct.sum()); n_ = len(q_)
                        lo, hi = (st.beta.ppf(.025, k_, n_ - k_ + 1) if k_ > 0 else 0.0), (st.beta.ppf(.975, k_ + 1, n_ - k_) if k_ < n_ else 1.0)
                        r['perf_by_group'][g] = dict(n=n_, acc=k_ / max(n_, 1), acc_ci_clopper_pearson=[float(lo), float(hi)], mean_conf=float(q_.conf.mean()),
                                                     abs_gap=float(abs(q_.correct.mean() - q_.conf.mean())))
            stn[f'{ver}_{subset_name}'] = r
    # power: minimum detectable standardised difference V-VI vs I-II at 80% power (two-sided a=.05, normal approx)
    n1 = int((c.fst_v1 == 'I-II').sum()); n2 = int((c.fst_v1 == 'V-VI').sum()); d80 = (1.96 + 0.8416) * np.sqrt(1 / n1 + 1 / n2)
    stn['mde_correct_only'] = dict(n_I_II=n1, n_V_VI=n2, cohen_d_80pct=float(d80), lsas_sd=float(c.lsas.std()), lsas_units=float(d80 * c.lsas.std()))
    R['skin'] = stn
    # ---- class-wise table (for detail)
    if detail:
        tab = []
        for k in CLASSES:
            q_ = c[c.pred_cls == k]; qa = d[d.pred_cls == k]; qw = d[(d.pred_cls == k) & (~d.correct)]
            tab.append(dict(cls=k, n_correct=len(q_), med_correct=q_.lsas.median(), ci_lo=bca_median_ci(q_.lsas.values)[0], ci_hi=bca_median_ci(q_.lsas.values)[1],
                            n_incorrect_pred=len(qw), med_incorrect=qw.lsas.median() if len(qw) else np.nan, n_all=len(qa), med_all=qa.lsas.median(), med_e_shuf_all=qa.e_shuf.median(),
                            med_area_all=qa.area224.median()))
        pd.DataFrame(tab).to_csv(RES / 'ref_class_table.csv', index=False)
        # pairwise class comparisons (Holm) on correct-only LSAS (original) and on all e_shuf
        pw = []
        for i in range(7):
            for j in range(i + 1, 7):
                a_, b_ = CLASSES[i], CLASSES[j]
                pw.append(dict(pair=f'{a_} vs {b_}', lsas=mwu(c[c.pred_cls == a_].lsas.values, c[c.pred_cls == b_].lsas.values),
                               e_shuf=mwu(d[d.pred_cls == a_].e_shuf.values, d[d.pred_cls == b_].e_shuf.values)))
        R['class_pairwise'] = pw
        d.to_csv(RES / 'ref_per_image_enriched.csv', index=False)
    return R, d


allR = {}; det = None
for s in SEEDS_AVAIL:
    R, dd = analyze(s, detail=(s == REF)); allR[s] = R
    if s == REF: det = dd
    print('seed', s, 'done: rho_obs', round(R['perm_null']['rho_obs'], 3), 'null', round(R['perm_null']['rho_null_mean'], 3),
          'e_shuf~area', round(R['size_measures']['e_shuf']['spearman_area']['rho'], 3), flush=True)
json.dump(allR, open(RES / 'xai_all_seeds.json', 'w'), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
