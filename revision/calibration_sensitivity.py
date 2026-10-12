"""Checks whether confidence-related conclusions depend on temperature scaling, since TS did not improve
test ECE (Section 4.2). Recomputes uncalibrated confidence (T=1) from the stored logits for all five seeds
and reruns the two correctness-prediction models and the confidence-LSAS correlations that use calibrated
confidence in analysis_xai.py (R['logit_correct'], R['logit_correct_excess'], R['spearman_conf_lsas']),
to confirm the sign/significance pattern is unchanged."""
import json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as st
from common import *

RES = OUT / 'results'


def spear(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 5: return dict(rho=np.nan, p=np.nan, n=int(m.sum()))
    r, p = st.spearmanr(x[m], y[m]); return dict(rho=float(r), p=float(p), n=int(m.sum()))


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


rows = []
per_seed = {}
for seed in SEEDS:
    od = OUT / f'seed{seed}'
    d = pd.read_csv(od / 'per_image.csv')
    lg = np.load(od / 'logits.npz')['test']
    conf_uncal = softmax(lg).max(1)
    assert len(conf_uncal) == len(d)
    d['conf_uncal'] = conf_uncal
    d['logA'] = np.log(d.area224)
    d['e_shuf'] = d.lsas - d.null_shuf_mean
    d['correct_i'] = d.correct.astype(int)
    groups = d.lesion_id.astype('category').cat.codes

    r_cal = spear(d.conf.values, d.lsas.values)
    r_un = spear(d.conf_uncal.values, d.lsas.values)

    m2_cal = smf.logit('correct_i ~ conf + logA + lsas', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
    m2_un = smf.logit('correct_i ~ conf_uncal + logA + lsas', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
    m3_cal = smf.logit('correct_i ~ conf + logA + e_shuf', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})
    m3_un = smf.logit('correct_i ~ conf_uncal + logA + e_shuf', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': groups})

    rec = dict(
        seed=seed,
        rho_conf_lsas_cal=r_cal['rho'], rho_conf_lsas_cal_p=r_cal['p'],
        rho_conf_lsas_uncal=r_un['rho'], rho_conf_lsas_uncal_p=r_un['p'],
        conf_coef_cal=float(m2_cal.params['conf']), conf_p_cal=float(m2_cal.pvalues['conf']),
        conf_coef_uncal=float(m2_un.params['conf_uncal']), conf_p_uncal=float(m2_un.pvalues['conf_uncal']),
        lsas_coef_cal=float(m2_cal.params['lsas']), lsas_p_cal=float(m2_cal.pvalues['lsas']),
        lsas_coef_uncal=float(m2_un.params['lsas']), lsas_p_uncal=float(m2_un.pvalues['lsas']),
        eshuf_coef_cal=float(m3_cal.params['e_shuf']), eshuf_p_cal=float(m3_cal.pvalues['e_shuf']),
        eshuf_coef_uncal=float(m3_un.params['e_shuf']), eshuf_p_uncal=float(m3_un.pvalues['e_shuf']),
        auc_conf_cal=float(st.pointbiserialr(d.correct_i, d.conf)[0]),
        auc_conf_uncal=float(st.pointbiserialr(d.correct_i, d.conf_uncal)[0]),
    )
    rows.append(rec)
    print(f'seed {seed}: conf coef cal p={rec["conf_p_cal"]:.2e} uncal p={rec["conf_p_uncal"]:.2e}; '
          f'lsas coef cal p={rec["lsas_p_cal"]:.3f} uncal p={rec["lsas_p_uncal"]:.3f}', flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / 'calibration_sensitivity.csv', index=False)

summary = dict(
    n_seeds=len(SEEDS),
    conf_sig_cal=int((res.conf_p_cal < 0.05).sum()), conf_sig_uncal=int((res.conf_p_uncal < 0.05).sum()),
    lsas_sig_cal=int((res.lsas_p_cal < 0.05).sum()), lsas_sig_uncal=int((res.lsas_p_uncal < 0.05).sum()),
    eshuf_sig_cal=int((res.eshuf_p_cal < 0.05).sum()), eshuf_sig_uncal=int((res.eshuf_p_uncal < 0.05).sum()),
    conf_coef_same_sign=bool(np.all(np.sign(res.conf_coef_cal) == np.sign(res.conf_coef_uncal))),
    rho_conf_lsas_cal_range=[float(res.rho_conf_lsas_cal.min()), float(res.rho_conf_lsas_cal.max())],
    rho_conf_lsas_uncal_range=[float(res.rho_conf_lsas_uncal.min()), float(res.rho_conf_lsas_uncal.max())],
    max_abs_rho_diff=float((res.rho_conf_lsas_cal - res.rho_conf_lsas_uncal).abs().max()),
    auc_conf_cal_range=[float(res.auc_conf_cal.min()), float(res.auc_conf_cal.max())],
    auc_conf_uncal_range=[float(res.auc_conf_uncal.min()), float(res.auc_conf_uncal.max())],
)
json.dump(summary, open(RES / 'calibration_sensitivity_summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
print('DONE')
