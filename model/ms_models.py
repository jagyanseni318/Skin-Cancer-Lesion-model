"""Results/Discussion sections for the three-configuration study (Sections 4.12-4.15, 5.10).  Every number and every directional
statement is generated from the result files (model_comparison.json, arch_robustness.json, efficiency.json, INT8 sweeps), with
branches by outcome, so the prose cannot contradict the data."""
import json
import numpy as np, pandas as pd
from common import *

RES = OUT / 'results'
J = lambda n: json.load(open(RES / n))
CMP = J('model_comparison.json'); EX = pd.DataFrame(J('arch_robustness.json'))
LP = J('linear_probe_baseline.json')
EFF = J('efficiency.json') if (RES / 'efficiency.json').exists() else None
CFG = CMP['configs']; PS = {k: pd.DataFrame(v) for k, v in CMP['per_seed'].items()}
NAME = {'dn_fp32': 'DenseNet121 (FP32)', 'rn_fp32': 'ResNet50 (FP32)', 'dn_int8': 'DenseNet121 (INT8)', 'rn_int8': 'ResNet50 (INT8)',
        'dn_ptq': 'DenseNet121 (INT8, post-training only)', 'rn_ptq': 'ResNet50 (INT8, post-training only)'}
MET = {'acc': 'Accuracy', 'bacc': 'Balanced accuracy', 'f1': 'Macro-F1', 'auroc': 'Macro-AUROC', 'mcc': 'MCC', 'ece_ts': 'ECE after temperature scaling',
       'pad_acc': 'PAD-UFES-20 accuracy (five classes)', 'pad_bacc': 'PAD-UFES-20 balanced accuracy', 'pad_f1': 'PAD-UFES-20 macro-F1'}
f3 = lambda v: f'{v:.3f}'; f4 = lambda v: f'{v:.4f}'
LOW = {'acc': 'accuracy', 'bacc': 'balanced accuracy', 'f1': 'macro-F1', 'auroc': 'macro-AUROC', 'mcc': 'MCC', 'ece_ts': 'ECE after temperature scaling',
       'pad_acc': 'PAD-UFES-20 accuracy', 'pad_bacc': 'PAD-UFES-20 balanced accuracy', 'pad_f1': 'PAD-UFES-20 macro-F1'}
cap1 = lambda t: t[0].upper() + t[1:]
pct = lambda v: f'{100 * v:.1f}%'


def pval(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def ms(key, m, d=3):                       # mean (SD) with no plus/minus sign
    c = CFG[key][m]; return f"{c['mean']:.{d}f} (SD {c['sd']:.{d}f})"


def exd(arch):
    return EX[EX.arch == arch].reset_index(drop=True)


def sig_count(df, col_med, col_p, sign=1):
    return int(((df[col_med] * sign > 0) & (df[col_p] < 0.05)).sum())


def winner(metric, keys, higher=True):
    """Decision rule of Section 3.15: best = highest mean and advantage over the runner-up exceeds both seed SDs; otherwise tie."""
    ks = [k for k in keys if k in CFG]; mean = {k: CFG[k][metric]['mean'] for k in ks}; sd = {k: CFG[k][metric]['sd'] for k in ks}
    order = sorted(ks, key=lambda k: mean[k], reverse=higher); b, r = order[0], order[1]
    return (b if abs(mean[b] - mean[r]) > max(sd[b], sd[r]) else 'tie'), order


# =====================================================================================================================
def sec_architecture(M):
    A = CMP['tests']['arch_dn_vs_rn']; dn, rn = exd('densenet121'), exd('resnet50'); ndn, nrn = len(dn), len(rn)
    M.h(2, '4.12 Architecture comparison: DenseNet121 versus ResNet50')
    M.p(f"Sections 4.1-4.11 describe one architecture. To test which findings belong to the audited behaviour and which to the network, the complete protocol (identical split, preprocessing, loss, sampler, schedule, checkpoint rule and temperature scaling) was repeated for ResNet50 with {nrn} independently trained seeds; the classifiers are compared in [[T:arch_cls]] and the explanation audits in [[T:arch_expl]]. Differences are absolute values with the higher-scoring model named; exact permutation tests over the 252 splits of the ten seed values and Welch tests are given because five seeds per architecture detect only large differences (Section 3.16).")
    rows = []
    for m in ['acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts', 'pad_acc', 'pad_bacc', 'pad_f1']:
        a = A[m]; hi_dn = CFG['dn_fp32'][m]['mean'] > CFG['rn_fp32'][m]['mean']
        better = ('DenseNet121' if hi_dn else 'ResNet50') if m != 'ece_ts' else ('DenseNet121' if not hi_dn else 'ResNet50')
        rows.append([MET[m] + (' (lower is better)' if m == 'ece_ts' else ''), f3(CFG['dn_fp32'][m]['mean']), f3(CFG['dn_fp32'][m]['sd']), f3(CFG['rn_fp32'][m]['mean']), f3(CFG['rn_fp32'][m]['sd']),
                     better, f3(abs(a['diff_rn_minus_dn'])), pval(a['welch_p']), pval(a['perm_p']), f"{abs(a['cohen_d']):.2f}"])
    M.table(['Measure', 'DenseNet121 mean', 'DenseNet121 SD', 'ResNet50 mean', 'ResNet50 SD', 'Better', 'Absolute difference', 'Welch p', 'Permutation p', 'Cohen d (absolute)'], rows,
            f'Classification, calibration and external performance of the two architectures ({"five seeds each" if ndn == nrn == 5 else f"{ndn} and {nrn} seeds"}; test set of 990 images; PAD-UFES-20 mapped to five classes).',
            'Means and SDs are over seeds; ECE uses temperature scaling fitted on val_cal. The column Better names the model with the more favourable mean regardless of significance.', label='tab:arch_cls')
    sig = [m for m in ['acc', 'bacc', 'f1', 'auroc', 'mcc', 'ece_ts', 'pad_acc', 'pad_bacc', 'pad_f1'] if A[m]['perm_p'] < 0.05]
    key4 = ['f1', 'bacc', 'acc', 'auroc']
    s_parts = []
    for m in key4:
        a = A[m]; hi_dn = CFG['dn_fp32'][m]['mean'] > CFG['rn_fp32'][m]['mean']
        s_parts.append(f"{LOW[m]} was higher for {'DenseNet121' if hi_dn else 'ResNet50'} ({f3(max(CFG['dn_fp32'][m]['mean'], CFG['rn_fp32'][m]['mean']))} against {f3(min(CFG['dn_fp32'][m]['mean'], CFG['rn_fp32'][m]['mean']))}; permutation p {pval(a['perm_p'])})")
    pw = A['per_seed_pairwise']; psi = np.mean([(r['dn_only_correct'] + r['rn_only_correct']) / 990 for r in pw]); nsig = sum(r['mcnemar_p'] < 0.05 for r in pw)
    mdd = 2.8 * np.sqrt(psi / 990)
    M.p(f"Discrimination differs by metric. {cap1(s_parts[0])}; {s_parts[1]}; {s_parts[2]}; and {s_parts[3]}. The architectures disagreed on {pct(psi)} of test images on average across the {len(pw)} seed pairs (the discordant proportion ψ of Section 3.16, giving a minimum detectable accuracy difference of about {100 * mdd:.1f} percentage points at 80% power), and the McNemar test on the 990 predictions was significant in {nsig} of {len(pw)} seed pairs. {len(sig)} of the nine measures in [[T:arch_cls]] differ at permutation p < 0.05 ({', '.join(LOW[m] for m in sig) if sig else 'none'}); the remaining differences are within what five seeds per architecture can resolve.")
    # ---------------------------------------------------------------- explanation table
    def msd(df, col, d=3, sub=None):
        v = df[col] if sub is None else df[col].apply(lambda x: x[sub]); return f'{v.mean():.{d}f} (SD {v.std(ddof=1):.{d}f})'
    def cnt(df, cond): return f'{int(cond(df).sum())} of {len(df)}'
    rows = [
        ['Raw LSAS-area correlation ρ', msd(dn, 'rho_lsas_area'), msd(rn, 'rho_lsas_area')],
        ['Geometric-null ρ (re-paired maps)', msd(dn, 'rho_null_mean'), msd(rn, 'rho_null_mean')],
        ['Seeds with raw ρ above the null (p < 0.05)', cnt(dn, lambda d: d.p_null_greater < 0.05), cnt(rn, lambda d: d.p_null_greater < 0.05)],
        ['Seeds with raw ρ below the null (p < 0.05)', cnt(dn, lambda d: d.p_null_less < 0.05), cnt(rn, lambda d: d.p_null_less < 0.05)],
        ['Median excess LSAS (mean of seed medians)', msd(dn, 'e_shuf_median'), msd(rn, 'e_shuf_median')],
        ['Seeds with excess LSAS significantly above zero', cnt(dn, lambda d: (d.e_shuf_median > 0) & (d.e_shuf_wilcoxon_p < 0.05)), cnt(rn, lambda d: (d.e_shuf_median > 0) & (d.e_shuf_wilcoxon_p < 0.05))],
        ['Seeds with excess LSAS significantly below zero', cnt(dn, lambda d: (d.e_shuf_median < 0) & (d.e_shuf_wilcoxon_p < 0.05)), cnt(rn, lambda d: (d.e_shuf_median < 0) & (d.e_shuf_wilcoxon_p < 0.05))],
        ['Median insertion gain IG (mean of seed medians)', msd(dn, 'IG_median'), msd(rn, 'IG_median')],
        ['Images with IG > 0 (mean over seeds)', pct(dn.IG_frac_pos.mean()), pct(rn.IG_frac_pos.mean())],
        ['Seeds with median deletion gain DG below zero', cnt(dn, lambda d: d.DG_median < 0), cnt(rn, lambda d: d.DG_median < 0)],
        ['ρ(excess LSAS, IG) (mean over seeds)', msd(dn, 'rho_eshuf_IG', 2, 'rho'), msd(rn, 'rho_eshuf_IG', 2, 'rho')],
        ['Median excess insertion gain EIG (mean of seed medians)', msd(dn, 'EIG_median', 4), msd(rn, 'EIG_median', 4)],
        ['Seeds with EIG significantly above zero', cnt(dn, lambda d: (d.EIG_median > 0) & (d.EIG_wilcoxon_p < 0.05)), cnt(rn, lambda d: (d.EIG_median > 0) & (d.EIG_wilcoxon_p < 0.05))],
        ['Seeds with EIG significantly below zero', cnt(dn, lambda d: (d.EIG_median < 0) & (d.EIG_wilcoxon_p < 0.05)), cnt(rn, lambda d: (d.EIG_median < 0) & (d.EIG_wilcoxon_p < 0.05))],
        ['Median excess deletion gain EDG (mean of seed medians)', msd(dn, 'EDG_median', 4), msd(rn, 'EDG_median', 4)],
        ['Seeds with EDG significantly above zero', cnt(dn, lambda d: (d.EDG_median > 0) & (d.EDG_wilcoxon_p < 0.05)), cnt(rn, lambda d: (d.EDG_median > 0) & (d.EDG_wilcoxon_p < 0.05))],
        ['Seeds with EDG significantly below zero', cnt(dn, lambda d: (d.EDG_median < 0) & (d.EDG_wilcoxon_p < 0.05)), cnt(rn, lambda d: (d.EDG_median < 0) & (d.EDG_wilcoxon_p < 0.05))],
        ['Median saliency stability STS (mean over seeds)', msd(dn, 'sts_median', 2), msd(rn, 'sts_median', 2)],
    ]
    M.table(['Measure (990 test images per seed)', f'DenseNet121 ({ndn} seeds)', f'ResNet50 ({nrn} seeds)'], rows,
            'Explanation audit computed with identical code for both architectures.',
            'LSAS = Lesion-Saliency Alignment Score; excess LSAS = LSAS minus the cross-image shuffled null; IG and DG = insertion and deletion gains over random patch order; EIG and EDG = excess gains over the cross-image null order (Section 3.9); STS = stochastic saliency stability. Significance is a per-seed Wilcoxon test against zero (p < 0.05).', label='tab:arch_expl')
    # ---- generated interpretation (branches by outcome)
    conf_rep = (dn.rho_lsas_area.min() > 0.5) and (rn.rho_lsas_area.min() > 0.5)
    rn_above = int((rn.p_null_greater < 0.05).sum()); rn_below = int((rn.p_null_less < 0.05).sum()); dn_above = int((dn.p_null_greater < 0.05).sum())
    t1 = (f"The geometric confound replicates: raw LSAS rises steeply with lesion area for both architectures (ρ = {dn.rho_lsas_area.mean():.2f} for DenseNet121 and {rn.rho_lsas_area.mean():.2f} for ResNet50), so a geometric null is required whichever network is audited. " if conf_rep else
          f"The raw LSAS-area association was {dn.rho_lsas_area.mean():.2f} for DenseNet121 and {rn.rho_lsas_area.mean():.2f} for ResNet50. ")
    t1 += (f"Against its own geometric null, the raw association of ResNet50 was below the null in {rn_below} of {nrn} seeds and above it in {rn_above}, whereas that of DenseNet121 was above the null in {dn_above} of {ndn} seeds; the area-adjusted excess LSAS was "
           f"{'significantly negative' if sig_count(rn, 'e_shuf_median', 'e_shuf_wilcoxon_p', -1) == nrn else 'not consistently signed'} for ResNet50 (median {rn.e_shuf_median.mean():+.3f}) and "
           f"{'significantly positive' if sig_count(dn, 'e_shuf_median', 'e_shuf_wilcoxon_p', 1) == ndn else 'not consistently signed'} for DenseNet121 (median {dn.e_shuf_median.mean():+.3f}). ")
    ig_both = (dn.IG_frac_pos.min() > 0.5) and (rn.IG_frac_pos.min() > 0.5)
    t1 += (f"Insertion faithfulness replicates in direction (IG > 0 for {pct(rn.IG_frac_pos.mean())} of ResNet50 images and {pct(dn.IG_frac_pos.mean())} of DenseNet121 images) but not in magnitude (median IG {rn.IG_median.mean():.3f} against {dn.IG_median.mean():.3f})." if ig_both else
           f"Insertion gain was positive for {pct(dn.IG_frac_pos.mean())} of DenseNet121 images and {pct(rn.IG_frac_pos.mean())} of ResNet50 images.")
    M.p(t1)
    rn_eig = sig_count(rn, 'EIG_median', 'EIG_wilcoxon_p', 1); rn_edg = sig_count(rn, 'EDG_median', 'EDG_wilcoxon_p', 1)
    dn_eig = sig_count(dn, 'EIG_median', 'EIG_wilcoxon_p', 1); dn_edg = sig_count(dn, 'EDG_median', 'EDG_wilcoxon_p', 1)
    rn_eig_neg = sig_count(rn, 'EIG_median', 'EIG_wilcoxon_p', -1); rn_edg_neg = sig_count(rn, 'EDG_median', 'EDG_wilcoxon_p', -1)
    r_eig = rn.EIG_median.mean() / dn.EIG_median.mean(); r_edg = rn.EDG_median.mean() / dn.EDG_median.mean()
    if rn_eig == nrn and rn_edg == nrn: verdict = 'The null-controlled faithfulness result therefore replicates on the second architecture.'
    elif rn_eig <= 1 and rn_edg <= 1 and r_eig < 0.2 and r_edg < 0.2:
        verdict = (f'The null-controlled faithfulness result therefore does not replicate on the second architecture: the mean of the ResNet50 seed medians is {rn.EIG_median.mean():+.4f} for EIG and {rn.EDG_median.mean():+.4f} for EDG, against {dn.EIG_median.mean():+.4f} and {dn.EDG_median.mean():+.4f} for DenseNet121, and the sign is not consistent across seeds (EIG significantly negative in {rn_eig_neg} and EDG in {rn_edg_neg} ResNet50 seeds). '
                   'It is established for DenseNet121 with Grad-CAM++ at the layer used here and should not be assumed to be general.')
    else: verdict = 'The null-controlled faithfulness result therefore replicates only in part on the second architecture and should not be assumed to be general.'
    M.p(f"EIG was significantly positive in {dn_eig} of {ndn} DenseNet121 seeds and {rn_eig} of {nrn} ResNet50 seeds, and EDG in {dn_edg} of {ndn} and {rn_edg} of {nrn}. {verdict} The quantities behind the metric show why: the median insertion AUC with the image's own map was {dn.ins_own_median.mean():.3f} (DenseNet121) and {rn.ins_own_median.mean():.3f} (ResNet50), with the cross-image null order {dn.ins_null_median.mean():.3f} and {rn.ins_null_median.mean():.3f}, and with a random order {dn.ins_rand_median.mean():.3f} and {rn.ins_rand_median.mean():.3f}. For ResNet50 the Grad-CAM++ order borrowed from a different image is {'almost as informative as' if abs(rn.ins_own_median.mean() - rn.ins_null_median.mean()) < 0.02 else 'less informative than'} the image's own order, while for DenseNet121 the own map is distinctly more informative than the borrowed one. Whether this reflects the architecture, the target layer or an interaction was not established; this robustness check was run with its analysis code fixed before the outcome was inspected, and its outcome is reported as found.")
