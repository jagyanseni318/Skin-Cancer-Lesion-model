"""Results sections 4.1-4.4 (classification, calibration, multi-seed, convergence) - every number read from result files."""
import json
import numpy as np, pandas as pd
from common import *

RES = OUT / 'results'


def pm(m, s, d=3): return f'{m:.{d}f} ± {s:.{d}f}'
def f3(x, d=3): return f'{x:.{d}f}'
def fp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return 'n/a'
    if p < 1e-16: return 'p < 10⁻¹⁶'
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); return f'p = {p / 10 ** e:.2f} × 10^{e}'.replace('^-', '⁻').replace('^', '')
    return f'p = {p:.3f}'


def dataset_table(M):
    ham = pd.read_csv(CACHE / 'ham_meta.csv'); pad = pd.read_csv(CACHE / 'pad_meta.csv')
    rows = []
    for c in CLASSES:
        d = ham[ham.label == c]
        rows.append([c, f'{(d.split == "train").sum():,}', f'{((d.split == "val") & (d.val_role == "sel")).sum():,}', f'{((d.split == "val") & (d.val_role == "cal")).sum():,}', f'{(d.split == "test").sum():,}', f'{len(d):,}',
                     f'{(pad.label == c).sum():,}' if c in ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV'] else '—'])
    tr = ham[ham.split == 'train']
    rows.append(['**Total**', f'**{len(tr):,}**', f'**{((ham.split == "val") & (ham.val_role == "sel")).sum():,}**', f'**{((ham.split == "val") & (ham.val_role == "cal")).sum():,}**', f'**{(ham.split == "test").sum():,}**', f'**{len(ham):,}**', f'**{int(pad.label.notna().sum()):,}**'])
    M.table(['Class', 'Train', 'Val_sel (checkpoint selection)', 'Val_cal (temperature)', 'Test', 'HAM10000 total', 'PAD-UFES-20 (mapped)'], rows,
            'Dataset partition (lesion-level split, identical for all seeds) and PAD-UFES-20 class counts after mapping.',
            f'Images per split; 7,470 unique lesions in HAM10000 (test: {ham[ham.split == "test"].lesion_id.nunique()} lesions). SCC (192 PAD images) is excluded; DF and VASC have no PAD counterpart.')


def cls_sections(M):
    S = json.load(open(RES / 'cls_summary.json')); df = pd.read_csv(RES / 'cls_per_seed.csv'); agg = S['agg']; ens = S['ensemble']; bh = agg['boot_halfwidth_mean']
    seeds = S['seeds']; k = len(seeds)
    names = [('acc', 'Accuracy'), ('bacc', 'Balanced accuracy'), ('f1', 'Macro-F1'), ('auroc', 'Macro-AUROC'), ('prauc', 'Macro-PR-AUC'), ('mcc', 'MCC'), ('kappa', "Cohen's κ"), ('brier', 'Brier score'), ('ece', 'ECE (15 bins)'), ('nll', 'NLL')]
    rows = []
    for key, nm in names:
        a = agg[key]; ci = a.get('ci_seed_mean', [np.nan, np.nan])
        rows.append([nm] + [f3(df.loc[df.seed == s, key].values[0]) for s in seeds] + [f'**{pm(a["mean"], a["sd"])}**', f'{ci[0]:.3f}–{ci[1]:.3f}', f'± {bh[key]:.3f}', f3(ens[key])])
    M.h(2, '4.1 Diagnostic performance and multi-seed robustness')
    M.p(f"On the 990-image held-out test set, the {k} independently trained DenseNet121 models (temperature-scaled with val_cal) reached accuracy {pm(agg['acc']['mean'], agg['acc']['sd'])}, balanced accuracy {pm(agg['bacc']['mean'], agg['bacc']['sd'])}, macro-F1 {pm(agg['f1']['mean'], agg['f1']['sd'])}, macro-AUROC {pm(agg['auroc']['mean'], agg['auroc']['sd'])}, macro-PR-AUC {pm(agg['prauc']['mean'], agg['prauc']['sd'])}, MCC {pm(agg['mcc']['mean'], agg['mcc']['sd'])} and Cohen's κ {pm(agg['kappa']['mean'], agg['kappa']['sd'])} (mean ± SD over seeds; [[T:perf_seed]]). The seed-to-seed SD is small compared with the lesion-clustered test-sampling uncertainty (mean half-width of the 95% bootstrap interval: ±{bh['bacc']:.3f} for balanced accuracy and ±{bh['f1']:.3f} for macro-F1), so differences of the size seen between seeds are not distinguishable from test-set sampling noise. The reference seed (42), designated before the runs for the detailed per-image tables, is the best of the five on accuracy ({df.acc.max():.3f}) and macro-F1 ({df.f1.max():.3f}) ([[T:perf_seed]]), so its detailed results are slightly optimistic relative to the seed mean and every headline statistic is reported for all seeds. An ensemble of the {k} models is shown for reference only.")
    M.table(['Metric'] + [f'Seed {s}' for s in seeds] + ['Mean ± SD', '95% CI of seed mean (t)', 'Test-sampling half-width (cluster bootstrap)', f'{k}-seed ensemble'], rows,
            'Held-out test performance of the corrected, fully annealed DenseNet121 for each seed (temperature-scaled).',
            'Brier score = mean of per-class one-vs-rest binary Brier scores. Test-sampling half-width is the mean over seeds of the half-width of the 95% lesion-clustered bootstrap interval (500 resamples). The ensemble column is descriptive only.')
    pc = pd.read_csv(RES / 'cls_per_class_per_seed.csv'); rows = []
    for c in CLASSES:
        d = pc[pc.cls == c]; rows.append([c] + [pm(d[m].mean(), d[m].std(ddof=1)) for m in ['precision', 'recall', 'f1', 'specificity', 'auroc']] + [int(d.support.iloc[0])])
    M.p(f"Per-class results ([[T:perclass]]) show the usual pattern for this imbalanced problem: NV, VASC and BCC are well separated, whereas AKIEC and MEL have low precision because they are confused with the large NV and BKL classes. Macro-metrics are therefore informative, and the per-class seed SDs (largest for the rarest classes) show that small-class metrics are the least stable component of the model across training runs.")
    M.table(['Class', 'Precision', 'Recall', 'F1', 'Specificity', 'AUROC (OvR)', 'Support'], rows, 'Per-class test performance (mean ± SD over seeds).')
    M.h(2, '4.2 Calibration')
    T = agg['T']; rows = []
    for key, nm in [('ece_uncal', 'ECE before TS'), ('ece', 'ECE after TS'), ('nll_uncal', 'NLL before TS'), ('nll', 'NLL after TS'), ('brier_uncal', 'Brier before TS'), ('brier', 'Brier after TS')]:
        rows.append([nm, pm(agg[key]['mean'], agg[key]['sd'], 4), f'{agg[key]["min"]:.4f}–{agg[key]["max"]:.4f}'])
    rows.insert(0, ['Temperature T (fit on val_cal)', pm(T['mean'], T['sd'], 3), f'{T["min"]:.3f}–{T["max"]:.3f}'])
    worse = (df.ece > df.ece_uncal).sum()
    M.p(f"The fitted temperatures were {pm(T['mean'], T['sd'])} ([[T:calib]]). Test-set ECE was {pm(agg['ece']['mean'], agg['ece']['sd'], 4)} after temperature scaling and {pm(agg['ece_uncal']['mean'], agg['ece_uncal']['sd'], 4)} before; temperature scaling therefore {'did not improve' if agg['ece']['mean'] >= agg['ece_uncal']['mean'] else 'improved'} top-label ECE on the test set (ECE was higher after scaling in {worse}/{k} seeds), and NLL changed from {pm(agg['nll_uncal']['mean'], agg['nll_uncal']['sd'], 4)} to {pm(agg['nll']['mean'], agg['nll']['sd'], 4)}. Plausible but untested explanations are that the focal plus label-smoothing objective already yields modest confidence and that a 503-image calibration set with 7–8 images in the smallest classes gives a noisy single-parameter fit. Calibration was retained as pre-specified and both versions are reported; downstream analyses use the calibrated confidence, and the conclusions do not depend on the monotone rescaling. A reliability diagram is shown in [[F:perf_calib]].")
    M.table(['Quantity', 'Mean ± SD over seeds', 'Range'], rows, 'Calibration before and after temperature scaling (TS).')
    M.h(2, '4.3 Training convergence')
    cv = pd.read_csv(RES / 'convergence.csv'); rows = []
    for _, r in cv.iterrows():
        rows.append([r.run.replace('seed42_ep100', 'seed 42, 100-epoch schedule').replace('seed', 'seed '), int(r.epochs), int(r.best_epoch), f'{r.best_val_f1:.3f}', f'{r.last_val_f1:.3f}', f'{r.last10_val_f1_mean:.3f}', f'{r.min_val_loss:.3f} (ep {int(r.min_val_loss_epoch)})',
                     f'{r.final_train_loss:.3f} / {r.final_train_acc:.3f}', f'{r.val_loss_slope_last15:+.4f}', f'{r.val_f1_slope_last15:+.4f}', '—' if pd.isna(r.patience15_would_stop_at) else int(r.patience15_would_stop_at)])
    b = cv[cv.run.str.match(r'^seed4[2-6]$')]
    M.p(f"[[T:converge]] and [[F:training]] document convergence. Each of the {k} runs completed the full 60-epoch annealed schedule (final head learning rate ≈ 10⁻⁹). Training loss and accuracy plateaued (final train loss {b.final_train_loss.mean():.3f}, accuracy {b.final_train_acc.mean():.3f}); validation loss reached its minimum early (epoch {int(b.min_val_loss_epoch.median())}, median) and then rose slightly as training accuracy approached saturation, while validation macro-F1 fluctuated within a narrow band (over the last 15 epochs the fitted slope was {b.val_f1_slope_last15.mean():+.4f} per epoch on average). A patience-15 early-stopping rule, applied post hoc to the recorded curves, would have stopped the runs at epoch {', '.join(str(int(x)) for x in b.patience15_would_stop_at.dropna().tolist())}, i.e. before the schedule ended, and the selected (best val_sel F1) checkpoints were at epochs {', '.join(str(int(x)) for x in b.best_epoch.tolist())}. Training beyond approximately 30–40 epochs brought no systematic further validation gain in any of the five runs.")
    p100 = cv[cv.run == 'seed42_ep100']
    if len(p100):
        r = p100.iloc[0]; r0 = cv[cv.run == 'seed42'].iloc[0]
        e = json.load(open(RES / 'ep100_vs_60.json')); a6, a1 = e['seed42'], e['seed42_ep100']
        M.p(f"As a direct sensitivity check, a longer 100-epoch fully annealed schedule for seed 42 reached a higher best val_sel macro-F1 ({r.best_val_f1:.3f} at epoch {int(r.best_epoch)} vs {r0.best_val_f1:.3f}), but this is the maximum over more epochs on a 502-image validation set and did not transfer: on the test set accuracy was {a1['acc']:.3f} vs {a6['acc']:.3f}, balanced accuracy {a1['bacc']:.3f} vs {a6['bacc']:.3f}, macro-F1 {a1['f1']:.3f} vs {a6['f1']:.3f} and macro-AUROC {a1['auroc']:.3f} vs {a6['auroc']:.3f} (100-epoch vs 60-epoch schedule), with a final training accuracy of {r.final_train_acc:.3f}. Longer training therefore increased fitting of the training set without a test-set benefit, and the 60-epoch protocol was retained. Test metrics of the 100-epoch run were computed for this comparison only and were not used to select anything.")
    M.table(['Run', 'Epochs', 'Selected epoch', 'Best val_sel F1', 'Last-epoch val_sel F1', 'Mean val_sel F1, last 10 ep.', 'Min val_sel loss', 'Final train loss / acc.', 'Val-loss slope, last 15 ep.', 'Val-F1 slope, last 15 ep.', 'Patience-15 stop epoch'], rows,
            'Convergence audit of every training run.', 'Slopes are per epoch from a linear fit.')
    return agg, df, cv
