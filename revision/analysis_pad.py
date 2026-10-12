"""PAD-UFES-20 external-validation audit: separates (i) evaluation-population change, (ii) label-space/metric-definition effects,
(iii) class-prevalence shift, (iv) residual appearance/label-semantics shift, from saved logits (no retraining)."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats as st
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, recall_score
from common import *
warnings.filterwarnings('ignore')
RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
ham = pd.read_csv(CACHE / 'ham_meta.csv'); te = ham[ham.split == 'test'].reset_index(drop=True); yh = te.label_idx.values
pad = pd.read_csv(CACHE / 'pad_meta.csv'); mapped = pad.label.notna().values; pm = pad[mapped].reset_index(drop=True); yp = pm.label_idx.astype(int).values
SEEDS_AVAIL = [s for s in SEEDS if (OUT / f'seed{s}' / 'logits.npz').exists()]
SH5 = [C2I[c] for c in ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV']]        # classes present in PAD after mapping
SH3 = [C2I[c] for c in ['BCC', 'MEL', 'NV']]                        # semantically most comparable classes


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece_top(P, y, nb=15):
    conf = P.max(1); acc = (P.argmax(1) == y).astype(float); e = 0.0; edges = np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def evalset(lg, y, T, cls_subset=None, present_only=True):
    """lg: raw 7-way logits. cls_subset: restrict output space to these class indices (re-normalised softmax).
    Macro metrics computed over classes present in y (present_only) -- also returns 7-class-with-zeros macro-F1 (legacy definition)."""
    if cls_subset is None: z = lg / T; cols = list(range(7))
    else: z = lg[:, cls_subset] / T; cols = list(cls_subset)
    P = softmax(z); pred = np.array(cols)[P.argmax(1)]; present = sorted(set(y.tolist()))
    conf = P.max(1); acc = (pred == y).astype(float); n = len(y)
    out = dict(n=n, acc=float(acc.mean()), bacc=float(balanced_accuracy_score(y, pred)),
               f1_present=float(f1_score(y, pred, labels=present, average='macro', zero_division=0)),
               f1_7zero=float(f1_score(y, pred, labels=list(range(7)), average='macro', zero_division=0)),
               ece=float(ece_top(P, np.array([cols.index(v) if v in cols else -1 for v in y]))) if all(v in cols for v in y) else float('nan'),
               mean_conf=float(conf.mean()))
    # AUROC / brier on present classes (renormalised probability over evaluated output space)
    yi = np.array([cols.index(v) for v in y]); Pn = P
    try:
        out['auroc'] = float(roc_auc_score(yi, Pn, multi_class='ovr', average='macro', labels=[cols.index(v) for v in present]))
    except Exception: out['auroc'] = float('nan')
    yb = np.eye(len(cols))[yi]; out['brier'] = float(np.mean([(Pn[:, [cols.index(v) for v in present]][:, k] - yb[:, cols.index(v)]) ** 2 for k, v in enumerate(present)]))
    out['recall'] = {CLASSES[v]: float((pred[y == v] == v).mean()) for v in present}
    out['pred_dist'] = {CLASSES[v]: float((pred == v).mean()) for v in cols}
    return out


rows = {}
for s in SEEDS_AVAIL:
    z = np.load(OUT / f'seed{s}' / 'logits.npz'); T = float(z['T']); lh = z['test']; lp = z['pad'][mapped]
    R = {}
    R['A_HAM_test_7class'] = evalset(lh, yh, T)
    m5 = np.isin(yh, SH5); R['B_HAM_test_5class_images_7way_output'] = evalset(lh[m5], yh[m5], T)                 # population effect
    R['C_HAM_test_5class_images_5way_output'] = evalset(lh[m5], yh[m5], T, SH5)                                     # + label-space effect
    R['D_PAD_mapped_7way_output'] = evalset(lp, yp, T)                                                              # original protocol
    R['E_PAD_mapped_5way_output'] = evalset(lp, yp, T, SH5)
    m3h = np.isin(yh, SH3); m3p = np.isin(yp, SH3)
    R['F_HAM_test_3class_3way'] = evalset(lh[m3h], yh[m3h], T, SH3); R['G_PAD_3class_3way'] = evalset(lp[m3p], yp[m3p], T, SH3)
    # prevalence effect: HAM per-class recall re-weighted to PAD prevalence (5-way output)
    rec_h = R['C_HAM_test_5class_images_5way_output']['recall']; prev_p = pd.Series(yp).value_counts(normalize=True)
    R['prevalence_reweighted_HAM_acc_5way'] = float(sum(prev_p[C2I[c]] * rec_h[c] for c in rec_h))
    prev_h5 = pd.Series(yh[m5]).value_counts(normalize=True)
    R['prevalence'] = dict(PAD={CLASSES[k]: float(v) for k, v in prev_p.items()}, HAM5={CLASSES[k]: float(v) for k, v in prev_h5.items()})
    # temperature refit oracle on PAD (diagnostic only, patient-level not needed for a 1-parameter fit but reported as an upper bound)
    import torch, torch.nn as nn
    def fitT(lg_, y_):
        l = torch.tensor(lg_[:, SH5]); t = nn.Parameter(torch.ones(1)); yy_ = torch.tensor([SH5.index(v) for v in y_])
        opt = torch.optim.LBFGS([t], lr=0.05, max_iter=100); ce = nn.CrossEntropyLoss()
        def clo(): opt.zero_grad(); l_ = ce(l / t, yy_); l_.backward(); return l_
        opt.step(clo); return float(t.item())
    Tp = fitT(lp, yp); R['PAD_refit_T_5way'] = Tp; R['PAD_5way_ECE_after_T_refit'] = evalset(lp, yp, Tp, SH5)['ece']
    R['PAD_conf_vs_acc'] = dict(mean_conf=R['E_PAD_mapped_5way_output']['mean_conf'], acc=R['E_PAD_mapped_5way_output']['acc'])
    # per-class confusion (7-way, PAD)
    cm = confusion_matrix(yp, lp.argmax(1), labels=list(range(7))); R['PAD_confusion_7way'] = cm.tolist()
    # Fitzpatrick-stratified PAD accuracy (5-way)
    fst = pm.fitspatrick.values; pred5 = np.array(SH5)[softmax(lp[:, SH5] / T).argmax(1)]; fs = {}
    for name, sel in [('I', fst == 1), ('II', fst == 2), ('III', fst == 3), ('IV', fst == 4), ('V-VI', (fst == 5) | (fst == 6)), ('missing', np.isnan(fst))]:
        k = int((pred5[sel] == yp[sel]).sum()); n_ = int(sel.sum())
        fs[name] = dict(n=n_, acc=k / max(n_, 1), ci=[float(st.beta.ppf(.025, k, n_ - k + 1)) if k > 0 else 0.0, float(st.beta.ppf(.975, k + 1, n_ - k)) if k < n_ else 1.0])
    R['PAD_by_fitzpatrick_acc_5way'] = fs
    rows[s] = R
    print('seed', s, 'A', round(R['A_HAM_test_7class']['acc'], 3), 'B', round(R['B_HAM_test_5class_images_7way_output']['acc'], 3), 'C', round(R['C_HAM_test_5class_images_5way_output']['acc'], 3),
          'D', round(R['D_PAD_mapped_7way_output']['acc'], 3), 'E', round(R['E_PAD_mapped_5way_output']['acc'], 3), 'prev-rew', round(R['prevalence_reweighted_HAM_acc_5way'], 3), flush=True)
json.dump(rows, open(RES / 'pad_audit.json', 'w'), indent=1, default=float)

# mapping table + exclusions (dataset-level facts)
pm_all = pd.read_csv(DATA / 'PAD-UFES-20' / 'metadata.csv')
facts = dict(n_total=int(len(pm_all)), n_patients=int(pm_all.patient_id.nunique()), n_lesions=int(pm_all[['patient_id', 'lesion_id']].drop_duplicates().shape[0]),
             diag_counts=pm_all.diagnostic.value_counts().to_dict(), n_mapped=int(mapped.sum()), n_excluded=int((~mapped).sum()),
             mapped_counts=pm.label.value_counts().to_dict(), n_patients_mapped=int(pm.patient_id.nunique()),
             biopsy_proven_frac=float(pm.biopsed.mean()), fitz_counts={str(k): int(v) for k, v in pm.fitspatrick.value_counts(dropna=False).items()},
             ham_test_counts=te.label.value_counts().to_dict(), native_res_median=[float(pm.h_native.median()), float(pm.w_native.median())],
             native_res_min=[int(pm.h_native.min()), int(pm.w_native.min())], native_res_max=[int(pm.h_native.max()), int(pm.w_native.max())],
             n_gt_1024=int((np.maximum(pm.h_native, pm.w_native) > 1024).sum()))
json.dump(facts, open(RES / 'pad_facts.json', 'w'), indent=1, default=float)
print(json.dumps(facts, indent=1, default=float))
