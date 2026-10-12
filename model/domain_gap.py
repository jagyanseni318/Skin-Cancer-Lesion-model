"""Domain-gap diagnostics + target-adaptation ladder for HAM10000-test -> PAD-UFES-20 (per seed model).
 (1) image statistics; (2) domain-classifier AUROC & Frechet distance in backbone-feature space;
 (3) adaptation ladder on a patient-level split of PAD (30% adapt / 70% evaluate, 20 repeats): zero-shot -> temperature -> class-bias (prior/label-shift)
     -> linear probe on frozen features. Recovery by bias-only correction estimates label-shift/calibration share; recovery needing a probe estimates representation/decision-boundary shift.
 (4) deviation check for the 1,024-px cap in PAD hair removal (prediction agreement hair-removal-cap vs. resize-only)."""
import json, sys, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, cv2
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from common import *
warnings.filterwarnings('ignore')
RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
seeds = [int(a) for a in sys.argv[1:]] or [42]
ham = pd.read_csv(CACHE / 'ham_meta.csv'); te = ham[ham.split == 'test'].reset_index(drop=True); yh = te.label_idx.values
pad = pd.read_csv(CACHE / 'pad_meta.csv'); mapped = pad.label.notna().values; pm = pad[mapped].reset_index(drop=True); yp = pm.label_idx.astype(int).values
X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); XP = np.load(CACHE / 'pad_x.npy', mmap_mode='r'); XPn = np.load(CACHE / 'pad_x_nohair.npy', mmap_mode='r')
SH5 = [C2I[c] for c in ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV']]
pad_idx = np.where(mapped)[0]; te_idx = te.cache_idx.values


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


@torch.no_grad()
def feats(model, arr, idx, bs=64):
    f1, f2, lg = [], [], []
    for i in range(0, len(idx), bs):
        x = norm_batch(arr[idx[i:i + bs]]).to(DEVICE); f = model.backbone(x); h = torch.relu(model.bn1(model.fc1(f)))
        f1.append(f.cpu().numpy()); f2.append(h.cpu().numpy()); lg.append(model.classifier(h).cpu().numpy())
    return np.concatenate(f1), np.concatenate(f2), np.concatenate(lg)


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1.dot(s2), disp=False); cs = cs.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def stats_of(arr, idx):
    out = []
    for i in idx:
        im = np.ascontiguousarray(arr[i]); lab = cv2.cvtColor(im, cv2.COLOR_RGB2LAB).astype(np.float32); g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        out.append([im[..., 0].mean(), im[..., 1].mean(), im[..., 2].mean(), lab[..., 0].mean() * 100 / 255, lab[..., 0].std() * 100 / 255,
                    lab[..., 1].mean() - 128, lab[..., 2].mean() - 128, cv2.Laplacian(g, cv2.CV_64F).var(), g.std(), (im.max(2) < 45).mean()])
    return pd.DataFrame(out, columns=['R', 'G', 'B', 'L', 'L_sd', 'a', 'b', 'lap_var', 'gray_sd', 'dark_frac'])


res = {}
sh = stats_of(X, te_idx); sp = stats_of(XP, pad_idx)
res['image_stats'] = {c: dict(ham_mean=float(sh[c].mean()), ham_sd=float(sh[c].std()), pad_mean=float(sp[c].mean()), pad_sd=float(sp[c].std()),
                              cohen_d=float((sp[c].mean() - sh[c].mean()) / np.sqrt((sp[c].var() + sh[c].var()) / 2))) for c in sh.columns}

for s in seeds:
    model = DenseNet121_TrustXAI().to(DEVICE); model.load_state_dict(torch.load(OUT / 'models' / f'seed{s}_best.pt', map_location=DEVICE)); model.eval()
    T = float(np.load(OUT / f'seed{s}' / 'logits.npz')['T'])
    fh, hh, lgh = feats(model, X, te_idx); fp, hp, lgp = feats(model, XP, pad_idx); fpn, hpn, lgpn = feats(model, XPn, pad_idx)
    R = {}
    # (4) 1,024-px cap / hair-removal deviation check
    a1 = softmax(lgp[:, SH5] / T).argmax(1); a2 = softmax(lgpn[:, SH5] / T).argmax(1); big = np.maximum(pm.h_native, pm.w_native).values > 1024
    R['hair_removal_deviation'] = dict(agree_all=float((a1 == a2).mean()), agree_big_images=float((a1[big] == a2[big]).mean()), n_big=int(big.sum()),
                                       acc_with_hair_removal=float((np.array(SH5)[a1] == yp).mean()), acc_resize_only=float((np.array(SH5)[a2] == yp).mean()))
    # (2) domain classifier + Frechet distance (1024-d pooled features; PCA-free, standardised logistic regression)
    Xd = np.vstack([fh, fp]); yd = np.r_[np.zeros(len(fh)), np.ones(len(fp))]
    grp = np.r_[te.lesion_id.values, pm.patient_id.values]
    skf = StratifiedKFold(5, shuffle=True, random_state=0); auc = []
    for tr, va in skf.split(Xd, yd):
        sc = StandardScaler().fit(Xd[tr]); clf = LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced').fit(sc.transform(Xd[tr]), yd[tr])
        auc.append(roc_auc_score(yd[va], clf.decision_function(sc.transform(Xd[va]))))
    R['domain_classifier_auroc'] = dict(mean=float(np.mean(auc)), sd=float(np.std(auc)))
    # class-matched Frechet: same-dataset baseline = HAM-val-like split of the test set
    rng = np.random.RandomState(0); perm = rng.permutation(len(fh)); h1, h2 = fh[perm[:len(fh) // 2]], fh[perm[len(fh) // 2:]]
    R['frechet'] = dict(ham_vs_pad=frechet(fh, fp), ham_half_vs_half=frechet(h1, h2))
    # (3) adaptation ladder on patient-level splits
    ladder = {k: [] for k in ['zero_shot', 'temp_only', 'bias_only', 'temp_bias', 'probe_1024', 'probe_512']}
    gss = GroupShuffleSplit(n_splits=20, train_size=0.3, random_state=11)
    lg5 = lgp[:, SH5]; y5 = np.array([SH5.index(v) for v in yp])
    def m_(pred, y_): return [float((pred == y_).mean()), float(balanced_accuracy_score(y_, pred)), float(f1_score(y_, pred, labels=list(range(5)), average='macro', zero_division=0))]
    for tr, ev in gss.split(lg5, y5, groups=pm.patient_id.values):
        if len(set(y5[tr])) < 5 or len(set(y5[ev])) < 5: continue
        ladder['zero_shot'].append(m_(lg5[ev].argmax(1), y5[ev]))
        # temperature on adaptation set
        l = torch.tensor(lg5[tr]); t = nn.Parameter(torch.ones(1)); yt = torch.tensor(y5[tr]); opt = torch.optim.LBFGS([t], lr=0.05, max_iter=100); ce = nn.CrossEntropyLoss()
        def clo(): opt.zero_grad(); l_ = ce(l / t, yt); l_.backward(); return l_
        opt.step(clo); ladder['temp_only'].append(m_((lg5[ev] / t.item()).argmax(1), y5[ev]))     # T alone cannot change argmax (kept for completeness)
        # class-bias vector (5 params) with weak L2 -> corrects label/prior shift and systematic class offset
        b = nn.Parameter(torch.zeros(5)); opt = torch.optim.LBFGS([b], lr=0.1, max_iter=200)
        def clo2(): opt.zero_grad(); l_ = ce(l + b, yt) + 1e-3 * (b ** 2).sum(); l_.backward(); return l_
        opt.step(clo2); ladder['bias_only'].append(m_((lg5[ev] + b.detach().numpy()).argmax(1), y5[ev]))
        t2 = nn.Parameter(torch.ones(1)); b2 = nn.Parameter(torch.zeros(5)); opt = torch.optim.LBFGS([t2, b2], lr=0.1, max_iter=300)
        def clo3(): opt.zero_grad(); l_ = ce(l * t2 + b2, yt) + 1e-3 * (b2 ** 2).sum(); l_.backward(); return l_
        opt.step(clo3); ladder['temp_bias'].append(m_((lg5[ev] * t2.item() + b2.detach().numpy()).argmax(1), y5[ev]))
        for nm, F_ in [('probe_1024', fp), ('probe_512', hp)]:
            sc = StandardScaler().fit(F_[tr]); clf = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(sc.transform(F_[tr]), y5[tr])
            ladder[nm].append(m_(clf.predict(sc.transform(F_[ev])), y5[ev]))
    R['adaptation_ladder'] = {k: dict(acc=[float(np.mean([r[0] for r in v])), float(np.std([r[0] for r in v]))], bacc=[float(np.mean([r[1] for r in v])), float(np.std([r[1] for r in v]))],
                                     f1=[float(np.mean([r[2] for r in v])), float(np.std([r[2] for r in v]))], n_splits=len(v)) for k, v in ladder.items()}
    # internal reference: same linear probe trained on HAM-val-like 30% of HAM test restricted to 5 classes, evaluated on the rest (patient=lesion groups)
    m5 = np.isin(yh, SH5); yh5 = np.array([SH5.index(v) for v in yh[m5]]); res_ = []
    for tr, ev in GroupShuffleSplit(n_splits=20, train_size=0.3, random_state=11).split(fh[m5], yh5, groups=te.lesion_id.values[m5]):
        if len(set(yh5[tr])) < 5 or len(set(yh5[ev])) < 5: continue
        sc = StandardScaler().fit(fh[m5][tr]); clf = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(sc.transform(fh[m5][tr]), yh5[tr])
        res_.append(m_(clf.predict(sc.transform(fh[m5][ev])), yh5[ev]))
    R['HAM_internal_probe_reference'] = dict(acc=[float(np.mean([r[0] for r in res_])), float(np.std([r[0] for r in res_]))], bacc=[float(np.mean([r[1] for r in res_])), float(np.std([r[1] for r in res_]))],
                                            f1=[float(np.mean([r[2] for r in res_])), float(np.std([r[2] for r in res_]))])
    res[f'seed{s}'] = R; print('seed', s, json.dumps({k: (v if not isinstance(v, dict) or k != 'adaptation_ladder' else {a: b['bacc'] for a, b in v.items()}) for k, v in R.items()}, default=float)[:900], flush=True)
    np.savez_compressed(OUT / f'seed{s}' / 'features_pad_test.npz', fh=fh.astype(np.float16), fp=fp.astype(np.float16))
json.dump(res, open(RES / f'domain_gap_{"_".join(map(str, seeds))}.json', 'w'), indent=1, default=float)
