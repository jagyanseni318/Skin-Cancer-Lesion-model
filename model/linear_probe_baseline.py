"""'Condition before' baseline for the classification stage: frozen ImageNet-pretrained backbone + class-balanced multinomial
logistic-regression head fitted on the training split (no end-to-end adaptation, no focal loss, no sampler).  The regularization
strength is chosen on val_sel; temperature on val_cal; the test split and PAD-UFES-20 are evaluated once.
Same preprocessing (cached 224x224 hair-removed images, ImageNet normalization) as the fine-tuned models."""
import json, warnings
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from common import *
from int8_lib import fit_temperature, softmax
warnings.filterwarnings('ignore')
RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
ham = pd.read_csv(CACHE / 'ham_meta.csv'); pad = pd.read_csv(CACHE / 'pad_meta.csv')
X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); XP = np.load(CACHE / 'pad_x.npy', mmap_mode='r')
tr = ham[ham.split == 'train']; vs = ham[(ham.split == 'val') & (ham.val_role == 'sel')]; vc = ham[(ham.split == 'val') & (ham.val_role == 'cal')]
te = ham[ham.split == 'test']; mapped = pad.label.notna().values
SH5 = [C2I[c] for c in ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV']]


@torch.no_grad()
def feats(model, arr, idx, bs=128):
    out = []
    for i in range(0, len(idx), bs):
        out.append(model.backbone(norm_batch(arr[idx[i:i + bs]]).to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def metr(lg, y, T):
    P = softmax(lg / T); pred = P.argmax(1)
    conf = P.max(1); e = 0.0; edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return dict(acc=float((pred == y).mean()), bacc=float(balanced_accuracy_score(y, pred)),
                f1=float(f1_score(y, pred, average='macro', labels=list(range(7)), zero_division=0)),
                auroc=float(roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(7)))),
                mcc=float(matthews_corrcoef(y, pred)), ece=float(e))


res = {}
for arch in ['densenet121', 'resnet50']:
    m = ARCHES[arch]().to(DEVICE).eval()          # ImageNet weights, untouched backbone
    F = {k: feats(m, X, d.cache_idx.values) for k, d in [('tr', tr), ('vs', vs), ('vc', vc), ('te', te)]}
    FP = feats(m, XP, np.arange(len(pad)))[mapped]
    sc = StandardScaler().fit(F['tr']); Z = {k: sc.transform(v) for k, v in F.items()}; ZP = sc.transform(FP)
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(Z['tr'], tr.label_idx.values)
        f1 = f1_score(vs.label_idx.values, clf.predict(Z['vs']), average='macro', labels=list(range(7)), zero_division=0)
        print(arch, 'C', C, 'val_sel macro-F1', round(f1, 4), flush=True)
        if best is None or f1 > best[0]: best = (f1, C, clf)
    clf = best[2]
    lg = lambda Zk: np.log(np.clip(clf.predict_proba(Zk), 1e-9, 1))
    T = fit_temperature(lg(Z['vc']), vc.label_idx.values)
    r = metr(lg(Z['te']), te.label_idx.values, T); r['C'] = best[1]; r['T'] = T
    yp = pad[mapped].label_idx.astype(int).values; lp = lg(ZP)[:, SH5] / T
    pp = softmax(lp); predp = np.array(SH5)[pp.argmax(1)]
    r['pad_acc'] = float((predp == yp).mean()); r['pad_bacc'] = float(balanced_accuracy_score(yp, predp))
    res[arch] = r; print(arch, r, flush=True)
json.dump(res, open(RES / 'linear_probe_baseline.json', 'w'), indent=1)
print('DONE')
