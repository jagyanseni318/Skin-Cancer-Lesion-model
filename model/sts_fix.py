"""Recompute MC-dropout stability columns for an already-evaluated seed (fixes eval-mode reset bug)."""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from common import *
seed = int(sys.argv[1]); od = OUT / f'seed{seed}'
ham = pd.read_csv(CACHE / 'ham_meta.csv'); te = ham[ham.split == 'test'].reset_index(drop=True); tidx = te.cache_idx.values
X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); d = pd.read_csv(od / 'per_image.csv'); T = float(np.load(od / 'logits.npz')['T']); pred = d.pred.values
model_s = DenseNet121_TrustXAI().to(DEVICE); model_s.load_state_dict(torch.load(OUT / 'models' / f'seed{seed}_best.pt', map_location=DEVICE)); model_s.eval(); model_s.use_mc_dropout = True
cam_s = GradCAMPlusPlus(model=model_s, target_layers=model_s.cam_layers())
for m in model_s.modules():
    if isinstance(m, nn.Dropout): m.train()
sts_idx = te.sample(300, random_state=42).index.values
sts = np.full(len(te), np.nan); agree = np.full(len(te), np.nan); ent = np.full(len(te), np.nan)
for i in sts_idx:
    xb = norm_batch(np.repeat(X[tidx[i]][None], 15, 0)).to(DEVICE)
    g = cam_s(input_tensor=xb, targets=[ClassifierOutputTarget(int(pred[i]))] * 15).reshape(15, -1)
    C = np.corrcoef(g)
    if np.isfinite(C).all(): sts[i] = C[np.triu_indices(15, 1)].mean()
    with torch.no_grad(): pp = torch.softmax(model_s(xb) / T, 1).cpu().numpy()
    agree[i] = (pp.argmax(1) == pred[i]).mean(); mp = pp.mean(0); ent[i] = -(mp * np.log(mp + 1e-12)).sum()
d['sts'] = sts; d['mc_pred_agree'] = agree; d['mc_entropy'] = ent; d.to_csv(od / 'per_image.csv', index=False)
print('STS', seed, np.nanmedian(sts), np.nanmean(sts < 0.5), np.nanmean(agree))
