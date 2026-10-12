"""Null-controlled faithfulness metric (new methodological contribution added in this revision).

Motivation: Section 4.5 shows that raw LSAS (a localization/geometry statistic) is confounded by
lesion-mask geometry and must be compared against a matched null before it is informative. The same
question has not been asked of perturbation-based faithfulness (deletion/insertion): a positive
insertion gain over a *uniformly random* patch order shows only that *some* spatially structured
revealing order beats *no* structure -- it does not show that THIS image's own Grad-CAM++ map is
better than a plausible-looking but wrong map (e.g. a map computed for a different lesion). This
script computes that stronger control directly: for each test image, the deletion/insertion curves
are repeated using the Grad-CAM++ maps of three other, randomly chosen test images (already cached in
cams_f16.npy, no re-inference of the explainer needed) as the patch-importance order applied to this
image's own pixels, mean-image baseline, all five seeds. The resulting quantities are:

    Excess Insertion Gain (EIG) = IG(observed CAM order) - IG(cross-image null-CAM order)
    Excess Deletion Gain  (EDG) = DG(cross-image null-CAM order) - DG(observed CAM order)

Positive EIG/EDG indicate the map carries image-specific, not merely image-generic, functional
information -- the faithfulness analogue of excess LSAS. This reuses the exact patch definition,
baseline, step grid and AUC rule already validated in eval_seed.py's faith_for(), so numbers are
directly comparable to Tables 10-11.
"""
import argparse, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--arch', default='densenet121', choices=list(ARCHES))
a = ap.parse_args()
arch_tag = '' if a.arch == 'densenet121' else f'_{a.arch}'
name = f'seed{a.seed}{arch_tag}'; od = OUT / name
t0 = time.time(); _trapz = getattr(np, 'trapezoid', None) or np.trapz

ham = pd.read_csv(CACHE / 'ham_meta.csv'); X = np.load(CACHE / 'ham_x.npy', mmap_mode='r')
te = ham[ham.split == 'test'].reset_index(drop=True); tidx = te.cache_idx.values; n = len(te)
d = pd.read_csv(od / 'per_image.csv'); assert len(d) == n
cams = np.load(od / 'cams_f16.npy').astype(np.float32)                 # (990,224,224), predicted-class maps
T = float(np.load(od / 'logits.npz')['T'])
pred = d.pred.values

model = ARCHES[a.arch]().to(DEVICE)
model.load_state_dict(torch.load(OUT / 'models' / f'{name}_best.pt', map_location=DEVICE)); model.eval()

PATCH = 16; G = IMG // PATCH; NP = G * G; STEPS = list(range(0, NP + 1, 14)); NPAIR = 3
frac = np.array(STEPS) / NP
STEPS_T = torch.tensor(STEPS, device=DEVICE)[:, None]


def patch_scores(H):
    return H.reshape(len(H), G, PATCH, G, PATCH).mean((2, 4)).reshape(len(H), NP)


def curve(xn, base, order, mode):
    rank = torch.empty(NP, dtype=torch.long, device=DEVICE); rank[torch.as_tensor(np.asarray(order).copy(), device=DEVICE)] = torch.arange(NP, device=DEVICE)
    pm = (rank[None, :] < STEPS_T).float().view(len(STEPS), 1, G, G)
    pix = F.interpolate(pm, scale_factor=PATCH, mode='nearest').bool()
    return torch.where(pix, base[None], xn[None]) if mode == 'del' else torch.where(pix, xn[None], base[None])


ps = patch_scores(cams)   # (990,196) patch-mean saliency per image, predicted class
rng = np.random.RandomState(3000 + a.seed)
# fixed, reproducible triple of "wrong" donor images per target image (never the image itself)
donors = np.zeros((n, NPAIR), dtype=int)
for i in range(n):
    pool = rng.permutation(n); pool = pool[pool != i][:NPAIR]; donors[i] = pool

out = np.zeros((n, 4), np.float32)   # del_nullcam, ins_nullcam (mean over 3 donors), + their SD across donors
auc = lambda v: float(_trapz(v, frac))
for i in range(n):
    xn = norm_batch(X[tidx[i]][None])[0].to(DEVICE); base = torch.zeros_like(xn); c = int(pred[i])
    orders = [('del', np.argsort(-ps[donors[i, k]], kind='stable')) for k in range(NPAIR)] + \
             [('ins', np.argsort(-ps[donors[i, k]], kind='stable')) for k in range(NPAIR)]
    batch = torch.cat([curve(xn, base, o, m) for m, o in orders])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        pr = torch.softmax(model(batch).float() / T, 1)[:, c].cpu().numpy().reshape(len(orders), len(STEPS))
    del_aucs = [auc(pr[k]) for k in range(NPAIR)]; ins_aucs = [auc(pr[NPAIR + k]) for k in range(NPAIR)]
    out[i] = [np.mean(del_aucs), np.mean(ins_aucs), np.std(del_aucs), np.std(ins_aucs)]
    if i % 200 == 0: print(f'[{name}] null-faith {i}/{n} {time.time()-t0:.0f}s', flush=True)

res = pd.DataFrame(out, columns=['mean_del_nullcam', 'mean_ins_nullcam', 'mean_del_nullcam_sd', 'mean_ins_nullcam_sd'])
res.insert(0, 'image_id', d.image_id.values)
res['mean_EIG'] = d.mean_ins_cam.values - res.mean_ins_nullcam.values         # excess insertion gain
res['mean_EDG'] = res.mean_del_nullcam.values - d.mean_del_cam.values         # excess deletion gain
# for reference, also recompute against the SAME-image random-order baseline already stored (sanity: null-cam should sit between cam and random)
res['mean_ins_rand_ref'] = d.mean_ins_rand.values; res['mean_del_rand_ref'] = d.mean_del_rand.values
res.to_csv(od / 'null_faithfulness.csv', index=False)
json.dump(dict(seed=a.seed, n_pairs=NPAIR, secs=time.time() - t0), open(od / 'null_faithfulness_summary.json', 'w'))
print(f'[{name}] DONE {time.time()-t0:.0f}s | median EIG={np.median(res.mean_EIG):.4f} median EDG={np.median(res.mean_EDG):.4f}', flush=True)
