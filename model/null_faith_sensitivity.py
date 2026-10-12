"""Sensitivity analysis for the null-controlled faithfulness metric (EIG/EDG): donor count, baseline
choice and donor-selection strategy, on the reference seed (42), reusing the cached Grad-CAM++ maps
and trained weights (no retraining). Answers the methodological question "why three random cross-image
donor maps constitute an appropriate null?" by showing the sign and approximate magnitude of EIG/EDG
under 1/3/5/10/20 random donors, under the blurred baseline, and under three alternative donor-selection
rules (same predicted class, different predicted class, lesion-area-matched)."""
import json, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy.stats import wilcoxon
from common import *

SEED = 42; name = f'seed{SEED}'; od = OUT / name; RES = OUT / 'results'
t0 = time.time(); _trapz = getattr(np, 'trapezoid', None) or np.trapz

ham = pd.read_csv(CACHE / 'ham_meta.csv'); X = np.load(CACHE / 'ham_x.npy', mmap_mode='r')
te = ham[ham.split == 'test'].reset_index(drop=True); tidx = te.cache_idx.values; n = len(te)
d = pd.read_csv(od / 'per_image.csv'); assert len(d) == n
cams = np.load(od / 'cams_f16.npy').astype(np.float32)
T = float(np.load(od / 'logits.npz')['T'])
pred = d.pred.values; area = d.area224.values

model = DenseNet121_TrustXAI().to(DEVICE)
model.load_state_dict(torch.load(OUT / 'models' / f'{name}_best.pt', map_location=DEVICE)); model.eval()

PATCH = 16; G = IMG // PATCH; NP = G * G; STEPS = list(range(0, NP + 1, 14))
frac = np.array(STEPS) / NP
STEPS_T = torch.tensor(STEPS, device=DEVICE)[:, None]


def patch_scores(H):
    return H.reshape(len(H), G, PATCH, G, PATCH).mean((2, 4)).reshape(len(H), NP)


def curve(xn, base, order, mode):
    rank = torch.empty(NP, dtype=torch.long, device=DEVICE); rank[torch.as_tensor(np.asarray(order).copy(), device=DEVICE)] = torch.arange(NP, device=DEVICE)
    pm = (rank[None, :] < STEPS_T).float().view(len(STEPS), 1, G, G)
    pix = F.interpolate(pm, scale_factor=PATCH, mode='nearest').bool()
    return torch.where(pix, base[None], xn[None]) if mode == 'del' else torch.where(pix, xn[None], base[None])


ps = patch_scores(cams)
auc = lambda v: float(_trapz(v, frac))
all_idx = np.arange(n)


def donors_random(rng, i, k):
    pool = rng.permutation(n); pool = pool[pool != i]; return pool[:k]


def donors_same_class(rng, i, k):
    cls = pred[i]; pool = all_idx[(pred == cls) & (all_idx != i)]
    if len(pool) < k: pool = np.concatenate([pool, rng.choice(all_idx[all_idx != i], k - len(pool), replace=False)])
    return rng.choice(pool, k, replace=False)


def donors_diff_class(rng, i, k):
    cls = pred[i]; pool = all_idx[(pred != cls)]
    return rng.choice(pool, k, replace=False)


def donors_area_matched(rng, i, k):
    order = all_idx[np.argsort(np.abs(area - area[i]))]
    order = order[order != i]
    return order[:k]


SEL = dict(random=donors_random, same_class=donors_same_class, diff_class=donors_diff_class, area_matched=donors_area_matched)


def run_config(donor_k, baseline, selection, rng_seed):
    rng = np.random.RandomState(rng_seed); sel_fn = SEL[selection]
    eig = np.zeros(n, np.float32); edg = np.zeros(n, np.float32)
    own_ins = d[f'{baseline}_ins_cam'].values; own_del = d[f'{baseline}_del_cam'].values
    for i in range(n):
        xn = norm_batch(X[tidx[i]][None])[0].to(DEVICE)
        base = torch.zeros_like(xn) if baseline == 'mean' else TF.gaussian_blur(xn[None], kernel_size=51, sigma=25.0)[0]
        c = int(pred[i]); donors = sel_fn(rng, i, donor_k)
        orders = [('del', np.argsort(-ps[dn], kind='stable')) for dn in donors] + \
                 [('ins', np.argsort(-ps[dn], kind='stable')) for dn in donors]
        batch = torch.cat([curve(xn, base, o, m) for m, o in orders])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            pr = torch.softmax(model(batch).float() / T, 1)[:, c].cpu().numpy().reshape(len(orders), len(STEPS))
        del_aucs = [auc(pr[kk]) for kk in range(donor_k)]; ins_aucs = [auc(pr[donor_k + kk]) for kk in range(donor_k)]
        null_del = float(np.mean(del_aucs)); null_ins = float(np.mean(ins_aucs))
        eig[i] = own_ins[i] - null_ins; edg[i] = null_del - own_del[i]
    return eig, edg


def stat(v):
    med = float(np.median(v)); q1, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
    fpos = float((v > 0).mean())
    try: p = float(wilcoxon(v).pvalue)
    except Exception: p = float('nan')
    return med, q1, q3, fpos, p


configs = (
    [(k, 'mean', 'random') for k in [1, 3, 5, 10, 20]] +
    [(3, 'blur', 'random')] +
    [(3, 'mean', sel) for sel in ['same_class', 'diff_class', 'area_matched']]
)

rows = []
for k, baseline, selection in configs:
    t1 = time.time()
    eig, edg = run_config(k, baseline, selection, rng_seed=9000 + k + hash((baseline, selection)) % 1000)
    em, eq1, eq3, efp, ep = stat(eig); dm, dq1, dq3, dfp, dp = stat(edg)
    rows.append(dict(donor_k=k, baseline=baseline, selection=selection,
                      EIG_median=em, EIG_q1=eq1, EIG_q3=eq3, EIG_frac_pos=efp, EIG_wilcoxon_p=ep,
                      EDG_median=dm, EDG_q1=dq1, EDG_q3=dq3, EDG_frac_pos=dfp, EDG_wilcoxon_p=dp))
    print(f'[sensitivity] k={k} baseline={baseline} sel={selection} EIG_med={em:+.4f} (p={ep:.1e}) '
          f'EDG_med={dm:+.4f} (p={dp:.1e}) {time.time()-t1:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(RES / 'null_faith_sensitivity.csv', index=False)

print(f'DONE total {time.time()-t0:.0f}s', flush=True)
