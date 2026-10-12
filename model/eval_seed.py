"""Per-seed evaluation: temperature scaling (val_cal), test + PAD logits, Grad-CAM++ (predicted class), LSAS, LSAS null
baselines, deletion/insertion faithfulness (Petsiuk et al. 2018 protocol, 16x16 patches), MC-dropout saliency stability.
Every test image (correct AND misclassified) is processed; a valid mask exists for all 990."""
import argparse, json, time, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from common import *

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--tag', default='')
ap.add_argument('--ckpt', default='best'); ap.add_argument('--skip_faith', action='store_true'); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--baselines', default='mean,blur')
ap.add_argument('--arch', default='densenet121', choices=list(ARCHES))
a = ap.parse_args()
arch_tag = '' if a.arch == 'densenet121' else f'_{a.arch}'
name = f'seed{a.seed}{a.tag}{arch_tag}'; od = OUT / (name + ('_smoke' if a.smoke else '')); od.mkdir(exist_ok=True, parents=True)
ham = pd.read_csv(CACHE / 'ham_meta.csv'); pad = pd.read_csv(CACHE / 'pad_meta.csv')
X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); MK = np.load(CACHE / 'ham_mask.npy', mmap_mode='r')
XP = np.load(CACHE / 'pad_x.npy', mmap_mode='r')
model = ARCHES[a.arch]().to(DEVICE)
model.load_state_dict(torch.load(OUT / 'models' / f'{name}_{a.ckpt}.pt', map_location=DEVICE)); model.eval()


@torch.no_grad()
def logits_of(arr, idx, bs=64):
    out = []
    for i in range(0, len(idx), bs):
        out.append(model(norm_batch(arr[idx[i:i + bs]]).to(DEVICE)).float().cpu())
    return torch.cat(out).numpy()


def fit_temperature(lg, y):
    lg = torch.tensor(lg); y = torch.tensor(y); T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50); ce = nn.CrossEntropyLoss()
    def clo():
        opt.zero_grad(); l = ce(lg / T, y); l.backward(); return l
    opt.step(clo); return float(T.item())


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


t0 = time.time(); _trapz = getattr(np, "trapezoid", None) or np.trapz
vcal = ham[(ham.split == 'val') & (ham.val_role == 'cal')]
lg_vcal = logits_of(X, vcal.cache_idx.values); T = fit_temperature(lg_vcal, vcal.label_idx.values)
te = ham[ham.split == 'test'].reset_index(drop=True)
if a.smoke: te = te.iloc[:a.smoke].reset_index(drop=True)
tidx = te.cache_idx.values
lg_te = logits_of(X, tidx); lg_pad = logits_of(XP, np.arange(len(pad) if not a.smoke else 64))
np.savez(od / 'logits.npz', T=T, val_cal=lg_vcal, test=lg_te, pad=lg_pad)
print(f'[{name}] T={T:.4f} logits done {time.time()-t0:.0f}s', flush=True)

P = softmax(lg_te / T); pred = P.argmax(1); conf = P.max(1); y = te.label_idx.values
masks = np.stack([MK[i] for i in tidx]).astype(np.float32)          # (990,224,224)
area = masks.mean((1, 2))

# ---- Grad-CAM++ for predicted class (as in original), batch mode
cam = GradCAMPlusPlus(model=model, target_layers=model.cam_layers())
cams = np.zeros((len(te), IMG, IMG), np.float32)
for i in range(0, len(te), 32):
    xb = norm_batch(X[tidx[i:i + 32]]).to(DEVICE)
    cams[i:i + 32] = cam(input_tensor=xb, targets=[ClassifierOutputTarget(int(c)) for c in pred[i:i + 32]])
np.save(od / 'cams_f16.npy', cams.astype(np.float16))
# CAM for the TRUE class on misclassified images (secondary analysis)
cams_true = np.zeros_like(cams)
for i in range(0, len(te), 32):
    xb = norm_batch(X[tidx[i:i + 32]]).to(DEVICE)
    cams_true[i:i + 32] = cam(input_tensor=xb, targets=[ClassifierOutputTarget(int(c)) for c in y[i:i + 32]])
cam.activations_and_grads.release(); del cam
torch.cuda.empty_cache()
print(f'[{name}] cams done {time.time()-t0:.0f}s', flush=True)


def lsas_of(H, M):
    return (H * M).sum((1, 2)) / (H.sum((1, 2)) + 1e-7)


lsas = lsas_of(cams, masks); lsas_true = lsas_of(cams_true, masks)
# ---- null baselines --------------------------------------------------------------------------
rng = np.random.RandomState(1000 + a.seed)
Hf = cams.reshape(len(te), -1); Mf = masks.reshape(len(te), -1)
S = (Hf @ Mf.T) / (Hf.sum(1, keepdims=True) + 1e-7)     # S[j,i] = LSAS of saliency j against mask i
n = len(te); np.save(od / 'S_full.npy', S.astype(np.float32)); np.fill_diagonal(S, np.nan)
null_shuf_mean = np.nanmean(S, 0)                        # per mask i: mean over other images' saliency
null_shuf_rank = np.array([(np.nansum(S[:, i] < lsas[i]) + 0.5 * np.nansum(S[:, i] == lsas[i])) / (n - 1) for i in range(n)])
K = 100 if not a.smoke else 20; null_shift = np.zeros((n, K), np.float32)
for i in range(n):
    dy = rng.randint(0, IMG, K); dx = rng.randint(0, IMG, K)
    for k in range(K):
        Hs = np.roll(cams[i], (dy[k], dx[k]), axis=(0, 1)); null_shift[i, k] = (Hs * masks[i]).sum() / (Hs.sum() + 1e-7)
null_shift_mean = null_shift.mean(1); null_shift_sd = null_shift.std(1)
null_shift_p = ((null_shift >= lsas[:, None]).sum(1) + 1) / (K + 1)
from sklearn.metrics import roc_auc_score
loc_auc = np.array([roc_auc_score(masks[i].ravel() > 0.5, cams[i].ravel()) if 0 < masks[i].sum() < masks[i].size else np.nan for i in range(n)])
print(f'[{name}] nulls done {time.time()-t0:.0f}s', flush=True)

# ---- deletion / insertion faithfulness ---------------------------------------------------------
PATCH = 16; G = IMG // PATCH; NP = G * G; STEPS = list(range(0, NP + 1, 14)); NR = 3
frac = np.array(STEPS) / NP


def patch_scores(H):    # (n,224,224)->(n,196) mean saliency per 16x16 patch
    return H.reshape(len(H), G, PATCH, G, PATCH).mean((2, 4)).reshape(len(H), NP)


def patch_mask_to_pix(pm):   # (NP,) bool -> (224,224) bool
    return np.kron(pm.reshape(G, G), np.ones((PATCH, PATCH))).astype(bool)


STEPS_T = torch.tensor(STEPS, device=DEVICE)[:, None]


def curve(xn, base, order, target, mode):
    """xn, base: (3,224,224) normalized tensors on DEVICE; order: patch order (most important first).
    'del': start from image, replace top-k patches by baseline; 'ins': start from baseline, reveal top-k patches."""
    rank = torch.empty(NP, dtype=torch.long, device=DEVICE); rank[torch.as_tensor(np.asarray(order).copy(), device=DEVICE)] = torch.arange(NP, device=DEVICE)
    pm = (rank[None, :] < STEPS_T).float().view(len(STEPS), 1, G, G)
    pix = F.interpolate(pm, scale_factor=PATCH, mode='nearest').bool()          # (S,1,224,224)
    return torch.where(pix, base[None], xn[None]) if mode == 'del' else torch.where(pix, xn[None], base[None])


def faith_for(base_kind):
    res = {}
    ps = patch_scores(cams); pm_mask = patch_scores(masks)   # lesion coverage per patch
    out = np.zeros((n, 3 + 3 + 2 + 2 + 2 + 2 + 2), np.float32)   # see cols below
    cols = ['del_cam', 'del_lecam', 'del_rand', 'ins_cam', 'ins_lecam', 'ins_rand', 'del_lesion_first', 'del_bg_first',
            'ins_lesion_first', 'ins_bg_first', 'p0', 'p_full_ins', 'p_base']
    out = np.zeros((n, len(cols)), np.float32); curves = np.zeros((n, 14, len(STEPS)), np.float16)
    rr = np.random.RandomState(2000 + a.seed)
    for i in range(n):
        xn = norm_batch(X[tidx[i]][None])[0].to(DEVICE)
        if base_kind == 'mean': base = torch.zeros_like(xn)
        else:
            import torchvision.transforms.functional as TF
            base = TF.gaussian_blur(xn[None], kernel_size=51, sigma=25.0)[0]
        c = int(pred[i]); od_cam = np.argsort(-ps[i], kind='stable'); od_le = od_cam[::-1].copy()
        od_les = np.argsort(-pm_mask[i], kind='stable'); od_bg = od_les[::-1].copy()
        rands = [rr.permutation(NP) for _ in range(NR)]
        orders = [('del', od_cam), ('del', od_le)] + [('del', r) for r in rands] + \
                 [('ins', od_cam), ('ins', od_le)] + [('ins', r) for r in rands] + \
                 [('del', od_les), ('del', od_bg), ('ins', od_les), ('ins', od_bg)]
        batch = torch.cat([curve(xn, base, o, c, m) for m, o in orders])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            pr = torch.softmax(model(batch).float() / T, 1)[:, c].cpu().numpy().reshape(len(orders), len(STEPS))
        curves[i] = pr
        auc = lambda v: float(_trapz(v, frac))
        out[i] = [auc(pr[0]), auc(pr[1]), np.mean([auc(pr[2 + r]) for r in range(NR)]),
                  auc(pr[2 + NR]), auc(pr[3 + NR]), np.mean([auc(pr[4 + NR + r]) for r in range(NR)]),
                  auc(pr[4 + 2 * NR]), auc(pr[5 + 2 * NR]), auc(pr[6 + 2 * NR]), auc(pr[7 + 2 * NR]),
                  pr[0, 0], pr[2 + NR, -1], pr[2 + NR, 0]]
    np.save(od / f'curves_{base_kind}.npy', curves)
    return pd.DataFrame(out, columns=[f'{base_kind}_{c}' for c in cols])


faith = pd.DataFrame(index=range(n))
if not a.skip_faith:
    for bk in a.baselines.split(','):
        faith = pd.concat([faith, faith_for(bk)], axis=1); print(f'[{name}] faithfulness {bk} done {time.time()-t0:.0f}s', flush=True)

# ---- MC-dropout saliency stability (15 passes; fixed subset of 300 test images) -------------------
sts_idx = te.sample(min(300, len(te)), random_state=42).index.values
model_s = ARCHES[a.arch]().to(DEVICE); model_s.load_state_dict(model.state_dict()); model_s.eval()
model_s.use_mc_dropout = True
for m in model_s.modules():
    if isinstance(m, nn.Dropout): m.train()
cam_s = GradCAMPlusPlus(model=model_s, target_layers=model_s.cam_layers())
for m in model_s.modules():   # GradCAM.__init__ calls model.eval(); dropout must be re-enabled AFTER construction
    if isinstance(m, nn.Dropout): m.train()
sts = np.full(n, np.nan); mc_pred_agree = np.full(n, np.nan); mc_ent = np.full(n, np.nan)
for i in sts_idx:
    xb = norm_batch(np.repeat(X[tidx[i]][None], 15, 0)).to(DEVICE)
    g = cam_s(input_tensor=xb, targets=[ClassifierOutputTarget(int(pred[i]))] * 15).reshape(15, -1)
    C = np.corrcoef(g)
    if np.isfinite(C).all(): sts[i] = C[np.triu_indices(15, 1)].mean()
    with torch.no_grad():
        pp = torch.softmax(model_s(xb) / T, 1).cpu().numpy()
    mc_pred_agree[i] = (pp.argmax(1) == pred[i]).mean(); mp = pp.mean(0); mc_ent[i] = -(mp * np.log(mp + 1e-12)).sum()
cam_s.activations_and_grads.release()
print(f'[{name}] STS done {time.time()-t0:.0f}s', flush=True)

df = pd.DataFrame(dict(image_id=te.image_id, lesion_id=te.lesion_id, label=te.label, label_idx=y, pred=pred, correct=(pred == y),
                       conf=conf, area224=area, area_full=te.mask_area_full.values, lsas=lsas, loc_auc=loc_auc, lsas_true_class=lsas_true,
                       null_shuf_mean=null_shuf_mean, null_shuf_rank=null_shuf_rank, null_shift_mean=null_shift_mean,
                       null_shift_sd=null_shift_sd, null_shift_p=null_shift_p, sts=sts, mc_pred_agree=mc_pred_agree,
                       mc_entropy=mc_ent, prob_true=P[np.arange(n), y]))
df = pd.concat([df, faith], axis=1); df.to_csv(od / 'per_image.csv', index=False)
np.save(od / 'null_shift.npy', null_shift.astype(np.float16))
json.dump(dict(seed=a.seed, T=T, secs=time.time() - t0), open(od / 'eval_summary.json', 'w'))
print(f'[{name}] DONE {time.time()-t0:.0f}s', flush=True)
