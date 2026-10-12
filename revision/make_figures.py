"""Publication figures generated exclusively from saved experiment outputs."""
import json, warnings
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats as st
from common import *
warnings.filterwarnings('ignore')
RES = OUT / 'results'; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})
C = dict(blue='#1f5fa8', orange='#d95f02', green='#1b9e77', grey='#7f7f7f', red='#c0392b', purple='#6a3d9a', black='#222222')
SEEDS_AVAIL = [s for s in SEEDS if (OUT / f'seed{s}' / 'per_image.csv').exists()]


def save(fig, name): fig.savefig(FIG / name, dpi=300, bbox_inches='tight'); plt.close(fig)


def binned(x, y, nb=15):
    q = np.quantile(x, np.linspace(0, 1, nb + 1)); idx = np.clip(np.digitize(x, q[1:-1]), 0, nb - 1)
    return np.array([np.median(x[idx == b]) for b in range(nb)]), np.array([np.median(y[idx == b]) for b in range(nb)])


# ---------- Fig 1: trust dimensions schematic
def fig_dimensions():
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    items = [('Diagnostic\nperformance', 'Acc, BAcc, macro-F1,\nAUROC, PR-AUC, MCC, $\\kappa$'), ('Calibration', 'ECE, Brier, NLL,\ntemperature scaling'),
             ('Explanation\nlocalization', 'LSAS vs. chance\nand ceiling'), ('Explanation\nfaithfulness', 'Deletion / insertion\n(RISE protocol)'),
             ('Explanation\nstability', 'MC-dropout\nsaliency correlation'), ('Lesion-size\ndependence', 'Null-controlled\ngeometry audit'),
             ('Skin-tone\nproxy', 'ITA strata, continuous\nITA, uncertainty'), ('External\ngeneralization', 'PAD-UFES-20:\nmapping vs. shift')]
    cols = [C['blue'], C['blue'], C['orange'], C['orange'], C['orange'], C['green'], C['purple'], C['red']]
    for i, ((t, d), c) in enumerate(zip(items, cols)):
        x = 0.3 + (i % 4) * 2.4; y = 2.7 - (i // 4) * 2.4
        ax.add_patch(FancyBboxPatch((x, y), 2.1, 1.95, boxstyle='round,pad=0.03', fc=c, ec='none', alpha=0.13)); ax.add_patch(FancyBboxPatch((x, y), 2.1, 1.95, boxstyle='round,pad=0.03', fc='none', ec=c, lw=1.4))
        ax.text(x + 1.05, y + 1.45, t, ha='center', va='center', fontsize=9, fontweight='bold', color=c); ax.text(x + 1.05, y + 0.6, d, ha='center', va='center', fontsize=7.5, color=C['black'])
    ax.text(5, 0.05, 'Distinct, non-interchangeable trust dimensions evaluated on one model family (5 seeds)', ha='center', fontsize=8, style='italic')
    save(fig, 'fig1_trust_dimensions.png')


# ---------- Fig 2: training curves
def fig_training():
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))
    for s in SEEDS_AVAIL:
        h = pd.read_csv(OUT / 'models' / f'seed{s}_history.csv')
        ax[0].plot(h.epoch, h.train_loss, lw=1); ax[1].plot(h.epoch, h.val_loss, lw=1); ax[2].plot(h.epoch, h.val_f1, lw=1, label=f'seed {s}'); ax[3].plot(h.epoch, h.lr_head, lw=1)
        b = int(h.val_f1.idxmax()); ax[2].plot(h.epoch[b], h.val_f1[b], 'k*', ms=6)
    p = OUT / 'models' / 'seed42_ep100_history.csv'
    if p.exists():
        h = pd.read_csv(p); ax[1].plot(h.epoch, h.val_loss, 'k--', lw=1); ax[2].plot(h.epoch, h.val_f1, 'k--', lw=1, label='seed 42, 100-ep schedule'); ax[0].plot(h.epoch, h.train_loss, 'k--', lw=1)
    for a, t in zip(ax, ['Training loss', 'Validation loss (val$_{sel}$)', 'Validation macro-F1 (val$_{sel}$)', 'Head learning rate (OneCycle)']): a.set_title(t); a.set_xlabel('Epoch')
    ax[2].legend(frameon=False, fontsize=6.5); ax[3].set_yscale('log'); fig.tight_layout(); save(fig, 'fig2_training_curves.png')


# ---------- Fig 3: performance + reliability
def fig_perf():
    pc = pd.read_csv(RES / 'cls_per_class_per_seed.csv'); rel = json.load(open(RES / 'reliability_seed42.json')); cms = json.load(open(RES / 'confusion_matrices.json'))
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3)); w = 0.38; x = np.arange(7)
    for k, (m, c) in enumerate([('recall', C['blue']), ('f1', C['orange'])]):
        g = pc.groupby('cls')[m].agg(['mean', 'std']).reindex(CLASSES); ax[0].bar(x + (k - .5) * w, g['mean'], w, yerr=g['std'], color=c, label=m.replace('f1', 'F1').capitalize(), capsize=2, ec='none')
        for s in SEEDS_AVAIL:
            v = pc[(pc.seed == s)].set_index('cls').reindex(CLASSES)[m].values; ax[0].scatter(x + (k - .5) * w, v, s=5, c='k', alpha=.45, zorder=3)
    ax[0].set_xticks(x); ax[0].set_xticklabels(CLASSES, rotation=30); ax[0].set_ylim(0, 1.05); ax[0].legend(frameon=False); ax[0].set_title('Per-class recall / F1 (mean and SD, dots = seeds)')
    cm = np.array(cms['42']); cmn = cm / cm.sum(1, keepdims=True); im = ax[1].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(7):
        for j in range(7): ax[1].text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='white' if cmn[i, j] > .5 else 'k')
    ax[1].set_xticks(range(7)); ax[1].set_yticks(range(7)); ax[1].set_xticklabels(CLASSES, rotation=45, fontsize=7); ax[1].set_yticklabels(CLASSES, fontsize=7); ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('True'); ax[1].set_title('Row-normalised confusion (seed 42)')
    for nm, c, lab in [('uncal', C['grey'], 'Before temperature scaling'), ('cal', C['blue'], 'After temperature scaling')]:
        b = [r for r in rel[nm] if r['n'] > 0]; ax[2].plot([r['conf'] for r in b], [r['acc'] for r in b], 'o-', c=c, ms=3.5, lw=1, label=lab)
    ax[2].plot([0, 1], [0, 1], 'k:', lw=1); ax[2].set_xlabel('Confidence'); ax[2].set_ylabel('Accuracy'); ax[2].set_title('Reliability diagram (seed 42)'); ax[2].legend(frameon=False)
    fig.tight_layout(); save(fig, 'fig3_performance_calibration.png')


# ---------- Fig 4/5: lesion-size audit
def fig_size(d):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5)); a = d.area224.values
    for cor, col, lab in [(True, C['blue'], 'correct'), (False, C['red'], 'misclassified')]:
        m = d.correct.values == cor; ax[0].scatter(a[m], d.lsas.values[m], s=5, alpha=.35, c=col, label=f'Observed LSAS ({lab})', edgecolors='none')
    ax[0].plot(*binned(a, d.lsas.values), c=C['black'], lw=1.8, label='Observed (binned median)'); ax[0].plot(*binned(a, d.null_shuf_mean.values), c=C['orange'], lw=1.8, label='Null: cross-image shuffled saliency')
    ax[0].plot(*binned(a, d.null_shift_mean.values), c=C['purple'], lw=1.5, ls='--', label='Null: toroidal shift'); ax[0].plot(*binned(a, d.oracle.values), c=C['green'], lw=1.8, label='Ceiling: lesion-aligned CAM-resolution oracle')
    ax[0].plot([0, 1], [0, 1], c=C['grey'], ls=':', lw=1, label='Uniform saliency (LSAS = area)'); ax[0].set_xlabel('Lesion-mask area fraction'); ax[0].set_ylabel('LSAS'); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].legend(frameon=False, fontsize=5.8, loc='upper left'); ax[0].set_title('LSAS vs. lesion area (seed 42)')
    ax[1].scatter(a, d.e_shuf.values, s=5, alpha=.35, c=np.where(d.correct.values, C['blue'], C['red']), edgecolors='none'); ax[1].axhline(0, c='k', lw=.8); ax[1].plot(*binned(a, d.e_shuf.values), c=C['black'], lw=1.8)
    ax[1].set_xlabel('Lesion-mask area fraction'); ax[1].set_ylabel('Excess LSAS = LSAS − null (shuffled)'); ax[1].set_title('Area-adjusted (excess) localization')
    order = ['small', 'medium', 'large']; pos = np.arange(3); wd = 0.26
    for k, (v, c, lab) in enumerate([('lsas', C['black'], 'Observed'), ('null_shuf_mean', C['orange'], 'Null (shuffled)'), ('oracle', C['green'], 'Ceiling (oracle)')]):
        bp = ax[2].boxplot([d[d['size'] == s][v] for s in order], positions=pos + (k - 1) * wd, widths=wd * .85, patch_artist=True, showfliers=False, medianprops=dict(color='k'))
        for b in bp['boxes']: b.set_facecolor(c); b.set_alpha(.5)
        ax[2].plot([], [], c=c, lw=6, alpha=.5, label=lab)
    ax[2].set_xticks(pos); ax[2].set_xticklabels([f'{s}\n(n={int((d["size"] == s).sum())})' for s in order]); ax[2].set_ylabel('LSAS'); ax[2].legend(frameon=False, fontsize=7); ax[2].set_title('Size strata: observed vs. null vs. ceiling')
    fig.tight_layout(); save(fig, 'fig4_lesion_size_audit.png')


def fig_correctness(d):
    fig, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))
    for a, (v, t) in zip(ax, [('lsas', 'LSAS'), ('e_shuf', 'Excess LSAS'), ('conf', 'Calibrated confidence'), ('area224', 'Lesion area fraction')]):
        dat = [d[d.correct][v].values, d[~d.correct][v].values]; vp = a.violinplot(dat, showmedians=True, showextrema=False)
        for b, c in zip(vp['bodies'], [C['blue'], C['red']]): b.set_facecolor(c); b.set_alpha(.5)
        vp['cmedians'].set_color('k'); a.set_xticks([1, 2]); a.set_xticklabels([f'correct\n(n={len(dat[0])})', f'misclassified\n(n={len(dat[1])})']); a.set_title(t)
    fig.tight_layout(); save(fig, 'fig5_correct_vs_misclassified.png')


# ---------- Fig 6: faithfulness
def fig_faith(d):
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.1)); frac = np.array(list(range(0, 197, 14))) / 196
    cur = np.load(OUT / 'seed42' / 'curves_mean.npy').astype(np.float32)   # (n,14,15): del cam, del lecam, del r1..3, ins cam, ins lecam, ins r1..3, del les, del bg, ins les, ins bg
    def mc(idx): v = cur[:, idx].mean(1) if isinstance(idx, list) else cur[:, idx]; return v.mean(0), 1.96 * v.std(0) / np.sqrt(len(v))
    for idx, c, lab in [(0, C['blue'], 'Grad-CAM++ (most important first)'), ([2, 3, 4], C['grey'], 'Random order'), (10, C['green'], 'Lesion pixels first (oracle mask)'), (11, C['orange'], 'Background first (oracle mask)')]:
        m, e = mc(idx); ax[0].plot(frac, m, c=c, label=lab); ax[0].fill_between(frac, m - e, m + e, color=c, alpha=.2)
    ax[0].set_title('Deletion (lower = more faithful)'); ax[0].set_xlabel('Fraction of patches removed'); ax[0].set_ylabel('Predicted-class probability'); ax[0].legend(frameon=False, fontsize=6)
    for idx, c in [(5, C['blue']), ([7, 8, 9], C['grey']), (12, C['green']), (13, C['orange'])]:
        m, e = mc(idx); ax[1].plot(frac, m, c=c); ax[1].fill_between(frac, m - e, m + e, color=c, alpha=.2)
    ax[1].set_title('Insertion (higher = more faithful)'); ax[1].set_xlabel('Fraction of patches revealed')
    ax[2].scatter(d.e_shuf, d.mean_DG, s=5, alpha=.35, c=np.where(d.correct, C['blue'], C['red']), edgecolors='none'); ax[2].axhline(0, c='k', lw=.8); ax[2].plot(*binned(d.e_shuf.values, d.mean_DG.values), c='k', lw=1.6)
    ax[2].set_xlabel('Excess LSAS (localization)'); ax[2].set_ylabel('Deletion gain  (AUC$_{rand}$ − AUC$_{CAM}$)'); ax[2].set_title('Localization vs. faithfulness')
    ax[3].scatter(d.e_shuf, d.mean_IG, s=5, alpha=.35, c=np.where(d.correct, C['blue'], C['red']), edgecolors='none'); ax[3].axhline(0, c='k', lw=.8); ax[3].plot(*binned(d.e_shuf.values, d.mean_IG.values), c='k', lw=1.6)
    ax[3].set_xlabel('Excess LSAS (localization)'); ax[3].set_ylabel('Insertion gain  (AUC$_{CAM}$ − AUC$_{rand}$)'); ax[3].set_title('Localization vs. faithfulness')
    fig.tight_layout(); save(fig, 'fig6_faithfulness.png')


def fig_sts(d):
    s = d.dropna(subset=['sts']); fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    ax[0].hist(s.sts, bins=25, color=C['blue'], ec='w'); ax[0].axvline(.5, c='r', ls='--'); ax[0].set_xlabel('MC-dropout saliency stability (mean pairwise Pearson r, 15 passes)'); ax[0].set_ylabel('Images'); ax[0].set_title(f'Stability (n={len(s)}, seed 42)')
    ax[1].scatter(s.e_shuf, s.sts, s=6, alpha=.5, c=np.where(s.correct, C['blue'], C['red']), edgecolors='none'); ax[1].set_xlabel('Excess LSAS'); ax[1].set_ylabel('Stability'); ax[1].set_title('Stability vs. localization')
    fig.tight_layout(); save(fig, 'fig7_stability.png')


def fig_skin(d):
    ita = pd.read_csv(OUT / 'ita_all.csv'); fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    ax[0].hist(ita.ita_v1.dropna().clip(-60, 90), bins=60, color=C['grey'], alpha=.6, label='v1 (original definition)'); ax[0].hist(ita.ita_v2.dropna().clip(-60, 90), bins=60, color=C['blue'], alpha=.45, label='v2 (vignette/halo-robust)')
    for t in [28, 55]: ax[0].axvline(t, c='k', ls='--', lw=.8)
    ax[0].set_xlabel('ITA (degrees)'); ax[0].set_ylabel('HAM10000 images (n=10,015)'); ax[0].legend(frameon=False); ax[0].set_title('ITA distribution (thresholds 28°, 55°)')
    G = ['I-II', 'III-IV', 'V-VI']
    for k, (v, t) in enumerate([('lsas', 'LSAS'), ('e_shuf', 'Excess LSAS')]):
        a = ax[1 + k]; dat = [d[d.fst_v1 == g][v].values for g in G]; bp = a.boxplot(dat, showfliers=False, patch_artist=True, medianprops=dict(color='k'))
        for b, c in zip(bp['boxes'], [C['orange'], C['blue'], C['purple']]): b.set_alpha(.45); b.set_facecolor(c)
        for i, v_ in enumerate(dat): a.scatter(np.random.RandomState(0).normal(i + 1, .05, len(v_)), v_, s=4, alpha=.3, c='k', edgecolors='none')
        a.set_xticklabels([f'{g}\n(n={len(x)})' for g, x in zip(G, dat)]); a.set_title(f'{t} by ITA proxy stratum (all test images, seed 42)'); a.set_ylabel(t)
    fig.tight_layout(); save(fig, 'fig8_skin_tone.png')


def fig_pad():
    P = json.load(open(RES / 'pad_audit.json')); seeds = sorted(P.keys()); fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    keys = [('A_HAM_test_7class', 'HAM test\n7-class\n(all images)'), ('B_HAM_test_5class_images_7way_output', 'HAM test\n5 mapped classes\n(7-way output)'), ('C_HAM_test_5class_images_5way_output', 'HAM test\n5 mapped classes\n(5-way output)')]
    vals = [[P[s][k]['acc'] for s in seeds] for k, _ in keys]; vals.append([P[s]['prevalence_reweighted_HAM_acc_5way'] for s in seeds]); vals.append([P[s]['E_PAD_mapped_5way_output']['acc'] for s in seeds]); vals.append([P[s]['D_PAD_mapped_7way_output']['acc'] for s in seeds])
    labs = [l for _, l in keys] + ['HAM recalls\nre-weighted to\nPAD prevalence', 'PAD-UFES-20\n(5-way output)', 'PAD-UFES-20\n(7-way output;\noriginal protocol)']
    cols = [C['blue'], C['blue'], C['blue'], C['orange'], C['red'], C['red']]
    ax[0].bar(range(6), [np.mean(v) for v in vals], yerr=[np.std(v, ddof=1) for v in vals], color=cols, alpha=.8, capsize=3)
    for i, v in enumerate(vals): ax[0].scatter([i] * len(v), v, s=6, c='k', zorder=3)
    ax[0].set_xticks(range(6)); ax[0].set_xticklabels(labs, fontsize=6.3); ax[0].set_ylabel('Accuracy'); ax[0].set_ylim(0, 1); ax[0].set_title('Stepwise accuracy decomposition (mean and SD, 5 seeds)')
    dg = json.load(open(RES / 'domain_gap_42.json')); k0 = 'seed42'; lad = dg[k0]['adaptation_ladder']
    names = [('zero_shot', 'Zero-shot'), ('bias_only', '+ class-bias\n(label shift)'), ('probe_1024', '+ linear probe\n(1024-d feat.)'), ('probe_512', '+ linear probe\n(512-d feat.)')]
    ax[1].bar(range(4), [lad[k]['bacc'][0] for k, _ in names], yerr=[lad[k]['bacc'][1] for k, _ in names], color=[C['red'], C['orange'], C['green'], C['green']], alpha=.8, capsize=3)
    ref = dg[k0]['HAM_internal_probe_reference']['bacc'][0]; ax[1].axhline(ref, c=C['blue'], ls='--', lw=1); ax[1].text(3.4, ref + .01, 'HAM probe reference', ha='right', fontsize=6.5, color=C['blue'])
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([n for _, n in names], fontsize=7); ax[1].set_ylabel('Balanced accuracy (PAD eval. patients)'); ax[1].set_ylim(0, 1); ax[1].set_title(f'Adaptation ladder on PAD-UFES-20 ({k0})')
    fig.tight_layout(); save(fig, 'fig9_pad_decomposition.png')


def fig_gallery(d):
    MK = np.load(CACHE / 'ham_mask.npy', mmap_mode='r'); X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); cams = np.load(OUT / 'seed42' / 'cams_f16.npy').astype(np.float32)
    te = pd.read_csv(CACHE / 'ham_meta.csv'); te = te[te.split == 'test'].reset_index(drop=True); rng = np.random.RandomState(3); picks = []
    for cor in [True, False]:
        for sz in ['small', 'large']:
            q = d[(d.correct == cor) & (d['size'] == sz)]
            hi = q.sort_values('e_shuf').iloc[-1:]; lo = q.sort_values('e_shuf').iloc[:1]
            picks += [(i, cor, sz) for i in list(hi.index) + list(lo.index)]
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
    for k, (i, cor, sz) in enumerate(picks[:8]):
        img = np.ascontiguousarray(X[te.cache_idx[i]]); m = MK[te.cache_idx[i]].astype(np.uint8); h = cams[i]; cm = (plt.cm.jet(h)[..., :3] * 255).astype(np.uint8)
        ov = (0.55 * img + 0.45 * cm).astype(np.uint8); cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov, cnt, -1, (255, 255, 255), 2)
        ax[0, k].imshow(img); ax[1, k].imshow(ov); ax[0, k].axis('off'); ax[1, k].axis('off')
        ax[0, k].set_title(f'{CLASSES[d.label_idx[i]]}→{CLASSES[d.pred[i]]} ({"correct" if cor else "wrong"})\n{sz}, area={d.area224[i]:.2f}', fontsize=7)
        ax[1, k].set_title(f'LSAS={d.lsas[i]:.2f}  null={d.null_shuf_mean[i]:.2f}\nexcess={d.e_shuf[i]:+.2f}', fontsize=7)
    fig.tight_layout(); save(fig, 'fig10_gallery.png')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['all']; d = None
    if (RES / 'ref_per_image_enriched.csv').exists(): d = pd.read_csv(RES / 'ref_per_image_enriched.csv')
    def do(n, f, *a):
        if 'all' in which or n in which:
            try: f(*a); print('made', n)
            except Exception as e: print('FAILED', n, repr(e))
    do('dim', fig_dimensions); do('train', fig_training); do('perf', fig_perf)
    if d is not None:
        do('size', fig_size, d); do('corr', fig_correctness, d); do('faith', fig_faith, d); do('sts', fig_sts, d); do('skin', fig_skin, d); do('gallery', fig_gallery, d)
    do('pad', fig_pad)
