"""Figure 1: architecture and workflow of the complete pipeline for the three model configurations
(M1 DenseNet121, M2 ResNet50, M3 INT8 versions).  Clean three-column layout with straight, non-crossing arrows:
explanation audits | external evaluation | INT8 deployment, converging on multi-seed aggregation and model comparison.
Written to fig1_trust_dimensions.png (the first figure of the manuscript)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
OUT = Path(__file__).resolve().parent / 'out'; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({'font.family': 'DejaVu Sans'})
C = dict(blue='#1f5fa8', orange='#d95f02', green='#1b9e77', grey='#7f7f7f', red='#c0392b', purple='#6a3d9a', black='#222222', lgrey='#eef0f2')


def box(ax, x, y, w, h, text, edge, fs=7.8, alpha=0.14, fc=None, lw=1.3, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=(fc or edge), ec='none', alpha=alpha, zorder=z))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc='none', ec=edge, lw=lw, zorder=z + 1))
    if text: ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=C['black'], zorder=z + 2, linespacing=1.35)


def arrow(ax, p0, p1, lw=1.3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=11, color=C['black'], lw=lw, shrinkA=0, shrinkB=0, zorder=6))


def line(ax, pts, lw=1.3):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C['black'], lw=lw, zorder=5, solid_capstyle='butt')


fig, ax = plt.subplots(figsize=(12.5, 9.0))
ax.set_xlim(0, 12.5); ax.set_ylim(1.0, 15.5); ax.axis('off')
# ---- Row A: data
box(ax, 0.3, 14.2, 3.3, 1.0, 'HAM10000\n10,015 images / 7,470 lesions\n(training + internal evaluation)', C['blue'])
box(ax, 4.4, 14.2, 3.7, 1.0, 'Lesion-level split (seed 42, fixed)\ntrain 8,020 / val 1,005 / test 990', C['grey'])
box(ax, 8.9, 14.2, 3.3, 1.0, 'PAD-UFES-20\n2,298 images\n(external evaluation only)', C['red'])
arrow(ax, (3.6, 14.7), (4.4, 14.7))
# ---- Row B: preprocessing and training of M1 and M2
box(ax, 0.5, 12.55, 3.8, 1.1, 'Pre-processing\nDullRazor hair removal, resize 224 x 224,\nImageNet normalisation', C['grey'])
box(ax, 5.0, 12.55, 2.7, 1.1, 'M1 DenseNet121\n5 seeds (42-46)\none-cycle, 60 epochs', C['blue'])
box(ax, 8.1, 12.55, 2.7, 1.1, 'M2 ResNet50\n5 seeds (42-46)\nidentical recipe', C['orange'])
line(ax, [(6.25, 14.2), (6.25, 13.95), (2.4, 13.95)]); arrow(ax, (2.4, 13.95), (2.4, 13.65))
arrow(ax, (4.3, 13.1), (5.0, 13.1))
# ---- bus from the two trained models to the three branches
line(ax, [(6.35, 12.55), (6.35, 12.25)]); line(ax, [(9.45, 12.55), (9.45, 12.25)]); line(ax, [(6.35, 12.25), (9.45, 12.25)])
line(ax, [(7.9, 12.25), (7.9, 12.1)]); line(ax, [(2.3, 12.1), (10.45, 12.1)])
for x in (2.3, 6.35, 10.45): arrow(ax, (x, 12.1), (x, 11.85))
# ---- Row C: three branch heads
box(ax, 0.3, 10.85, 4.0, 1.0, 'Grad-CAM++ saliency map (predicted class)\nM1: final dense block\nM2: final bottleneck convolution', C['orange'], fs=7.6)
box(ax, 4.55, 10.85, 3.6, 1.0, 'Temperature scaling\nfitted on val_cal (disjoint from\ncheckpoint-selection val_sel)', C['grey'], fs=7.6)
box(ax, 8.4, 10.85, 3.8, 1.0, 'M3: INT8 post-training quantization\nconfiguration chosen on val_sel;\ncalibration on val_cal only', C['green'], fs=7.6)
# ---- Column 1: four parallel audits in a container
box(ax, 0.2, 5.55, 4.2, 4.85, '', C['grey'], alpha=0.07, lw=1.0, z=0)
ax.text(2.3, 10.2, 'Four parallel explanation-reliability audits', ha='center', va='center', fontsize=7.4, style='italic', color=C['black'], zorder=5)
aud = [('Null-controlled localization\nLSAS vs. shuffled / shift null, CAM ceiling,\npermutation test -> excess LSAS', C['green']),
       ('Perturbation faithfulness\ndeletion / insertion; DG, IG,\nnull-controlled EIG, EDG', C['purple']),
       ('Stochastic stability\nMC dropout, 15 passes, 300 images -> STS', C['blue']),
       ('Skin-tone-proxy audit\nITA strata, continuous ITA, power analysis', C['red'])]
for i, (t, c) in enumerate(aud):
    box(ax, 0.4, 8.75 - i * 1.05, 3.8, 0.95, t, c, fs=6.9)
arrow(ax, (2.3, 10.85), (2.3, 10.4))
# ---- Column 2: external evaluation
box(ax, 4.55, 9.1, 3.6, 1.25, 'External evaluation, PAD-UFES-20\n(no retraining; 2,106 images\nafter class mapping)', C['red'], fs=7.6)
box(ax, 4.55, 7.3, 3.6, 1.35, 'Stepwise decomposition\npopulation -> label space -> prevalence\n-> residual (sums to the total drop)', C['red'], fs=7.6)
box(ax, 4.55, 5.55, 3.6, 1.2, 'Domain classifier, Frechet distance,\nadaptation ladder', C['red'], fs=7.6)
arrow(ax, (6.35, 10.85), (6.35, 10.35)); arrow(ax, (6.35, 9.1), (6.35, 8.65)); arrow(ax, (6.35, 7.3), (6.35, 6.75))
# ---- Column 3: INT8 deployment
box(ax, 8.4, 9.1, 3.8, 1.25, 'Quantization-aware refinement\n(only if outside the pre-specified\ntolerance); best epoch on val_sel', C['green'], fs=7.6)
box(ax, 8.4, 7.3, 3.8, 1.35, 'INT8 evaluation (identical metrics)\nsize, CPU latency, agreement with\nFP32 parent; paired tests', C['green'], fs=7.6)
arrow(ax, (10.3, 10.85), (10.3, 10.35)); arrow(ax, (10.3, 9.1), (10.3, 8.65))
# ---- aggregation and output
box(ax, 0.9, 3.6, 10.7, 1.2, 'Multi-seed aggregation and model comparison (5 seeds each): mean (SD), paired tests,\nlesion-clustered bootstrap CIs, criterion-wise winners for M1, M2 and M3', C['grey'], fs=8.2)
arrow(ax, (2.3, 5.55), (2.3, 4.8)); arrow(ax, (6.35, 5.55), (6.35, 4.8)); arrow(ax, (10.3, 7.3), (10.3, 4.8))
box(ax, 2.1, 1.7, 8.3, 1.3, 'Trust report: classification, calibration, null-controlled localization, faithfulness, stability,\nskin-tone-proxy evidence, external generalization, INT8 efficiency -- separate, non-interchangeable\ndimensions', C['black'], fc=C['lgrey'], alpha=None, fs=8.0)
arrow(ax, (6.25, 3.6), (6.25, 3.0))
plt.tight_layout()
fig.savefig(FIG / 'fig1_trust_dimensions.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('Figure 1 (pipeline diagram) regenerated ->', FIG / 'fig1_trust_dimensions.png')
