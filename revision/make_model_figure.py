"""Figure 12: bar-chart comparison of the three model configurations (M1 DenseNet121, M2 ResNet50, M3 INT8 versions).
Colour = architecture (validated two-slot categorical palette, blue/orange); hatching = INT8 (secondary encoding, so identity never rests on colour alone).
One axis per panel (no dual axes); bars are five-seed means with SD whiskers; efficiency bars are single deterministic measurements."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common import *

RES = OUT / 'results'; FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True, parents=True)
CMP = json.load(open(RES / 'model_comparison.json'))['configs']; EFF = json.load(open(RES / 'efficiency.json'))
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 120, 'hatch.linewidth': 0.9})
fin = lambda k: (k + '_int8') if (k + '_int8') in CMP else (k + '_ptq')
SERIES = [('dn_fp32', 'M1 DenseNet121 (FP32)', BLUE, None), ('rn_fp32', 'M2 ResNet50 (FP32)', ORANGE, None),
          (fin('dn'), 'M3 DenseNet121 (INT8)', BLUE, '////'), (fin('rn'), 'M3 ResNet50 (INT8)', ORANGE, '////')]
n = len(SERIES); w = 0.19


def bars(ax, items, metric_keys, ylim=None, labels=True, fmt='{:.2f}'):
    x = np.arange(len(items))
    for i, (k, lab, col, hatch) in enumerate(SERIES):
        m = [CMP[k][mk]['mean'] for mk in metric_keys]; s = [CMP[k][mk]['sd'] for mk in metric_keys]
        xs = x + (i - (n - 1) / 2) * w
        ax.bar(xs, m, width=w * 0.86, color=col, hatch=hatch, edgecolor=SURF if hatch is None else 'white', linewidth=0.0 if hatch is None else 0.0, zorder=3)
        ax.errorbar(xs, m, yerr=s, fmt='none', ecolor=INK2, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=4)
        if labels:
            for xx, mm, ss in zip(xs, m, s): ax.text(xx, mm + ss + 0.012, fmt.format(mm).lstrip('0') if mm < 1 else fmt.format(mm), ha='center', va='bottom', fontsize=5.6, color=INK2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(items, fontsize=7.4); ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    if ylim: ax.set_ylim(*ylim)
    ax.tick_params(axis='x', length=0); ax.tick_params(axis='y', length=2)


fig = plt.figure(figsize=(11.6, 3.9), facecolor=SURF)
gs = fig.add_gridspec(1, 4, width_ratios=[4, 3.1, 1.45, 1.45], wspace=0.34)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1], sharey=a); c = fig.add_subplot(gs[2]); d = fig.add_subplot(gs[3])
bars(a, ['Accuracy', 'Balanced\naccuracy', 'Macro-\nF1', 'Macro-\nAUROC'], ['acc', 'bacc', 'f1', 'auroc'], (0, 1.14))
bars(b, ['PAD\naccuracy', 'PAD balanced\naccuracy', 'ECE after\nscaling'], ['pad_acc', 'pad_bacc', 'ece_ts'])
a.set_ylabel('Score (five-seed mean, SD whiskers)'); a.set_title('A  HAM10000 test set', loc='left', fontsize=8.5, fontweight='bold', color=INK)
b.set_title('B  PAD-UFES-20, calibration', loc='left', fontsize=8.5, fontweight='bold', color=INK); plt.setp(b.get_yticklabels(), visible=False)
# efficiency (deterministic single measurements)
keys = [s[0] for s in SERIES]
def eff_panel(ax, field, title, ylabel, fmt):
    vals = []
    for k, lab, col, hatch in SERIES:
        kk = k if (k in EFF and field in EFF[k]) else ('dn_fp32' if k.startswith('dn') else 'rn_fp32'); vals.append(EFF[kk][field])
    xs = np.arange(n) * 0.5
    for xx, v, (k, lab, col, hatch) in zip(xs, vals, SERIES):
        ax.bar(xx, v, width=0.42, color=col, hatch=hatch, edgecolor='white', linewidth=0, zorder=3)
        ax.text(xx, v + max(vals) * 0.02, fmt.format(v), ha='center', va='bottom', fontsize=6, color=INK2)
    ax.set_xticks([]); ax.set_ylim(0, max(vals) * 1.18); ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold', color=INK); ax.set_ylabel(ylabel); ax.tick_params(axis='y', length=2)
eff_panel(c, 'size_mb', 'C  Model size', 'Serialized size (MB)', '{:.0f}')
eff_panel(d, 'cpu_b32_ms', 'D  CPU latency', 'ms per image (batch 32)', '{:.0f}')
handles = [Patch(facecolor=col, hatch=hatch, edgecolor='white' if hatch else col, label=lab) for k, lab, col, hatch in SERIES]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06), handlelength=1.6, columnspacing=1.8)
fig.savefig(FIG / 'fig12_model_comparison.png', dpi=300, bbox_inches='tight', facecolor=SURF); plt.close(fig)
print('Figure 12 saved')
