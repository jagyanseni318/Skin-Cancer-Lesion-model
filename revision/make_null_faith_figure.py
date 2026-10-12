"""Figure 11: null-controlled faithfulness (excess insertion/deletion gain), reference seed 42."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True, parents=True)
C = dict(blue='#1f5fa8', orange='#d95f02', green='#1b9e77', grey='#7f7f7f', red='#c0392b', purple='#6a3d9a', black='#222222')
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})

REF = 42
nf = pd.read_csv(OUT / f'seed{REF}' / 'null_faithfulness.csv')
pi = pd.read_csv(OUT / f'seed{REF}' / 'per_image.csv')
d = nf.merge(pi[['image_id', 'lsas', 'null_shuf_mean', 'correct', 'area224']], on='image_id')
d['e_shuf'] = d.lsas - d.null_shuf_mean
allseeds = pd.read_csv(OUT / 'results' / 'null_faith_per_seed.csv')


def binned(x, y, nb=12):
    q = np.quantile(x, np.linspace(0, 1, nb + 1)); idx = np.clip(np.digitize(x, q[1:-1]), 0, nb - 1)
    return np.array([np.median(x[idx == b]) for b in range(nb)]), np.array([np.median(y[idx == b]) for b in range(nb)])


fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.2))
for a, col, lab in [(ax[0], 'mean_EIG', 'Excess insertion gain (EIG)'), (ax[1], 'mean_EDG', 'Excess deletion gain (EDG)')]:
    a.hist(d[col], bins=30, color=C['purple'] if 'EIG' in col else C['green'], ec='white', alpha=.85)
    a.axvline(0, c='k', lw=1); a.axvline(d[col].median(), c=C['red'], ls='--', lw=1.3, label=f'median {d[col].median():+.3f}')
    a.set_xlabel(lab); a.set_ylabel('Test images'); a.set_title(f'{lab} (reference seed)'); a.legend(frameon=False, fontsize=7)

ax[2].scatter(d.e_shuf, d.mean_EIG, s=6, alpha=.35, c=np.where(d.correct, C['blue'], C['red']), edgecolors='none')
ax[2].plot(*binned(d.e_shuf.values, d.mean_EIG.values), c='k', lw=1.8); ax[2].axhline(0, c='k', lw=.8)
ax[2].set_xlabel('Excess LSAS'); ax[2].set_ylabel('Excess insertion gain (EIG)'); ax[2].set_title('Faithfulness vs. localization (both null-controlled)')

ord_ = ['Seed 42', 'Seed 43', 'Seed 44', 'Seed 45', 'Seed 46']; xs = np.arange(5)
w = 0.35
bp1 = ax[3].boxplot([pd.read_csv(OUT / f'seed{s}' / 'null_faithfulness.csv').mean_EIG for s in SEEDS], positions=xs - w / 2, widths=w * .85, patch_artist=True, showfliers=False, medianprops=dict(color='k'))
bp2 = ax[3].boxplot([pd.read_csv(OUT / f'seed{s}' / 'null_faithfulness.csv').mean_EDG for s in SEEDS], positions=xs + w / 2, widths=w * .85, patch_artist=True, showfliers=False, medianprops=dict(color='k'))
for b in bp1['boxes']: b.set_facecolor(C['purple']); b.set_alpha(.55)
for b in bp2['boxes']: b.set_facecolor(C['green']); b.set_alpha(.55)
ax[3].axhline(0, c='k', lw=.8); ax[3].set_xticks(xs); ax[3].set_xticklabels(['42', '43', '44', '45', '46']); ax[3].set_xlabel('Seed')
ax[3].plot([], [], c=C['purple'], lw=6, alpha=.55, label='EIG'); ax[3].plot([], [], c=C['green'], lw=6, alpha=.55, label='EDG')
ax[3].legend(frameon=False, fontsize=7); ax[3].set_title('Consistency across seeds')
fig.tight_layout(); fig.savefig(FIG / 'fig11_null_faithfulness.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('Figure 11 saved')
