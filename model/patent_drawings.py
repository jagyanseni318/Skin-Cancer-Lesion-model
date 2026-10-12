"""Patent-style black-and-white line drawings (FIG. 1 system block diagram, FIG. 2 method flowchart) and grayscale versions of experimental figures."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from pathlib import Path
OUT = Path(__file__).resolve().parent / 'out'
PD = OUT / 'patent'; PD.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8})


def box(ax, x, y, w, h, text, num, dashed=False, fs=7.5):
    ax.add_patch(Rectangle((x, y), w, h, fc='white', ec='black', lw=1.2, ls='--' if dashed else '-'))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)
    ax.text(x + w - 0.05, y + h + 0.07, num, ha='right', va='bottom', fontsize=7.5, fontweight='bold')


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color='black', lw=1.0, shrinkA=0, shrinkB=0))


def fig1():
    fig, ax = plt.subplots(figsize=(8.2, 6.2)); ax.set_xlim(0, 16.4); ax.set_ylim(0, 12.4); ax.axis('off')
    box(ax, 0.3, 10.2, 3.2, 1.6, 'Dermoscopic\nimage source', '100')
    box(ax, 4.4, 10.2, 3.2, 1.6, 'Pre-processing\nunit\n(hair removal, resize,\nnormalise)', '102')
    box(ax, 8.5, 10.2, 3.2, 1.6, 'Neural-network\nclassifier\n(predicted class)', '104')
    box(ax, 12.6, 10.2, 3.5, 1.6, 'Confidence\ncalibration unit\n(temperature scaling)', '106')
    for a, b in [(3.5, 4.4), (7.6, 8.5), (11.7, 12.6)]: arrow(ax, a, 11.0, b, 11.0)
    box(ax, 8.5, 7.7, 3.2, 1.6, 'Saliency-map\ngenerator\n(Grad-CAM++)', '108'); arrow(ax, 10.1, 10.2, 10.1, 9.3)
    box(ax, 0.3, 7.7, 3.2, 1.6, 'Lesion-mask\nsource\n(segmentation)', '110')
    box(ax, 4.4, 4.9, 3.6, 2.0, 'Localization scorer\n(mask-overlap score,\nLSAS)', '112'); arrow(ax, 8.5, 8.5, 6.6, 6.9); arrow(ax, 1.9, 7.7, 5.0, 6.9)
    box(ax, 0.3, 4.9, 3.2, 2.0, 'Reference\ngenerator:\nnull + ceiling\n(shuffle / shift /\noracle)', '114'); arrow(ax, 3.5, 5.9, 4.4, 5.9); arrow(ax, 8.5, 8.3, 3.0, 6.9)
    box(ax, 12.6, 7.7, 3.5, 1.6, 'Faithfulness\nevaluator\n(insertion / deletion)', '116'); arrow(ax, 11.7, 8.5, 12.6, 8.5)
    box(ax, 12.6, 5.3, 3.5, 1.6, 'Stability evaluator\n(stochastic passes)', '118'); arrow(ax, 10.1, 7.7, 12.6, 6.1)
    box(ax, 12.6, 2.9, 3.5, 1.6, 'Domain-shift\nmonitor', '120', dashed=True); arrow(ax, 14.35, 10.2, 14.35, 9.3) if False else None
    box(ax, 5.3, 1.6, 5.6, 1.9, 'Trust aggregator\n(excess localization, faithfulness,\nstability, calibrated confidence,\ndomain flag)', '122')
    arrow(ax, 6.2, 4.9, 6.9, 3.5); arrow(ax, 12.6, 8.0, 10.0, 3.5); arrow(ax, 12.6, 5.8, 10.6, 3.2); arrow(ax, 12.6, 3.7, 10.9, 2.6); arrow(ax, 14.35, 10.2, 14.35, 4.5) if False else None
    ax.plot([16.1, 16.3, 16.3], [11.0, 11.0, 3.7], 'k', lw=1.0); ax.annotate('', xy=(10.9, 2.9), xytext=(16.3, 3.7), arrowprops=dict(arrowstyle='-|>', color='black', lw=1.0)) if False else None
    ax.plot([16.3, 16.3], [3.7, 1.0], 'k', lw=1.0); ax.plot([16.3, 8.1], [1.0, 1.0], 'k', lw=1.0); arrow(ax, 8.1, 1.0, 8.1, 1.6)
    box(ax, 0.3, 0.2, 5.0, 1.4, 'Output interface: display, suppress,\nannotate or defer explanation', '124'); arrow(ax, 5.3, 2.5, 3.0, 1.6)
    ax.text(0.2, 12.2, 'FIG. 1', fontsize=11, fontweight='bold'); ax.text(8.2, 0.05, 'System 10', ha='center', fontsize=8, style='italic')
    fig.savefig(PD / 'patent_fig1_system.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def fig2():
    steps = ['S200  Receive dermoscopic image and\nlesion mask M; obtain predicted class c\nand calibrated confidence',
             'S202  Generate saliency map H for class c\nat target layer of resolution r × r,\nupsample and normalise',
             'S204  Compute mask-overlap score\nLSAS = Σ_M H / Σ H',
             'S206  Compute reference expectation E:\nmean overlap of M with maps of other\nimages (or shifted copies of H)',
             'S208  Compute ceiling Cmax: overlap of M pooled\nto r × r and upsampled with M',
             'S210  Compute excess e = LSAS − E and,\noptionally, attainable fraction LSAS / Cmax',
             'S212  Compute insertion/deletion gains for H versus\nrandom patch order; compute stability under\nstochastic passes',
             'S214  Aggregate e, gains, stability and calibrated\nconfidence into reliability indicator',
             'S216  Display / suppress / annotate / defer\nthe explanation according to indicator']
    fig, ax = plt.subplots(figsize=(6.2, 8.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 22.6); ax.axis('off')
    y = 21.4
    for i, t in enumerate(steps):
        h = 1.9 if i not in (6,) else 2.2
        ax.add_patch(Rectangle((1.0, y - h), 8.0, h, fc='white', ec='black', lw=1.2)); ax.text(5.0, y - h / 2, t, ha='center', va='center', fontsize=7.6)
        if i < len(steps) - 1: arrow(ax, 5.0, y - h, 5.0, y - h - 0.55)
        y -= h + 0.55
    ax.text(0.1, 22.3, 'FIG. 2', fontsize=11, fontweight='bold')
    fig.savefig(PD / 'patent_fig2_flow.png', dpi=300, bbox_inches='tight'); plt.close(fig)


def gray(src, dst):
    Image.open(OUT / 'figures' / src).convert('L').save(PD / dst)


fig1(); fig2()
for s, d in [('fig4_lesion_size_audit.png', 'patent_fig3.png'), ('fig6_faithfulness.png', 'patent_fig4.png'), ('fig7_stability.png', 'patent_fig5.png'),
             ('fig9_pad_decomposition.png', 'patent_fig6.png'), ('fig10_gallery.png', 'patent_fig7.png'), ('fig1_trust_dimensions.png', 'patent_fig8.png')]:
    gray(s, d)
print('drawings done')
