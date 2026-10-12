"""Target-domain evaluation and adaptation on PAD-UFES-20 (reviewer comment on the 0.278 accuracy).

Every number is computed from result files: out/results/pad_finetune.json (pad_finetune.py), the adaptation ladder in
domain_gap_*.json (domain_gap.py), and the PAD-UFES-20 label counts in cache/pad_meta.csv. Nothing is typed in."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / 'out' / 'results'
FT = RES / 'pad_finetune.json'
CLASSES5 = ['AKIEC', 'BCC', 'BKL', 'MEL', 'NV']


def available():
    if not FT.exists(): return False
    d = json.loads(FT.read_text(encoding='utf-8'))
    return len(d['seeds']) == 5


def ms(v, d=3):
    v = np.asarray(v, float); return f'{v.mean():.{d}f} (SD {v.std(ddof=1):.{d}f})'          # sample SD across the seed models, as everywhere else in the paper


def load():
    d = json.loads(FT.read_text(encoding='utf-8')); seeds = sorted(d['seeds'], key=int)
    mean = lambda key, j: np.array([np.mean([r[j] for r in d['seeds'][s][key]]) for s in seeds])        # per-seed mean over splits
    out = {'seeds': seeds, 'protocol': d['protocol'], 'n_splits': len(d['seeds'][seeds[0]]['fine_tuned']),
           'n_adapt': float(np.mean([np.mean(d['seeds'][s]['n_adapt']) for s in seeds])), 'n_eval': float(np.mean([np.mean(d['seeds'][s]['n_eval']) for s in seeds]))}
    for key, nm in (('zero_shot', 'zs'), ('fine_tuned', 'ft')):
        for j, m in enumerate(('acc', 'bacc', 'f1')): out[f'{nm}_{m}'] = mean(key, j)
    # ladder (Table 17) values, 20 splits, same protocol
    dg = {}
    for f in ('domain_gap_42.json', 'domain_gap_43_44_45.json', 'domain_gap_46.json'):
        dg.update({k: v for k, v in json.loads((RES / f).read_text(encoding='utf-8')).items() if k.startswith('seed')})
    out['ladder'] = {k: {m: np.array([dg[s]['adaptation_ladder'][k][m][0] for s in sorted(dg)]) for m in ('acc', 'bacc', 'f1')} for k in ('zero_shot', 'bias_only', 'probe_1024', 'probe_512')}
    for m in ('acc', 'bacc', 'f1'):
        out[f'zs_cached_{m}'] = out[f'zs_{m}']; out[f'zs_{m}'] = out['ladder']['zero_shot'][m]      # paired comparator = zero-shot row of the adaptation ladder (Table 17)
    out['zs_diff_max'] = float(max(np.abs(out[f'zs_cached_{m}'] - out[f'zs_{m}']).max() for m in ('acc', 'bacc', 'f1')))
    out['gain_acc'] = out['ft_acc'] - out['zs_acc']; out['gain_bacc'] = out['ft_bacc'] - out['zs_bacc']
    pad = pd.read_csv(HERE / 'cache' / 'pad_meta.csv'); lab = pad.label.dropna(); cnt = lab.value_counts()
    p = cnt.max() / cnt.sum(); out['majority'] = dict(cls=cnt.idxmax(), acc=float(p), bacc=0.2, f1=float((2 * p / (1 + p)) / 5), n=int(cnt.sum()), biopsy=None)
    return out


def adapted_parameters():
    """Number of parameters trained in the partial fine-tuning (dense block 4, final batch norm, head), counted from the architecture of common.DenseNet121_TrustXAI."""
    from torchvision import models
    f = models.densenet121(weights=None).features
    head = (1024 * 512 + 512) + 2 * 512 + (512 * 7 + 7)
    return sum(p.numel() for p in f.denseblock4.parameters()) + sum(p.numel() for p in f.norm5.parameters()) + head


def sec_adaptation(M, PA):
    A = load(); mj = A['majority']; L = A['ladder']
    E = [PA[s]['E_PAD_mapped_5way_output'] for s in sorted(PA)]
    zero_shot_acc_mean = float(np.mean([e['acc'] for e in E])); mean_conf = float(np.mean([e['mean_conf'] for e in E]))
    rec = {c: float(np.mean([e['recall'][c] for e in E])) for c in CLASSES5}; pnv = float(np.mean([e['pred_dist']['NV'] for e in E]))
    Eacc = np.array([e['acc'] for e in E]); Ebacc = np.array([e['bacc'] for e in E]); Ef1 = np.array([e['f1_present'] for e in E])
    rows = [[f"Majority-class rule (always {mj['cls']}; all {mj['n']:,} mapped images)", 'none', f"{mj['acc']:.3f}", f"{mj['bacc']:.3f}", f"{mj['f1']:.3f}"],
            [f"Zero-shot, all {mj['n']:,} mapped images ([[T:pad_decomp]], condition E)", 'none', ms(Eacc), ms(Ebacc), ms(Ef1)],
            [f"Zero-shot, mean over the {A['n_splits']} evaluation subsets (paired comparator)", 'none', ms(A['zs_acc']), ms(A['zs_bacc']), ms(A['zs_f1'])],
            ['+ 5-parameter class-bias correction', '5 parameters', ms(L['bias_only']['acc']), ms(L['bias_only']['bacc']), ms(L['bias_only']['f1'])],
            ['+ linear probe on frozen 512-d head features', 'linear classifier', ms(L['probe_512']['acc']), ms(L['probe_512']['bacc']), ms(L['probe_512']['f1'])],
            ['+ linear probe on frozen 1,024-d backbone features', 'linear classifier', ms(L['probe_1024']['acc']), ms(L['probe_1024']['bacc']), ms(L['probe_1024']['f1'])],
            ['+ partial fine-tuning (dense block 4, final batch norm and head)', f'{adapted_parameters() / 1e6:.1f} million parameters', ms(A['ft_acc']), ms(A['ft_bacc']), ms(A['ft_f1'])]]
    lo, hi = A['gain_acc'].min() * 100, A['gain_acc'].max() * 100
    M.p(f"__Is target-domain evaluation or adaptation needed?__ The data support both, and they answer different questions. "
        f"__Evaluation.__ The zero-shot accuracy of {zero_shot_acc_mean:.3f} is a deterministic evaluation of the trained models ([[T:pad_decomp]]) and not a usable estimate of what the model offers on clinical photographs. "
        f"It is lower than the {mj['acc']:.3f} that a rule predicting only the most frequent class ({mj['cls']}) would reach on the same images ([[T:pad_ft]]), and the mean confidence of the model ({mean_conf:.3f}) is far above its accuracy. "
        f"The shortfall is concentrated in the classes that dominate PAD-UFES-20: mean recall was {rec['AKIEC']:.2f} for AKIEC and {rec['BCC']:.2f} for BCC, which make up most of the images, while NV, the majority class of HAM10000, received {100 * pnv:.0f}% of all predictions. "
        f"A model trained on dermoscopic images therefore needs its own evaluation on target-domain patients, with patient-level splits and class-balanced metrics, before any use on such images. "
        f"__Adaptation.__ [[T:pad_ft]] shows what a small labelled target sample recovers. The adaptation set is 30% of the PAD-UFES-20 patients (on average {A['n_adapt']:.0f} images) and the evaluation set the remaining 70% ({A['n_eval']:.0f} images), with {A['n_splits']} patient-level splits per seed model and five seed models. "
        f"Fine-tuning dense block 4, the final batch normalisation and the classifier head raised accuracy from {ms(A['zs_acc'])} to {ms(A['ft_acc'])}, balanced accuracy from {ms(A['zs_bacc'])} to {ms(A['ft_bacc'])} and macro-F1 from {ms(A['zs_f1'])} to {ms(A['ft_f1'])}; "
        f"the accuracy gain was positive in all five seed models ({lo:.1f} to {hi:.1f} points). Linear probes on frozen features, which adapt far fewer parameters, reached {ms(L['probe_512']['acc'])} (512-d) and {ms(L['probe_1024']['acc'])} (1,024-d), and a class-bias correction alone {ms(L['bias_only']['acc'])} with lower balanced accuracy. "
        f"Most of the loss is therefore recoverable with a small target sample.")
    M.table(['Level (PAD-UFES-20, five classes, evaluation patients)', 'Parameters adapted', 'Accuracy', 'Balanced accuracy', 'Macro-F1'], rows,
            f"Zero-shot performance, reference rule and target-domain adaptation on PAD-UFES-20 (mean and SD across the five seed models of the mean over patient-level 30%/70% splits).",
            f"Notes: the second zero-shot row and all rows below it use the same {A['n_splits']} patient-level splits per seed model (30% of patients for adaptation, 70% for evaluation; the splits of the adaptation ladder, Section 3.12); the first zero-shot row is computed over all images, so the two zero-shot rows differ only by patient sampling. Cells give the mean and sample SD across the five seed models. "
            f"The paired comparator is the zero-shot row of [[T:pad_ladder]]; recomputing it on the cached-feature path used for fine-tuning agreed to within {A['zs_diff_max']:.3f} in every seed and measure. The majority-class row is one fixed rule (balanced accuracy 0.200 by construction). Fine-tuning used AdamW, 8 epochs, cosine decay, the study's layer-wise learning rates and class-weighted cross-entropy on the five mappable logits; earlier layers stayed frozen.", label='tab:pad_ft')
    M.p(f"__How to read these numbers.__ The adapted accuracy of about {A['ft_acc'].mean():.2f} is a different quantity from the zero-shot accuracy of {zero_shot_acc_mean:.3f}: it requires labelled target-domain patients and cannot be quoted as an improvement of the zero-shot result, which is unchanged. "
        f"It is also an optimistic research estimate for a new site, because adaptation and evaluation patients come from the same dataset and the adaptation labels are available. Balanced accuracy ({ms(A['ft_bacc'])}) remains well below accuracy, so rare classes stay the weak point. "
        f"Full fine-tuning, adaptation without target labels and adaptation on a different clinical site were not tested, so no statement is made about them, and none of this shows clinical adequacy.")


def numbers():
    A = load()
    return dict(ft_acc=A['ft_acc'].mean(), ft_acc_sd=A['ft_acc'].std(ddof=1), ft_bacc=A['ft_bacc'].mean(), ft_bacc_sd=A['ft_bacc'].std(ddof=1), zs_acc=A['zs_acc'].mean(), zs_acc_sd=A['zs_acc'].std(ddof=1),
                maj=A['majority']['acc'], gain_min=A['gain_acc'].min(), n_splits=A['n_splits'])
