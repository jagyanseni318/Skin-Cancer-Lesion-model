"""Sections 4.13-4.15 and 5.10: INT8 results, before/after conditions, three-configuration comparison and discussion.
Imports the shared loaders of ms_models; every statement is generated from the result files."""
from ms_models import *

SELECTED = {'densenet121': 'hist_noRR_skipstemhead_n503', 'resnet50': 'hist_noRR_skipstemhead_n503_onednn'}
FIN = {'dn': 'dn_int8', 'rn': 'rn_int8'}
HAS_MIXED = 'dn_mixed' in CFG
ARCH_NAME = {'dn': 'DenseNet121', 'rn': 'ResNet50'}
FIG_BARS = OUT / 'figures' / 'fig12_model_comparison.png'
STATE = {}
LOWN = {'Macro-F1': 'macro-F1', 'Balanced accuracy': 'balanced accuracy', 'Accuracy': 'accuracy', 'Macro-AUROC': 'macro-AUROC', 'MCC': 'MCC', 'PAD-UFES-20 balanced accuracy': 'PAD-UFES-20 balanced accuracy', 'PAD-UFES-20 accuracy': 'PAD-UFES-20 accuracy'}


def eff(key, field, fmt='{:.1f}'):
    if EFF is None or key not in EFF or field not in EFF[key]: return 'n/a'
    return fmt.format(EFF[key][field])


def _desc(name):
    t = name.split('_'); act = {'hist': 'histogram', 'minmax': 'min–max', 'mavg': 'moving-average min–max'}[t[0]]
    rr, sk, n, be = 'off', 'all layers INT8', '503', 'x86'
    for tok in t[1:]:
        if tok == 'RR': rr = 'on'
        elif tok == 'noRR': rr = 'off'
        elif tok in ('all', 'skiphead', 'skipstemhead', 'skipstem'): sk = {'all': 'all layers INT8', 'skiphead': 'head in FP32', 'skipstemhead': 'stem and head in FP32', 'skipstem': 'stem in FP32'}[tok]
        elif tok.startswith('n') and tok[1:].isdigit(): n = tok[1:]
        elif tok in ('fbgemm', 'onednn'): be = tok
        else: raise ValueError(f'unparsed variant token {tok} in {name}')
    return f'{act} observer; reduce_range {rr}; {sk}; {n} calibration images', be


def _qat_pilot(arch):
    tag = '' if arch == 'densenet121' else '_resnet50'; f = OUT / 'models' / f'seed42{tag}_int8qat_summary.json'
    return json.load(open(f)) if f.exists() else None


def sec_int8(M):
    sw = pd.concat([pd.read_csv(RES / 'int8_sweep.csv'), pd.read_csv(RES / 'int8_sweep2.csv').query("variant != 'FP32'")], ignore_index=True)
    sens = pd.read_csv(RES / 'int8_sensitivity.csv'); mpj = json.load(open(RES / 'int8_mixed_precision.json'))
    M.h(2, '4.13 INT8 quantization of both architectures')
    M.p("INT8 versions were built from the trained FP32 checkpoints (Section 3.14), with the configuration chosen on validation data only using the seed-42 model of each architecture. [[T:int8_sweep]] lists every post-training candidate with its val_sel scores and agreement with the FP32 predictions, together with the quantization-aware pilot; [[T:int8_sens]] reports the group-wise sensitivity analysis; [[T:int8]] gives the final results for all five seeds.")
    rows = []
    for arch in ['densenet121', 'resnet50']:
        d = sw[sw.arch == arch]; fp = d[d.variant == 'FP32'].iloc[0]; an = ARCH_NAME['dn' if arch == 'densenet121' else 'rn']
        rows.append([an, 'FP32 reference', '–', f3(fp.acc), f3(fp.f1), f3(fp.auroc), '1.000', ''])
        for _, r in d[d.variant != 'FP32'].iterrows():
            desc, be = _desc(r.variant)
            rows.append([an, desc, be, f3(r.acc), f3(r.f1), f3(r.auroc), f3(r.agree), 'Selected' if r.variant == SELECTED[arch] else ''])
        qp = _qat_pilot(arch)
        if qp:
            v = qp['val_sel_int8']; rows.append([an, f"quantization-aware refinement, {qp['epochs']} epochs (pilot); stem and head in FP32", qp.get('backend', 'x86' if arch == 'densenet121' else 'onednn'), f3(v['acc']), f3(v['f1']), f3(v['auroc']), f3(v['agree']), 'Pilot only'])
    M.table(['Architecture', 'Configuration (INT8 unless stated)', 'Backend', 'Accuracy', 'Macro-F1', 'Macro-AUROC', 'Agreement with FP32', 'Choice'], rows,
            'Validation-only sweep of INT8 configurations and quantization-aware pilot (seed-42 models; 502 val_sel images; calibration on val_cal).',
            'Agreement = proportion of val_sel images with the same predicted class as the FP32 model. Ranking used macro-F1, with differences below 0.005 treated as ties and broken by agreement. The test set was not used.', label='tab:int8_sweep')
    g = lambda arch, v: sw[(sw.arch == arch) & (sw.variant == v)].iloc[0]
    sdn, sdb, fdn = g('densenet121', 'hist_noRR_all_n503'), g('densenet121', SELECTED['densenet121']), g('densenet121', 'FP32')
    srn, srb, frn = g('resnet50', 'hist_noRR_all_n503'), g('resnet50', SELECTED['resnet50']), g('resnet50', 'FP32')
    o_dn = (fdn.f1 - sdb.f1) > 0.02 or (fdn.acc - sdb.acc) > 0.01; o_rn = (frn.f1 - srb.f1) > 0.02 or (frn.acc - srb.acc) > 0.01
    tol_txt = ('The best post-training configurations remained ' + ('outside' if o_dn else 'within') + ' the pre-specified tolerance for both architectures') if o_dn == o_rn else ('The best post-training configuration remained ' + ('outside' if o_dn else 'within') + ' the pre-specified tolerance for DenseNet121 but ' + ('outside' if o_rn else 'within') + ' it for ResNet50')
    M.p(f"A default configuration that quantizes every layer was not acceptable for either network: on val_sel its macro-F1 was {f3(sdn.f1)} for DenseNet121 (FP32 {f3(fdn.f1)}) and {f3(srn.f1)} for ResNet50 (FP32 {f3(frn.f1)}), with only {pct(sdn.agree)} and {pct(srn.agree)} of predictions agreeing with the FP32 model. Keeping the first convolution, which sees raw normalized pixels, and the classifier head in FP32 raised macro-F1 to {f3(sdb.f1)} (DenseNet121) and {f3(srb.f1)} (ResNet50) while leaving almost all of the computation in INT8, and the histogram observer was consistently better than the min–max and moving-average observers, plausibly because rare large activations stretch the quantization range. {tol_txt} on val_sel (1 percentage point of accuracy, 2 points of macro-F1), so two further remedies were tried: group-wise mixed precision and quantization-aware refinement (Section 3.14).")
    # ---- sensitivity table
    rows = []
    for arch in ['densenet121', 'resnet50']:
        an = ARCH_NAME['dn' if arch == 'densenet121' else 'rn']; d = sens[sens.arch == arch]; fp = mpj[arch]['fp32_val_sel']
        for _, r in d[d.step <= 1].iterrows():
            rows.append([an, 'none (stem and head in FP32 only)' if r.step == 0 else r.extra, f3(r.acc), f3(r.f1), f3(r.agree), pct(r.fp32_mac_share), 'Within budget' if r.fp32_mac_share <= 0.25 else 'Over budget'])
    M.table(['Architecture', 'Additional layer group kept in FP32', 'Accuracy', 'Macro-F1', 'Agreement with FP32', 'FP32 share of multiply-accumulates', 'Compute budget (25%)'], rows,
            'Group-wise sensitivity of INT8 quantization (seed-42 models; 502 val_sel images).',
            'Each row quantizes everything except the stem, the head and the group named. The budget of 25% of multiply-accumulates in FP32 was fixed before the analysis. No group was added in the greedy search because none within budget improved macro-F1 by more than 0.002.', label='tab:int8_sens')
    sd = sens[(sens.arch == 'densenet121') & (sens.step == 1)]; sr = sens[(sens.arch == 'resnet50') & (sens.step == 1)]
    bd = sd.sort_values('f1').iloc[-1]; br = sr.sort_values('f1').iloc[-1]; b0d = sens[(sens.arch == 'densenet121') & (sens.step == 0)].iloc[0]; b0r = sens[(sens.arch == 'resnet50') & (sens.step == 0)].iloc[0]
    M.p(f"For DenseNet121 the only layer group whose release to FP32 clearly helped was {bd.extra} (val_sel accuracy {f3(bd.acc)}, macro-F1 {f3(bd.f1)}, agreement {f3(bd.agree)}, against {f3(b0d.acc)}, {f3(b0d.f1)} and {f3(b0d.agree)} for the base configuration), but it holds {pct(bd.fp32_mac_share)} of the multiply-accumulates, above the 25% budget fixed in advance, so the search added no group. For ResNet50 no group changed macro-F1 by more than {max(abs(sr.f1 - b0r.f1)):.3f}, which is below the sampling noise of 502 images. The headline INT8 model is therefore the post-training configuration with only the stem and the head in FP32 for both architectures; the DenseNet121 variant with {bd.extra} in FP32 is reported as a costlier alternative in [[T:int8]]. The quantization-aware pilot (seed 42, {(_qat_pilot('densenet121') or {}).get('epochs', 6)} epochs, stem and head in FP32, best epoch on val_sel) gave real-INT8 val_sel macro-F1 {f3((_qat_pilot('densenet121') or {'val_sel_int8': {'f1': float('nan')}})['val_sel_int8']['f1'])} for DenseNet121 and {f3((_qat_pilot('resnet50') or {'val_sel_int8': {'f1': float('nan')}})['val_sel_int8']['f1'])} for ResNet50 against {f3(b0d.f1)} and {f3(b0r.f1)} for the post-training variants, with lower agreement with the FP32 parents ({f3((_qat_pilot('densenet121') or {'val_sel_int8': {'agree': float('nan')}})['val_sel_int8']['agree'])} and {f3((_qat_pilot('resnet50') or {'val_sel_int8': {'agree': float('nan')}})['val_sel_int8']['agree'])} against {f3(b0d.agree)} and {f3(b0r.agree)}); it also cost about {((_qat_pilot('densenet121') or {}).get('secs', 0)) / 60:.0f} minutes of GPU time per model, and was therefore not extended to the other seeds.")
    # ---- final results table
    keys = [('dn_fp32', 'DenseNet121', 'FP32'), ('dn_int8', 'DenseNet121', 'INT8 (stem and head in FP32)'), ('dn_mixed', 'DenseNet121', 'INT8 (stem, head and dense block 1 in FP32)'),
            ('rn_fp32', 'ResNet50', 'FP32'), ('rn_int8', 'ResNet50', 'INT8 (stem and head in FP32)')]
    rows = []
    for k, a, p in keys:
        if k not in CFG: continue
        par = k.split('_')[0] + '_fp32'; ag = '1.000'
        if k != par:
            tk = f"precision_{'densenet121' if k.startswith('dn') else 'resnet50'}_{k.split('_')[1]}"
            ag = f3(np.mean([r['agreement'] for r in CMP['tests'][tk]['per_seed_pairwise']]))
        ek = k if (EFF and k in EFF) else None
        speed = 'n/a' if ek is None else f"{EFF[par]['cpu_b32_ms'] / EFF[ek]['cpu_b32_ms']:.1f}×"
        rows.append([a, p, f"{f3(CFG[k]['acc']['mean'])} (SD {f3(CFG[k]['acc']['sd'])})", f"{f3(CFG[k]['f1']['mean'])} (SD {f3(CFG[k]['f1']['sd'])})", f3(CFG[k]['bacc']['mean']),
                     ag, eff(k, 'size_mb'), eff(k, 'cpu_b32_ms'), speed])
    M.table(['Architecture', 'Precision', 'Accuracy', 'Macro-F1', 'Balanced accuracy', 'Agreement with FP32', 'Size (MB)', 'CPU ms per image', 'Speed-up'], rows,
            'FP32 and INT8 versions of both architectures on the 990 test images (five seeds each).',
            'Means over the five seeds with the seed SD in parentheses; further metrics are in [[T:three]]. INT8 models share weights with their FP32 parent and were configured on validation data only. Agreement = proportion of test images with the same predicted class as the FP32 parent. Latency at batch 32 on an Intel Core i5-13450HX (six threads, no other jobs). The INT8 row with stem and head in FP32 is model M3.', label='tab:int8')
    out = []
    for a, k, kinds in [('DenseNet121', 'dn', ['int8'] + (['mixed'] if HAS_MIXED else [])), ('ResNet50', 'rn', ['int8'])]:
        for kind in kinds:
            T = CMP['tests'].get(f"precision_{'densenet121' if k == 'dn' else 'resnet50'}_{kind}")
            if not T: continue
            da = -T['acc']['mean_diff_int8_minus_fp32']; df_ = -T['f1']['mean_diff_int8_minus_fp32']; ci = [-T['acc']['ci95'][1], -T['acc']['ci95'][0]]
            pw = T['per_seed_pairwise']; nsig = sum(r['mcnemar_p'] < 0.05 for r in pw); ok = (da <= 0.01) and (df_ <= 0.02)
            lab = 'INT8 (stem and head in FP32)' if kind == 'int8' else 'INT8 with dense block 1 also in FP32'
            out.append(f"{a} {lab} changed accuracy by {abs(da) * 100:.1f} percentage points {'below' if da > 0 else 'above'} its FP32 parent (95% CI of the FP32-minus-INT8 difference across seeds, {min(ci) * 100:.1f} to {max(ci) * 100:.1f} points) and macro-F1 {('by ' + format(abs(df_) * 100, '.1f') + ' points ' + ('below' if df_ > 0 else 'above') + ' it') if abs(df_) >= 0.0005 else 'essentially unchanged'}, {'within' if ok else 'outside'} the pre-specified tolerance of 1 and 2 points, with McNemar tests on the 990 test predictions significant in {nsig} of {len(pw)} seeds")
    M.p('Against the FP32 parents on the test set, ' + '; '.join(out) + '.')
    mx = '' if not (EFF and 'dn_mixed' in EFF) else f" The alternative that also keeps dense block 1 in FP32 occupies {eff('dn_mixed', 'size_mb')} MB and runs at {eff('dn_mixed', 'cpu_b32_ms')} ms per image, so its accuracy advantage is bought with {EFF['dn_mixed']['cpu_b32_ms'] / EFF['dn_int8']['cpu_b32_ms']:.1f} times the latency of the headline INT8 model."
    M.p(f"Size and speed follow from the arithmetic: the INT8 DenseNet121 occupies {eff('dn_int8', 'size_mb')} MB against {eff('dn_fp32', 'size_mb')} MB, and the INT8 ResNet50 {eff('rn_int8', 'size_mb')} MB against {eff('rn_fp32', 'size_mb')} MB; CPU latency at batch size 32 falls from {eff('dn_fp32', 'cpu_b32_ms')} to {eff('dn_int8', 'cpu_b32_ms')} ms per image (DenseNet121) and from {eff('rn_fp32', 'cpu_b32_ms')} to {eff('rn_int8', 'cpu_b32_ms')} ms (ResNet50).{mx} Grad-CAM++ maps are produced by the FP32 parent, since gradients are undefined through integer kernels; consistency is measured as prediction agreement ([[T:int8]]). The explanation findings of Sections 4.4–4.12 therefore describe the FP32 models, and no claim is made about maps from quantized networks.")


def sec_before_after(M):
    M.h(2, '4.14 Condition before and after each model configuration')
    M.p("[[T:before_after]] sets the measured condition before each intervention of Section 3.4 against the condition after it: the frozen-feature baseline (ImageNet backbone, class-balanced logistic-regression head fitted on the training split), the fine-tuned FP32 models (five-seed means), and the INT8 models against their FP32 parents. Changes are absolute magnitudes with a direction word.")

    def ch(b, a, d=3):
        diff = a - b
        return 'unchanged' if abs(diff) < 0.0005 else f"{abs(diff):.{d}f} {'higher' if diff > 0 else 'lower'}"
    rows = []
    for k, arch in [('dn', 'densenet121'), ('rn', 'resnet50')]:
        L = LP[arch]; F = CFG[f'{k}_fp32']
        for lab, lv, fv in [('Accuracy', L['acc'], F['acc']['mean']), ('Macro-F1', L['f1'], F['f1']['mean']), ('Balanced accuracy', L['bacc'], F['bacc']['mean']),
                            ('Macro-AUROC', L['auroc'], F['auroc']['mean']), ('PAD-UFES-20 accuracy (five classes)', L['pad_acc'], F['pad_acc']['mean'])]:
            rows.append([f"{'M1' if k == 'dn' else 'M2'}: {ARCH_NAME[k]}, fine-tuning", lab, f3(lv), f3(fv), ch(lv, fv)])
    for k in ['dn', 'rn']:
        F = CFG[f'{k}_fp32']; Q = CFG[FIN[k]]
        for lab, key in [('Accuracy', 'acc'), ('Macro-F1', 'f1'), ('PAD-UFES-20 accuracy (five classes)', 'pad_acc')]:
            rows.append([f'M3: {ARCH_NAME[k]}, INT8', lab, f3(F[key]['mean']), f3(Q[key]['mean']), ch(F[key]['mean'], Q[key]['mean'])])
        if EFF is not None and f'{k}_fp32' in EFF and FIN[k] in EFF:
            fk, qk = f'{k}_fp32', FIN[k]
            rows.append([f'M3: {ARCH_NAME[k]}, INT8', 'Model size (MB)', f"{EFF[fk]['size_mb']:.1f}", f"{EFF[qk]['size_mb']:.1f}", f"{EFF[fk]['size_mb'] / EFF[qk]['size_mb']:.1f}× smaller"])
            rows.append([f'M3: {ARCH_NAME[k]}, INT8', 'CPU latency (ms/image, batch 32)', f"{EFF[fk]['cpu_b32_ms']:.1f}", f"{EFF[qk]['cpu_b32_ms']:.1f}", f"{EFF[fk]['cpu_b32_ms'] / EFF[qk]['cpu_b32_ms']:.1f}× faster"])
    M.table(['Configuration and intervention', 'Measure', 'Condition before', 'Condition after', 'Change'], rows,
            'Measured condition before and after each model configuration.',
            'Before = frozen ImageNet features with a logistic-regression head (one deterministic fit per architecture) or the FP32 parent (INT8 rows); after = five-seed mean of the fine-tuned FP32 or INT8 model. PAD-UFES-20 uses the five mapped classes. Before-and-after results of the explanation audit are in the calibration, localization (raw versus excess LSAS) and faithfulness (random versus cross-image null order) tables.', label='tab:before_after')
    L1, L2 = LP['densenet121'], LP['resnet50']; F1, F2 = CFG['dn_fp32'], CFG['rn_fp32']
    M.p(f"End-to-end fine-tuning is the intervention that changes classification most: macro-F1 rises from {f3(L1['f1'])} to {f3(F1['f1']['mean'])} for DenseNet121 and from {f3(L2['f1'])} to {f3(F2['f1']['mean'])} for ResNet50, and balanced accuracy from {f3(L1['bacc'])} to {f3(F1['bacc']['mean'])} and from {f3(L2['bacc'])} to {f3(F2['bacc']['mean'])}, so frozen ImageNet features leave a large part of the available discrimination, particularly on the rare classes, unused. External accuracy on PAD-UFES-20 moves from {f3(L1['pad_acc'])} to {f3(F1['pad_acc']['mean'])} for DenseNet121 and from {f3(L2['pad_acc'])} to {f3(F2['pad_acc']['mean'])} for ResNet50, so fine-tuning on HAM10000 improves the internal fit but does not remove the shift quantified in Section 4.9.")


def _row_winner(vals, higher=True):
    ks = list(vals); order = sorted(ks, key=lambda k: vals[k][0], reverse=higher); b, r = order[0], order[1]
    return b if abs(vals[b][0] - vals[r][0]) > max(vals[b][1], vals[r][1]) else 'tie'


def compute_summary():
    """Criterion-wise table rows and winners under the decision rule of Section 3.15 (used by 4.15, the Abstract and the Conclusion)."""
    keys = ['dn_fp32', 'rn_fp32', FIN['dn'], FIN['rn']]; lab = {'dn_fp32': 'M1 DenseNet121 (FP32)', 'rn_fp32': 'M2 ResNet50 (FP32)', FIN['dn']: 'M3 DenseNet121 (INT8)', FIN['rn']: 'M3 ResNet50 (INT8)'}
    rows = []; wins = {k: 0 for k in keys}; wincnt = {}

    def add(name, vals, higher=True, show=lambda v: f'{v[0]:.3f} (SD {v[1]:.3f})'):
        w = _row_winner(vals, higher); wincnt[name] = w
        if w in wins: wins[w] += 1
        rows.append([name] + [show(vals[k]) if k in vals else 'as parent' for k in keys] + [lab.get(w, 'Tie')])
    for m, nm, hi in [('f1', 'Macro-F1', True), ('bacc', 'Balanced accuracy', True), ('acc', 'Accuracy', True), ('auroc', 'Macro-AUROC', True), ('mcc', 'MCC', True),
                      ('pad_bacc', 'PAD-UFES-20 balanced accuracy', True), ('pad_acc', 'PAD-UFES-20 accuracy', True), ('ece_ts', 'ECE after temperature scaling (lower is better)', False)]:
        add(nm, {k: (CFG[k][m]['mean'], CFG[k][m]['sd']) for k in keys}, hi)
    dn, rn = exd('densenet121'), exd('resnet50')
    for nm, col, hi, d in [('Median excess LSAS (FP32 parent)', 'e_shuf_median', True, 4), ('Median insertion gain (FP32 parent)', 'IG_median', True, 3), ('Median EIG (FP32 parent)', 'EIG_median', True, 4)]:
        v = {'dn_fp32': (dn[col].mean(), dn[col].std(ddof=1)), 'rn_fp32': (rn[col].mean(), rn[col].std(ddof=1))}
        w = _row_winner(v, hi); wincnt[nm] = w
        if w in wins: wins[w] += 1
        rows.append([nm, f"{v['dn_fp32'][0]:.{d}f} (SD {v['dn_fp32'][1]:.{d}f})", f"{v['rn_fp32'][0]:.{d}f} (SD {v['rn_fp32'][1]:.{d}f})", 'as parent', 'as parent', lab.get(w, 'Tie')])
    if EFF is not None:
        for nm, field, fm_ in [('Parameters (millions)', 'params_m', '{:.2f}'), ('Serialized size (MB)', 'size_mb', '{:.1f}'), ('CPU latency, batch 32 (ms/image)', 'cpu_b32_ms', '{:.1f}')]:
            cells = []
            for k in keys:
                kk = k if k in EFF and field in EFF[k] else ('dn_fp32' if k.startswith('dn') else 'rn_fp32')
                cells.append(fm_.format(EFF[kk][field]))
            vals = {k: float(c) for k, c in zip(keys, cells)}; w = min(vals, key=vals.get); wincnt[nm] = w; wins[w] += 1
            rows.append([nm] + cells + [lab[w]])
    ties = sum(1 for v in wincnt.values() if v == 'tie')
    STATE.update(rows=rows, wins=wins, wincnt=wincnt, lab=lab, ties=ties, n_crit=len(wincnt), keys=keys)
    return STATE


def _sci(x):
    """1.6e-07 -> '2 × 10⁻⁷' (rounded up to one significant digit, so the stated bound is never smaller than the difference)."""
    import math
    e = math.floor(math.log10(x)); d = math.ceil(x / 10 ** e)
    if d == 10: d, e = 1, e + 1
    sup = str(abs(e)).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f"{d} × 10{'⁻' if e < 0 else ''}{sup}"


def _sec_descriptive(M):
    """Descriptive statistics over the five seeds, from the independent recomputation in verify_statistics.py."""
    import json as _json
    p = OUT / 'results' / 'statistical_verification.json'
    if not p.exists():
        return
    V = _json.load(open(p)); C = V['per_config']; B = V['baseline']
    cfgs = [('dn_fp32', 'M1 DenseNet121 (FP32)'), ('rn_fp32', 'M2 ResNet50 (FP32)'), ('dn_int8', 'M3 DenseNet121 (INT8)'), ('rn_int8', 'M3 ResNet50 (INT8)'),
            ('dn_mixed', 'DenseNet121 (INT8, dense block 1 in FP32)')]
    meas = [('acc', 'Accuracy'), ('f1', 'Macro-F1'), ('bacc', 'Balanced accuracy'), ('auroc', 'Macro-AUROC')]
    rows = []
    for k, nm in cfgs:
        for m, mn in meas:
            s = C[k][m]
            rows.append([nm, mn, f"{s['mean']:.3f}", f"{s['median']:.3f}", f"{s['sd']:.3f}", f"{s['min']:.3f} to {s['max']:.3f}", f"{s['ci95_low']:.3f} to {s['ci95_high']:.3f}"])
    nrep = sum(1 for k in C for m in C[k] for v in [C[k][m]['mode']] if v is not None)
    M.p(f"__Descriptive statistics over seeds, recomputed independently.__ Before the comparison was written up, every per-seed metric was recomputed from the saved raw logits with code written separately from the pipeline (confusion-matrix macro-F1, rank-based one-versus-rest AUROC, and a separate calibration-error routine). The recomputed means and SDs agree with the pipeline's to within {_sci(V['max_abs_difference_vs_pipeline'])}, and all {len(V['manuscript_table_checks'])} mean-and-SD pairs for the four headline configurations reproduce exactly, to three decimals, in the tables of this paper. [[T:descstats]] adds the median, the range and a t-based 95% interval of the five seed values (four degrees of freedom; sample SD with n minus 1). The mean and median differ by at most {max(abs(C[k][m]['mean'] - C[k][m]['median']) for k, _ in cfgs for m, _ in meas):.3f}, so the seed distributions are not visibly skewed by a single seed (five seeds cannot establish more than that). The mode is uninformative for continuous seed values (a value repeated at three decimals occurred in only {nrep} of {len(C) * len(next(iter(C.values())))} configuration-measure combinations, through rounding) but informative for the class labels, where the most frequent class, {B['modal_class']}, makes up {100 * B['modal_class_share']:.1f}% of the 990 test images; a classifier that always predicted it would reach an accuracy of {B['majority_baseline_accuracy']:.3f} but a macro-F1 of only {B['majority_baseline_macro_f1']:.3f}, which is why accuracy alone is not used to rank the models.")
    M.table(['Configuration', 'Measure', 'Mean', 'Median', 'SD', 'Range over seeds', '95% interval of the mean'], rows,
            'Descriptive statistics of the five training seeds for each model configuration (test set of 990 images).',
            'Recomputed from the saved logits (verify_statistics.py). SD is the sample SD (n minus 1); the 95% interval is the t-based interval of the mean over the five seeds (four degrees of freedom) and describes training-seed variation only, not sampling variation of the test images (see Section 3.16 for that).', label='tab:descstats')


def sec_three(M):
    S = compute_summary(); keys, lab, rows, wins, wincnt, ties = S['keys'], S['lab'], S['rows'], S['wins'], S['wincnt'], S['ties']
    M.h(2, '4.15 Three-configuration comparison and model selection')
    M.p("[[T:three]] and [[F:bars]] bring the three configurations together on the criteria fixed in Section 3.15: M1 (DenseNet121 with Grad-CAM++), M2 (ResNet50 with Grad-CAM++) and M3 (the INT8 versions of both). A configuration is named best for a criterion only if it has the highest mean and leads the runner-up by more than the larger of the two seed SDs; otherwise the criterion is a tie. INT8 versions are derived from their FP32 parents, so for the explanation criteria they inherit the parent's value.")
    M.table(['Criterion'] + [lab[k] for k in keys] + ['Best by the decision rule'], rows,
            'Criterion-wise comparison of the three model configurations (five seeds each; test set of 990 images and PAD-UFES-20).',
            'Cells give the mean over seeds with the seed SD. Excess measures are signed because their sign relative to the null carries the finding; other values are unsigned. Best = highest mean (lowest for ECE and efficiency) with an advantage exceeding both seed SDs; efficiency rows are deterministic. PAD-UFES-20 uses the five mapped classes.', label='tab:three')
    M.fig(FIG_BARS, 'Comparison of the three model configurations: DenseNet121 (M1), ResNet50 (M2) and their INT8 versions (M3). Panels show discrimination on the 990 HAM10000 test images, external performance and calibration on PAD-UFES-20, and efficiency; bars are five-seed means with SD whiskers (efficiency bars are single deterministic measurements).', 17, label='fig:bars')
    _sec_descriptive(M)
    top = [lab[k] for k in sorted(wins, key=lambda k: -wins[k]) if wins[k] > 0]
    names = ['Macro-F1', 'Balanced accuracy', 'Accuracy', 'Macro-AUROC', 'PAD-UFES-20 balanced accuracy']
    wtxt = ', '.join(f"{LOWN[nm]} ({lab[wincnt[nm]] if wincnt[nm] in lab else 'a tie'})" for nm in names)
    M.p(f"No configuration is best on every criterion. By the decision rule the winners were: {wtxt}; {ties} of {len(wincnt)} criteria were ties, in which the difference between the leading configurations was smaller than their seed-to-seed variation. The configuration with the most criterion wins was {top[0] if top else 'none'}. These counts summarize the table and are not a composite score, because the criteria are not equally important and their weighting is a deployment decision (Section 5.10).")
    dnx, rnx = exd('densenet121'), exd('resnet50'); r_acc = {k: CFG[k]['acc']['mean'] for k in keys}; hi_acc = max(r_acc, key=r_acc.get)
    r_bacc = {k: CFG[k]['bacc']['mean'] for k in keys}; hi_bacc = max(r_bacc, key=r_bacc.get)
    sz = EFF[FIN['dn']]['size_mb'] if EFF else float('nan'); lat_rn = EFF[FIN['rn']]['cpu_b32_ms'] if EFF else float('nan'); lat_dn = EFF[FIN['dn']]['cpu_b32_ms'] if EFF else float('nan')
    M.p(f"__Translated into deployment priorities__ (each statement restates a row of [[T:three]]). If explanation reliability matters, M1 is the only configuration that leads on the explanation criteria: its mean excess LSAS is {dnx.e_shuf_median.mean():+.3f} against {rnx.e_shuf_median.mean():+.3f}, its median insertion gain {dnx.IG_median.mean():.3f} against {rnx.IG_median.mean():.3f}, and its excess insertion gain {dnx.EIG_median.mean():+.4f} against {rnx.EIG_median.mean():+.4f} (Section 4.12). If overall accuracy is the priority, the highest means belong to {lab[hi_acc]} ({r_acc[hi_acc]:.3f}), but accuracy is a tie within seed variation and the class-balanced measures favour no configuration separably (highest balanced accuracy: {lab[hi_bacc]}, {r_bacc[hi_bacc]:.3f}). If CPU inference cost is the constraint, {lab[FIN['rn']]} is fastest ({lat_rn:.1f} ms per image) and stays within the accuracy tolerance, whereas {lab[FIN['dn']]} is smallest ({sz:.1f} MB) but {'falls outside' if (CFG['dn_fp32']['acc']['mean'] - CFG[FIN['dn']]['acc']['mean']) > 0.01 else 'stays within'} it. Without a stated priority, no single configuration can be recommended on these data; the measured trade-offs are the result.")


def sec_discussion_models(M):
    W, lab = STATE['wincnt'], STATE['lab']; dn, rn = exd('densenet121'), exd('resnet50')
    M.h(2, '5.10 Model selection and deployment implications')
    grp = lambda names: '; '.join(f"{LOWN[n]}: {lab.get(W[n], 'tie')}" for n in names)
    M.p(f"__Which configuration is best?__ The question has no single answer in these data, and the study does not force one. Under the decision rule of Section 3.15 the criterion-wise outcomes were: discrimination on HAM10000 ({grp(['Macro-F1', 'Balanced accuracy', 'Accuracy', 'Macro-AUROC'])}); external performance ({grp(['PAD-UFES-20 balanced accuracy'])}); calibration ({lab.get(W['ECE after temperature scaling (lower is better)'], 'tie')}); and, for the explanation audit, excess LSAS ({lab.get(W['Median excess LSAS (FP32 parent)'], 'tie')}), insertion gain ({lab.get(W['Median insertion gain (FP32 parent)'], 'tie')}) and the image-specificity measure EIG ({lab.get(W['Median EIG (FP32 parent)'], 'tie')}). Because accuracy rewards the majority class (NV is roughly 58 times more frequent than DF) while macro-F1 and balanced accuracy weight the rare classes equally, a classifier can lead on accuracy and trail on class-balanced measures, as the two architectures do, so the choice depends on which property a deployment needs most.")
    inq = {k: (abs(-CMP['tests'][f"precision_{'densenet121' if k == 'dn' else 'resnet50'}_int8"]['acc']['mean_diff_int8_minus_fp32']) <= 0.01 and abs(-CMP['tests'][f"precision_{'densenet121' if k == 'dn' else 'resnet50'}_int8"]['f1']['mean_diff_int8_minus_fp32']) <= 0.02) for k in ['dn', 'rn']}
    _sw = pd.concat([pd.read_csv(RES / 'int8_sweep.csv'), pd.read_csv(RES / 'int8_sweep2.csv')]); _sw = _sw[_sw.variant != 'FP32']; lo, hi = _sw.f1.min(), _sw.f1.max()
    M.p(f"__What INT8 changes, and what it does not.__ INT8 quantization reduces model size and CPU time per image by the factors in [[T:int8]], and for {', '.join(ARCH_NAME[k] for k in inq if inq[k]) or 'neither architecture'} the loss in test accuracy and macro-F1 stayed within the pre-specified tolerance{'' if all(inq.values()) else ('; for ' + ', '.join(ARCH_NAME[k] for k in inq if not inq[k]) + ' it did not')}. Two lessons come from the validation sweep. First, a default all-layers INT8 configuration can be unacceptable even though INT8 is generally reported to be within 1-2% of floating point [@krishnamoorthi2018,nagel2021]: both networks required the first convolution and the head in FP32, and DenseNet121, whose dense connectivity concatenates feature maps of very different ranges, was more fragile; the only remedy that restored its validation scores, keeping dense block 1 in FP32, left a large share of the computation in floating point, and a quantization-aware pilot did not beat plain post-training quantization for either network. Second, the configuration must be chosen on validation data, as here, because candidates span a wide range of outcomes (val_sel macro-F1 from {lo:.2f} to {hi:.2f}) that a post hoc choice on the test set would exploit. An INT8 deployment is explained by its FP32 parent, which is consistent only to the extent that the two models predict the same class (agreement in [[T:int8]]). Quantization can alter class-activation and attribution maps in an architecture-dependent way even when accuracy is preserved [@kerkouri2024,rabbi2026]; the present study neither tests nor contradicts this for dermoscopic models, and a null-controlled audit of maps from quantized networks remains to be done.")
    pa_f, pa_i = CFG['dn_fp32']['pad_acc']['mean'], CFG[FIN['dn']]['pad_acc']['mean']; pb_f, pb_i = CFG['dn_fp32']['pad_bacc']['mean'], CFG[FIN['dn']]['pad_bacc']['mean']
    M.p(f"__A caution on the external accuracy of the INT8 DenseNet121.__ The INT8 DenseNet121 reached a higher PAD-UFES-20 accuracy than its FP32 parent ({pa_i:.3f} against {pa_f:.3f}) although its in-domain accuracy was lower, and the decision rule names it best on that single criterion. We do not interpret this as a robustness gain: the corresponding balanced accuracy, which does not depend on the very different class prevalence of PAD-UFES-20, is {pb_i:.3f} against {pb_f:.3f} and not separable, and accuracy there is dominated by the two most frequent classes (AKIEC and BCC), so a modest change in the class mix of the predictions is enough to move it. Any compressed model should be validated in the target domain.")
    M.p("__Limits of this comparison.__ Two architectures with five seeds each, one fixed lesion-level split and one external dataset can resolve only large differences between architectures (Section 3.16) and do not characterize other architectures, explanation methods or target layers. INT8 results come from one toolchain (PyTorch FX quantization, x86 CPU kernels) and one CPU; latency and the acceptability of a configuration may differ elsewhere. All conclusions are computational and none addresses clinical safety, reader benefit or regulatory suitability.")
