"""Sensitivity-guided mixed-precision INT8 (validation data only; seed-42 model of each architecture).
Starting from the selected post-training configuration (stem and head in FP32), flip ONE further layer group at a time to FP32,
measure val_sel macro-F1/accuracy/agreement, then add groups greedily (largest macro-F1 gain, ties by agreement) until the
tolerance is met (accuracy within 1 point and macro-F1 within 2 points of FP32), three extra groups are used, or the FP32 share of
multiply-accumulates would exceed 25%.  The chosen group list is saved and applied unchanged to every seed."""
import sys, json, time
from int8_lib import *

torch.set_num_threads(6)
RES = OUT / 'results'
ham = pd.read_csv(CACHE / 'ham_meta.csv'); X = np.load(CACHE / 'ham_x.npy', mmap_mode='r')
vsel = ham[(ham.split == 'val') & (ham.val_role == 'sel')]; vcal = ham[(ham.split == 'val') & (ham.val_role == 'cal')]
ysel = vsel.label_idx.values; sel_idx = vsel.cache_idx.values; cal_idx = vcal.cache_idx.values
BASE = {'densenet121': dict(act='hist', reduce_range=False, skip='stem_head', n=503, backend='x86'),
        'resnet50': dict(act='hist', reduce_range=False, skip='stem_head', n=503, backend='onednn')}
GROUPS = {'densenet121': {'denseblock1': 'backbone.features.denseblock1', 'transition1': 'backbone.features.transition1', 'denseblock2': 'backbone.features.denseblock2',
                          'transition2': 'backbone.features.transition2', 'denseblock3': 'backbone.features.denseblock3', 'transition3': 'backbone.features.transition3',
                          'denseblock4': 'backbone.features.denseblock4', 'norm5': 'backbone.features.norm5'},
          'resnet50': {'layer1': 'backbone.layer1', 'layer2': 'backbone.layer2', 'layer3': 'backbone.layer3', 'layer4': 'backbone.layer4'}}
archs = sys.argv[1].split(',') if len(sys.argv) > 1 else ['densenet121', 'resnet50']


def macs_by_module(model):
    macs = {}; hooks = []
    def mk(name):
        def h(m, i, o):
            if isinstance(m, nn.Conv2d): macs[name] = o.shape[2] * o.shape[3] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            else: macs[name] = m.in_features * m.out_features
        return h
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)): hooks.append(m.register_forward_hook(mk(n)))
    with torch.no_grad(): model(norm_batch(X[sel_idx[:1]]))
    for h in hooks: h.remove()
    return macs


out = {}; rows = []
for arch in archs:
    fp = load_fp32(arch, 42); lg_fp = logits_cpu(fp, X, sel_idx); m_fp = metrics_from_logits(lg_fp, ysel); macs = macs_by_module(fp); tot = sum(macs.values())
    names = [n for n, _ in fp.named_modules()]
    base_skip = set(sum([SKIP_SETS[BASE[arch]['skip']]], []))
    share = lambda skipped: sum(v for k, v in macs.items() if any(k == s or k.startswith(s + '.') for s in skipped)) / tot
    leaves = lambda prefix: [n for n in names if n == prefix or n.startswith(prefix + '.')]

    def evaluate(extra_groups):
        skip_names = [n for g in extra_groups for n in leaves(GROUPS[arch][g])]
        v = dict(BASE[arch], skip_names=skip_names); torch.backends.quantized.engine = v['backend']
        q = build_ptq(fp, X, cal_idx[:v['n']], v); lg = logits_cpu(q, X, sel_idx); m = metrics_from_logits(lg, ysel)
        fp32_share = share(list(base_skip) + [GROUPS[arch][g] for g in extra_groups])
        return dict(m, agree=float((lg.argmax(1) == lg_fp.argmax(1)).mean()), fp32_mac_share=float(fp32_share))

    t0 = time.time(); r0 = evaluate([]); rows.append(dict(arch=arch, step=0, extra='(none: stem and head in FP32)', **r0))
    print(arch, 'FP32', m_fp, '| base', {k: round(v, 3) for k, v in r0.items()}, flush=True)
    chosen = []; cur = r0; ok = lambda r: (m_fp['acc'] - r['acc'] <= 0.01) and (m_fp['f1'] - r['f1'] <= 0.02)
    for step in range(1, 4):
        if ok(cur): break
        cands = {}
        for g in GROUPS[arch]:
            if g in chosen: continue
            r = evaluate(chosen + [g]); cands[g] = r
            rows.append(dict(arch=arch, step=step, extra='+'.join(chosen + [g]), **r))
            print(f"  step {step} + {g}: acc={r['acc']:.3f} f1={r['f1']:.3f} agree={r['agree']:.3f} fp32_share={r['fp32_mac_share']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        feas = {g: r for g, r in cands.items() if r['fp32_mac_share'] <= 0.25}
        if not feas: print('  no group fits the 25% FP32 compute budget'); break
        best = max(feas, key=lambda g: (round(feas[g]['f1'] / 0.005), feas[g]['agree']))
        if feas[best]['f1'] <= cur['f1'] + 0.002: print('  no further gain'); break
        chosen.append(best); cur = feas[best]; print(f'  -> keep {best} in FP32; val_sel f1={cur["f1"]:.3f}', flush=True)
    out[arch] = dict(chosen=chosen, skip_names=[n for g in chosen for n in leaves(GROUPS[arch][g])], fp32_val_sel=m_fp, final_val_sel=cur, tolerance_met=bool(ok(cur)), base_val_sel=r0)
    json.dump(out, open(RES / 'int8_mixed_precision.json', 'w'), indent=1); pd.DataFrame(rows).to_csv(RES / 'int8_sensitivity.csv', index=False)
print('SENSITIVITY DONE', flush=True)
