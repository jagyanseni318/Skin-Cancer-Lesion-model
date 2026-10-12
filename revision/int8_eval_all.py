"""Produce INT8 logits (val_cal, val_sel, test, PAD-UFES-20) for every seed of both architectures.
mode ptq : post-training quantization with the validation-selected configuration (calibration on val_cal only)
mode qat : load the quantization-aware-refined INT8 model written by int8_qat.py
Temperature is refitted on the INT8 val_cal logits.  Output: out/seed{S}{arch tag}_int8{ptq|}/logits.npz"""
import sys, json, time
from int8_lib import *

mode = sys.argv[1]; archs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['densenet121', 'resnet50']
seeds = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else SEEDS
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 4)
CHOSEN = {'densenet121': dict(act='hist', reduce_range=False, skip='stem_head', n=503, backend='x86'),
          'resnet50': dict(act='hist', reduce_range=False, skip='stem_head', n=503, backend='onednn')}
ham = pd.read_csv(CACHE / 'ham_meta.csv'); pad = pd.read_csv(CACHE / 'pad_meta.csv')
X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); XP = np.load(CACHE / 'pad_x.npy', mmap_mode='r')
vcal = ham[(ham.split == 'val') & (ham.val_role == 'cal')]; vsel = ham[(ham.split == 'val') & (ham.val_role == 'sel')]; te = ham[ham.split == 'test']
for arch in archs:
    v = CHOSEN[arch]; tag = '' if arch == 'densenet121' else f'_{arch}'
    for seed in seeds:
        t0 = time.time(); name = f'seed{seed}{tag}'; od = OUT / (name + {'ptq': '_int8ptq', 'mixed': '_int8mixed'}.get(mode, '_int8')); od.mkdir(exist_ok=True, parents=True)
        torch.backends.quantized.engine = v['backend']
        if mode == 'ptq':
            fp = load_fp32(arch, seed); q = build_ptq(fp, X, vcal.cache_idx.values[:v['n']], v)
        elif mode == 'mixed':
            fp = load_fp32(arch, seed); names = [n for n, _ in fp.named_modules() if n == 'backbone.features.denseblock1' or n.startswith('backbone.features.denseblock1.')]
            v = dict(v, skip_names=names); q = build_ptq(fp, X, vcal.cache_idx.values[:v['n']], v)
        else:
            f = OUT / 'models' / f'{name}_int8qat.pt'
            if not f.exists(): print('missing', f); continue
            q = torch.jit.load(str(f)); q.eval()
        lg = dict(val_cal=logits_cpu(q, X, vcal.cache_idx.values), val_sel=logits_cpu(q, X, vsel.cache_idx.values),
                  test=logits_cpu(q, X, te.cache_idx.values), pad=logits_cpu(q, XP, np.arange(len(pad))))
        T = fit_temperature(lg['val_cal'], vcal.label_idx.values)
        np.savez(od / 'logits.npz', T=T, **lg)
        m = metrics_from_logits(lg['test'], te.label_idx.values, T)
        json.dump(dict(name=name, mode=mode, config=v, T=T, test=m, secs=time.time() - t0), open(od / 'eval_summary.json', 'w'), indent=1)
        print(f'[{name}-{mode}] T={T:.3f} test acc={m["acc"]:.4f} f1={m["f1"]:.4f} auroc={m["auroc"]:.4f} {time.time()-t0:.0f}s', flush=True)
print('INT8 EVAL DONE', mode, flush=True)
