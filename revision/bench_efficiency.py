"""Efficiency of every configuration: parameters, serialized size, CPU latency (batch 1 and 32; six threads) and FP32 GPU latency.
Run only when no other job is using the CPU or GPU.  Seed-42 checkpoints (sizes and latencies do not depend on the seed)."""
import json, os, statistics, time
from int8_lib import *
import torch

RES = OUT / 'results'; torch.set_num_threads(6)
X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); ham = pd.read_csv(CACHE / 'ham_meta.csv')
vcal = ham[(ham.split == 'val') & (ham.val_role == 'cal')]; te = ham[ham.split == 'test']; idx = te.cache_idx.values[:64]; xb32 = norm_batch(X[idx[:32]]); xb1 = xb32[:1]
CHOSEN = {'densenet121': dict(act='hist', reduce_range=False, skip='stem_head', n=503, backend='x86'),
          'resnet50': dict(act='hist', reduce_range=False, skip='stem_head', n=503, backend='onednn')}


def med_latency(fn, x, reps=15, warm=3):
    with torch.no_grad():
        for _ in range(warm): fn(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) / len(x) * 1000.0


def size_mb(obj, tmp, script=False):
    if script: torch.jit.save(obj, tmp)
    else: torch.save(obj.state_dict(), tmp)
    s = os.path.getsize(tmp) / 1e6; os.remove(tmp); return s


out = {}
for arch, k in [('densenet121', 'dn'), ('resnet50', 'rn')]:
    fp = load_fp32(arch, 42); params = sum(p.numel() for p in fp.parameters()) / 1e6; tmp = str(OUT / 'models' / '_tmp_size.pt')
    out[f'{k}_fp32'] = dict(params_m=params, size_mb=size_mb(fp, tmp), cpu_b1_ms=med_latency(fp, xb1), cpu_b32_ms=med_latency(fp, xb32))
    if torch.cuda.is_available():
        g = fp.to(DEVICE); xg = xb32.to(DEVICE)
        def gf(x):
            g(x); torch.cuda.synchronize()
        out[f'{k}_fp32']['gpu_b32_ms'] = med_latency(gf, xg)
    v = CHOSEN[arch]; torch.backends.quantized.engine = v['backend']
    q = build_ptq(load_fp32(arch, 42), X, vcal.cache_idx.values[:v['n']], v)
    out[f'{k}_int8'] = dict(size_mb=size_mb(q, tmp), cpu_b1_ms=med_latency(q, xb1), cpu_b32_ms=med_latency(q, xb32))
    if arch == 'densenet121':
        f2 = load_fp32(arch, 42); names = [n for n, _ in f2.named_modules() if n == 'backbone.features.denseblock1' or n.startswith('backbone.features.denseblock1.')]
        qm = build_ptq(f2, X, vcal.cache_idx.values[:v['n']], dict(v, skip_names=names))
        out['dn_mixed'] = dict(size_mb=size_mb(qm, tmp), cpu_b1_ms=med_latency(qm, xb1), cpu_b32_ms=med_latency(qm, xb32))
    print(arch, json.dumps({kk: v_ for kk, v_ in out.items() if kk.startswith(k)}, indent=1), flush=True)
out['cpu'] = 'Intel Core i5-13450HX (AVX2), 6 threads'; out['gpu'] = 'NVIDIA GeForce RTX 4050 Laptop GPU'
json.dump(out, open(RES / 'efficiency.json', 'w'), indent=1)
print('EFFICIENCY DONE')
