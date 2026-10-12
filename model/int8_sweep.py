"""Validation-only sweep of post-training INT8 configurations for both architectures (seed 42 models).
Ranking metric: macro-F1 and prediction agreement with the FP32 model on val_sel; calibration images come from val_cal.
Nothing in this script reads test-set images or labels."""
import sys, json, time
from int8_lib import *

torch.set_num_threads(6)
RES = OUT / 'results'; RES.mkdir(exist_ok=True, parents=True)
archs = sys.argv[1].split(',') if len(sys.argv) > 1 else ['densenet121', 'resnet50']
ham = pd.read_csv(CACHE / 'ham_meta.csv'); X = np.load(CACHE / 'ham_x.npy', mmap_mode='r')
vsel = ham[(ham.split == 'val') & (ham.val_role == 'sel')]; vcal = ham[(ham.split == 'val') & (ham.val_role == 'cal')]
ysel = vsel.label_idx.values; sel_idx = vsel.cache_idx.values; cal_idx_all = vcal.cache_idx.values

VARIANTS = [
    dict(name='hist_noRR_all_n503', act='hist', reduce_range=False, skip='none', n=503),
    dict(name='hist_RR_all_n503', act='hist', reduce_range=True, skip='none', n=503),
    dict(name='minmax_all_n503', act='minmax', reduce_range=False, skip='none', n=503),
    dict(name='mavg_all_n503', act='mavg', reduce_range=False, skip='none', n=503),
    dict(name='hist_noRR_skiphead_n503', act='hist', reduce_range=False, skip='head', n=503),
    dict(name='hist_noRR_skipstemhead_n503', act='hist', reduce_range=False, skip='stem_head', n=503),
    dict(name='hist_noRR_all_n128', act='hist', reduce_range=False, skip='none', n=128),
]
rows = []
for arch in archs:
    fp = load_fp32(arch, 42)
    lg_fp = logits_cpu(fp, X, sel_idx); m_fp = metrics_from_logits(lg_fp, ysel)
    print(arch, 'FP32 val_sel', m_fp, flush=True)
    rows.append(dict(arch=arch, variant='FP32', **m_fp, agree=1.0, mean_abs_dlogit=0.0, secs=0.0))
    for v in VARIANTS:
        t0 = time.time()
        try:
            q = build_ptq(fp, X, cal_idx_all[:v['n']], v)
            lg = logits_cpu(q, X, sel_idx); m = metrics_from_logits(lg, ysel)
            agree = float((lg.argmax(1) == lg_fp.argmax(1)).mean()); dl = float(np.abs(lg - lg_fp).mean())
            rows.append(dict(arch=arch, variant=v['name'], **m, agree=agree, mean_abs_dlogit=dl, secs=time.time() - t0))
            print(arch, v['name'], f"acc={m['acc']:.3f} f1={m['f1']:.3f} auroc={m['auroc']:.3f} agree={agree:.3f} dlogit={dl:.3f} {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(arch, v['name'], 'FAILED', repr(e)[:200], flush=True)
        pd.DataFrame(rows).to_csv(RES / 'int8_sweep.csv', index=False)
print('SWEEP DONE', flush=True)
