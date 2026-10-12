"""Quantization-aware refinement (QAT) of an FP32 checkpoint, then conversion to a real INT8 CPU model.
Training data and loss are identical to FP32 training (train split only); the best epoch is chosen on val_sel macro-F1
(fake-quant simulation); the test set is never used here.  Output: out/models/seed{S}{tag}_int8qat.pt (TorchScript)."""
import argparse, copy, time, json
from int8_lib import *
from torch.ao.quantization import get_default_qat_qconfig_mapping, disable_observer
from torch.utils.data import DataLoader, WeightedRandomSampler
from train import DS, train_tf, val_tf, gpu_norm, wseed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='densenet121', choices=list(ARCHES)); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--skip', default='none', choices=list(SKIP_SETS))
    ap.add_argument('--obs_epochs', type=int, default=2); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max_steps', type=int, default=0); ap.add_argument('--backend', default='x86')
    a = ap.parse_args()
    tag = '' if a.arch == 'densenet121' else f'_{a.arch}'; name = f'seed{a.seed}{tag}'
    ham = pd.read_csv(CACHE / 'ham_meta.csv'); xp = str(CACHE / 'ham_x.npy'); X = np.load(xp, mmap_mode='r')
    tr = ham[ham.split == 'train']; vs = ham[(ham.split == 'val') & (ham.val_role == 'sel')]
    set_seed(a.seed); g = torch.Generator(); g.manual_seed(a.seed)
    cnt = tr.label_idx.value_counts().sort_index(); w = [1.0 / cnt[l] for l in tr.label_idx]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True, generator=g)
    tl = DataLoader(DS(xp, tr.cache_idx.values, tr.label_idx.values, train_tf), batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                    worker_init_fn=wseed, generator=g, persistent_workers=True)
    vl = DataLoader(DS(xp, vs.cache_idx.values, vs.label_idx.values, val_tf), batch_size=32, shuffle=False, num_workers=2)
    crit = CombinedLoss(make_alpha(tr)).to(DEVICE)

    fp = load_fp32(a.arch, a.seed)
    torch.backends.quantized.engine = a.backend
    qm = get_default_qat_qconfig_mapping(a.backend)
    for n in SKIP_SETS[a.skip]: qm.set_module_name(n, None)
    prep = qfx.prepare_qat_fx(copy.deepcopy(fp).train(), qm, (torch.randn(1, 3, IMG, IMG),)).to(DEVICE)


    def freeze_bn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)): m.eval()
        if hasattr(m, 'freeze_bn_stats'): m.freeze_bn_stats()


    def evaluate(model):
        model.eval(); P = []; L = []
        with torch.no_grad():
            for x, y in vl:
                P.append(model(gpu_norm(x.to(DEVICE))).float().cpu().numpy()); L.append(y.numpy())
        lg = np.concatenate(P); y = np.concatenate(L); return metrics_from_logits(lg, y), lg


    m0, _ = evaluate(prep); print(f'[{name}] QAT-prepared (before refinement, fake-quant) val_sel {m0}', flush=True)
    opt = torch.optim.AdamW(prep.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(tl) if not a.max_steps else a.max_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    best = (m0['f1'], copy.deepcopy(prep.state_dict()), 0); step = 0; t00 = time.time(); hist = [dict(epoch=0, **m0)]
    for ep in range(1, a.epochs + 1):
        prep.train(); prep.apply(freeze_bn)
        if ep > a.obs_epochs: prep.apply(disable_observer)
        t0 = time.time(); tot = n = 0
        for x, y in tl:
            x, y = gpu_norm(x.to(DEVICE, non_blocking=True)), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True); loss = crit(prep(x).float(), y)
            if not torch.isfinite(loss): continue
            loss.backward(); nn.utils.clip_grad_norm_(prep.parameters(), 1.0); opt.step(); sched.step(); step += 1
            tot += loss.item() * len(y); n += len(y)
            if a.max_steps and step >= a.max_steps: break
        m, _ = evaluate(prep); hist.append(dict(epoch=ep, loss=tot / max(n, 1), **m))
        if m['f1'] > best[0]: best = (m['f1'], copy.deepcopy(prep.state_dict()), ep)
        print(f'[{name}] QAT ep{ep} loss={tot/max(n,1):.4f} val_sel acc={m["acc"]:.4f} f1={m["f1"]:.4f} auroc={m["auroc"]:.4f} best_ep={best[2]} {time.time()-t0:.0f}s', flush=True)
        if a.max_steps and step >= a.max_steps: break
    prep.load_state_dict(best[1]); prep.apply(disable_observer)
    torch.backends.quantized.engine = a.backend
    q = qfx.convert_fx(copy.deepcopy(prep).cpu().eval())
    lg_q = logits_cpu(q, X, vs.cache_idx.values); mq = metrics_from_logits(lg_q, vs.label_idx.values)
    lg_fp = logits_cpu(fp, X, vs.cache_idx.values)
    mq['agree'] = float((lg_q.argmax(1) == lg_fp.argmax(1)).mean())
    print(f'[{name}] converted real-INT8 val_sel {mq}', flush=True)
    out = OUT / 'models' / f'{name}_int8qat.pt'
    try:
        ts = torch.jit.trace(q, torch.randn(1, 3, IMG, IMG), strict=False); torch.jit.save(ts, str(out)); fmt = 'torchscript'
    except Exception as e:
        torch.save(q, str(out)); fmt = 'pickle-graphmodule'; print('trace failed, saved pickled GraphModule:', repr(e)[:120])
    json.dump(dict(name=name, arch=a.arch, seed=a.seed, epochs=a.epochs, lr=a.lr, skip=a.skip, best_epoch=best[2], fmt=fmt, val_sel_int8=mq,
                   val_sel_before=m0, hist=hist, secs=time.time() - t00), open(OUT / 'models' / f'{name}_int8qat_summary.json', 'w'), indent=1)
    print(f'[{name}] QAT DONE {time.time()-t00:.0f}s format={fmt}', flush=True)



if __name__ == '__main__':
    main()
