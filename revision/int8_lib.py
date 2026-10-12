"""INT8 integration for the trust-audit pipeline (post-training static quantization and quantization-aware refinement).

All quantization uses PyTorch's built-in FX graph-mode quantization (torch.ao.quantization.quantize_fx; x86/fbgemm kernels, CPU).
Selection of the quantization configuration uses ONLY validation data (val_sel for ranking, val_cal for calibration and
temperature); the test set is touched once, with the configuration already fixed."""
import copy, time
import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.ao.quantization.quantize_fx as qfx
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig_mapping
from torch.ao.quantization.observer import (HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver,
                                            PerChannelMinMaxObserver)
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from common import *

torch.backends.quantized.engine = 'x86'


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


@torch.no_grad()
def logits_cpu(model, X, idx, bs=64):
    out = []
    for i in range(0, len(idx), bs):
        out.append(model(norm_batch(X[idx[i:i + bs]])).float())
    return torch.cat(out).numpy()


def metrics_from_logits(lg, y, T=1.0):
    P = softmax(lg / T); pred = P.argmax(1)
    m = dict(acc=float((pred == y).mean()), bacc=float(balanced_accuracy_score(y, pred)),
             f1=float(f1_score(y, pred, average='macro', labels=list(range(7)), zero_division=0)))
    try: m['auroc'] = float(roc_auc_score(y, P, multi_class='ovr', average='macro', labels=list(range(7))))
    except Exception: m['auroc'] = float('nan')
    return m


def fit_temperature(lg, y):
    lg = torch.tensor(lg); y = torch.tensor(y); T = nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50); ce = nn.CrossEntropyLoss()
    def clo():
        opt.zero_grad(); l = ce(lg / T, y); l.backward(); return l
    opt.step(clo); return float(T.item())


def load_fp32(arch, seed, device='cpu'):
    tag = '' if arch == 'densenet121' else f'_{arch}'
    m = ARCHES[arch](); m.load_state_dict(torch.load(OUT / 'models' / f'seed{seed}{tag}_best.pt', map_location=device)); return m.eval()


def make_qconfig(act='hist', reduce_range=False):
    a = {'hist': HistogramObserver.with_args(reduce_range=reduce_range),
         'minmax': MinMaxObserver.with_args(reduce_range=reduce_range),
         'mavg': MovingAverageMinMaxObserver.with_args(reduce_range=reduce_range)}[act]
    w = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    return QConfig(activation=a, weight=w)


# modules kept in FP32 for the mixed-precision variants (names exist in both architectures except stem names)
SKIP_SETS = {
    'none': [],
    'head': ['fc1', 'classifier'],
    'stem_head': ['fc1', 'classifier', 'backbone.features.conv0', 'backbone.conv1'],
    'stem': ['backbone.features.conv0', 'backbone.conv1'],
}


def qconfig_mapping(variant):
    """variant = dict(act, reduce_range, skip, backend)"""
    qm = QConfigMapping().set_global(make_qconfig(variant.get('act', 'hist'), variant.get('reduce_range', False)))
    for n in SKIP_SETS[variant.get('skip', 'none')] + list(variant.get('skip_names', [])):
        qm.set_module_name(n, None)
    return qm


def build_ptq(fp32, X, calib_idx, variant, bs=32):
    torch.backends.quantized.engine = variant.get('backend', 'x86')
    ex = (torch.randn(1, 3, IMG, IMG),)
    prep = qfx.prepare_fx(copy.deepcopy(fp32).eval(), qconfig_mapping(variant), ex)
    with torch.no_grad():
        for i in range(0, len(calib_idx), bs):
            prep(norm_batch(X[calib_idx[i:i + bs]]))
    return qfx.convert_fx(prep)


def model_size_mb(model, tmp):
    torch.save(model.state_dict(), tmp); import os; s = os.path.getsize(tmp) / 1e6; os.remove(tmp); return s


def latency_ms(model, X, idx, bs, reps=5, threads=None):
    if threads: torch.set_num_threads(threads)
    xb = norm_batch(X[idx[:bs]])
    with torch.no_grad():
        model(xb)
        t0 = time.time()
        for _ in range(reps): model(xb)
    return (time.time() - t0) / reps / bs * 1000.0
