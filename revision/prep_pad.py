"""PAD-UFES-20 cache: images whose longest side exceeds 1024 px are area-downscaled to 1024 px BEFORE DullRazor
(full-resolution Telea inpainting of ~10-MP smartphone images was computationally prohibitive); then resize to 224x224.
Also stores native resolution for the domain-gap audit, and an un-hair-removed 224 version for the deviation check."""
import numpy as np, cv2, pandas as pd
from multiprocessing import Pool
from common import *

def proc(p):
    b = cv2.imread(p); h, w = b.shape[:2]; s = max(h, w)
    raw = cv2.cvtColor(cv2.resize(b, (IMG, IMG), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB) if s <= 1024 else None
    if s > 1024:
        b = cv2.resize(b, (round(w * 1024 / s), round(h * 1024 / s)), interpolation=cv2.INTER_AREA)
        raw = cv2.cvtColor(cv2.resize(b, (IMG, IMG), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    hr = cv2.cvtColor(cv2.resize(dull_razor(b), (IMG, IMG), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return hr, raw, h, w

if __name__ == '__main__':
    pad = load_pad()
    with Pool(9) as pool:
        R = pool.map(proc, list(pad.path), chunksize=8)
    np.save(CACHE / 'pad_x.npy', np.stack([r[0] for r in R])); np.save(CACHE / 'pad_x_nohair.npy', np.stack([r[1] for r in R]))
    pad['h_native'] = [r[2] for r in R]; pad['w_native'] = [r[3] for r in R]
    pad.to_csv(CACHE / 'pad_meta.csv', index=False); print('done', len(pad))
