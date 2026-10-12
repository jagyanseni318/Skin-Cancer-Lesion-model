"""Cache hair-removed, 224x224-resized RGB arrays (identical to the original pipeline's dull_razor -> Resize(224,224)),
and 224x224 binary lesion masks (original: cv2.resize bilinear, >127)."""
import numpy as np, cv2, pandas as pd
from multiprocessing import Pool
from common import *


def proc_img(p):
    b = cv2.imread(p)
    if b is None:
        return None
    b = dull_razor(b); r = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    return cv2.resize(r, (IMG, IMG), interpolation=cv2.INTER_LINEAR)


def proc_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE); full_frac = float((m > 127).mean())
    m = cv2.resize(m, (IMG, IMG)); return (m > 127).astype(np.uint8), full_frac


if __name__ == '__main__':
    ham = load_ham(); pad = load_pad()
    mask_dir = DATA / 'HAM10000_masks' / 'HAM10000_segmentations_lesion_tschandl'
    with Pool(9) as pool:
        X = pool.map(proc_img, list(ham.image_path), chunksize=32)
        assert all(x is not None for x in X); np.save(CACHE / 'ham_x.npy', np.stack(X)); del X
        M = pool.map(proc_mask, [str(mask_dir / f'{i}_segmentation.png') for i in ham.image_id], chunksize=64)
        np.save(CACHE / 'ham_mask.npy', np.stack([m for m, _ in M]))
        ham['mask_area_full'] = [f for _, f in M]
        XP = pool.map(proc_img, list(pad.path), chunksize=16)
        assert all(x is not None for x in XP); np.save(CACHE / 'pad_x.npy', np.stack(XP))
    ham.to_csv(CACHE / 'ham_meta.csv', index=False); pad.to_csv(CACHE / 'pad_meta.csv', index=False)
    print('done', len(ham), len(pad))
