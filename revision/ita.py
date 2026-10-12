"""Individual Typology Angle (ITA) proxy for all 10,015 HAM10000 images.
v1 = original definition (all non-lesion pixels of the 224x224 hair-removed image, mean L / mean b).
v2 = vignette-robust: excludes the black dermoscope vignette (dark pixels connected to the image border) and a 5-px halo
     around the lesion, uses the median of L and b. Both use ITA = atan2(L-50, b) in degrees (Chardon 1991)."""
import numpy as np, cv2, pandas as pd
from common import *

X = np.load(CACHE / 'ham_x.npy', mmap_mode='r'); MK = np.load(CACHE / 'ham_mask.npy', mmap_mode='r')
ham = load_ham(); rows = []
kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)); k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
for i in range(len(ham)):
    img = np.ascontiguousarray(X[i]); m = MK[i].astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32); L = lab[..., 0] * (100 / 255.0); b = lab[..., 2] - 128.0
    sk = m == 0
    ita1 = np.degrees(np.arctan2(L[sk].mean() - 50.0, b[sk].mean())) if sk.sum() >= 50 else np.nan
    gray = img.max(2); dark = (gray < 45).astype(np.uint8)
    nl, lbl, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    vig = np.zeros_like(dark)
    for l in range(1, nl):
        x0, y0, w, h, ar = st[l]
        if ar > 20 and (x0 == 0 or y0 == 0 or x0 + w == IMG or y0 + h == IMG): vig[lbl == l] = 1
    vig = cv2.dilate(vig, k3); halo = cv2.dilate(m, kern)
    sk2 = (vig == 0) & (halo == 0)
    ita2 = np.degrees(np.arctan2(np.median(L[sk2]) - 50.0, np.median(b[sk2]))) if sk2.sum() >= 50 else np.nan
    rows.append((ham.image_id[i], ita1, ita2, float(vig.mean()), int(sk.sum()), int(sk2.sum())))
df = pd.DataFrame(rows, columns=['image_id', 'ita_v1', 'ita_v2', 'vignette_frac', 'n_skin_v1', 'n_skin_v2'])
bucket = lambda t: np.where(np.isnan(t), None, np.where(t > 55, 'I-II', np.where(t > 28, 'III-IV', 'V-VI')))
df['fst_v1'] = bucket(df.ita_v1.values); df['fst_v2'] = bucket(df.ita_v2.values)
df.to_csv(OUT / 'ita_all.csv', index=False)
print(df.fst_v1.value_counts().to_dict(), df.fst_v2.value_counts().to_dict(), df.vignette_frac.describe())
