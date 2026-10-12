"""Display-only rendering of the Grad-CAM++ map for the quick-predict GUI.

The map itself is computed by inference/gradcam.py (unchanged: same layer, same class, same normalisation). Its native
resolution is the 7x7 grid of the last convolution stage, so the 224x224 map is blocky. For viewing only, this module
(1) smooths the map with a Gaussian of about one grid cell, (2) draws the overlay with a per-pixel opacity that follows the
map, so low-contribution areas stay fully visible, and (3) outlines the region holding the upper half of the map's range.
Nothing here feeds back into any probability, threshold, stored value or the research-output file."""
import cv2
import numpy as np

MAX_SIDE = 900            # display size of the longer image side (pixels)
CELL = 224 / 7            # one grid cell of the native map, in 224-px units
OUTLINE_LEVEL = 0.5


def _size(h, w):
    s = min(1.0, MAX_SIDE / max(h, w))
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def smooth_map(cam224):
    """Smooth the 224x224 map (display only) and re-scale it to [0, 1]."""
    cam = cv2.GaussianBlur(np.asarray(cam224, np.float32), (0, 0), sigmaX=CELL * 0.45)
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


def render(original_bgr, cam224):
    """Return PNG bytes: (original, heat map with outline, overlay with outline), all at the same display size."""
    h, w = original_bgr.shape[:2]
    W, H = _size(h, w)
    base = cv2.resize(original_bgr, (W, H), interpolation=cv2.INTER_AREA)
    cam = np.clip(cv2.resize(smooth_map(cam224), (W, H), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (0.10 + 0.70 * cam)[..., None]                       # low map value -> original image shows through
    overlay = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    mask = (cam >= OUTLINE_LEVEL).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thick = max(2, int(round(max(W, H) / 300)))
    for img in (heat, overlay):
        cv2.drawContours(img, contours, -1, (255, 255, 255), thick + 2, cv2.LINE_AA)   # white halo under a cyan line
        cv2.drawContours(img, contours, -1, (255, 220, 0), thick, cv2.LINE_AA)           # BGR cyan-blue
    enc = lambda im: cv2.imencode('.png', im)[1].tobytes()
    return enc(base), enc(heat), enc(overlay)
