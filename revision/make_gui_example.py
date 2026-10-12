"""Figures 13-14 and gui_example.json: the quick-predict research GUI run on de-identified clinical images.

    python make_gui_example.py <image1> [<image2> ...]     # Figure 13, 14, ... in the order given

Runs the image through the same code path as the GUI (clinical_eval_restored/quick_predict/app.py, test client, so the
frozen model, quality gate and Grad-CAM++ are exactly those of the GUI), renders the response page with headless Edge or
Chrome, and writes out/figures/fig13_gui_output.png plus out/results/gui_example.json (class probabilities, model and
manifest identifiers, SHA-256 of the image; no pixels and no file name). Inference only: nothing is trained, fitted or
tuned, and the result feeds no analysis in the paper."""
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE / 'clinical_eval_restored'
OUT = HERE / 'out'
BROWSERS = [r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Google\Chrome\Application\chrome.exe']
WIDTH = 700                      # layout width (px) of each half; below 860 px the page stacks into one column


def gui_html(img_bytes):
    sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / 'quick_predict'))
    import app as quick
    client = quick.create_app().test_client()
    r = client.post('/', data={'image': (io.BytesIO(img_bytes), 'case.jpg')}, content_type='multipart/form-data')
    assert r.status_code == 200
    return r.data.decode('utf-8')


def screenshot(html, png=None, height=2600):
    """Render the page with headless Edge/Chrome and return the BGR image (the PNG is written in a temporary folder)."""
    import time
    browser = next(b for b in BROWSERS if Path(b).exists())
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        f = Path(td) / 'page.html'; f.write_text(html, encoding='utf-8'); shot = Path(td) / 'shot.png'
        subprocess.run([browser, '--headless=new', '--disable-gpu', '--hide-scrollbars', f'--user-data-dir={td}/profile', '--no-first-run',
                        '--virtual-time-budget=3000', f'--window-size={WIDTH},{height}', f'--screenshot={shot}', f.as_uri()],
                       check=True, capture_output=True, timeout=120)
        for _ in range(60):                               # the browser can return before the PNG has been written
            if shot.exists() and shot.stat().st_size > 0: break
            time.sleep(0.5)
        time.sleep(0.5)
        assert shot.exists(), 'screenshot was not produced'
        return cv2.imdecode(np.fromfile(str(shot), dtype=np.uint8), cv2.IMREAD_COLOR)


def page_with(html, keys):
    """The same page, reduced to the result cards whose text contains one of keys (a figure view; nothing is recomputed)."""
    head = re.sub(r'<header>.*?</header>', '', html[:html.index('<main>')], flags=re.S)
    cards = re.findall(r'<section class="card.*?</section>', html, flags=re.S)
    keep = [c for c in cards if any(k in c for k in keys)]
    assert keep, keys
    return head + '<main>' + ''.join(keep) + '</main></body></html>'


def crop_to_content(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = np.where(g.std(axis=1) > 6)[0]
    return img[:min(img.shape[0], int(rows.max()) + 40)]


def run_one(path, fig_name):
    data = Path(path).read_bytes()
    html = gui_html(data)
    probs = re.findall(r'<tr[^>]*><td>(\w+)</td><td>[^<]+</td><td class="num">([\d.]+)%', html)
    top = re.search(r'class="big">(\w+)', html).group(1)
    pct = float(re.search(r'class="pct">([\d.]+)%', html).group(1))
    assert len(probs) == 7 and probs[0][0] == top and abs(float(probs[0][1]) - pct) < 0.11, 'unexpected page content'
    left = crop_to_content(screenshot(page_with(html, ('AI prediction', 'Model probability for each class', 'research flag')), OUT / 'figures' / '_a.png'))
    right = crop_to_content(screenshot(page_with(html, ('Model explanation (',)), OUT / 'figures' / '_b.png'))
    h = max(left.shape[0], right.shape[0]); bg = tuple(int(v) for v in left[-1, 3])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=bg)
    gap = np.full((h, 18, 3), 255, np.uint8)
    cv2.imwrite(str(OUT / 'figures' / fig_name), np.hstack([pad(left), gap, pad(right)]))
    dec = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
    return {'figure': fig_name, 'image_sha256': hashlib.sha256(data).hexdigest(), 'image_width': int(dec.shape[1]), 'image_height': int(dec.shape[0]),
            'predicted_class': top, 'model_probability_percent': pct, 'class_probabilities_percent': {c: float(v) for c, v in probs},
            'quality_gate_passed': 'Image quality is insufficient' not in html, 'binary_flag_shown': 'above threshold' in html or 'below threshold' in html}


def main(paths):
    man = json.loads((PKG / 'model' / 'model_manifest.json').read_text(encoding='utf-8'))
    (OUT / 'results').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    ex = [run_one(p, f'fig{13 + i}_gui_output.png') for i, p in enumerate(paths)]
    res = {'model_version': man['model_version'], 'checkpoint_sha256': man['checkpoint_sha256'], 'model_name': man['model_name'], 'examples': ex}
    json.dump(res, open(OUT / 'results' / 'gui_example.json', 'w'), indent=1)
    for e in ex: print(e['figure'], e['predicted_class'], e['model_probability_percent'])


if __name__ == '__main__':
    main(sys.argv[1:])
