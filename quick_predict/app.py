"""Quick-predict GUI: upload one image, get the frozen model's research output with its model probabilities.

    python quick_predict/app.py            # http://127.0.0.1:5090  (this computer only)

This is the lightweight sibling of the reader-study system (run.py): no sign-in, no database, nothing stored.
An uploaded image is processed in memory and discarded when the response has been built. Every model, preprocessing,
calibration, threshold and Grad-CAM++ setting is read from model/model_manifest.json through the same hash-verified
inference package as the reader-study system (inference/predictor.py), so the two cannot disagree. The GUI cannot
train, fine-tune, recalibrate or change the model, and it opens no network connection.

Output wording: AI prediction, model probability, predicted class, research output. The page never states a
diagnosis."""
import base64
import datetime
import hashlib
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from flask import Flask, render_template, request                       # noqa: E402

from inference import config, golden                                    # noqa: E402
from inference.gradcam import METHOD, ExplanationError, compute_cam   # noqa: E402
from inference.manifest import INTEGRITY_FAILED, IntegrityError        # noqa: E402
from inference.predictor import FrozenPredictor, InferenceError        # noqa: E402
from inference.preprocessing import decode_image, to_rgb224            # noqa: E402
from inference.quality_check import QUALITY_MESSAGE, UploadRejected, assess_quality, validate_upload   # noqa: E402
from inference import lock                                              # noqa: E402
from display import render                                              # noqa: E402  (display-only rendering, same folder)

# Standard HAM10000 names of the seven class codes stored in the manifest (display only).
CLASS_NAMES = {'AKIEC': 'Actinic keratosis / intraepithelial carcinoma', 'BCC': 'Basal cell carcinoma',
               'BKL': 'Benign keratosis-like lesion', 'DF': 'Dermatofibroma', 'MEL': 'Melanoma',
               'NV': 'Melanocytic nevus', 'VASC': 'Vascular lesion'}

EXTERNAL_CAUTION = ('The model was trained and calibrated on dermoscopic images (HAM10000). In the accompanying study the same model '
                    'reached a five-class accuracy of only about 0.28 on a different dataset of clinical (non-dermoscopic) photographs '
                    '(PAD-UFES-20), so its probabilities should not be relied on for images unlike its training data.')


def _startup():
    """Verify the frozen package (checksums, strict load, golden test). Inference stays blocked unless everything passes."""
    state = {'ok': False, 'message': ''}
    try:
        predictor = FrozenPredictor()
        bad = [r for r in golden.run(predictor, include_cam=False) if not r[1]]
        if bad:
            state['message'] = 'Golden test failed: inference is blocked until the model package is verified.'
            return None, state
        state['ok'] = True
        return predictor, state
    except IntegrityError:
        state['message'] = INTEGRITY_FAILED
    except Exception as e:                                              # any other fault also blocks inference
        state['message'] = f'Model could not be loaded: {type(e).__name__}'
    return None, state


def _png_uri(png_bytes):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes).decode('ascii')


def _model_info(p):
    m = p.manifest
    bm = m['binary_mapping']
    return {'name': m['model_name'], 'version': m['model_version'], 'checkpoint_sha256': m['checkpoint_sha256'],
            'preprocessing_version': m['preprocessing_version'], 'temperature': p.temperature,
            'calibration_method': m['calibration_method'], 'gradcam_method': m['gradcam_method'],
            'gradcam_layer': m['gradcam_target_layer'], 'threshold': p.threshold,
            'binary_confirmed': bool(bm['confirmed_by_investigator']),
            'malignant_group': bm['malignant'], 'benign_group': bm['benign'], 'created_at': m['created_at'],
            'frozen': m['frozen']}


def _analyse(p, data, filename):
    """Return a dict for the template. Raises UploadRejected for a bad file; everything else is reported in the dict."""
    validate_upload(data, filename)
    img = decode_image(data)
    q = assess_quality(img, to_rgb224(img))
    out = {'quality': q.to_dict(), 'prediction': None, 'images': None, 'explanation_error': None, 'error': None}
    if not q.passed:
        out['error'] = QUALITY_MESSAGE
        return out
    try:
        pred, rgb = p.predict(img)
    except lock.FrozenModelError:
        raise
    except InferenceError:
        out['error'] = 'AI inference could not be completed.'
        return out
    ranked = sorted(pred.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top1, top2 = ranked[0], ranked[1]
    out['ranked'] = [{'code': c, 'name': CLASS_NAMES.get(c, c), 'p': v, 'is_top': i == 0} for i, (c, v) in enumerate(ranked)]
    out['margin'] = top1[1] - top2[1]
    out['second'] = {'code': top2[0], 'name': CLASS_NAMES.get(top2[0], top2[0]), 'p': top2[1]}
    out['top_name'] = CLASS_NAMES.get(top1[0], top1[0])
    out['spread'] = top1[1] <= 0.5
    out['prediction'] = pred
    try:
        cam = compute_cam(p, rgb, pred.predicted_index)
        base, heat, overlay = render(img, cam)
        out['images'] = {'original': _png_uri(base), 'heatmap': _png_uri(heat), 'overlay': _png_uri(overlay)}
    except ExplanationError:
        out['explanation_error'] = 'Explanation generation unavailable for this case.'
    return out


def _research_output(p, pred, data, q, info):
    """Machine-readable copy of what is on screen (no file name, no pixel data)."""
    doc = {'research_output': True, 'notice': config.SHORT_NOTICE,
           'generated_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           'image_sha256': hashlib.sha256(data).hexdigest(),
           'ai_prediction': {'predicted_class': pred.predicted_class, 'model_probability': pred.predicted_probability,
                             'probabilities_calibrated': pred.probabilities},
           'model': {'name': info['name'], 'version': info['version'], 'checkpoint_sha256': info['checkpoint_sha256'],
                     'preprocessing_version': info['preprocessing_version'], 'calibration_method': info['calibration_method'],
                     'temperature': info['temperature'], 'gradcam_method': info['gradcam_method'],
                     'gradcam_target_layer': info['gradcam_layer']},
           'image_quality': q, 'inference_ms': round(pred.inference_ms, 1)}
    if info['binary_confirmed']:
        doc['binary_research_flag'] = {'p_malignant_group': pred.p_malignant, 'threshold': pred.threshold,
                                       'above_threshold': pred.above_threshold, 'malignant_group': info['malignant_group']}
    raw = json.dumps(doc, indent=2).encode('utf-8')
    return 'data:application/json;base64,' + base64.b64encode(raw).decode('ascii')


def create_app():
    app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_BYTES + 1024 * 1024
    predictor, state = _startup()
    info = _model_info(predictor) if predictor else None

    @app.after_request
    def headers(resp):
        resp.headers['Cache-Control'] = 'no-store'
        resp.headers['X-Content-Type-Options'] = 'nosniff'
        resp.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'"
        return resp

    @app.route('/', methods=['GET', 'POST'])
    def index():
        ctx = {'state': state, 'info': info, 'notice': config.RESEARCH_NOTICE, 'short_notice': config.SHORT_NOTICE,
               'explanation_notice': config.EXPLANATION_NOTICE, 'explanation_detail': config.EXPLANATION_DETAIL,
               'external_caution': EXTERNAL_CAUTION, 'result': None, 'upload_error': None, 'json_uri': None,
               'status': 'idle'}
        if request.method == 'POST' and predictor:
            ctx['status'] = 'running'
            f = request.files.get('image')
            if f is None or not f.filename:
                ctx['upload_error'] = 'Choose an image file first.'
                ctx['status'] = 'idle'
            else:
                data = f.read()
                try:
                    res = _analyse(predictor, data, f.filename)
                    ctx['result'] = res
                    ctx['status'] = 'success' if res['prediction'] is not None else 'idle'
                    if res['prediction'] is not None:
                        ctx['json_uri'] = _research_output(predictor, res['prediction'], data, res['quality'], info)
                except UploadRejected as e:
                    ctx['upload_error'] = str(e)
                    ctx['status'] = 'idle'
                except lock.FrozenModelError:
                    ctx['upload_error'] = 'Model integrity verification failed.'
                    ctx['status'] = 'idle'
        return render_template('index.html', **ctx)

    @app.errorhandler(413)
    def too_large(_):
        return render_template('index.html', state=state, info=info, notice=config.RESEARCH_NOTICE, short_notice=config.SHORT_NOTICE,
                               explanation_notice=config.EXPLANATION_NOTICE, explanation_detail=config.EXPLANATION_DETAIL,
                               external_caution=EXTERNAL_CAUTION, result=None, json_uri=None,
                               upload_error=f'File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.'), 413

    return app


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5090)
    ap.add_argument('--host', default='127.0.0.1', help='default: this computer only')
    a = ap.parse_args()
    if a.host not in ('127.0.0.1', 'localhost', '::1'):
        sys.exit('The quick-predict GUI has no sign-in, so it only runs on 127.0.0.1. Use run.py (with sign-in and TLS) for shared use.')
    app = create_app()
    print(f'http://{a.host}:{a.port}   (Ctrl+C to stop)')
    app.run(host=a.host, port=a.port, debug=False, threaded=False)


if __name__ == '__main__':
    main()
