"""Section 4.16: the research GUI and two illustrative outputs (Figures 13-14). Every number is read from the frozen manifest
(clinical_eval_restored/model/model_manifest.json), out/results/gui_example.json (written by make_gui_example.py) and
out/results/model_comparison.json; nothing is typed in."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'clinical_eval_restored' / 'model' / 'model_manifest.json'
EXAMPLE = HERE / 'out' / 'results' / 'gui_example.json'
FIG13 = HERE / 'out' / 'figures' / 'fig13_gui_output.png'
CLASS_NAMES = {'AKIEC': 'actinic keratosis/intraepithelial carcinoma', 'BCC': 'basal cell carcinoma', 'BKL': 'benign keratosis-like lesion',
               'DF': 'dermatofibroma', 'MEL': 'melanoma', 'NV': 'melanocytic nevus', 'VASC': 'vascular lesion'}


FIGS = [HERE / 'out' / 'figures' / 'fig13_gui_output.png', HERE / 'out' / 'figures' / 'fig14_gui_output.png']


EXAM = [HERE / 'out' / 'figures' / 'exam_7_deidentified.png', HERE / 'out' / 'figures' / 'exam_8_deidentified.png']


def available():
    return MANIFEST.exists() and EXAMPLE.exists() and all(f.exists() for f in FIGS)


def _describe(e):
    probs = e['class_probabilities_percent']; ranked = sorted(probs, key=probs.get, reverse=True); top, nxt = ranked[0], ranked[1]
    assert top == e['predicted_class'] and abs(probs[top] - e['model_probability_percent']) < 0.11 and e['quality_gate_passed']
    return top, nxt, probs


def sec_gui(M, pad_acc):
    man = json.loads(MANIFEST.read_text(encoding='utf-8')); ex = json.loads(EXAMPLE.read_text(encoding='utf-8'))['examples']
    assert len(ex) == 2
    (t1, n1, p1), (t2, n2, p2) = _describe(ex[0]), _describe(ex[1])
    T = man['calibration_parameters']['temperature']
    M.h(2, '4.16 Research interface and illustrative outputs')
    M.p(f"__Interface.__ To make the frozen primary model (M1, DenseNet121 with Grad-CAM++, seed 42, FP32; version {man['model_version']}) usable on a single image, "
        f"we built a local research interface (a web application on the same computer; it opens no network connection and stores nothing). Before any image is processed it verifies the checkpoint "
        f"(SHA-256 {man['checkpoint_sha256'][:12]}…), loads the weights with strict matching and reproduces reference outputs for public HAM10000 images; if any check fails, inference is blocked. "
        f"The image is then passed through the preprocessing of the study (version {man['preprocessing_version']}: hair removal, resizing to {man['input_size']} × {man['input_size']} pixels, ImageNet normalisation), "
        f"the logits are divided by the temperature fitted on the HAM10000 calibration split (T = {T:.4f}), and the interface reports the predicted class, the calibrated model probability of each of the seven classes "
        f"and a {man['gradcam_method']} explanation for the predicted class from the layer {man['gradcam_target_layer']}. Images that are too small, too blurred or too dark or bright, by limits derived from HAM10000 only, are refused without a prediction. "
        f"The interface cannot train, fine-tune, recalibrate or change the threshold of the model, and every setting is read from a machine-readable model manifest (model_manifest.json: architecture, checkpoint hash, class mapping, input size, normalisation, "
        f"preprocessing version, calibration, Grad-CAM++ layer, software versions and the frozen flag). Its outputs are labelled as model probabilities for research use and it never reports a diagnosis; because the benign/malignant grouping of the seven classes has not been confirmed by the investigators, it shows no binary flag.")
    M.p(f"__Illustrative outputs.__ [[F:gui]] and [[F:gui2]] show the interface for two de-identified clinical images of pigmented skin lesions, used under the institutional authorization described in the Declarations. "
        f"The first is a close-up clinical photograph; the second is a magnified, dermoscopy-style close-up. "
        f"For the first image the AI prediction was {t1} ({CLASS_NAMES[t1]}) with a model probability of {p1[t1]:.1f}%; the next most probable class was {n1} ({CLASS_NAMES[n1]}) at {p1[n1]:.1f}%. "
        f"For the second image the AI prediction was {t2} ({CLASS_NAMES[t2]}) with a model probability of {p2[t2]:.1f}%; the next most probable class was {n2} ({CLASS_NAMES[n2]}) at {p2[n2]:.1f}%. "
        f"In both, the Grad-CAM++ outline, which encloses the area holding the upper half of the map's range, includes the lesion and part of the surrounding skin. For display only, the 7 × 7 map is enlarged and smoothed and the overlay lets the original image show through where the map is low; "
        f"the map values and the probabilities are unchanged by this.")
    M.fig(FIGS[0], "Output of the research interface for the first de-identified clinical image (close-up clinical photograph): AI prediction and calibrated model probability of the seven classes (left) and the Grad-CAM++ explanation for the predicted class (right; overlay and heat map are smoothed for display only). Illustrative research output; not a clinical result.", 16.5, label='fig:gui')
    M.fig(FIGS[1], "Output of the research interface for the second de-identified clinical image (magnified, dermoscopy-style close-up), in the same format as the first. Illustrative research output; not a clinical result.", 16.5, label='fig:gui2')
    if all(f.exists() for f in EXAM):
        M.p("__Examination setting.__ [[F:exam1]] and [[F:exam2]] show the setting of the clinical evaluation: a dermatologist inspects a pigmented lesion with a handheld dermatoscope while the dermoscopic view is shown on a monitor. "
            "To keep the images de-identified, the clinician's face was blurred in both and an on-image location and time stamp was removed from the first; they are not the images analysed in [[F:gui]] and [[F:gui2]].")
        M.fig(EXAM[0], "Examination setting of the clinical evaluation (first photograph): a dermatologist examines a pigmented lesion on the shoulder with a handheld dermatoscope; the monitor shows the dermoscopic view. The clinician's face is blurred and the on-image location and time stamp is removed.", 9.0, label='fig:exam1')
        M.fig(EXAM[1], "Examination setting of the clinical evaluation (second photograph): a dermatologist examines a pigmented lesion on the thigh with a handheld dermatoscope; the monitor shows the dermoscopic view. The clinician's face is blurred.", 9.0, label='fig:exam2')
    M.p(f"__What these examples do and do not show.__ They demonstrate the output format and the safeguards of the interface. They are two images, part of no analysis, table or claim in this paper, and no reference diagnosis is reported for either, "
        f"so they say nothing about whether the predictions are correct and no accuracy can be inferred. The first image comes from a different modality than the training data; for such a shift, "
        f"Section 4.9 measured a five-class accuracy of {pad_acc:.3f} on PAD-UFES-20. The second resembles the training images more closely, but its acquisition conditions have not been compared with those of HAM10000. "
        f"The probabilities shown are the model's scores, possibly under domain shift, not the probability that a lesion is of the predicted class. "
        f"A prospective reader study with reference diagnoses, in which a clinician's assessment is recorded before and after the model output, is required before any statement about clinical usefulness (Section 6).")
