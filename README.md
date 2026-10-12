# Explanation Reliability Beyond Diagnostic Accuracy in Dermoscopic Skin-Lesion Classification

A reproducible research pipeline for evaluating **diagnostic performance, explanation reliability, calibration, faithfulness, stability, and cross-dataset generalization** in dermoscopic skin-lesion classification.

The framework evaluates DenseNet121 on HAM10000 and investigates generalization to PAD-UFES-20 using lesion-level data splitting, multiple random seeds, Grad-CAM++, explanation baselines, and statistical analyses.

## Key Features

- **Reproducible evaluation:** Five model seeds (42–46) with a fixed lesion-level train/validation/test split.
- **Diagnostic performance:** Classification metrics, calibration, and temperature scaling.
- **Explainable AI:** Grad-CAM++ explanations and LSAS-based analysis.
- **Explanation faithfulness:** Deletion/insertion tests and comparison against null and ceiling baselines.
- **Explanation stability:** Monte Carlo dropout with corrected stochastic-pass evaluation.
- **Skin-tone proxy analysis:** Individual Typology Angle (ITA), evaluated using two definitions.
- **Cross-dataset generalization:** HAM10000-to-PAD-UFES-20 evaluation and domain-gap analysis.
- **Robustness and sensitivity:** Per-seed analyses and a 100-epoch convergence check.
- **Research transparency:** Saved numerical outputs, reproducible analysis scripts, and manuscript-generation utilities.

## Research Objectives

This repository supports the investigation of five questions:

1. Does strong diagnostic accuracy imply reliable visual explanations?
2. How faithful and stable are the explanations produced by the model?
3. How do explanation characteristics vary with lesion size, skin-tone proxy, and classification correctness?
4. How much does performance change when the model is evaluated on a different dataset?
5. Are the reported findings consistent across random seeds and training-duration sensitivity checks?

These questions are evaluated through the implemented experimental pipeline. Their answers should be interpreted using the generated results rather than assumed from the framework design.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── model/
    ├── common.py
    ├── prep_cache.py
    ├── prep_pad.py
    ├── ita.py
    ├── train.py
    ├── eval_seed.py
    ├── sts_fix.py
    ├── domain_gap.py
    ├── analysis_cls.py
    ├── analysis_xai.py
    ├── analysis_pad.py
    ├── make_figures.py
    ├── ms_lib.py
    ├── ms_static.py
    ├── ms_results_cls.py
    ├── ms_results_xai.py
    ├── build_manuscript.py
    ├── build_audit.py
    └── out/
        └── results/
```

### Core modules

| File | Purpose |
|---|---|
| `common.py` | Shared configuration, architecture, loss, data splits, and loaders |
| `prep_cache.py` | HAM10000 preprocessing and image/mask caching |
| `prep_pad.py` | PAD-UFES-20 preprocessing |
| `ita.py` | ITA skin-tone proxy calculations |
| `train.py` | DenseNet121 training |
| `eval_seed.py` | Classification, calibration, explanations, faithfulness, and stability evaluation |
| `sts_fix.py` | Recomputes corrected MC-dropout stability results |
| `domain_gap.py` | Domain-gap measurements and patient-level adaptation analysis |
| `analysis_cls.py` | Aggregates classification, calibration, and convergence results |
| `analysis_xai.py` | Aggregates explainability and subgroup analyses |
| `analysis_pad.py` | Decomposes cross-dataset performance differences |
| `make_figures.py` | Generates manuscript figures |
| `build_manuscript.py` | Optionally assembles the manuscript |
| `build_audit.py` | Optionally generates results-integrity and submission audits |

The `model/out/results/` directory contains the numerical outputs used for the manuscript. Model weights, per-seed evaluation artifacts, figures, and preprocessing caches are generated locally and are not included because of their storage requirements.

## Datasets

### HAM10000

The primary dataset for dermoscopic skin-lesion classification and explanation analysis.

- Source: [HAM10000 dataset](https://doi.org/10.7910/DVN/DBW86T)
- Access: Public research dataset
- Preparation: Image resizing, DullRazor hair removal, ImageNet normalization, and caching

### PAD-UFES-20

The external dataset used to evaluate cross-dataset generalization.

- Source: [PAD-UFES-20 dataset](https://doi.org/10.17632/zr7vgbcyr2.1)
- Preparation: Dataset-specific preprocessing, including downscaling images larger than 1024 pixels before hair removal

Download both datasets from their official repositories and follow `data/README.md` for the expected directory structure.

**Important:** Verify dataset access conditions, licensing, label mappings, and preprocessing requirements before running the experiments.

## Experimental Configuration

| Setting | Configuration |
|---|---|
| Model architecture | DenseNet121 |
| Primary dataset | HAM10000 |
| External dataset | PAD-UFES-20 |
| Model seeds | 42, 43, 44, 45, 46 |
| Split seed | 42 |
| Standard training duration | 60 epochs |
| Sensitivity experiment | 100 epochs |
| Explanation method | Grad-CAM++ |
| Stability analysis | Monte Carlo dropout |
| Calibration | Temperature scaling |
| Training hardware | NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM |
| Software environment | Python 3.12.2, PyTorch 2.7.1, CUDA 11.8 |

The fixed lesion-level split is shared across the five model seeds. Model seeds control model initialization and other stochastic training or analysis procedures; they do not change the data split.

## Installation

### Requirements

- Python 3.12.2
- NVIDIA GPU with compatible CUDA support for the training pipeline
- Git
- Sufficient disk space for datasets, cached arrays, model weights, and evaluation artifacts

The complete pipeline was reported to run on an NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM. Actual runtime depends on hardware, dataset placement, and software configuration.

### Install dependencies

Clone the repository and enter the project directory:

```bash
git clone https://github.com/jagyanseni318/Skin-Cancer-Lesion-model.git
cd Skin-Cancer-Lesion-model
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that PyTorch detects the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is unavailable, check the installed PyTorch build, NVIDIA driver, and CUDA compatibility before starting training.

## Reproducing the Experiments

Run the following commands from inside the `model/` directory. Ensure that both datasets are correctly placed before preprocessing.

### 1. Prepare datasets

```bash
cd model

python prep_cache.py
python prep_pad.py
python ita.py
```

These scripts prepare the image caches and compute the ITA skin-tone proxies.

### 2. Train and evaluate all five seeds

Run the following commands in a Bash-compatible shell:

```bash
for seed in 42 43 44 45 46; do
    python train.py --seed "$seed" --epochs 60 --workers 4
    python eval_seed.py --seed "$seed"
    python sts_fix.py "$seed"
done
```

Each seed produces model and evaluation artifacts used by the subsequent analyses.

### 3. Run the convergence sensitivity experiment

```bash
python train.py --seed 42 --epochs 100 --workers 4 --tag _ep100

python eval_seed.py --seed 42 --tag _ep100 --skip_faith
```

This experiment evaluates a longer training schedule for seed 42. Faithfulness computation is skipped for this sensitivity run.

### 4. Analyze cross-dataset generalization

```bash
python domain_gap.py 42 43 44 45 46
```

### 5. Aggregate results and generate figures

```bash
python analysis_cls.py
python analysis_xai.py
python analysis_pad.py
python make_figures.py
```

The resulting outputs are saved under `model/out/results/` and the configured figure/output directories.

**Reproducibility note:** The commands above reflect the documented experimental workflow. For an exact reproduction, verify that the checked-out code, dataset versions, dependency lock versions, and command-line options match those used for the reported results.

## Evaluation Methodology

### Classification and calibration

The evaluation pipeline measures diagnostic performance and calibration, including temperature scaling. Reported metrics should be interpreted alongside class distribution, uncertainty, and variability across seeds.

### Explanation reliability

Grad-CAM++ explanations are evaluated through complementary analyses:

- **Faithfulness:** Deletion/insertion tests assess how predictions respond to changes in image regions.
- **Baseline comparison:** Null and ceiling baselines provide reference points for interpreting explanation scores.
- **Stability:** Monte Carlo dropout evaluates variability across stochastic forward passes.
- **Subgroup analysis:** Explanation characteristics are examined by lesion size, ITA skin-tone proxy, and correct versus incorrect predictions.

### External generalization

The HAM10000-to-PAD-UFES-20 analysis investigates domain differences and decomposes performance changes into population, label-space, prevalence, and residual components.

These analyses do not, by themselves, establish clinical validity or fairness across demographic groups. Interpretation depends on dataset representativeness, label compatibility, statistical uncertainty, and the limitations of proxy measurements.

## Reproducibility and Randomness

- The train/validation/test split uses a fixed split seed of `42`.
- Five model seeds (`42–46`) evaluate variability across stochastic training runs.
- Explainability and faithfulness analyses use dedicated seeded random-number generators.
- Random baselines, including cross-image shuffling, toroidal shifts, and random patch ordering, are designed to be reproducible.
- Numerical outputs are retained in `model/out/results/` to support inspection without rerunning the full pipeline.

A fixed seed improves reproducibility but does not guarantee identical results across different hardware, CUDA versions, library builds, or nondeterministic GPU operations.

## Known Issue and Correction

An earlier version of `eval_seed.py` instantiated the Grad-CAM++ evaluator after enabling dropout layers for Monte Carlo sampling. The evaluator called `model.eval()`, disabling dropout and causing the stochastic passes to become numerically identical.

The corrected implementation re-enables dropout after constructing the Grad-CAM++ evaluator.

The accompanying `sts_fix.py` script recalculates the affected stability results for previously evaluated seeds without repeating the full classification, Grad-CAM++, LSAS, and faithfulness pipeline.

This correction is documented to preserve transparency and make the stability analysis auditable.

## Outputs and Manuscript Generation

The repository includes the numerical outputs used to prepare the manuscript.

Optional manuscript and audit utilities include:

- `build_manuscript.py` — assembles the manuscript in supported document formats.
- `build_audit.py` — generates a results/integrity/submission audit.
- `make_figures.py` — recreates the manuscript figures.

These utilities are not required to reproduce the numerical analyses. Their successful execution depends on the expected output files and the dependencies defined by the implementation.

## Computational Cost

The reported runtime on a single NVIDIA RTX 4050 Laptop GPU (6 GB VRAM) was approximately:

- **Training:** 1 hour per seed
- **Evaluation:** 15–20 minutes per seed

These figures are estimates from the documented environment, not guaranteed runtimes for other systems.

## Data Availability and Ethics

HAM10000 and PAD-UFES-20 are publicly available research datasets. No new patient data were collected for this study.

Users remain responsible for following the datasets' access conditions, licenses, and applicable research-ethics requirements. The pipeline is intended for research evaluation and is **not a clinically validated diagnostic system**.

## Limitations

- External evaluation depends on compatibility between dataset labels and class definitions.
- ITA is used as a skin-tone proxy and should not be treated as a direct measurement of an individual's skin tone or demographic identity.
- Explanation metrics capture specific mathematical properties and do not establish that an explanation is clinically meaningful.
- Results from five model seeds do not capture every possible source of uncertainty.
- Cross-dataset performance differences may reflect multiple sources of domain shift and should not automatically be attributed to one factor.

## Citation

If you use this code or build upon the experiments, cite the associated research paper once its publication details are available.

**Manuscript title:** *Explanation Reliability Beyond Diagnostic Accuracy in Dermoscopic Skin-Lesion Classification*

Add the final authors, publication venue, year, DOI, and BibTeX entry after the manuscript is formally published or assigned a persistent identifier.

## License

A license has not yet been specified in this repository. Add an appropriate license file before redistributing the code or assigning a public archival DOI. Do not assume that the code is MIT-licensed until that license has been explicitly adopted.

## Acknowledgments

We acknowledge the researchers who developed and publicly released HAM10000 and PAD-UFES-20, enabling reproducible research in skin-lesion classification and explainable artificial intelligence.

---

**Reproducibility first. Diagnostic accuracy alone is not enough; explanation quality, stability, calibration, and external generalization also deserve careful evaluation.**
