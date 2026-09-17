# MCOF-multiomics

**Multi-Channel attention-based Omics Fusion (MCOF)** for multi-omics classification and candidate biomarker prioritization.

Code accompanying manuscript **ijms-4520828**, *MCOF: A Multi-Channel Attention-based Omics Fusion Framework for Cancer Subtyping and Candidate Biomarker Prioritization*.

## Which implementation corresponds to the manuscript?

The main model is **MCOF v2, `--model_type mcof_se`**, implemented by `MCOFSEV2` in `code/main_model/models_v2.py`. It uses omics-specific projectors, a sample-wise squeeze-and-excitation gate, latent weighted summation, and a 1D convolutional classifier. The archived entry point also contains development models, and defaults to `mcof_attn` and seed 42. **Those defaults are not the publication recipe.** Use the explicit launcher below, which selects `mcof_se` and seed 1 without modifying the archived core files.

| Directory | Contents |
|---|---|
| `code/main_model/` | Frozen v2 main training, model, preprocessing, and legacy IG modules |
| `code/strict_ablation/` | Separate `mcof_no_se` and `mcof_no_conv` implementations and summary code |
| `code/baselines/` | KNN, SVM, Gaussian NB, RF, DIABLO, and split generation |
| `code/mogonet/` | MOGONET source, original notices, and the study-specific formal runner |
| `code/revision_analysis/` | Single-omics, omitted-omics, PAM50 exclusion, feature-count sensitivity, per-class analysis, IG robustness, and statistics |
| `code/attribution/` | July attribution/annotation and ranking-summary variants; kept separate from the frozen main code |
| `code/efficiency/`, `code/figures/` | Timing, aggregation, table, and figure scripts |
| `splits/` | Original split membership/order, with the outcome-label column removed |
| `configs/`, `provenance/` | Explicit main recipe, source hashes, archived revision settings, protocol records, and file disposition |
| `scripts/`, `tests/` | New release-only launch, integrity, synthetic-data, and compatibility checks |

## Install and check the package

Use a separate Python environment. The recorded server runtime for the supplied revision jobs was Python 3.9.23 with PyTorch 2.8.0+cu128; this is **not** a complete historical dependency lock. See `environments/recorded_runtime.json` and `docs/REPRODUCIBILITY.md`.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python tests/test_models.py
python scripts/synthetic_smoke.py
```

`requirements.txt` lists dependencies, not reconstructed historical versions. For a GPU installation, install the appropriate PyTorch build for the local platform using https://pytorch.org/get-started/locally/ before installing remaining dependencies. Captum is optional and changes the backend chosen by the archived attribution scripts; see `docs/ATTRIBUTION.md`.

The last two commands use synthetic data or random model weights. They check implementation and packaging, **not** clinical accuracy or reproduction of the reported results. R/mixOmics checks are separate:

```bash
Rscript environments/check_R_dependencies.R
```

## Main-model training

Obtain data with the required permissions, preserve the original row/column order, and read `data/README.md`. For one dataset:

```bash
python scripts/run_main.py \
  --data-dir /path/to/authorized_data/BRCA \
  --output-dir runs/main/BRCA/mcof_se \
  --dry-run
```

Inspect the printed command, then remove `--dry-run` to train. Add `--cpu` for CPU execution. The launcher reads `configs/mcof_se.json`, explicitly selects `mcof_se`, and delegates to the unmodified archived trainer. It refuses to overwrite a nonempty output directory. Main defaults include five stratified 80:20 holdouts, five inner folds, seed 1, hidden width 128, convolutional channels 64, kernel width 5, dropout 0.30, batch size 64, learning rate 0.001, weight decay 0.0001, maximum 300 epochs, and patience 30.

The original main trainer generates splits internally; it does **not** accept `--split_file`. The stored split CSVs are used by baseline and revision entry points. `scripts/check_input_splits.py` can check membership against local labels before training. It does not prove participant identity or reconstruct missing source measurements.

For baselines, strict ablation, attribution, and sensitivity analyses, use the commands in **[docs/RUNNING_ANALYSES.md](docs/RUNNING_ANALYSES.md)**. Do not interchange same-named modules from different directories.

## Evaluation scope and provenance

The inputs are previously reduced candidate panels, not original raw-omics cohorts. Downstream train-partition fitting does not retroactively remove any outcome information used to construct those panels. The raw-to-panel differential-expression workflow is not reconstructed by `DATA_preprocessing_v2.py`; that file harmonizes tables and exports aligned inputs. The available archive does not contain a complete executable trace of upstream screening, excluded cancer subtypes, or original data snapshots. The benchmark and rankings are conditional on the panels; fully nested evaluation and independent validation are separate requirements.

The package was prepared from the supplied 171-file code review collection and the separately recovered PAM50 file. Core files retain their collected bytes; site-specific absolute path prefixes in other scripts/configurations were replaced with documented placeholders. No model, loss, metric, or feature-selection algorithm was changed during packaging. Hash equality establishes equality of collected copies, not independent proof of historical execution. See **[docs/PROVENANCE.md](docs/PROVENANCE.md)** and **[docs/ANALYSIS_MAP.md](docs/ANALYSIS_MAP.md)**.

Original source data, checkpoints, individual predictions, and complete IG result tables are not redistributed in this code package. Generated results require the authorized input data and, where applicable, checkpoints/result tables. Synthetic demonstrations must never be reported as study results.

## Attribution and citation

Please retain source and license notices. MOGONET source is accompanied by its upstream MIT license. No new license for the authors' own code has been assigned during repository preparation: see `LICENSE_STATUS.md` and `THIRD_PARTY_NOTICES.md`. This repository should not be described as entirely MIT-licensed unless the rights holders separately authorize that choice.

Use the metadata in `CITATION.cff`. No acceptance status, publication DOI, or successful remote release is asserted by this package.
