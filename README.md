# MCOF-multiomics

**MCOF (Multi-Channel attention-based Omics Fusion)** combines omics-specific projection, sample-wise channel recalibration, latent fusion, and one-dimensional convolution for multi-omics classification, with integrated gradients for candidate biomarker prioritization.

Code accompanying manuscript **ijms-4520828**. The manuscript model is **MCOF v2 (`mcof_se`)**.

## Quick start

Run from the repository root in a separate Python environment. 

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python tests/test_models.py
```

These checks verify package integrity and synthetic-model behaviour, not reproduction of study results.

Prepare authorized inputs following the [data instructions](data/README.md), preserving sample and feature order. Use the launcher below to select the manuscript settings (`mcof_se`, seed 1), rather than the archived training script's defaults:

```bash
python scripts/run_main.py \
  --data-dir /path/to/authorized_data/BRCA \
  --output-dir runs/main/BRCA/mcof_se \
  --dry-run
```

Inspect the command, then remove `--dry-run` to train. Add `--cpu` for CPU execution. Training settings are in [configs/mcof_se.json](configs/mcof_se.json).


Citation metadata: [CITATION.cff](CITATION.cff). Licensing: [LICENSE_STATUS.md](LICENSE_STATUS.md) . MOGONET retains its MIT license; no new license has been assigned to the authors' MCOF code.
