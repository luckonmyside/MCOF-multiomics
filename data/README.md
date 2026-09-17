# Required input data

This directory intentionally contains no cohort matrices or patient records. Dataset access and redistribution remain subject to the original providers' conditions. The study context identifies TCGA BRCA/STAD via UCSC Xena, ROSMAP via AD Knowledge Portal/Synapse, and SCZ proteomics/metabolomics via PXD024474 and MSV000086975. These source names are **not** a pinned reconstruction of the exact historical download snapshot.

Each authorized analysis-ready dataset directory contains:

```text
BRCA/
  1_all.csv
  2_all.csv
  labels_all.csv
  1_featname.csv
  2_featname.csv
```

Matrices are numeric, sample rows by feature columns, with **no header and no row-name column**. `labels_all.csv` is one label per row, also without a header. Each `m_featname.csv` is one feature name per input column in the same order, without a header. ROSMAP additionally requires `3_all.csv` and `3_featname.csv`. The old optional `featname_all.csv` is not a replacement for verifying modality-specific annotations.

| Dataset | Samples | Ordered omics blocks | Primary panel widths | Encoded outcome classes in the supplied protocol |
|---|---:|---|---|---|
| BRCA | 1002 | RNA, miRNA | 1000, 1000 | Basal-like, HER2-enriched, Luminal A, Luminal B |
| STAD | 217 | RNA, miRNA | 1000, 1000 | CIN, GS, MSI |
| ROSMAP | 351 | mRNA, miRNA, DNA methylation | 200, 200, 200 | Control, Alzheimer disease |
| SCZ | 104 | Protein, Metabolite | 200, 200 | Control, Schizophrenia |

Class order is zero-based in the above ordering, as defined in the supplied revision `scripts/common.py`. Raw labels are encoded by the relevant reader; verify the actual mapping rather than inferring it from a newly downloaded cohort.

Preserve row alignment across every modality and labels, and feature ordering within each modality. Stored split indices have meaning only under the original sample ordering. The public split files do not include the original `encoded_label` column or a participant-to-row mapping. Do not substitute three-view, five-class upstream MOGONET BRCA example data for this two-view, four-class BRCA task.

`code/main_model/DATA_preprocessing_v2.py` aligns sample IDs, optionally transposes inputs, optionally merges duplicate features, and exports numbered CSVs. It does not recreate the historical differential-expression selection. Do not generate a new panel with that utility and claim it is the archived study panel.

Load only trusted checkpoint files: the archived checkpoint loader explicitly uses `torch.load(..., weights_only=False)` to restore preprocessing objects. Checkpoints are not distributed here and may contain more than neural-network weights.
