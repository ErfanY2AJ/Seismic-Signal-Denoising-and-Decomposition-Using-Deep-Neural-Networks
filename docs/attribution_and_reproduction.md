# Attribution, permissions, and reproducibility

## 1. People and roles

- **Erfan Yavari:** documentation and organization of this GitHub-ready package; English explanation and step-by-step walkthrough of the supplied code.
- **Rahaleh Mohseni:** named on the title page of the Persian-language course report supplied for this package. The source document should not be relabeled as solely authored by Erfan Yavari.
- **Weiqiang Zhu, S. Mostafa Mousavi, and Gregory C. Beroza:** creators/authors of the original **DeepDenoiser** research method and publication (2019).

## 2. Upstream code and license

The supplied Python notebooks match examples from the official AI4EPS teaching material or DeepDenoiser project. Their reuse is attributed to the original project. The official DeepDenoiser repository is published under the **MIT license** and lists **Copyright (c) 2021 Weiqiang Zhu**. The full upstream license is reproduced in `UPSTREAM_MIT_LICENSE.txt` for convenience. Preserve that notice with substantial reuse of the upstream source.

- Repo: https://github.com/AI4EPS/DeepDenoiser
- License: https://github.com/AI4EPS/DeepDenoiser/blob/main/LICENSE
- Tutorial: https://ai4eps.github.io/homepage/eps207-observational-seismology/lectures/codes/denoising/

## 3. Figures from the report

`assets/report_figures/` contains figure extracts from the user-supplied course PDF. They depict research and outcomes documented in the source report and related paper. **They have not been redrawn as independently developed results.** The academic article is published by IEEE, and software's MIT license does not automatically confer redistribution rights to journal figure artwork. **Before hosting the extracted images publicly, confirm applicable reuse permission or replace them with original drawings and plots.** A citation is essential but not alone a license.

`assets/notebook_outputs/` contains graphics embedded as saved outputs in the two provided Jupyter notebook files. Their presence does not establish that they were executed in a new environment as part of this package.

## 4. What is reproducible with the files supplied?

- **Text/code review:** yes; both `.ipynb` files are included intact.
- **Interactive inference:** depends on an active compatible DeepDenoiser API; the old endpoint embedded in the notebook might no longer respond.
- **Archived notebook outputs:** yes; PNGs are provided for GitHub viewing.
- **Batch figures from code:** not without input and prediction `.npz` arrays, which were not included.
- **Training and publication-level benchmarks:** no; datasets, splits, trained checkpoints, and training scripts are not included.

No new model was trained, and no test metrics were calculated while preparing this documentation.
