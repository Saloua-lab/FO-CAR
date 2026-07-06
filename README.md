# FO-CAR: Fuzzy Operators-based Context-Aware Recommendation

This repository contains the reference implementation of **FO-CAR**, a context-aware
recommendation model that combines **Shapley value-based context feature weighting**
with **Choquet integral-based neighbor selection (CINS)**, together with all the
scripts used for the experiments reported in the paper.



## Method overview

FO-CAR predicts ratings in three steps:

1. **Shapley-based context feature weighting** — the predictive importance of each
   contextual dimension is estimated by Monte Carlo permutation sampling of a
   cooperative game whose payoff is the reduction in prediction error (Algorithm 1
   in the paper).
2. **Neighborhood generation** — candidate neighbors (co-raters of the target item)
   are screened with a Shapley-weighted contextual similarity (*CTXFeatSim*).
3. **CINS: Choquet integral-based neighbor selection** — a local Sugeno
   λ-measure is fitted to the candidates' context-match vectors and Pearson
   correlations; neighbors are selected by their Choquet relevance score and
   weighted by Pearson × Choquet for the final prediction (Algorithm 2 in the paper).

## Repository structure

| Folder | Content |
|---|---|
| `LDOS/` | FO-CAR pipeline for the LDOS-CoMoDa dataset (plus supporting classes: `LdosData`, `ShapleyWeighting`, `Neighbor2`, `User2`) |
| `InCarMusic/` | FO-CAR pipeline for the InCarMusic dataset |
| `Movie/` | FO-CAR pipeline for the DePaul Movie dataset |
| `Food/` | FO-CAR pipeline for the Food dataset |
| `RichMovielens/` | FO-CAR pipeline for the Rich MovieLens dataset |
| `MTVContext/` | FO-CAR pipeline for the Rich MTV dataset |
| `Amazon/` | FO-CAR pipeline for the Amazon-Books rating-prediction experiment |
| `FullCtx/` | Full-context (`r^C`) coverage and metrics analysis for all six datasets |
| `Robustnes/Java programs/` | Robustness experiments: irrelevant-neighbor injection, injection-rate and ε sweeps, neighbor-weight distribution |
| `Datasets Preprocessing/` | Python scripts to clean/build each dataset into the CSV format expected by the pipelines |
| `Figures_generates/` | Python scripts generating the figures of the paper |

Each `RecomMain2_FOCAR_*.java` file is a self-contained pipeline (data loading,
Steps 1–3, evaluation) for one dataset.

## Requirements

- **Java 8+** (the pipelines are plain Java, no external dependencies)
- **Python 3.8+** with `pandas` and `matplotlib` (preprocessing and figure scripts)
- **[CARSKit](https://github.com/irecsys/CARSKit)** (optional) to reproduce the
  baseline results (ItemKNN, UserKNN, SVD++, CAMF variants, DCW, ...)

## Data

The datasets are **not redistributed** in this repository. Please obtain them from
their original sources:

| Dataset | Source |
|---|---|
| LDOS-CoMoDa | Košir et al. — request from the LDOS lab (University of Ljubljana) |
| InCarMusic | Baltrunas et al. |
| DePaul Movie | Zheng et al. — available with the CARSKit distribution |
| Food | Ono et al. |
| Rich MovieLens / Rich MTV | Zammali et al. (context-enriched datasets) |
| Amazon-Books | Standard Amazon review data |

Run the corresponding script in `Datasets Preprocessing/` to produce the cleaned
CSV (format: `user,item,rating,<context dimensions...>`; missing context values
encoded as `-1`), then update the `DATA_CSV` path at the top of the matching
`RecomMain2_FOCAR_*.java` file.

## Running an experiment

Example for Rich MovieLens:

```bash
cd RichMovielens
javac RecomMain2_FOCAR_RichML.java
java RecomMain2_FOCAR_RichML
```

The program prints per-fold diagnostics and writes the final averaged metrics
(MAE, RMSE, Precision@5/@10, Recall@5/@10, NDCG@5/@10 over 10 seeds × 5-fold CV)
to the output file configured at the top of the class.

## Hyperparameters (paper settings)

| Parameter | Value | Description |
|---|---|---|
| `SHAPLEY_SAMPLES` (T) | 200 | sampled training instances (Step 1) |
| `SHAPLEY_ITER` (K) | 30 | permutations per instance (Step 1) |
| `EPSILON` (ε) | ≈ 1/N | CINS Choquet relevance threshold (Step 3) |
| Gradient descent | 100 iterations, η = 0.05/M | local Sugeno density fitting |
| Evaluation | 10 seeds × 5-fold CV | relevance threshold: rating ≥ 4 (1–5 scale) |

Baseline hyperparameters follow the CARSKit configuration files (see the paper's
experimental-setup section).

## Citation

If you use this code, please cite:

```bibtex
@article{focar2026,
  title   = {[Paper title]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026}
}
```

## License

[MIT](LICENSE) — see the LICENSE file. The datasets remain subject to the terms
of their respective original distributors.
