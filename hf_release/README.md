---
license: mit
tags:
- actuarial-science
- insurance
- generalized-linear-models
- genetic-algorithm
- pure-premium
- graybox
---

# Café Poliçe: graybox GLM + metaheuristic pure-premium pricing

Results of the article *"Rakamların Diliyle Güven"*. Code: [000MAK/cafepolice-graybox-meta](https://github.com/000MAK/cafepolice-graybox-meta).

This card holds the fitted config and test metrics, not a trained weight file: the model is two interpretable GLMs (Poisson frequency + Gamma severity) whose `alpha` hyperparameters were tuned with a genetic algorithm.

Data: `freMTPL2` (CC0, openml) — frequency `freMTPL2freq` (ClaimNb/Exposure, 50,000 subsample) and severity `freMTPL2sev` (ClaimAmount, joined via `IDpol`, 26,444 claims). Falls back to labelled synthetic data if unavailable.

| | |
|---|---|
| GA best alphas | frequency 0.0034, severity 0.803 (log10 in [−6, 3]) |
| Split | 70% train / 15% validation / 15% test (seed 42) |
| Claim frequency (test, annualised) | 24.6% |
| Frequency Poisson deviance (test) | 0.3353 |
| Severity MAE / MSE (test) | 1,883 / 1.05×10⁸ |
| Mean pure premium (test) | 131.2 |

Reproduce: `python src/graybox_engine.py --test` (offline) or `--real` (freMTPL2).

Files: `metrics.json` (all numbers), `sonuc_tablosu.csv`, and charts `01_frekans_fit.png`, `02_siddet_fit.png`, `03_ga_yakinsama.png`.

Limitations: frequency side downsampled; no trend or dispersion covariates; GA is a heuristic. A method illustration, not a pricing recommendation.
