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

Data: clearly-labelled synthetic demo (seed 42, 20,000 policies); the engine can also pull `freMTPL2` (CC0) for the frequency side.

| | |
|---|---|
| GA best alphas | frequency 8.06, severity 0.23 (log10 in [−6, 3]) |
| Split | 70% train / 15% validation / 15% test (seed 42) |
| Claim frequency (test) | 4.51% |
| Frequency Poisson deviance (test) | 0.2297 |
| Severity MAE / MSE (test) | 1,235 / 2.65×10⁶ |
| Pure premium MAE / MSE (test) | 168.7 / 4.03×10⁵ |
| Mean pure premium (test) | 76.9 |

Reproduce: `python src/graybox_engine.py --test` (offline) or `--real` (tries freMTPL2 frequency).

Limitations: synthetic demo data (labelled); no trend or dispersion covariates; GA is a heuristic. A method illustration, not a pricing recommendation.
