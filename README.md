# Café Poliçe: graybox GLM + metaheuristic pure-premium pricing

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

A frequency–severity (Poisson + Gamma) "graybox" pricing model whose GLM regularisation is tuned with a genetic algorithm instead of a grid search, evaluated on a held-out test slice. Code and the Turkish article *"Rakamların Diliyle Güven"* (Café Poliçe).

Model card on Hugging Face: [makactuary/cafepolice-graybox-meta](https://huggingface.co/makactuary/cafepolice-graybox-meta)

## What this is

- **White-box**: `PoissonRegressor` (frequency, log-link) × `GammaRegressor` (severity, log-link) = pure premium. Every coefficient is readable — the actuarial standard.
- **Black-box search kept in check**: a genetic algorithm tunes the `alpha` (L2 regularisation) of both GLMs on the validation set, minimising total Poisson + Gamma deviance. Continuous, adaptive search instead of a grid.
- **Honest evaluation**: 70% train / 15% validation / 15% test, fixed seed, metrics on the untouched test slice (Poisson deviance, MSE, MAE).

## Data

`freMTPL2` (open, CC0) drives the model: **frequency** from `freMTPL2freq` (ClaimNb / Exposure, downsampled to 50,000 policies) and **severity** from `freMTPL2sev` (ClaimAmount, joined to policy features via `IDpol`, 26,444 claims). If the download fails the engine falls back to **clearly-labelled synthetic data** (seed 42). The data source is printed in every run.

## Method

1. **Frequency**: `PoissonRegressor(alpha, log-link)` on claim counts with exposure weights.
2. **Severity**: `GammaRegressor(alpha, log-link)` on positive claim amounts.
3. **Pure premium**: expected frequency × expected severity.
4. **Metaheuristic**: GA over `[log10(alpha_freq), log10(alpha_sev)]` ∈ [−6, 3], tournament selection, blend crossover, Gaussian mutation, 10 generations × 16 individuals.
5. **Test**: Poisson deviance, MSE and MAE on the held-out 15%.

## Results (freMTPL2, CC0)

| | |
|---|---|
| Data | freMTPL2: frequency 50,000 (subsample), severity 26,444 |
| GA best alphas | frequency 0.0034, severity 0.803 |
| Claim frequency (test, annualised) | 24.6% |
| Frequency Poisson deviance (test) | 0.3353 |
| Severity MAE / MSE (test) | 1,883 / 1.05×10⁸ |
| Mean pure premium (test) | 131.2 |

Charts: `outputs/charts/01_frekans_fit.png`, `02_siddet_fit.png`, `03_ga_yakinsama.png`. Full numbers in `outputs/metrics.json` and `outputs/sonuc_tablosu.csv`.

## Run

```bash
pip install -r requirements.txt
python src/graybox_engine.py --test     # offline, deterministic self-check
python src/graybox_engine.py            # real freMTPL2, fallback synthetic
python src/graybox_engine.py --real     # force freMTPL2
python article/build_article.py         # makale.md -> makale.docx
```

## Article

`article/makale.md` (Turkish, 3 pages), `article/makale.docx` and `article/makale.pdf`, signed *Mehmet Ali Kaya*.

## Limitations

A single frequency–severity GLM pair without trend or dispersion covariates beyond the offset; the frequency side is downsampled for runtime; GA is a heuristic, not a proof of global optimality. A method illustration, not a pricing recommendation.

## Licence

Code: MIT. Data: freMTPL2 (CC0, openml).
