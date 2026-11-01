# Café Poliçe: graybox GLM + metaheuristic pure-premium pricing

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

A frequency–severity (Poisson + Gamma) "graybox" pricing model whose GLM regularisation is tuned with a genetic algorithm instead of a grid search, evaluated on a held-out test slice. Code and the Turkish article *"Rakamların Diliyle Güven"* (Café Poliçe).

Model card on Hugging Face: [makactuary/cafepolice-graybox-meta](https://huggingface.co/makactuary/cafepolice-graybox-meta)

## What this is

- **White-box**: `PoissonRegressor` (frequency, log-link) × `GammaRegressor` (severity, log-link) = pure premium. Every coefficient is readable — the actuarial standard.
- **Black-box search kept in check**: a genetic algorithm tunes the `alpha` (L2 regularisation) of both GLMs on the validation set, minimising total Poisson + Gamma deviance. Continuous, adaptive search instead of a grid.
- **Honest evaluation**: 70% train / 15% validation / 15% test, fixed seed, metrics on the untouched test slice (Poisson deviance, MSE, MAE).

## Data

The engine tries `freMTPL2` (open, CC0) for the frequency side at runtime. Because freMTPL2's severity is a separate file that cannot be joined at policy level, the end-to-end demo runs on **clearly-labelled synthetic data** (seed 42, 20,000 policies). Nothing is presented as real; the data source is printed in every run.

## Method

1. **Frequency**: `PoissonRegressor(alpha, log-link)` on claim counts with exposure weights.
2. **Severity**: `GammaRegressor(alpha, log-link)` on positive claim amounts.
3. **Pure premium**: expected frequency × expected severity.
4. **Metaheuristic**: GA over `[log10(alpha_freq), log10(alpha_sev)]` ∈ [−6, 3], tournament selection, blend crossover, Gaussian mutation, 12 generations × 20 individuals.
5. **Test**: Poisson deviance, MSE and MAE on the held-out 15%.

## Results (synthetic demo, seed 42)

| | |
|---|---|
| GA best alphas | frequency 8.06, severity 0.23 |
| Claim frequency (test) | 4.51% |
| Frequency Poisson deviance (test) | 0.2297 |
| Severity MAE / MSE (test) | 1,235 / 2.65×10⁶ |
| Pure premium MAE / MSE (test) | 168.7 / 4.03×10⁵ |
| Mean pure premium (test) | 76.9 |

## Run

```bash
pip install -r requirements.txt
python src/graybox_engine.py --test     # offline, deterministic self-check
python src/graybox_engine.py            # full run + JSON summary
python src/graybox_engine.py --real     # try freMTPL2 frequency (needs network)
```

## Limitations

Synthetic demo data (labelled); a single frequency–severity GLM pair without trend, dispersion or exposure-varying covariates beyond the offset; GA is a heuristic, not a proof of global optimality. A method illustration, not a pricing recommendation.

## Licence

Code: MIT.
