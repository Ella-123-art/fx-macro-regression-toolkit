# FX Macro Regression Toolkit

A small toolkit of regression diagnostics and modelling techniques applied
to daily FX log-returns against macroeconomic predictors — built as part
of coursework on regression under heteroskedasticity, outliers, and
non-linearity.

## Overview

Daily FX returns are noisy and generally close to unpredictable from
macro factors alone, but the *process* of testing that carefully is where
the value lies. This project walks through four complementary techniques
on a G10 FX pair regressed against five macro drivers:

| Predictor | Role |
|---|---|
| `US10Y` | US 10-Year Treasury yield log-return — interest rate differential, a key FX driver |
| `SP500` | S&P 500 index log-return — global risk appetite |
| `GOLD` | Gold spot price log-return — safe-haven demand |
| `OIL` | WTI crude oil log-return — commodity / terms of trade |
| `VIX` | CBOE VIX log-return — market fear / risk-off flows |

The FX pair analysed is deterministically selected from a numeric seed,
so results are reproducible for a given seed value.

## Methods covered

1. **Heteroskedasticity detection + WLS** — Breusch-Pagan test for
   non-constant error variance, followed by Weighted Least Squares
   correction (weights estimated from a regression of absolute OLS
   residuals on fitted values).

2. **Robust regression (Huber & Bisquare)** — M-estimation via Iteratively
   Reweighted Least Squares, comparing Huber's soft downweighting of
   large residuals against Bisquare's hard zeroing of extreme outliers.

3. **Lasso with 10-fold cross-validation** — L1-penalised regression on
   standardised predictors, examining which macro factors survive
   variable selection and how the penalised model generalises to a
   held-out test set.

4. **LOWESS with span sensitivity** — Local regression at four candidate
   smoothing spans (0.20 / 0.35 / 0.50 / 0.75), comparing in-sample fit
   and predicted FX return at a neutral market state (VIX log-return = 0).

## Data

Daily log-returns spanning 2015-01-05 to 2025-03-31 — a period covering
two full interest rate cycles, the COVID-19 shock (March 2020), the 2022
USD surge, and the 2023 banking stress. `FE610_M2_fx_data.csv` contains
the FX pairs and macro predictors used here.

## Usage

\`\`\`bash
pip install -r requirements.txt
python fx_macro_regression.py
\`\`\`

Each of the four functions can also be imported and used independently:

\`\`\`python
from fx_macro_regression import (
    assign_fx_pair,
    detect_heteroskedasticity_and_fit_wls,
    fit_robust_regression,
    fit_lasso_cv,
    fit_loess_with_spans,
)

fx_pair, df, y, X = assign_fx_pair(seed=1037340)
result = detect_heteroskedasticity_and_fit_wls(y, X["US10Y"])
\`\`\`

## Key takeaway

Across all four methods, the macro predictors show only a weak,
statistically fragile relationship with daily FX returns — heteroskedasticity
correction flips significance conclusions, Lasso's out-of-sample R² is
negative, and smoothing span barely moves the local fit. This is a fairly
typical (and informative) result for daily-frequency FX return modelling:
the exercise is as much about correctly diagnosing *how little* signal is
there as it is about finding a "best" model.

## Requirements

See `requirements.txt`. Built with `pandas`, `numpy`, `statsmodels`, and
`scikit-learn`.