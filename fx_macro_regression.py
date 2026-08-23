python
"""
FX Macro Regression Toolkit
============================

A set of regression diagnostics and modelling tools applied to daily FX
log-returns against macroeconomic predictors (US 10Y yield, S&P 500, Gold,
Oil, VIX).

Covers four techniques commonly needed when modelling noisy financial
return data:

    1. Heteroskedasticity detection (Breusch-Pagan) + Weighted Least Squares
    2. Robust regression (Huber and Bisquare/Tukey biweight M-estimation)
    3. Lasso regression with cross-validated regularisation
    4. Local regression (LOWESS) with span sensitivity analysis

Data
----
Daily FX log-returns for eight G10 currency pairs (2015-01-05 to
2025-03-31) alongside five macro predictors: US10Y, SP500, GOLD, OIL, VIX.
An individual FX pair is deterministically assigned from a numeric seed,
so results are reproducible for a given seed.

Usage
-----
    python fx_macro_regression.py
"""

import random
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FX_POOL = ["AUDUSD", "CADUSD", "CHFUSD", "EURUSD",
           "GBPUSD", "JPYUSD", "NOKUSD", "NZDUSD"]
PRED_COLS = ["US10Y", "SP500", "GOLD", "OIL", "VIX"]


def assign_fx_pair(seed, data_path="FE610_M2_fx_data.csv"):
    """Load the dataset and deterministically assign an FX pair from seed."""
    df_raw = pd.read_csv(data_path, index_col=0, parse_dates=True)
    random.seed(seed)
    fx_pair = random.choice(FX_POOL)
    df = df_raw.copy()
    y = df[fx_pair]
    X = df[PRED_COLS]
    return fx_pair, df, y, X


# ---------------------------------------------------------------------------
# Part 1: Heteroskedasticity detection + Weighted Least Squares
# ---------------------------------------------------------------------------
def detect_heteroskedasticity_and_fit_wls(y, x):
    """
    Detect heteroskedasticity with the Breusch-Pagan test and correct
    using Weighted Least Squares.

    Parameters
    ----------
    y : pd.Series — FX pair daily log-returns
    x : pd.Series — single predictor (e.g. US10Y daily log-returns)

    Returns
    -------
    dict with keys: 'bp_pvalue', 'wls_intercept', 'wls_slope'
    """
    X_ols = sm.add_constant(x)
    ols_model = sm.OLS(y, X_ols).fit()
    resid = ols_model.resid
    fitted = ols_model.fittedvalues

    _, bp_pvalue, _, _ = het_breuschpagan(resid, ols_model.model.exog)

    abs_resid = np.abs(resid)
    X_aux = sm.add_constant(fitted)
    aux_model = sm.OLS(abs_resid, X_aux).fit()
    fitted_abs_resid = np.clip(aux_model.fittedvalues, 1e-6, None)
    weights = 1 / (fitted_abs_resid ** 2)

    wls_model = sm.WLS(y, X_ols, weights=weights).fit()

    return {
        "bp_pvalue": round(float(bp_pvalue), 6),
        "wls_intercept": round(float(wls_model.params.iloc[0]), 6),
        "wls_slope": round(float(wls_model.params.iloc[1]), 6),
    }


# ---------------------------------------------------------------------------
# Part 2: Robust regression (Huber and Bisquare)
# ---------------------------------------------------------------------------
def fit_robust_regression(y, x):
    """
    Fit Huber and Bisquare (Tukey biweight) robust regression models.

    Parameters
    ----------
    y : pd.Series — FX pair daily log-returns
    x : pd.Series — single predictor (e.g. US10Y daily log-returns)

    Returns
    -------
    dict with keys: 'huber_intercept', 'huber_slope', 'bisquare_slope'
    """
    X_design = sm.add_constant(x)

    huber_model = sm.RLM(y, X_design, M=sm.robust.norms.HuberT()).fit()
    bisquare_model = sm.RLM(y, X_design, M=sm.robust.norms.TukeyBiweight()).fit()

    return {
        "huber_intercept": round(float(huber_model.params.iloc[0]), 6),
        "huber_slope": round(float(huber_model.params.iloc[1]), 6),
        "bisquare_slope": round(float(bisquare_model.params.iloc[1]), 6),
    }


# ---------------------------------------------------------------------------
# Part 3: Lasso regression with cross-validation
# ---------------------------------------------------------------------------
def fit_lasso_cv(y, X, seed):
    """
    Fit Lasso regression with 10-fold cross-validation on standardised
    predictors.

    Parameters
    ----------
    y    : pd.Series    — FX pair daily log-returns
    X    : pd.DataFrame — macro predictors
    seed : int          — random seed for the train/test split and CV

    Returns
    -------
    dict with keys: 'optimal_alpha', 'n_zero_coefs', 'test_r2'
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso_model = LassoCV(cv=10, random_state=seed).fit(X_train_scaled, y_train)

    return {
        "optimal_alpha": round(float(lasso_model.alpha_), 8),
        "n_zero_coefs": int(np.sum(lasso_model.coef_ == 0)),
        "test_r2": round(float(lasso_model.score(X_test_scaled, y_test)), 6),
    }


# ---------------------------------------------------------------------------
# Part 4: LOWESS with multiple spans
# ---------------------------------------------------------------------------
def fit_loess_with_spans(y, x):
    """
    Fit LOWESS at four candidate spans against a single predictor.
    Returns in-sample RSS at each span and predicted y at x=0 for the
    smallest and largest spans.

    Parameters
    ----------
    y : pd.Series — FX pair daily log-returns
    x : pd.Series — single predictor (e.g. VIX daily log-returns)

    Returns
    -------
    dict with keys: 'ssr_020', 'ssr_035', 'ssr_050', 'ssr_075',
                    'pred_vix0_020', 'pred_vix0_075'
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    spans = [0.20, 0.35, 0.50, 0.75]
    ssr_map, pred_map = {}, {}

    for span in spans:
        ft = lowess(y_arr, x_arr, frac=span, it=4)
        fitted_vals = np.interp(x_arr, ft[:, 0], ft[:, 1])
        rss = np.sum((y_arr - fitted_vals) ** 2)
        ssr_map[span] = round(float(rss), 6)
        pred_map[span] = round(float(np.interp(0.0, ft[:, 0], ft[:, 1])), 6)

    return {
        "ssr_020": ssr_map[0.20],
        "ssr_035": ssr_map[0.35],
        "ssr_050": ssr_map[0.50],
        "ssr_075": ssr_map[0.75],
        "pred_vix0_020": pred_map[0.20],
        "pred_vix0_075": pred_map[0.75],
    }


if __name__ == "__main__":
    STUDENT_ID = 1037340  # replace with your own numeric seed

    fx_pair, df, y, X = assign_fx_pair(STUDENT_ID)
    print(f"Assigned FX pair: {fx_pair}\n")

    print("--- Part 1: Heteroskedasticity + WLS ---")
    print(detect_heteroskedasticity_and_fit_wls(y, X["US10Y"]))

    print("\n--- Part 2: Robust regression ---")
    print(fit_robust_regression(y, X["US10Y"]))

    print("\n--- Part 3: Lasso with 10-fold CV ---")
    print(fit_lasso_cv(y, X, STUDENT_ID))

    print("\n--- Part 4: LOWESS span sensitivity ---")
    print(fit_loess_with_spans(y, X["VIX"]))