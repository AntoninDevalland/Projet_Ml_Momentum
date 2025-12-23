import pandas as pd 
import os 
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_squared_error


import pandas as pd

def rolling_window_forecast(
    y: pd.Series,
    fit_predict_fn,
    window: int,
    X: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Rolling 1-step ahead forecast engine (1-step) with optional regressors X.

    Convention:
    - forecast for date t is produced using info up to t-1, and stored at index t.

    Parameters
    ----------
    y : pd.Series
        Target series.
    fit_predict_fn : callable
        If X is None:  fit_predict_fn(y_win) -> (yhat_next, diag)
        If X provided: fit_predict_fn(y_win, X_win) -> (yhat_next, diag)
    window : int
        Window length (number of rows in y_win / X_win).
    X : pd.DataFrame | None
        Regressors aligned with y on the same index/frequency.

    Returns
    -------
    fcast : pd.Series
        Forecasts aligned on y.index (forecast for t stored at t).
    diag : pd.DataFrame
        Diagnostics aligned on forecasted date (same index as fcast non-NaN points).
    """
    # Align / clean
    if X is None:
        y2 = y.dropna().copy()
        X2 = None
    else:
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        y2 = df["y"].copy()
        X2 = df.drop(columns=["y"]).copy()

    fcast = pd.Series(index=y2.index, dtype=float)
    diag_rows = []

    # end = index of last in-window observation used to forecast end+1
    for end in range(window - 1, len(y2) - 1):
        y_win = y2.iloc[end - window + 1 : end + 1]

        if X2 is None:
            yhat_next, diag = fit_predict_fn(y_win)
        else:
            X_win = X2.iloc[end - window + 1 : end + 1]
            yhat_next, diag = fit_predict_fn(y_win, X_win)

        forecast_date = y2.index[end + 1]
        fcast.loc[forecast_date] = float(yhat_next)

        diag = {} if diag is None else dict(diag)
        diag_rows.append({"date": forecast_date, **diag})

    diag_df = pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame()
    return fcast, diag_df


def har_fit_predict(
    y_win: pd.Series,
    X_win: pd.DataFrame,
) -> tuple[float, dict]:
    """
    HAR: y_{t+1} = a + B' X_t + e_{t+1}
    y_win : Series of target (same freq as X_win)
    X_win : DataFrame of regressors (e.g. [D, W, M] or [D, W])
    Returns 1-step ahead forecast and in-sample diagnostics.
    """

    # build (X_t, y_{t+1}) pairs
    X = X_win.iloc[:-1].to_numpy()
    Y = y_win.iloc[1:].to_numpy()

    n = len(Y)
    k = X.shape[1] + 1  # +1 for intercept

    if n <= k:
        raise ValueError("Window too small for HAR estimation.")

    Xmat = np.column_stack([np.ones(n), X])

    # OLS
    beta, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)

    # In-sample fit + R2
    Yhat = Xmat @ beta
    resid = Y - Yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((Y - Y.mean())**2))
    r2_is = 1 - sse / sst if sst > 0 else np.nan

    # classic OLS t-stats (same logic as your AR(1))
    sigma2 = sse / (n - k)
    XtX_inv = np.linalg.pinv(Xmat.T @ Xmat)  # robust to collinearity
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstats = np.where(se > 0, beta / se, np.nan)

    # 1-step ahead forecast
    x_last = X_win.iloc[-1].to_numpy()
    yhat_next = float(beta[0] + beta[1:] @ x_last)

    # diagnostics dict
    diag = {
        "alpha": float(beta[0]),
        "R2_IS": float(r2_is),
        "n_pairs": int(n),
    }

    for j, name in enumerate(X_win.columns, start=1):
        diag[f"beta_{name}"] = float(beta[j])
        diag[f"t_{name}"] = float(tstats[j])

    return yhat_next, diag


def ar1_fit_predict(y_win: pd.Series) -> tuple[float, dict]:
    """
    AR(1): y_t = a + b y_{t-1} + e_t, estimated by OLS on the window.
    Returns forecast for next point (t+1) and diagnostics (R2_IS, beta, t_beta).
    """
    X = y_win.iloc[:-1].to_numpy()
    Y = y_win.iloc[1:].to_numpy()

    n = len(X)
    Xmat = np.column_stack([np.ones_like(X), X])  # const + lag

    beta, *_ = np.linalg.lstsq(Xmat, Y, rcond=None)
    a, b = beta

    # In-window fitted + R2 (classic SCE/SCT)
    Yhat = Xmat @ beta
    resid = Y - Yhat
    sse = float(np.sum(resid**2))                 # SCR
    sst = float(np.sum((Y - Y.mean())**2))        # SCT
    r2_is = 1 - sse/sst if sst > 0 else np.nan

    # t-stat for b (classic OLS)
    k = 2
    sigma2 = sse / (n - k)
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat)
    se_b = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    t_beta = float(b / se_b) if se_b > 0 else np.nan

    # 1-step ahead forecast
    yhat_next = a + b * y_win.iloc[-1]

    return float(yhat_next), {
        "alpha": float(a),
        "beta": float(b),
        "t_beta": t_beta,
        "R2_IS": float(r2_is),
        "n_pairs": int(n),
    }

def forecast_quality_oos(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """
    Standard OOS forecast metrics (finance).
    """
    df = pd.concat(
        [y_true.rename("y"), y_pred.rename("yhat")],
        axis=1
    ).dropna()

    err = df["y"] - df["yhat"]

    mspe = np.mean(err**2)
    rmse = np.sqrt(mspe)
    mae  = np.mean(np.abs(err))

    # Historical-mean benchmark (Welch–Goyal)
    y_bar = df["y"].mean()
    mspe_hm = np.mean((df["y"] - y_bar)**2)
    r2_oos = 1 - mspe / mspe_hm

    return pd.Series({
        "N_OOS": len(df),
        "MSPE": mspe,
        "RMSE": rmse,
        "MAE": mae,
        "R2_OOS": r2_oos
    })
