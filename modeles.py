from __future__ import annotations
import pandas as pd 
import os 
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score, mean_squared_error
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def rolling_window_forecast(
    y: pd.Series,
    fit_predict_fn,
    window: int,
    X: pd.DataFrame | None = None,
    *,
    horizon: int = 1,
    step: int = 1,
    **model_kwargs,
) -> pd.Series:
    """
    Rolling forecast engine (walk-forward).

    - horizon : horizon de prévision (nb de jours futurs)
    - step    : fréquence de recalcul (1=tous les jours, H=non-overlapping)

    La prévision produite à la date t est TOUJOURS indexée à t + horizon.
    """

    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be positive integers.")

    # Align / clean
    if X is None:
        y2 = y.dropna().copy()
        X2 = None
    else:
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        y2 = df["y"].copy()
        X2 = df.drop(columns=["y"]).copy()

    start_end = window - 1
    last_end = len(y2) - horizon - 1
    if last_end < start_end:
        raise ValueError("Not enough data for given window and horizon.")

    dates = []
    preds = []

    for end in range(start_end, last_end + 1, step):
        y_win = y2.iloc[end - window + 1 : end + 1]

        if X2 is None:
            out = fit_predict_fn(y_win, horizon=horizon, **model_kwargs)
        else:
            X_win = X2.iloc[end - window + 1 : end + 1]
            out = fit_predict_fn(y_win, X_win, horizon=horizon, **model_kwargs)

        # accepte yhat seul ou (yhat, diag)
        yhat = out[0] if isinstance(out, tuple) else out

        forecast_date = y2.index[end + horizon]  #  fin d'horizon

        dates.append(forecast_date)
        preds.append(float(yhat))

    fcast = pd.Series(preds, index=pd.Index(dates, name="date"), name="forecast")
    return fcast


def expanding_window_forecast(
    y: pd.Series,
    fit_predict_fn,
    min_window: int,
    X: pd.DataFrame | None = None,
    *,
    horizon: int = 1,
    step: int = 1,
    **model_kwargs,
) -> pd.Series:
    """
    Expanding forecast engine (walk-forward).

    - horizon : horizon de prévision (nb de jours futurs)
    - step    : fréquence de recalcul (1=tous les jours, H=non-overlapping)

    La prévision produite à la date t est TOUJOURS indexée à t + horizon.
    """

    if horizon < 1 or step < 1:
        raise ValueError("horizon and step must be positive integers.")

    # Align / clean
    if X is None:
        y2 = y.dropna().copy()
        X2 = None
    else:
        df = pd.concat([y.rename("y"), X], axis=1).dropna()
        y2 = df["y"].copy()
        X2 = df.drop(columns=["y"]).copy()

    start_end = min_window - 1
    last_end = len(y2) - horizon - 1
    if last_end < start_end:
        raise ValueError("Not enough data for given min_window and horizon.")

    dates = []
    preds = []

    for end in range(start_end, last_end + 1, step):
        y_win = y2.iloc[: end + 1]  # expanding window

        if X2 is None:
            out = fit_predict_fn(y_win, horizon=horizon, **model_kwargs)
        else:
            X_win = X2.iloc[: end + 1]
            out = fit_predict_fn(y_win, X_win, horizon=horizon, **model_kwargs)

        yhat = out[0] if isinstance(out, tuple) else out

        forecast_date = y2.index[end + horizon]  # TOUJOURS fin d'horizon

        dates.append(forecast_date)
        preds.append(float(yhat))

    fcast = pd.Series(preds, index=pd.Index(dates, name="date"), name="forecast")
    return fcast

def har_fit_predict(
    y_win: pd.Series,
    X_win: pd.DataFrame,
    horizon: int = 1) -> tuple[float, dict]:
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

    return yhat_next


def ar1_fit_predict(y_win: pd.Series, horizon: int = 1) -> tuple[float, dict]:
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

    return float(yhat_next)


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


def garch_fit_predict(
    r_win: pd.Series,
    horizon: int,
    p: int = 1,
    o: int =0,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    vol:str = "GARCH"
) -> float:
    """
    Fit GARCH(p,q) sur des rendements journaliers (en décimal) et prédit la
    realized variance (proxy) sur `horizon` jours:

        RV_hat_{t->t+h} = sum_{i=1..h} E_t[r_{t+i}^2] ~= sum_{i=1..h} sigma^2_{t+i|t}

    Paramètres
    ----------
    r_win : pd.Series
        Rendements en décimal (0.001 = 0.1%).
    horizon : int
        Horizon de prévision (nb de jours de trading), ex: 5.
    p, q : int
        Ordres GARCH.
    mean : {"constant","zero"}
    dist : {"normal","t","skewt"}

    Retour
    ------
    RV_hat : float
        Prévision de variance cumulée sur `horizon` jours (en unités décimales^2).
        Ex: 0.0004 ~ (2%)^2 sur l'horizon si tu prends sqrt ensuite.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    r_win = r_win.dropna().astype(float)
    if len(r_win) < max(50, 10 * (p + q + 1)):
        raise ValueError("Window too small for stable GARCH estimation.")

    # arch est plus stable en pourcents
    r_pct = 100.0 * r_win

    mean_spec = "Constant" if mean.lower() == "constant" else "Zero"
    dist_key = dist.lower()
    if dist_key not in {"normal", "t", "skewt"}:
        raise ValueError("dist must be one of {'normal','t','skewt'}.")
    dist_spec = {"normal": "normal", "t": "t", "skewt": "skewt"}[dist_key]

    am = arch_model(
        r_pct,
        mean=mean_spec,
        vol=vol,
        p=p,
        o = o,
        q=q,
        dist=dist_spec,
        rescale=False,  # déjà en %
    )
    res = am.fit(disp="off")

    if vol.upper() == "EGARCH" and horizon > 1:
        f = res.forecast(horizon=horizon, method="simulation", simulations=2000, reindex=False)
    else:
        f = res.forecast(horizon=horizon, reindex=False)
        
    vars_h = f.variance.iloc[-1].to_numpy()  # longueur = horizon
    var_H_pct2 = float(np.sum(vars_h))       # variance cumulée sur H jours en (%^2)

    # Retour en unités décimales^2 (car r_pct = 100 * r_dec)
    RV_hat = var_H_pct2 / (100.0 ** 2)

    return RV_hat

def msar1_fit_predict(y_win: pd.Series, k_regimes: int = 2, switching_variance: bool = True,
                     trend: str = "c", maxiter: int = 200, disp: bool = False):
    """
    Fit MS-AR(1) on y_win and return 1-step-ahead forecast + diagnostics.
    """
    y_arr = y_win.astype(float).to_numpy()

    # AR(1) with Markov-switching intercept (trend='c'), and optionally switching variance
    mod = MarkovRegression(
        endog=y_arr,
        k_regimes=k_regimes,
        trend=trend,           # 'c' for intercept, 'n' for none
        order=1,
        switching_variance=switching_variance,
    )

    try:
        res = mod.fit(disp=disp, maxiter=maxiter)

        # 1-step ahead forecast (end of sample + 1)
        # predict() uses index in "observation number" units here
        yhat_next = float(res.predict(start=len(y_arr), end=len(y_arr))[0])

        # Regime probas at the last in-sample point
        # filtered_marginal_probabilities: (T x k_regimes)
        p_last = res.filtered_marginal_probabilities[-1]
        diag = {
            "converged": bool(getattr(res.mle_retvals, "converged", True)),
            "llf": float(res.llf),
            **{f"p_regime_{j}": float(p_last[j]) for j in range(k_regimes)}
        }
        return yhat_next, diag

    except Exception as e:
        # In rolling estimation, some windows can fail to converge; don't kill the whole backtest
        return np.nan, {"converged": False, "error": str(e)}