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
    **model_kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Rolling 1-step ahead forecast engine with optional regressors X.

    Additional keyword arguments (**model_kwargs) are forwarded
    to fit_predict_fn.
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

    for end in range(window - 1, len(y2) - 1):
        y_win = y2.iloc[end - window + 1 : end + 1]

        if X2 is None:
            yhat_next, diag = fit_predict_fn(y_win, **model_kwargs)
        else:
            X_win = X2.iloc[end - window + 1 : end + 1]
            yhat_next, diag = fit_predict_fn(y_win, X_win, **model_kwargs)

        forecast_date = y2.index[end + 1]
        fcast.loc[forecast_date] = float(yhat_next)

        diag = {} if diag is None else dict(diag)
        diag_rows.append({"date": forecast_date, **diag})

    diag_df = pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame()
    return fcast, diag_df

def expanding_window_forecast(
    y: pd.Series,
    fit_predict_fn,
    min_window: int,
    X: pd.DataFrame | None = None,
    **model_kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Expanding 1-step ahead forecast engine with optional regressors X.

    Uses an expanding window: start with `min_window` observations, then
    keep adding one observation each step.

    Additional keyword arguments (**model_kwargs) are forwarded
    to fit_predict_fn.
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

    # end is the last in-sample index position; we forecast end+1
    for end in range(min_window - 1, len(y2) - 1):
        y_win = y2.iloc[: end + 1]

        if X2 is None:
            yhat_next, diag = fit_predict_fn(y_win, **model_kwargs)
        else:
            X_win = X2.iloc[: end + 1]
            yhat_next, diag = fit_predict_fn(y_win, X_win, **model_kwargs)

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


def garch_fit_predict(
    r_win: pd.Series,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    mean: str = "constant",
    dist: str = "normal",
) -> tuple[float, dict]:
    """
    Rolling GARCH(p,q) on returns.

    Fits a univariate GARCH(p,q) on r_win (returns), then produces an
    h-step-ahead *volatility* forecast for the horizon:
        sigma_{t,t+h} = sqrt(sum_{i=1..h} h_{t+i})

    Inputs
    ------
    r_win : pd.Series
        Returns in decimal units (e.g., 0.001 = 0.1%).
    horizon : int
        Forecast horizon in trading days (1=daily, 5=weekly-ish, 20=monthly-ish).
    p, q : int
        GARCH orders.
    mean : {"constant","zero"}
        Mean specification.
    dist : {"normal","t","skewt"}
        Innovation distribution.

    Returns
    -------
    sigma_hat : float
        Forecast volatility over `horizon` days in decimal units.
        (Same units as r_win: e.g., 0.02 = 2% over the horizon.)
    diag : dict
        Basic in-sample diagnostics and fitted params.
    """
    # Lazy import so your notebook only needs arch when you call this.

    r_win = r_win.dropna()
    if len(r_win) < max(50, 10 * (p + q + 1)):
        raise ValueError("Window too small for stable GARCH estimation.")

    # arch is more numerically stable with percent returns
    r_pct = 100.0 * r_win.astype(float)

    mean_spec = "Constant" if mean.lower() == "constant" else "Zero"
    dist_spec = {"normal": "normal", "t": "t", "skewt": "skewt"}[dist.lower()]

    am = arch_model(
        r_pct,
        mean=mean_spec,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist_spec,
        rescale=False,  # we already scaled to %
    )
    res = am.fit(disp="off")

    # Forecast: variance path for 1..horizon (in (%ret)^2)
    f = res.forecast(horizon=horizon, reindex=False)
    vars_h = f.variance.iloc[-1].to_numpy()  # length=horizon
    var_H = float(np.sum(vars_h))
    sigma_H_pct = float(np.sqrt(var_H))      # % over horizon

    # Back to decimal units
    sigma_hat = sigma_H_pct / 100.0

    # In-sample diagnostics
    # Note: res.conditional_volatility is in % (same scaling as r_pct)
    cond_vol_pct = res.conditional_volatility
    resid = res.std_resid  # standardized residuals
    diag = {
        "model": f"GARCH({p},{q})",
        "mean": mean_spec,
        "dist": dist_spec,
        "n_obs": int(res.nobs),
        "horizon": int(horizon),
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "last_cond_vol_1d": float(cond_vol_pct.iloc[-1] / 100.0),  # decimal, 1-day
        "resid_std_mean": float(np.nanmean(resid)),
        "resid_std_var": float(np.nanvar(resid)),
    }

    # Parameters + (robust) t-stats if available
    params = res.params
    diag.update({f"param_{k}": float(v) for k, v in params.items()})

    try:
        tstats = res.tvalues
        diag.update({f"t_{k}": float(v) for k, v in tstats.items()})
    except Exception:
        pass

    return sigma_hat, diag



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