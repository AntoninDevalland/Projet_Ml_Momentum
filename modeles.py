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
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import itertools

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

def msar1_fit_predict(
    y_win: pd.Series,
    horizon: int = 1,
    k_regimes: int = 2,
    switching_variance: bool = False,
    trend: str = "n",
    maxiter: int = 200,
    disp: bool = False,
    debug: bool = False,
):
    if horizon != 1:
        raise ValueError("Cette fonction est faite pour horizon=1 (t+1).")

    y = y_win.astype(float).to_numpy()
    if (len(y) < 120) or (not np.isfinite(y).all()):
        return np.nan

    # winsorize léger
    lo, hi = np.quantile(y, [0.001, 0.999])
    y = np.clip(y, lo, hi)

    # centrage (stabilité)
    y_mean = y.mean()
    y0 = y - y_mean

    mod = MarkovRegression(
        endog=y0,
        k_regimes=k_regimes,
        trend=trend,
        order=1,
        switching_variance=switching_variance,
    )

    try:
        res = mod.fit(disp=disp, maxiter=maxiter, em_iter=10, search_reps=20, search_iter=5)
        yhat_last = float(np.asarray(res.predict())[-1])  # stable
        return yhat_last + y_mean
    except Exception as e:
        if debug:
            print("MSAR t+1 failed:", repr(e))
        return np.nan

    
def rebalance_dates_mask(
    signal: pd.Series,
    rebalance_freq: str = "M",  # "M", "MS", "W-FRI", "W-MON", "D"
) -> pd.Series:
    """
    Renvoie une série booléenne indexée comme `signal` :
    True = date de rebalancement, False sinon.

    - "D" : tous les jours
    - "M" : dernier jour de trading du mois
    - "MS": premier jour de trading du mois (approx via resample, voir note)
    - "W-FRI", "W-MON" : rebal hebdo sur le jour indiqué
    """
    signal = signal.dropna()
    idx = signal.index

    if rebalance_freq == "D":
        return pd.Series(True, index=idx, name="rebalance")

    dummy = pd.Series(1.0, index=idx)

    # Dates candidates générées par resample (puis intersect avec idx)
    rb_dates = dummy.resample(rebalance_freq).last().dropna().index
    rb_dates = rb_dates.intersection(idx)

    rb = pd.Series(False, index=idx, name="rebalance")
    rb.loc[rb_dates] = True
    return rb

def mom_vol_target_weights(
    log_var_mom_fcst: pd.Series,      # log(sigma^2) prévu
    rb: pd.Series,                    # série booléenne True = date de rebalancement
    target_vol_annual: float = 0.10,  # cible vol annualisée
    w_max: float = 2.0,               # cap levier
    no_trade_band: float | None = 0.10,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Stratégie baseline :
    Portefeuille = w_mom * MOM + (1 - w_mom) * DEF

    Le poids w_mom est déterminé uniquement par la volatilité prévue du facteur MOM.
    Cette version prend directement en entrée le masque de rebalancement rb (True/False).

    Timing (à gérer hors fonction si besoin):
    - si log_var_mom_fcst[t] correspond à la prévision pour t+1 mais est datée à t,
      alors appliquer un shift(1) sur les poids au moment du backtest.
    """

    # Alignement et nettoyage
    df = pd.concat({"logvar": log_var_mom_fcst, "rb": rb}, axis=1).dropna()
    idx = df.index
    rb_aligned = df["rb"].astype(bool)

    # Vol prévue
    sigma_hat = np.sqrt(np.exp(df["logvar"].astype(float)))

    # Cible daily
    sigma_star_daily = target_vol_annual / np.sqrt(trading_days)

    # Poids MOM aux dates de rebalance
    w_mom_rb = pd.Series(np.nan, index=idx)
    w_mom_rb.loc[rb_aligned] = sigma_star_daily / sigma_hat.loc[rb_aligned]

    # Bornes
    w_mom_rb = w_mom_rb.clip(lower=0.0, upper=w_max)

    # No-trade band (appliqué uniquement aux dates rb=True)
    if no_trade_band is not None:
        last = np.nan
        for t in idx:
            if not rb_aligned.loc[t]:
                continue
            cur = w_mom_rb.loc[t]
            if np.isnan(cur):
                continue
            if np.isnan(last):
                last = cur
                w_mom_rb.loc[t] = last
            else:
                if abs(cur - last) < no_trade_band:
                    w_mom_rb.loc[t] = last
                else:
                    last = cur
                    w_mom_rb.loc[t] = last

    # Poids constants entre rebalances
    w_mom = w_mom_rb.ffill().fillna(0.0)
    w_def = 1.0 - w_mom

    return pd.DataFrame({"w_mom": w_mom, "w_def": w_def}, index=idx)

def performance_metrics(
    returns: pd.Series,
    rf: pd.Series | None = None,
    trading_days: int = 252,
) -> pd.Series:
    """
    Calcule des métriques de performance classiques à partir d'une série de rendements.

    Paramètres
    ----------
    returns : pd.Series
        Rendements du portefeuille (daily).
    rf : pd.Series | None, optional
        Rendements sans risque (daily). Si None, rf = 0.
    trading_days : int
        Nombre de jours de trading par an (252 par défaut).

    Retour
    ------
    pd.Series
        Série contenant les métriques de performance.
    """

    returns = returns.dropna()

    if rf is None:
        rf = pd.Series(0.0, index=returns.index)
    else:
        rf = rf.reindex(returns.index).fillna(0.0)

    excess_ret = returns - rf

    # --- Moyennes et volatilités ---
    mean_daily = returns.mean()
    vol_daily = returns.std()

    mean_annual = mean_daily * trading_days
    vol_annual = vol_daily * np.sqrt(trading_days)

    # --- Sharpe ratio ---
    sharpe = np.nan
    if vol_annual > 0:
        sharpe = (excess_ret.mean() * trading_days) / vol_annual

    # --- Drawdown ---
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = cum_ret / running_max - 1
    max_drawdown = drawdown.min()

    # --- Autres métriques utiles ---
    skew = returns.skew()
    kurt = returns.kurtosis()

    hit_ratio = (returns > 0).mean()

    metrics = pd.Series(
        {
            "Mean annual return": mean_annual,
            "Annual volatility": vol_annual,
            "Sharpe ratio": sharpe,
            "Max drawdown": max_drawdown,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Hit ratio": hit_ratio,
        }
    )

    return metrics


def sklearn_walk_forward(model, X, y, min_train_size, rolling_window=False, gap=0, refit_step=10):
    """
    Effectue un Backtest Walk-Forward (fenêtre glissante ou grandissante) avec Scikit-Learn.
    Permet un ré-entraînement périodique pour accélérer le calcul.

    Parameters
    ----------
    model : sklearn estimator (ou Pipeline)
        Le modèle à entraîner (ex: RandomForest, ElasticNet, LightGBM...).
    X : pd.DataFrame
        Les features (variables explicatives).
    y : pd.Series
        La target (variable cible). Doit être alignée temporellement avec X.
    min_train_size : int
        Taille de l'historique initial nécessaire avant de faire la 1ère prédiction.
    rolling_window : bool, default=False
        - False (Expanding) : On garde tout l'historique depuis le début (fenêtre grandissante).
        - True (Rolling) : On garde une fenêtre fixe de taille `min_train_size` (fenêtre glissante).
    gap : int, default=0
        Nombre de jours à exclure entre la fin du train et le test
    refit_step : int, default=10
        Fréquence de ré-entraînement du modèle (en jours).

    Returns
    -------
    pd.DataFrame
        DataFrame indexé par la date contenant :
        - 'Actual': La vraie valeur réalisée.
        - 'Pred': La valeur prédite par le modèle.
    """

    n_samples = len(X)
    n_splits = n_samples - min_train_size
    max_train = min_train_size if rolling_window else None

    # On garde le test_size=1 pour avoir une prédiction quotidienne
    # Le gap assure la sécurité par rapport à l'horizon de prévision
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1, gap=gap, max_train_size=max_train)

    predictions = []
    actuals = []
    dates = []

    # On itère sur tous les jours
    for i, (train_idx, test_idx) in tqdm(enumerate(tscv.split(X)), total=n_splits):

        # Maj du modèle (Uniquement si on est sur un pas de refit)
        if i % refit_step == 0:
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            model.fit(X_train, y_train)

        # Prédiction
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        pred = model.predict(X_test)
        predictions.append(pred[0])
        actuals.append(y_test.values[0])
        dates.append(y_test.index[0])

    return pd.DataFrame({"Actual": actuals, "Pred": predictions}, index=dates)


def engineer_features(df_raw, price_cols=None):
    """
    Génère automatiquement les features techniques et macro.
    - price_cols : liste des colonnes à traiter comme des prix
    - Les autres colonnes sont traitées comme des niveaux (Taux, VIX...)
    """
    df = df_raw.copy()
    if price_cols is None: price_cols = []

    # Séparation automatique si non fournie
    rate_cols = [c for c in df.columns if c not in price_cols]

    X_out = pd.DataFrame(index=df.index)

    # A. Traitement des PRIX (Rendements, Tendance)
    for col in price_cols:
        if col in df.columns:
            X_out[f'{col}_ret1d'] = df[col].pct_change(fill_method=None)
            X_out[f'{col}_vol21'] = df[col].pct_change(fill_method=None).rolling(21).std()
            X_out[f'{col}_vol5'] = df[col].pct_change(fill_method=None).rolling(5).std()
            X_out[f'{col}_vol1'] = df[col].pct_change(fill_method=None).rolling(2).std()
            X_out[f'{col}_trend'] = (df[col] / df[col].rolling(200).mean()) - 1 # Distance MM200

    # B. Traitement des TAUX (Variations, Cycle)
    for col in rate_cols:
        if col in df.columns:
            X_out[f'{col}_chg1d'] = df[col].diff(1)
            X_out[f'{col}_gap200'] = df[col] - df[col].rolling(200).mean() # Écart au cycle long
            X_out[f'{col}_vol21'] = df[col].pct_change(fill_method=None).rolling(21).std()
            X_out[f'{col}_vol5'] = df[col].pct_change(fill_method=None).rolling(5).std()
            X_out[f'{col}_vol1'] = df[col].pct_change(fill_method=None).rolling(2).std()
            X_out[f'{col}_level'] = df[col]

    return X_out.replace([np.inf, -np.inf], np.nan).dropna()



def add_targeted_lags(df, suffixes, extra_cols=None, lags=[1, 2, 3, 5]):
    """
    Ajoute des lags (t-1, t-2...) sur les colonnes qui :
    1. Finissent par un des 'suffixes' donnés (ex: _ret1d)
    2. Sont listées dans 'extra_cols' (ex: RV 1j)
    """
    df_out = df.copy()
    if extra_cols is None: extra_cols = []

    # Identifier les colonnes à lagger
    cols_to_lag = []
    for col in df.columns:
        # Condition 1 : Suffixe
        if any(col.endswith(s) for s in suffixes):
            cols_to_lag.append(col)
        # Condition 2 : Colonne explicite (si présente)
        elif col in extra_cols:
            cols_to_lag.append(col)

    # Création des lags
    print(f"Lagging de {len(cols_to_lag)} variables sur {df.shape[1]}")

    for col in cols_to_lag:
        for lag in lags:
            new_col_name = f"{col}_lag{lag}"
            df_out[new_col_name] = df[col].shift(lag)

    return df_out.dropna()