import numpy as np
import pandas as pd
from scipy.stats import norm, t


def parametric_var_cvar(
    returns: pd.Series,
    sigma: pd.Series,
    alpha: float = 0.99,
    dist: str = "normal",
    nu: float | None = None
):
    """
    Parametric VaR & CVaR with distribution awareness.

    Parameters
    ----------
    returns : pd.Series
        Actual returns (decimal)
    sigma : pd.Series
        Forecasted volatility (decimal)
    alpha : float
        Confidence level (e.g. 0.99)
    dist : str
        'normal' or 't'
    nu : float
        Degrees of freedom for Student-t (required if dist='t')

    Returns
    -------
    aligned_returns : pd.Series
    var : pd.Series
        Positive VaR loss magnitude
    cvar : pd.Series
        Positive CVaR loss magnitude
    """

    if dist == "normal":
        z = norm.ppf(alpha)
        var = z * sigma
        cvar = (norm.pdf(z) / (1 - alpha)) * sigma

    elif dist == "t":
        if nu is None:
            raise ValueError("nu must be provided for Student-t VaR")

        z = t.ppf(alpha, nu)
        var = z * sigma
        cvar = (
            (t.pdf(z, nu) / (1 - alpha)) *
            ((nu + z**2) / (nu - 1)) *
            sigma
        )

    else:
        raise ValueError("dist must be 'normal' or 't'")

    aligned_returns = returns.loc[sigma.index]

    return aligned_returns, var, cvar


def var_breaches(
    returns: pd.Series,
    var: pd.Series
):
    """
    VaR breach occurs when realized loss exceeds VaR.
    """
    breaches = returns < -var
    breach_rate = breaches.mean()
    return breaches, breach_rate


def calibrate_volatility(
    returns: pd.Series,
    sigma: pd.Series
):
    """
    Variance-matching calibration for volatility forecasts.
    Fixes over-conservative ML VaR.
    """
    scale = returns.std() / sigma.mean()
    return sigma * scale
