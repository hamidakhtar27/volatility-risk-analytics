from arch.univariate import arch_model
import pandas as pd


def garch_vol(returns):
    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        q=1,
        dist="normal"
    )
    res = model.fit(disp="off")
    return pd.Series(
        res.conditional_volatility / 100,
        index=returns.index,
        name="garch"
    )


def egarch_vol(returns):
    model = arch_model(
        returns * 100,
        vol="EGarch",
        p=1,
        q=1,
        dist="normal"
    )
    res = model.fit(disp="off")
    return pd.Series(
        res.conditional_volatility / 100,
        index=returns.index,
        name="egarch"
    )


def tgarch_vol(returns):
    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        o=1,
        q=1,
        dist="normal"
    )
    res = model.fit(disp="off")
    return pd.Series(
        res.conditional_volatility / 100,
        index=returns.index,
        name="tgarch"
    )


def garch_t_vol(returns):
    model = arch_model(
        returns * 100,
        vol="Garch",
        p=1,
        q=1,
        dist="t"
    )
    res = model.fit(disp="off")
    return pd.Series(
        res.conditional_volatility / 100,
        index=returns.index,
        name="garch_t"
    )
