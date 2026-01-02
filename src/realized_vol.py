import pandas as pd
import numpy as np


def realized_volatility(
    returns: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Compute rolling realized volatility.

    sigma_t = sqrt(sum_{i=1}^{window} r_{t-i}^2)
    """
    rv = returns["log_return"].rolling(window).apply(
        lambda x: np.sqrt((x ** 2).sum()),
        raw=True
    )
    rv = rv.dropna()
    return rv.to_frame(name="realized_vol")
