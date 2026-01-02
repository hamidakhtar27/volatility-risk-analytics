import pandas as pd
import numpy as np


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price series.

    r_t = log(P_t / P_{t-1})
    """
    returns = np.log(price_df["price"] / price_df["price"].shift(1))
    returns = returns.dropna()
    return returns.to_frame(name="log_return")
