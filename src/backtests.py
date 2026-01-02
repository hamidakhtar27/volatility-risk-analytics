import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_pof_test(breaches, alpha):
    """
    Kupiec Proportion of Failures test
    """
    breaches = breaches.astype(int)
    T = len(breaches)
    x = breaches.sum()
    p = 1 - alpha

    if x == 0 or x == T:
        return {
            "LR_stat": np.nan,
            "p_value": np.nan,
            "breaches": x,
            "expected": T * p
        }

    LR = -2 * (
        (T - x) * np.log(1 - p) + x * np.log(p)
        - ((T - x) * np.log(1 - x / T) + x * np.log(x / T))
    )

    return {
        "LR_stat": LR,
        "p_value": 1 - chi2.cdf(LR, df=1),
        "breaches": x,
        "expected": T * p
    }


def christoffersen_test(breaches):
    """
    Christoffersen Independence Test (FIXED indexing)
    """
    breaches = breaches.astype(int)

    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(breaches)):
        prev = breaches.iloc[i - 1]
        curr = breaches.iloc[i]

        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0

    L0 = (
        (n00 + n10) * safe_log(1 - pi)
        + (n01 + n11) * safe_log(pi)
    )

    L1 = (
        n00 * safe_log(1 - pi0)
        + n01 * safe_log(pi0)
        + n10 * safe_log(1 - pi1)
        + n11 * safe_log(pi1)
    )

    LR = -2 * (L0 - L1)

    return {
        "LR_stat": LR,
        "p_value": 1 - chi2.cdf(LR, df=1)
    }


def rolling_basel_traffic_light(breaches, window=250):
    """
    Rolling Basel Traffic Light (correct regulatory usage)
    """
    results = []

    for i in range(window, len(breaches)):
        window_breaches = breaches.iloc[i - window:i].sum()

        if window_breaches <= 4:
            status = "GREEN"
        elif window_breaches <= 9:
            status = "YELLOW"
        else:
            status = "RED"

        results.append(status)

    return pd.Series(results, index=breaches.index[window:])
