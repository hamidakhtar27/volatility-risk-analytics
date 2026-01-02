import pandas as pd


def stress_test_period(returns, start, end):
    """
    Stress testing over a fixed crisis window
    """
    stress_returns = returns.loc[start:end]

    return {
        "start": start,
        "end": end,
        "days": len(stress_returns),
        "worst_day": stress_returns.min(),
        "cumulative_loss": stress_returns.sum(),
        "avg_daily_loss": stress_returns.mean()
    }


def run_standard_stress_tests(returns):
    """
    Standard industry stress scenarios
    """
    return {
        "GFC_2008": stress_test_period(
            returns, "2008-09-01", "2009-03-31"
        ),
        "COVID_2020": stress_test_period(
            returns, "2020-02-15", "2020-04-30"
        )
    }
