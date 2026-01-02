"""
VOLATILITY & RISK ANALYTICS SYSTEM
=================================
End-to-end institutional-grade volatility and tail-risk framework.

✔ Loads market data
✔ Computes returns & realized volatility
✔ Fits GARCH / EGARCH / TGARCH / GARCH-t
✔ ML walk-forward volatility forecasting (RF + XGB)
✔ Parametric 99% VaR (Gaussian & Student-t)
✔ VaR breach visualization
✔ Kupiec & Christoffersen regulatory tests
✔ Basel traffic light (250-day rolling)
✔ Stress-test cumulative drawdowns (FIXED CORRECTLY)
✔ Saves ALL figures & reports automatically
"""

# =================================================
# IMPORTS
# =================================================
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from src.data_loader import download_price_data
from src.returns import compute_log_returns
from src.realized_vol import realized_volatility

from src.garch_models import (
    garch_vol,
    egarch_vol,
    tgarch_vol,
    garch_t_vol
)

from src.ml_models import (
    create_volatility_features,
    walk_forward_ml_forecast
)

from src.risk_metrics import (
    parametric_var_cvar,
    var_breaches,
    calibrate_volatility
)

from src.backtests import (
    kupiec_pof_test,
    christoffersen_test
)

from src.stress_tests import run_standard_stress_tests


# =================================================
# GLOBAL PLOT STYLE
# =================================================
mpl.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.frameon": False,
    "savefig.dpi": 150
})


# =================================================
# MAIN PIPELINE
# =================================================
def main():

    print("\nVOLATILITY & RISK ANALYTICS SYSTEM")
    print("=================================\n")

    # -------------------------------------------------
    # Output directories
    # -------------------------------------------------
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)

    # =================================================
    # 1. LOAD DATA
    # =================================================
    prices = download_price_data("SPY")

    # =================================================
    # 2. RETURNS
    # =================================================
    returns_df = compute_log_returns(prices)
    returns = returns_df["log_return"]

    # =================================================
    # 3. REALIZED VOLATILITY
    # =================================================
    rv = realized_volatility(returns_df)

    # =================================================
    # 4. GARCH FAMILY MODELS
    # =================================================
    print("Fitting GARCH-family models...\n")

    garch_sigma   = garch_vol(returns).loc[rv.index]
    egarch_sigma  = egarch_vol(returns).loc[rv.index]
    tgarch_sigma  = tgarch_vol(returns).loc[rv.index]
    garch_t_sigma = garch_t_vol(returns).loc[rv.index]

    # =================================================
    # 5. ML VOLATILITY (WALK-FORWARD)
    # =================================================
    ml_data = create_volatility_features(returns_df, rv)

    rf_preds, _, rf_rmse = walk_forward_ml_forecast(
        ml_data, model_type="rf"
    )

    xgb_preds, _, xgb_rmse = walk_forward_ml_forecast(
        ml_data, model_type="xgb"
    )

    print("ML MODEL PERFORMANCE")
    print("-------------------")
    print(f"Random Forest RMSE : {rf_rmse:.6f}")
    print(f"XGBoost RMSE       : {xgb_rmse:.6f}")

    # =================================================
    # 6. PARAMETRIC VAR (99%)
    # =================================================
    alpha = 0.99

    # ---- GARCH Gaussian ----
    _, var_garch, _ = parametric_var_cvar(
        returns, garch_sigma, alpha, dist="normal"
    )
    r_g, v_g = returns.align(var_garch, join="inner")
    _, garch_rate = var_breaches(r_g, v_g)

    # ---- GARCH Student-t ----
    _, var_garch_t, _ = parametric_var_cvar(
        returns, garch_t_sigma, alpha, dist="t", nu=8
    )
    r_gt, v_gt = returns.align(var_garch_t, join="inner")
    _, garch_t_rate = var_breaches(r_gt, v_gt)

    # ---- ML Student-t ----
    ml_sigma_raw = pd.Series(
        xgb_preds,
        index=rv.index[-len(xgb_preds):]
    )

    ml_sigma_cal = calibrate_volatility(
        returns.loc[ml_sigma_raw.index],
        ml_sigma_raw
    )

    ret_ml, var_ml, _ = parametric_var_cvar(
        returns, ml_sigma_cal, alpha, dist="t", nu=8
    )

    r_ml, v_ml = returns.align(var_ml, join="inner")
    ml_breaches, ml_rate = var_breaches(r_ml, v_ml)

    print("\nVaR BACKTEST RESULTS (99%)")
    print("-------------------------")
    print(f"GARCH (Gaussian)   : {garch_rate:.4f}")
    print(f"GARCH (Student-t)  : {garch_t_rate:.4f}")
    print(f"ML (Calibrated-t)  : {ml_rate:.4f}")
    print(f"Expected           : {1 - alpha:.4f}")

    # Save VaR summary
    with open("reports/results/var_summary.txt", "w") as f:
        f.write("VaR BACKTEST RESULTS (99%)\n")
        f.write(f"GARCH Gaussian   : {garch_rate:.4f}\n")
        f.write(f"GARCH Student-t  : {garch_t_rate:.4f}\n")
        f.write(f"ML Student-t     : {ml_rate:.4f}\n")
        f.write(f"Expected         : {1-alpha:.4f}\n")

    # =================================================
    # 7. REGULATORY BACKTESTS
    # =================================================
    kupiec = kupiec_pof_test(ml_breaches, alpha)
    christ = christoffersen_test(ml_breaches)

    print("\nREGULATORY BACKTESTS — ML MODEL")
    print("--------------------------------")
    print(f"Kupiec p-value       : {kupiec['p_value']:.4f}")
    print(f"Christoffersen p-val : {christ['p_value']:.4f}")

    # =================================================
    # 8. VOLATILITY COMPARISON PLOT
    # =================================================
    plt.figure()
    plt.plot(rv.index, rv["realized_vol"], color="black", linewidth=2, label="Realized Vol")
    plt.plot(garch_sigma, label="GARCH(1,1)")
    plt.plot(egarch_sigma, label="EGARCH(1,1)")
    plt.plot(tgarch_sigma, label="TGARCH(1,1)")
    plt.plot(garch_t_sigma, label="GARCH-t")
    plt.title("Realized Volatility vs GARCH-family Models")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/volatility_models.png")
    plt.close()

    # =================================================
    # 9. VAR BREACH PLOT (ML)
    # =================================================
    plt.figure()
    plt.plot(r_ml.index, r_ml, color="grey", alpha=0.6, label="Returns")
    plt.plot(v_ml.index, -v_ml, color="blue", linewidth=2, label="99% VaR (ML, t)")
    plt.scatter(
        ml_breaches[ml_breaches].index,
        r_ml[ml_breaches],
        color="red",
        s=30,
        label="VaR Breaches",
        zorder=5
    )
    plt.title("ML Tail-Risk Validation: VaR Breaches (99%)")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/var_breaches.png")
    plt.close()

    # =================================================
    # 10. BASEL TRAFFIC LIGHT (250-DAY)
    # =================================================
    rolling_breaches = ml_breaches.rolling(250).sum()

    plt.figure()
    plt.plot(rolling_breaches, color="black", linewidth=1.5)
    plt.axhline(4, color="green", linestyle="--", label="Green (≤4)")
    plt.axhline(9, color="orange", linestyle="--", label="Yellow (5–9)")
    plt.axhline(10, color="red", linestyle="--", label="Red (≥10)")
    plt.fill_between(rolling_breaches.index, 0, 4, color="green", alpha=0.08)
    plt.fill_between(rolling_breaches.index, 4, 9, color="orange", alpha=0.08)
    plt.fill_between(
        rolling_breaches.index,
        9,
        rolling_breaches.max(),
        color="red",
        alpha=0.08
    )
    plt.title("Basel Traffic Light: Rolling 250-Day VaR Breaches (99%)")
    plt.ylabel("Number of Breaches")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/basel_traffic_light.png")
    plt.close()

    # =================================================
    # 11. STRESS TEST CUMULATIVE DRAWDOWNS (FINAL FIX)
    # =================================================
    stress_results = run_standard_stress_tests(returns)

    plt.figure()
    for scenario, res in stress_results.items():
        start = pd.to_datetime(res["start"])
        end = pd.to_datetime(res["end"])

        stress_returns = returns.loc[start:end]
        cumulative_dd = (1 + stress_returns).cumprod() - 1

        plt.plot(
            cumulative_dd,
            linewidth=2,
            label=scenario.replace("_", " ")
        )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Stress Test: Cumulative Drawdowns")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/stress_drawdowns.png")
    plt.close()

    print("\nALL FIGURES SAVED TO reports/figures/")
    print("SYSTEM EXECUTION COMPLETE ✅\n")


# =================================================
# ENTRY POINT
# =================================================
if __name__ == "__main__":
    main()
