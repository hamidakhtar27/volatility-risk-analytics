import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def create_volatility_features(
    returns: pd.DataFrame,
    realized_vol: pd.DataFrame,
    lags: int = 5
) -> pd.DataFrame:
    """
    Create time-series safe ML features for volatility forecasting.

    Target:
        Next-day realized volatility
    """
    df = pd.DataFrame(index=returns.index)

    # Lagged returns
    for i in range(1, lags + 1):
        df[f"ret_lag_{i}"] = returns["log_return"].shift(i)

    # Lagged realized volatility
    for i in range(1, lags + 1):
        df[f"rv_lag_{i}"] = realized_vol["realized_vol"].shift(i)

    # Rolling statistics
    df["ret_std_5"] = returns["log_return"].rolling(5).std()
    df["ret_std_21"] = returns["log_return"].rolling(21).std()

    # Target: next-day realized volatility
    df["target"] = realized_vol["realized_vol"].shift(-1)

    df = df.dropna()
    return df


def walk_forward_ml_forecast(
    data,
    model_type="rf",
    window=750,          # ~3 years
    step=5               # retrain weekly
):
    """
    Industry-style walk-forward ML volatility forecasting.
    """

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    X = data.drop(columns=["target"])
    y = data["target"]

    predictions = []
    actuals = []

    for i in range(window, len(X), step):
        X_train = X.iloc[i - window:i]
        y_train = y.iloc[i - window:i]

        X_test = X.iloc[i:i + step]
        y_test = y.iloc[i:i + step]

        if model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=80,
                max_depth=6,
                n_jobs=1,              # IMPORTANT
                random_state=42
            )

        elif model_type == "xgb":
            model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=42
            )
        else:
            raise ValueError("Invalid model_type")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        predictions.extend(preds)
        actuals.extend(y_test)

        if i % 250 == 0:
            print(f"ML progress: {i}/{len(X)}")

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return np.array(predictions), np.array(actuals), rmse
