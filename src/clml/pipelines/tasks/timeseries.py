"""Time-series forecasting task runners."""

import numpy as np
import pandas as pd

from clml.constants import (
    AIRLINE_SEASONAL_PERIODS,
    ARTIFACT_FORECAST_CSV,
    ARTIFACT_FORECAST_PNG,
    ARTIFACT_SERIES_CSV,
    MAPE_DENOMINATOR_FLOOR,
    TIMESERIES_HOLDOUT,
    TIMESERIES_HORIZON_DIVISOR,
)
from clml.data.adapters import write_frame
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks._shared import _finish
from clml.reporting.plots import plot_timeseries_forecast


def run_timeseries(ctx: RunContext) -> RunResult:
    series = _target_series(ctx.bundle.frame, ctx.bundle.info.target_column)
    horizon = min(TIMESERIES_HOLDOUT, max(1, len(series) // TIMESERIES_HORIZON_DIVISOR))
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    fitted = _fit_timeseries_model(ctx.spec.name, train)
    forecast = _forecast(fitted, ctx.spec.name, horizon)
    forecast.index = test.index
    metrics = _forecast_metrics(test, forecast)
    output = pd.DataFrame({"actual": test, "forecast": forecast})
    write_frame(ctx.run_dir / ARTIFACT_FORECAST_CSV, output, include_index=True)
    write_frame(
        ctx.run_dir / ARTIFACT_SERIES_CSV,
        pd.DataFrame({"observed": series}),
        include_index=True,
    )
    plot_timeseries_forecast(series, forecast, ctx.run_dir / ARTIFACT_FORECAST_PNG, ctx.spec.title)
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, fitted, metrics, {}, {})


def _target_series(frame: pd.DataFrame, target_column: str | None) -> pd.Series:
    if target_column is None:
        raise ValueError("Time-series methods require a target column.")
    series = frame[target_column].astype(float).copy()
    if "date" in frame.columns:
        index = pd.to_datetime(frame["date"], errors="coerce")
        if index.notna().all():
            series.index = index
            frequency = pd.infer_freq(index)
            if frequency:
                series = series.asfreq(frequency)
    return series


def _fit_timeseries_model(name: str, train: pd.Series):
    match name:
        case "statsmodels_exponential_smoothing":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            return ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=AIRLINE_SEASONAL_PERIODS,
            ).fit(optimized=True)
        case "statsmodels_sarimax":
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            return SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, AIRLINE_SEASONAL_PERIODS),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        case "statsmodels_autoreg":
            from statsmodels.tsa.ar_model import AutoReg

            return AutoReg(train, lags=AIRLINE_SEASONAL_PERIODS, old_names=False).fit()
        case _:
            raise ValueError(f"Unsupported time-series method: {name}")


def _forecast(fitted, name: str, horizon: int) -> pd.Series:
    if name == "statsmodels_autoreg":
        start = len(fitted.model.endog)
        end = start + horizon - 1
        return pd.Series(fitted.predict(start=start, end=end))
    return pd.Series(fitted.forecast(horizon))


def _forecast_metrics(actual: pd.Series, forecast: pd.Series) -> dict[str, float | int]:
    errors = actual.to_numpy(dtype=float) - forecast.to_numpy(dtype=float)
    denominator = np.maximum(np.abs(actual.to_numpy(dtype=float)), MAPE_DENOMINATOR_FLOOR)
    return {
        "horizon": int(len(actual)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors) / denominator)),
    }
