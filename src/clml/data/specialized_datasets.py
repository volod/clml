import numpy as np
import pandas as pd

from clml import constants
from clml.data.types import DatasetBundle, DatasetInfo


def _anomaly() -> DatasetBundle:
    rng = np.random.default_rng(constants.CATALOG_SEED)
    normal = rng.normal(0, 1, size=(constants.ANOMALY_N_NORMAL, constants.ANOMALY_N_FEATURES))
    outliers = rng.normal(
        constants.ANOMALY_OUTLIER_MEAN,
        constants.ANOMALY_OUTLIER_STD,
        size=(constants.ANOMALY_N_OUTLIERS, constants.ANOMALY_N_FEATURES),
    )
    x = np.vstack([normal, outliers])
    y = np.hstack([np.zeros(len(normal), dtype=int), np.ones(len(outliers), dtype=int)])
    feature_columns = [f"feature_{idx}" for idx in range(x.shape[1])]
    frame = pd.DataFrame(x, columns=feature_columns)
    frame["target"] = y
    info = DatasetInfo(
        name="anomaly",
        task="anomaly",
        description="Generated tabular anomaly detection data; target 1 means outlier.",
        feature_columns=feature_columns,
        target_column="target",
        source="numpy random normal mixtures",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _production_planning() -> DatasetBundle:
    frame = pd.DataFrame(
        {
            "product": ["standard", "premium", "economy", "enterprise", "custom"],
            "expected_profit_per_unit": [38.0, 72.0, 24.0, 115.0, 96.0],
            "labor_hours_per_unit": [1.4, 2.2, 0.9, 3.4, 2.9],
            "machine_hours_per_unit": [0.8, 1.4, 0.5, 2.1, 1.8],
            "material_kg_per_unit": [3.0, 4.8, 2.1, 6.2, 5.5],
            "demand_max_units": [480, 260, 700, 140, 180],
        }
    )
    feature_columns = frame.columns.tolist()
    info = DatasetInfo(
        name="production_planning",
        task="linear_programming",
        description=(
            "Product-mix optimization table with per-unit profit, resource usage, "
            "and demand bounds for a constrained linear program."
        ),
        feature_columns=feature_columns,
        target_column=None,
        source="generated operations research benchmark",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _marketing_mix() -> DatasetBundle:
    frame = pd.DataFrame(
        {
            "channel": ["search", "social", "video", "email", "partners", "display"],
            "current_spend": [110000, 85000, 130000, 35000, 70000, 45000],
            "max_spend": [220000, 190000, 260000, 90000, 160000, 120000],
            "baseline_sales": [4200, 3500, 5100, 1900, 2800, 2100],
            "response_scale": [90000, 70000, 130000, 28000, 75000, 55000],
            "saturation_sales": [9200, 7800, 11800, 3900, 6800, 4700],
        }
    )
    feature_columns = frame.columns.tolist()
    info = DatasetInfo(
        name="marketing_mix",
        task="nonlinear_optimization",
        description=(
            "Marketing channel allocation table with bounded spend and saturating "
            "sales-response curves for nonlinear constrained optimization."
        ),
        feature_columns=feature_columns,
        target_column=None,
        source="generated marketing mix optimization benchmark",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _portfolio_assets() -> DatasetBundle:
    frame = pd.DataFrame(
        {
            "asset": ["bonds", "large_cap", "small_cap", "reit", "commodities", "intl_equity"],
            "expected_return": [0.035, 0.082, 0.105, 0.071, 0.058, 0.088],
            "volatility": [0.045, 0.145, 0.22, 0.18, 0.19, 0.17],
            "liquidity_score": [0.98, 0.96, 0.78, 0.72, 0.84, 0.88],
            "max_weight": [0.65, 0.55, 0.28, 0.25, 0.22, 0.35],
            "sector": ["fixed_income", "equity", "equity", "real_assets", "real_assets", "equity"],
        }
    )
    info = DatasetInfo(
        name="portfolio_assets",
        task="convex_optimization",
        description=(
            "Asset allocation table with expected returns, volatility, liquidity, "
            "sector labels, and maximum weights for convex portfolio examples."
        ),
        feature_columns=frame.columns.tolist(),
        target_column=None,
        source="generated portfolio optimization benchmark",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _customer_survival() -> DatasetBundle:
    rng = np.random.default_rng(constants.CATALOG_SEED)
    rows = constants.SURVIVAL_N_ROWS
    age = rng.integers(constants.SURVIVAL_AGE_MIN, constants.SURVIVAL_AGE_MAX, rows)
    monthly_spend = rng.lognormal(
        mean=constants.SURVIVAL_SPEND_LOG_MEAN, sigma=constants.SURVIVAL_SPEND_LOG_STD, size=rows
    )
    tenure_months = rng.integers(1, constants.SURVIVAL_TENURE_MAX, rows)
    support_tickets = rng.poisson(
        constants.SURVIVAL_TICKETS_RATE
        + np.clip(
            constants.SURVIVAL_TICKETS_SPEND_REF - monthly_spend,
            0,
            constants.SURVIVAL_TICKETS_SPEND_MAX,
        )
        / constants.SURVIVAL_TICKETS_SPEND_REF
    )
    contract = rng.choice(
        ["monthly", "annual", "two_year"], rows, p=constants.SURVIVAL_CONTRACT_PROBS
    )
    segment = rng.choice(
        ["consumer", "small_business", "enterprise"], rows, p=constants.SURVIVAL_SEGMENT_PROBS
    )
    log_hazard = (
        constants.SURVIVAL_HAZARD_INTERCEPT
        + (contract == "monthly") * constants.SURVIVAL_MONTHLY_COEF
        - (contract == "two_year") * constants.SURVIVAL_TWO_YEAR_COEF
        + support_tickets * constants.SURVIVAL_TICKETS_COEF
        - np.log(monthly_spend) * constants.SURVIVAL_LOG_SPEND_COEF
        - tenure_months * constants.SURVIVAL_TENURE_COEF
        + (segment == "consumer") * constants.SURVIVAL_CONSUMER_COEF
        + rng.normal(0, constants.SURVIVAL_HAZARD_NOISE_STD, rows)
    )
    churn_rate = np.exp(log_hazard)
    churn_time = rng.exponential(1 / churn_rate)
    censor_time = rng.uniform(constants.SURVIVAL_CENSOR_MIN, constants.SURVIVAL_CENSOR_MAX, rows)
    duration = np.minimum(churn_time, censor_time).clip(
        constants.SURVIVAL_DURATION_MIN, constants.SURVIVAL_DURATION_MAX
    )
    event = (churn_time <= censor_time).astype(int)
    frame = pd.DataFrame(
        {
            "age": age,
            "monthly_spend": monthly_spend.round(2),
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "contract": contract,
            "segment": segment,
            "duration": duration.round(2),
            "event": event,
        }
    )
    info = DatasetInfo(
        name="customer_survival",
        task="survival",
        description=(
            "Synthetic customer churn time-to-event data with censoring, contracts, "
            "segments, spend, support load, and observed churn events."
        ),
        feature_columns=[col for col in frame.columns if col not in {"duration", "event"}],
        target_column="event",
        source="generated survival analysis benchmark",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _streaming_churn() -> DatasetBundle:
    rng = np.random.default_rng(constants.CATALOG_SEED)
    rows = constants.STREAMING_CHURN_N_ROWS
    index = np.arange(rows)
    trend = index / max(rows - 1, 1)
    usage_minutes = rng.gamma(shape=3.0, scale=38.0, size=rows) * (1.0 - 0.25 * trend)
    support_tickets = rng.poisson(0.4 + 1.4 * trend, size=rows)
    payment_failures = rng.poisson(0.08 + 0.35 * trend, size=rows)
    days_since_login = rng.gamma(shape=2.0, scale=4.0 + 7.0 * trend, size=rows)
    account_age_days = rng.integers(20, 1_100, size=rows)
    plan_value = rng.choice([19.0, 49.0, 99.0], size=rows, p=[0.45, 0.40, 0.15])
    linear_risk = (
        -2.2
        + constants.STREAMING_TREND_SCALE * trend
        - 0.010 * usage_minutes
        + 0.45 * support_tickets
        + 0.85 * payment_failures
        + 0.055 * days_since_login
        - 0.0008 * account_age_days
        - 0.006 * plan_value
        + rng.normal(0, constants.STREAMING_NOISE_STD, rows)
    )
    probability = 1 / (1 + np.exp(-linear_risk))
    target = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "event_index": index,
            "usage_minutes_7d": usage_minutes.round(2),
            "support_tickets_30d": support_tickets,
            "payment_failures_90d": payment_failures,
            "days_since_login": days_since_login.round(2),
            "account_age_days": account_age_days,
            "plan_value": plan_value,
            "target": target,
        }
    )
    info = DatasetInfo(
        name="streaming_churn",
        task="incremental_classification",
        description=(
            "Synthetic ordered churn stream with mild concept drift; rows are meant to be "
            "processed sequentially for online learning."
        ),
        feature_columns=[col for col in frame.columns if col != "target"],
        target_column="target",
        source="generated benchmark for incremental learning",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _airline_passengers() -> DatasetBundle:
    rng = np.random.default_rng(constants.CATALOG_SEED)
    periods = constants.AIRLINE_N_PERIODS
    index = np.arange(periods)
    month = index % constants.AIRLINE_SEASONAL_PERIODS
    dates = pd.date_range(constants.AIRLINE_DATE_START, periods=periods, freq="MS")
    seasonal = constants.AIRLINE_SEASONAL_AMPLITUDE * (
        np.sin(2 * np.pi * month / constants.AIRLINE_SEASONAL_PERIODS)
        + 0.35 * np.cos(4 * np.pi * month / constants.AIRLINE_SEASONAL_PERIODS)
    )
    passengers = (
        constants.AIRLINE_BASE_PASSENGERS
        + constants.AIRLINE_MONTHLY_TREND * index
        + seasonal
        + rng.normal(0, constants.AIRLINE_NOISE_STD, periods)
    )
    frame = pd.DataFrame(
        {
            "date": dates.astype(str),
            "month_index": index,
            "month": month + 1,
            "target": passengers.round(2),
        }
    )
    info = DatasetInfo(
        name="airline_passengers",
        task="timeseries",
        description=(
            "Synthetic monthly airline-passenger series with linear trend, annual "
            "seasonality, and moderate noise."
        ),
        feature_columns=["date", "month_index", "month"],
        target_column="target",
        source="generated benchmark inspired by airline passenger forecasting examples",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)
