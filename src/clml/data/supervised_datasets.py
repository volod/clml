import numpy as np
import pandas as pd

from clml import constants
from clml.data.types import DatasetBundle, DatasetInfo


def _credit_risk() -> DatasetBundle:
    rng = np.random.default_rng(constants.CATALOG_SEED)
    rows = constants.CREDIT_RISK_N_ROWS
    age = rng.integers(constants.CREDIT_RISK_AGE_MIN, constants.CREDIT_RISK_AGE_MAX, rows)
    income = rng.lognormal(
        mean=constants.CREDIT_RISK_INCOME_LOG_MEAN,
        sigma=constants.CREDIT_RISK_INCOME_LOG_STD,
        size=rows,
    )
    loan_amount = rng.lognormal(
        mean=constants.CREDIT_RISK_LOAN_LOG_MEAN,
        sigma=constants.CREDIT_RISK_LOAN_LOG_STD,
        size=rows,
    )
    debt_to_income = np.clip(
        loan_amount / income
        + rng.normal(
            constants.CREDIT_RISK_DTI_NOISE_MEAN, constants.CREDIT_RISK_DTI_NOISE_STD, rows
        ),
        constants.CREDIT_RISK_DTI_MIN,
        constants.CREDIT_RISK_DTI_MAX,
    )
    credit_history_years = np.clip(
        (age - constants.CREDIT_RISK_MIN_CREDIT_AGE)
        * rng.beta(
            constants.CREDIT_RISK_CREDIT_HIST_BETA_A, constants.CREDIT_RISK_CREDIT_HIST_BETA_B, rows
        ),
        0,
        constants.CREDIT_RISK_CREDIT_HIST_MAX,
    )
    delinquencies_2y = rng.poisson(
        np.clip(
            debt_to_income * constants.CREDIT_RISK_DELINQ_POISSON_SCALE,
            constants.CREDIT_RISK_DELINQ_POISSON_MIN,
            constants.CREDIT_RISK_DELINQ_CLIP_MAX,
        )
    )
    employment_length = np.clip(
        rng.normal(
            constants.CREDIT_RISK_EMP_LENGTH_MEAN, constants.CREDIT_RISK_EMP_LENGTH_STD, rows
        ),
        0,
        constants.CREDIT_RISK_EMP_LENGTH_MAX,
    )
    utilization = np.clip(
        rng.beta(constants.CREDIT_RISK_UTIL_BETA_A, constants.CREDIT_RISK_UTIL_BETA_B, rows)
        + debt_to_income * constants.CREDIT_RISK_UTIL_DTI_WEIGHT,
        0,
        1,
    )
    home_ownership = rng.choice(
        ["rent", "mortgage", "own"], rows, p=constants.CREDIT_RISK_HOME_PROBS
    )
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "medical", "small_business", "auto"],
        rows,
        p=constants.CREDIT_RISK_PURPOSE_PROBS,
    )
    region = rng.choice(
        ["north", "south", "west", "midwest"], rows, p=constants.CREDIT_RISK_REGION_PROBS
    )
    application_dates = pd.Timestamp(constants.CREDIT_RISK_DATE_START) + pd.to_timedelta(
        rng.integers(0, constants.CREDIT_RISK_DATE_DAYS, rows), unit="D"
    )
    linear_risk = (
        constants.CREDIT_RISK_INTERCEPT
        + debt_to_income * constants.CREDIT_RISK_DTI_COEF
        + utilization * constants.CREDIT_RISK_UTIL_COEF
        + delinquencies_2y * constants.CREDIT_RISK_DELINQ_COEF
        - np.log(income) * constants.CREDIT_RISK_LOG_INCOME_COEF
        - credit_history_years * constants.CREDIT_RISK_CREDIT_HIST_COEF
        + (home_ownership == "rent") * constants.CREDIT_RISK_RENT_COEF
        + (purpose == "small_business") * constants.CREDIT_RISK_SMALL_BIZ_COEF
        + (purpose == "medical") * constants.CREDIT_RISK_MEDICAL_COEF
        + rng.normal(0, constants.CREDIT_RISK_NOISE_STD, rows)
    )
    probability = 1 / (1 + np.exp(-linear_risk))
    target = rng.binomial(1, probability)
    frame = pd.DataFrame(
        {
            "age": age,
            "annual_income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "debt_to_income": debt_to_income.round(3),
            "credit_history_years": credit_history_years.round(1),
            "delinquencies_2y": delinquencies_2y,
            "employment_length": employment_length.round(1),
            "revolving_utilization": utilization.round(3),
            "application_date": application_dates.astype(str),
            "home_ownership": home_ownership,
            "loan_purpose": purpose,
            "region": region,
            "target": target,
        }
    )
    feature_columns = [col for col in frame.columns if col != "target"]
    info = DatasetInfo(
        name="credit_risk",
        task="classification",
        description=(
            "Synthetic but realistic credit-risk classification data with imbalance, "
            "mixed feature types, nonlinearity, and correlated financial predictors."
        ),
        feature_columns=feature_columns,
        target_column="target",
        source="generated benchmark inspired by public credit risk datasets",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _housing_prices() -> DatasetBundle:
    rng = np.random.default_rng(constants.CATALOG_SEED)
    rows = constants.HOUSING_N_ROWS
    living_area = rng.lognormal(
        mean=constants.HOUSING_LIVING_AREA_LOG_MEAN,
        sigma=constants.HOUSING_LIVING_AREA_LOG_STD,
        size=rows,
    )
    bedrooms = np.clip(
        np.round(
            living_area / constants.HOUSING_LIVING_AREA_PER_BEDROOM
            + rng.normal(constants.HOUSING_BEDROOM_MEAN, constants.HOUSING_BEDROOM_STD, rows)
        ),
        constants.HOUSING_BEDROOM_MIN,
        constants.HOUSING_BEDROOM_MAX,
    ).astype(int)
    bathrooms = np.clip(
        np.round(
            bedrooms * constants.HOUSING_BATHROOM_RATIO
            + rng.normal(constants.HOUSING_BATHROOM_MEAN, constants.HOUSING_BATHROOM_STD, rows),
            1,
        ),
        constants.HOUSING_BATHROOM_MIN,
        constants.HOUSING_BATHROOM_MAX,
    )
    age = rng.integers(0, constants.HOUSING_AGE_MAX, rows)
    lot_area = rng.lognormal(
        mean=constants.HOUSING_LOT_LOG_MEAN, sigma=constants.HOUSING_LOT_LOG_STD, size=rows
    )
    distance_to_center = rng.gamma(
        shape=constants.HOUSING_DIST_SHAPE, scale=constants.HOUSING_DIST_SCALE, size=rows
    )
    school_score = np.clip(
        rng.normal(constants.HOUSING_SCHOOL_MEAN, constants.HOUSING_SCHOOL_STD, rows)
        - distance_to_center * constants.HOUSING_SCHOOL_DIST_WEIGHT,
        constants.HOUSING_SCHOOL_MIN,
        constants.HOUSING_SCHOOL_MAX,
    )
    neighborhood = rng.choice(
        ["central", "suburban", "rural", "waterfront", "industrial_edge"],
        rows,
        p=constants.HOUSING_NEIGHBORHOOD_PROBS,
    )
    condition = rng.choice(
        ["needs_work", "average", "good", "renovated"],
        rows,
        p=constants.HOUSING_CONDITION_PROBS,
    )
    listing_dates = pd.Timestamp(constants.HOUSING_DATE_START) + pd.to_timedelta(
        rng.integers(0, constants.HOUSING_DATE_DAYS, rows), unit="D"
    )
    neighborhood_effect = (
        pd.Series(neighborhood)
        .map(
            {
                "central": constants.HOUSING_CENTRAL_EFFECT,
                "suburban": constants.HOUSING_SUBURBAN_EFFECT,
                "rural": constants.HOUSING_RURAL_EFFECT,
                "waterfront": constants.HOUSING_WATERFRONT_EFFECT,
                "industrial_edge": constants.HOUSING_INDUSTRIAL_EFFECT,
            }
        )
        .to_numpy()
    )
    condition_effect = (
        pd.Series(condition)
        .map(
            {
                "needs_work": constants.HOUSING_NEEDS_WORK_EFFECT,
                "average": 0,
                "good": constants.HOUSING_GOOD_EFFECT,
                "renovated": constants.HOUSING_RENOVATED_EFFECT,
            }
        )
        .to_numpy()
    )
    price = (
        constants.HOUSING_BASE_PRICE
        + living_area * constants.HOUSING_LIVING_AREA_COEF
        + lot_area * constants.HOUSING_LOT_AREA_COEF
        + bathrooms * constants.HOUSING_BATHROOM_COEF
        + school_score * constants.HOUSING_SCHOOL_COEF
        - age * constants.HOUSING_AGE_DEPRECIATION
        - distance_to_center * constants.HOUSING_DISTANCE_COEF
        + neighborhood_effect
        + condition_effect
        + rng.normal(0, constants.HOUSING_PRICE_NOISE_STD, rows)
    )
    luxury = (neighborhood == "waterfront") & (
        living_area > np.quantile(living_area, constants.HOUSING_LUXURY_QUANTILE)
    )
    price[luxury] *= constants.HOUSING_LUXURY_MULTIPLIER
    outlier_idx = rng.choice(rows, size=constants.HOUSING_OUTLIER_COUNT, replace=False)
    price[outlier_idx] *= rng.uniform(
        constants.HOUSING_OUTLIER_MULT_MIN, constants.HOUSING_OUTLIER_MULT_MAX, len(outlier_idx)
    )
    frame = pd.DataFrame(
        {
            "living_area_sqft": living_area.round(0),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "home_age": age,
            "lot_area_sqft": lot_area.round(0),
            "distance_to_center_km": distance_to_center.round(2),
            "school_score": school_score.round(1),
            "listing_date": listing_dates.astype(str),
            "neighborhood": neighborhood,
            "condition": condition,
            "target": price.round(2),
        }
    )
    feature_columns = [col for col in frame.columns if col != "target"]
    info = DatasetInfo(
        name="housing_prices",
        task="regression",
        description=(
            "Synthetic but realistic housing regression data with skewed prices, "
            "categorical location effects, nonlinear luxury uplift, and outliers."
        ),
        feature_columns=feature_columns,
        target_column="target",
        source="generated benchmark inspired by public housing datasets",
        rows=len(frame),
        columns=len(frame.columns),
    )
    return DatasetBundle(info=info, frame=frame)


def _monotone_1d() -> DatasetBundle:
    rng = np.random.default_rng(constants.CATALOG_SEED)
    x = np.sort(rng.uniform(0, constants.MONOTONE_X_MAX, constants.MONOTONE_N_SAMPLES))
    y = (
        constants.MONOTONE_LINEAR_COEF * x
        + np.log1p(x) * constants.MONOTONE_LOG_COEF
        + rng.normal(0, constants.MONOTONE_NOISE_STD, constants.MONOTONE_N_SAMPLES)
    )
    frame = pd.DataFrame({"x": x, "target": y})
    info = DatasetInfo(
        name="monotone_1d",
        task="regression_1d",
        description=(
            "Synthetic 1-D dataset with a monotonically increasing log-linear relationship "
            "and moderate noise — designed for isotonic regression demonstration."
        ),
        feature_columns=["x"],
        target_column="target",
        source="generated benchmark for 1-D monotone regression",
        rows=constants.MONOTONE_N_SAMPLES,
        columns=2,
    )
    return DatasetBundle(info=info, frame=frame)
