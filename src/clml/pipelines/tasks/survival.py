"""Survival analysis task runners: Cox PH and Kaplan-Meier."""

import pandas as pd
from sklearn.model_selection import train_test_split

from clml.config.settings import get_settings
from clml.constants import (
    COX_PENALIZER,
    KM_PLOT_DPI,
    KM_PLOT_FIGSIZE,
    SURVIVAL_EVAL_MONTH,
    TEST_SIZE,
)
from clml.data.adapters import write_frame, write_series
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks._shared import _finish
from clml.reporting.plots import plot_named_bars


def _survival_design(frame: pd.DataFrame) -> pd.DataFrame:
    design = pd.get_dummies(frame, columns=["contract", "segment"], drop_first=True, dtype=float)
    return design.astype(float)


def run_survival_cox(ctx: RunContext) -> RunResult:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    frame = _survival_design(ctx.bundle.frame)
    train, test = train_test_split(
        frame, test_size=TEST_SIZE, random_state=get_settings().random_state
    )
    fitter = CoxPHFitter(penalizer=COX_PENALIZER)
    fitter.fit(train, duration_col="duration", event_col="event")
    risk_score = fitter.predict_partial_hazard(test)
    metrics: dict = {
        "concordance_index_train": float(fitter.concordance_index_),
        "concordance_index_test": float(
            concordance_index(test["duration"], -risk_score, test["event"])
        ),
        "observed_event_fraction": float(test["event"].mean()),
    }
    write_frame(ctx.run_dir / "cox_coefficients.csv", fitter.summary, include_index=True)
    write_frame(
        ctx.run_dir / "survival_predictions.csv",
        test.assign(risk_score=risk_score.to_numpy()),
    )
    plot_named_bars(
        fitter.summary.index.tolist(),
        fitter.summary["coef"].tolist(),
        ctx.run_dir / "cox_coefficients.png",
        title="Cox model coefficients",
        ylabel="Coefficient",
    )
    return _finish(ctx.spec, ctx.bundle, ctx.run_dir, fitter, metrics, {}, {})


def run_survival_kaplan_meier(ctx: RunContext) -> RunResult:
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    frame = ctx.bundle.frame.copy()
    kmf = KaplanMeierFitter()
    plt.figure(figsize=KM_PLOT_FIGSIZE)
    survival_at_eval: dict[str, float] = {}
    for contract, group in frame.groupby("contract"):
        kmf.fit(group["duration"], event_observed=group["event"], label=str(contract))
        kmf.plot_survival_function(ci_show=False)
        survival_at_eval[str(contract)] = float(
            kmf.survival_function_at_times(SURVIVAL_EVAL_MONTH).iloc[0]
        )
    plt.title("Kaplan-Meier survival by contract")
    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.tight_layout()
    plt.savefig(ctx.run_dir / "kaplan_meier_curves.png", dpi=KM_PLOT_DPI)
    plt.close()
    logrank = multivariate_logrank_test(frame["duration"], frame["contract"], frame["event"])
    metrics: dict = {
        "logrank_p_value": float(logrank.p_value),
        "event_fraction": float(frame["event"].mean()),
        **{f"survival_{SURVIVAL_EVAL_MONTH}m_{k}": v for k, v in survival_at_eval.items()},
    }
    write_series(
        ctx.run_dir / f"survival_at_{SURVIVAL_EVAL_MONTH}_months.csv",
        pd.Series(survival_at_eval, name=f"survival_probability_{SURVIVAL_EVAL_MONTH}m"),
        include_index=True,
    )
    return _finish(
        ctx.spec,
        ctx.bundle,
        ctx.run_dir,
        {"estimator": "lifelines.KaplanMeierFitter"},
        metrics,
        {},
        {},
    )
