"""Optimization task runners: linear programming, nonlinear, and cvxpy portfolio."""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from clml.constants import (
    LP_BINDING_SLACK_TOL,
    LP_LABOR_CAPACITY,
    LP_MACHINE_CAPACITY,
    LP_MATERIAL_CAPACITY,
    MARKETING_BUDGET,
    PORTFOLIO_CROSS_SECTOR_CORR,
    PORTFOLIO_MIN_LIQUIDITY,
    PORTFOLIO_RISK_AVERSION,
    PORTFOLIO_SAME_SECTOR_CORR,
    PORTFOLIO_TARGET_RETURN,
    SLSQP_FTOL,
    SLSQP_MAX_ITER,
)
from clml.data.adapters import write_frame
from clml.pipelines._context import RunContext, RunResult
from clml.pipelines.tasks._shared import _finish
from clml.reporting.plots import plot_marketing_response_curves, plot_named_bars, response_at_spend

# ---------------------------------------------------------------------------
# Shared portfolio helpers
# ---------------------------------------------------------------------------


def _portfolio_covariance(frame: pd.DataFrame) -> np.ndarray:
    volatility = frame["volatility"].to_numpy(dtype=float)
    sectors = frame["sector"].tolist()
    correlation = np.full((len(frame), len(frame)), PORTFOLIO_CROSS_SECTOR_CORR)
    for i, si in enumerate(sectors):
        for j, sj in enumerate(sectors):
            if i == j:
                correlation[i, j] = 1.0
            elif si == sj:
                correlation[i, j] = PORTFOLIO_SAME_SECTOR_CORR
    return np.outer(volatility, volatility) * correlation


def _write_portfolio_outputs(
    frame: pd.DataFrame, weights: np.ndarray, covariance: np.ndarray, run_dir
) -> pd.DataFrame:
    clean_weights = np.maximum(weights, 0)
    clean_weights = clean_weights / clean_weights.sum()
    allocation = frame[
        ["asset", "expected_return", "volatility", "liquidity_score", "sector"]
    ].copy()
    allocation["weight"] = clean_weights
    allocation["return_contribution"] = allocation["weight"] * allocation["expected_return"]
    write_frame(run_dir / "allocation.csv", allocation)
    write_frame(
        run_dir / "covariance.csv",
        pd.DataFrame(covariance, index=frame["asset"], columns=frame["asset"]),
        include_index=True,
    )
    plot_named_bars(
        allocation["asset"].tolist(),
        allocation["weight"].mul(100).tolist(),
        run_dir / "portfolio_weights.png",
        title="Optimized portfolio weights",
        ylabel="Weight (%)",
    )
    plot_named_bars(
        allocation["asset"].tolist(),
        allocation["return_contribution"].mul(100).tolist(),
        run_dir / "return_contribution.png",
        title="Expected return contribution",
        ylabel="Contribution (%)",
    )
    return allocation


def _portfolio_metrics(allocation: pd.DataFrame, covariance: np.ndarray) -> dict[str, float]:
    weights = allocation["weight"].to_numpy(dtype=float)
    expected_return = float(allocation["return_contribution"].sum())
    volatility = float(np.sqrt(weights @ covariance @ weights))
    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_like_ratio": expected_return / volatility,
        "max_weight": float(weights.max()),
        "liquidity_score": float(weights @ allocation["liquidity_score"].to_numpy(dtype=float)),
    }


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------


def run_linear_programming(ctx: RunContext) -> RunResult:
    frame = ctx.bundle.frame.copy()
    resource_columns = ["labor_hours_per_unit", "machine_hours_per_unit", "material_kg_per_unit"]
    capacities = {
        "labor_hours_per_unit": LP_LABOR_CAPACITY,
        "machine_hours_per_unit": LP_MACHINE_CAPACITY,
        "material_kg_per_unit": LP_MATERIAL_CAPACITY,
    }
    c = -frame["expected_profit_per_unit"].to_numpy(dtype=float)
    a_ub = frame[resource_columns].to_numpy(dtype=float).T
    b_ub = np.array([capacities[col] for col in resource_columns], dtype=float)
    bounds = [(0.0, float(value)) for value in frame["demand_max_units"]]

    solution = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not solution.success:
        raise RuntimeError(f"Linear program failed: {solution.message}")

    allocation = solution.x
    resource_usage = a_ub @ allocation
    resource_report = pd.DataFrame(
        {
            "resource": resource_columns,
            "capacity": b_ub,
            "used": resource_usage,
            "slack": b_ub - resource_usage,
            "utilization": resource_usage / b_ub,
        }
    )
    allocation_report = frame[["product", "expected_profit_per_unit", "demand_max_units"]].copy()
    allocation_report["optimized_units"] = allocation
    allocation_report["expected_profit"] = allocation * frame["expected_profit_per_unit"]
    write_frame(ctx.run_dir / "allocation.csv", allocation_report)
    write_frame(ctx.run_dir / "resource_usage.csv", resource_report)
    metrics: dict = {
        "max_expected_profit": float(-solution.fun),
        "total_units": float(np.sum(allocation)),
        "binding_constraints": int((resource_report["slack"] < LP_BINDING_SLACK_TOL).sum()),
        "average_resource_utilization": float(resource_report["utilization"].mean()),
    }
    plot_named_bars(
        allocation_report["product"].tolist(),
        allocation_report["optimized_units"].tolist(),
        ctx.run_dir / "optimized_units.png",
        title="Optimized production plan",
        ylabel="Units",
    )
    plot_named_bars(
        resource_report["resource"].tolist(),
        resource_report["utilization"].mul(100).tolist(),
        ctx.run_dir / "resource_utilization.png",
        title="Resource utilization",
        ylabel="Utilization (%)",
    )
    return _finish(
        ctx.spec,
        ctx.bundle,
        ctx.run_dir,
        {"solver": "scipy.optimize.linprog", "success": solution.success},
        metrics,
        {},
        {},
    )


def run_nonlinear_optimization(ctx: RunContext) -> RunResult:
    frame = ctx.bundle.frame.copy()
    current_spend = frame["current_spend"].to_numpy(dtype=float)
    max_spend = frame["max_spend"].to_numpy(dtype=float)
    initial_spend = current_spend * min(1.0, MARKETING_BUDGET / current_spend.sum())

    def expected_sales(spend: np.ndarray) -> float:
        values = [
            response_at_spend(row, float(spend[i])) for i, (_, row) in enumerate(frame.iterrows())
        ]
        return float(np.sum(values))

    result = minimize(
        fun=lambda spend: -expected_sales(spend),
        x0=initial_spend,
        bounds=[(0.0, float(v)) for v in max_spend],
        constraints=[
            {"type": "ineq", "fun": lambda spend: MARKETING_BUDGET - float(np.sum(spend))}
        ],
        method="SLSQP",
        options={"maxiter": SLSQP_MAX_ITER, "ftol": SLSQP_FTOL},
    )
    if not result.success:
        raise RuntimeError(f"Nonlinear optimization failed: {result.message}")

    optimized_spend = result.x
    rows_iter = list(frame.iterrows())
    optimized_sales = [
        response_at_spend(row, float(optimized_spend[i])) for i, (_, row) in enumerate(rows_iter)
    ]
    current_sales = [
        response_at_spend(row, float(current_spend[i])) for i, (_, row) in enumerate(rows_iter)
    ]
    allocation_report = frame[["channel", "current_spend", "max_spend"]].copy()
    allocation_report["optimized_spend"] = optimized_spend
    allocation_report["current_expected_sales"] = current_sales
    allocation_report["optimized_expected_sales"] = optimized_sales
    allocation_report["incremental_sales"] = (
        allocation_report["optimized_expected_sales"] - allocation_report["current_expected_sales"]
    )
    write_frame(ctx.run_dir / "allocation.csv", allocation_report)
    metrics: dict = {
        "optimized_expected_sales": float(np.sum(optimized_sales)),
        "current_expected_sales": float(np.sum(current_sales)),
        "incremental_sales": float(np.sum(optimized_sales) - np.sum(current_sales)),
        "budget_used": float(np.sum(optimized_spend)),
        "budget_slack": float(MARKETING_BUDGET - np.sum(optimized_spend)),
    }
    plot_named_bars(
        allocation_report["channel"].tolist(),
        allocation_report["optimized_spend"].tolist(),
        ctx.run_dir / "optimized_spend.png",
        title="Optimized marketing spend",
        ylabel="Spend",
    )
    plot_named_bars(
        allocation_report["channel"].tolist(),
        allocation_report["incremental_sales"].tolist(),
        ctx.run_dir / "incremental_sales.png",
        title="Incremental expected sales",
        ylabel="Sales",
    )
    plot_marketing_response_curves(frame, optimized_spend, ctx.run_dir / "response_curves.png")
    return _finish(
        ctx.spec,
        ctx.bundle,
        ctx.run_dir,
        {"solver": "scipy.optimize.minimize", "success": result.success},
        metrics,
        {},
        {},
    )


def run_cvxpy_portfolio(ctx: RunContext) -> RunResult:
    import cvxpy as cp

    frame = ctx.bundle.frame.copy()
    returns = frame["expected_return"].to_numpy(dtype=float)
    covariance = _portfolio_covariance(frame)
    max_weight = frame["max_weight"].to_numpy(dtype=float)
    liquidity = frame["liquidity_score"].to_numpy(dtype=float)
    weights = cp.Variable(len(frame))
    objective = cp.Maximize(
        returns @ weights - PORTFOLIO_RISK_AVERSION * cp.quad_form(weights, covariance)
    )
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
        liquidity @ weights >= PORTFOLIO_MIN_LIQUIDITY,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if weights.value is None:
        raise RuntimeError("CVXPY portfolio optimization failed.")
    allocation = _write_portfolio_outputs(frame, weights.value, covariance, ctx.run_dir)
    metrics = _portfolio_metrics(allocation, covariance)
    metrics["objective_value"] = float(problem.value)
    return _finish(
        ctx.spec,
        ctx.bundle,
        ctx.run_dir,
        {"solver": "cvxpy", "status": problem.status},
        metrics,
        {},
        {},
    )


def run_cvxpy_quadratic(ctx: RunContext) -> RunResult:
    import cvxpy as cp

    frame = ctx.bundle.frame.copy()
    returns = frame["expected_return"].to_numpy(dtype=float)
    covariance = _portfolio_covariance(frame)
    max_weight = frame["max_weight"].to_numpy(dtype=float)
    weights = cp.Variable(len(frame))
    objective = cp.Minimize(cp.quad_form(weights, covariance))
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
        returns @ weights >= PORTFOLIO_TARGET_RETURN,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if weights.value is None:
        raise RuntimeError("CVXPY quadratic program failed.")
    allocation = _write_portfolio_outputs(frame, weights.value, covariance, ctx.run_dir)
    metrics = _portfolio_metrics(allocation, covariance)
    metrics["target_return"] = PORTFOLIO_TARGET_RETURN
    metrics["objective_value"] = float(problem.value)
    return _finish(
        ctx.spec,
        ctx.bundle,
        ctx.run_dir,
        {"solver": "cvxpy", "status": problem.status},
        metrics,
        {},
        {},
    )
