#!/usr/bin/env python3
"""
Ensemble and paired region-block bootstrap analysis for geographic CV.

The five neural seeds are averaged for every held-out region-month.
Mean and Ridge have one prediction per region-month.

Uncertainty is estimated by resampling complete regional units, preserving
the twelve-month dependence structure within each region.

The principal comparison is against the ordinary MLP ensemble.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "tables"
    / "spatial_cv_total_sm_predictions.csv.gz"
)

SPATIAL_SUMMARY_PATH = (
    ROOT
    / "results"
    / "tables"
    / "spatial_cv_total_sm_summary.csv"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

REFERENCE_CONFIGURATION = "MLP"

CONFIGURATION_ORDER = [
    "Mean",
    "Ridge",
    "MLP",
    "PINN_P_lambda10",
    "PINN_P_PET_lambda1",
    "PINN_P_PET_lambda10",
]

EXPECTED_ROWS_PER_CONFIGURATION = 2_064
EXPECTED_REGION_COUNT = 172
EXPECTED_MONTHS_PER_REGION = 12
EXPECTED_NEURAL_SEEDS = 5

BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 20260806


def calculate_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(
        truth,
        dtype=float,
    ).reshape(-1)

    prediction = np.asarray(
        prediction,
        dtype=float,
    ).reshape(-1)

    if truth.size == 0 or prediction.size == 0:
        return {
            "rmse_mm": float("nan"),
            "mae_mm": float("nan"),
            "r_squared": float("nan"),
            "mean_bias_mm": float("nan"),
        }

    return {
        "rmse_mm": float(
            np.sqrt(
                mean_squared_error(
                    truth,
                    prediction,
                )
            )
        ),
        "mae_mm": float(
            mean_absolute_error(
                truth,
                prediction,
            )
        ),
        "r_squared": float(
            r2_score(
                truth,
                prediction,
            )
        ),
        "mean_bias_mm": float(
            np.mean(prediction - truth)
        ),
    }


def construct_ensemble_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    grouping_columns = [
        "configuration",
        "model_family",
        "lambda_precipitation",
        "lambda_pet",
        "test_fold",
        "validation_fold",
        "region_id",
        "month",
        "native_cell_count",
        "core_region_ge_4_cells",
    ]

    ensemble = (
        predictions.groupby(
            grouping_columns,
            as_index=False,
            observed=True,
            dropna=False,
        )
        .agg(
            observed_SM_total_mm=(
                "observed_SM_total_mm",
                "first",
            ),
            minimum_observed_SM_total_mm=(
                "observed_SM_total_mm",
                "min",
            ),
            maximum_observed_SM_total_mm=(
                "observed_SM_total_mm",
                "max",
            ),
            ensemble_predicted_SM_total_mm=(
                "predicted_SM_total_mm",
                "mean",
            ),
            prediction_std_across_seeds_mm=(
                "predicted_SM_total_mm",
                "std",
            ),
            seed_count=("seed", "nunique"),
        )
    )

    observed_spread = (
        ensemble[
            "maximum_observed_SM_total_mm"
        ]
        - ensemble[
            "minimum_observed_SM_total_mm"
        ]
    ).abs()

    if float(observed_spread.max()) > 1e-8:
        raise RuntimeError(
            "Observed targets differ across model seeds."
        )

    ensemble[
        "prediction_std_across_seeds_mm"
    ] = ensemble[
        "prediction_std_across_seeds_mm"
    ].fillna(0.0)

    ensemble["residual_mm"] = (
        ensemble["observed_SM_total_mm"]
        - ensemble[
            "ensemble_predicted_SM_total_mm"
        ]
    )

    ensemble = ensemble.drop(
        columns=[
            "minimum_observed_SM_total_mm",
            "maximum_observed_SM_total_mm",
        ]
    )

    for configuration in CONFIGURATION_ORDER:
        configuration_data = ensemble.loc[
            ensemble["configuration"]
            == configuration
        ]

        if len(configuration_data) != (
            EXPECTED_ROWS_PER_CONFIGURATION
        ):
            raise RuntimeError(
                f"{configuration}: expected "
                f"{EXPECTED_ROWS_PER_CONFIGURATION} "
                f"ensemble rows, found "
                f"{len(configuration_data)}."
            )

        expected_seed_count = (
            1
            if configuration
            in ["Mean", "Ridge"]
            else EXPECTED_NEURAL_SEEDS
        )

        if not (
            configuration_data["seed_count"]
            == expected_seed_count
        ).all():
            raise RuntimeError(
                f"{configuration}: unexpected seed count."
            )

    return (
        ensemble.sort_values(
            [
                "configuration",
                "region_id",
                "month",
            ]
        )
        .reset_index(drop=True)
    )


def calculate_ensemble_summary(
    ensemble: pd.DataFrame,
    spatial_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for configuration in CONFIGURATION_ORDER:
        subset = ensemble.loc[
            ensemble["configuration"]
            == configuration
        ]

        metrics = calculate_metrics(
            subset["observed_SM_total_mm"],
            subset[
                "ensemble_predicted_SM_total_mm"
            ],
        )

        core = subset.loc[
            subset["core_region_ge_4_cells"]
        ]

        core_metrics = calculate_metrics(
            core["observed_SM_total_mm"],
            core[
                "ensemble_predicted_SM_total_mm"
            ],
        )

        rows.append(
            {
                "configuration": configuration,
                "model_family": (
                    subset["model_family"].iloc[0]
                ),
                "lambda_precipitation": float(
                    subset[
                        "lambda_precipitation"
                    ].iloc[0]
                ),
                "lambda_pet": float(
                    subset["lambda_pet"].iloc[0]
                ),
                "ensemble_seed_count": int(
                    subset["seed_count"].iloc[0]
                ),
                "ensemble_rmse_mm": (
                    metrics["rmse_mm"]
                ),
                "ensemble_mae_mm": (
                    metrics["mae_mm"]
                ),
                "ensemble_r_squared": (
                    metrics["r_squared"]
                ),
                "ensemble_mean_bias_mm": (
                    metrics["mean_bias_mm"]
                ),
                "core_ensemble_rmse_mm": (
                    core_metrics["rmse_mm"]
                ),
                "core_ensemble_mae_mm": (
                    core_metrics["mae_mm"]
                ),
                "core_ensemble_r_squared": (
                    core_metrics["r_squared"]
                ),
                "mean_prediction_std_across_seeds_mm": float(
                    subset[
                        "prediction_std_across_seeds_mm"
                    ].mean()
                ),
                "maximum_prediction_std_across_seeds_mm": float(
                    subset[
                        "prediction_std_across_seeds_mm"
                    ].max()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    violation_columns = spatial_summary[
        [
            "configuration",
            "median_pooled_test_p_violation",
            "median_pooled_test_pet_violation",
        ]
    ].copy()

    summary = summary.merge(
        violation_columns,
        on="configuration",
        how="left",
        validate="one_to_one",
    )

    best_rmse = float(
        summary["ensemble_rmse_mm"].min()
    )

    summary["accuracy_gap_from_best_percent"] = (
        100.0
        * (
            summary["ensemble_rmse_mm"]
            - best_rmse
        )
        / best_rmse
    )

    summary["within_one_percent_of_best"] = (
        summary[
            "accuracy_gap_from_best_percent"
        ] <= 1.0
    )

    ordering = {
        configuration: index
        for index, configuration
        in enumerate(CONFIGURATION_ORDER)
    }

    summary["_order"] = (
        summary["configuration"]
        .map(ordering)
    )

    return (
        summary.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def calculate_region_metrics(
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        configuration,
        region_id,
    ), group in ensemble.groupby(
        ["configuration", "region_id"],
        observed=True,
    ):
        if len(group) != EXPECTED_MONTHS_PER_REGION:
            raise RuntimeError(
                f"{configuration}, {region_id}: "
                "expected 12 months."
            )

        metrics = calculate_metrics(
            group["observed_SM_total_mm"],
            group[
                "ensemble_predicted_SM_total_mm"
            ],
        )

        rows.append(
            {
                "configuration": configuration,
                "region_id": region_id,
                "test_fold": int(
                    group["test_fold"].iloc[0]
                ),
                "native_cell_count": int(
                    group[
                        "native_cell_count"
                    ].iloc[0]
                ),
                "core_region_ge_4_cells": bool(
                    group[
                        "core_region_ge_4_cells"
                    ].iloc[0]
                ),
                "regional_rmse_mm": (
                    metrics["rmse_mm"]
                ),
                "regional_mae_mm": (
                    metrics["mae_mm"]
                ),
                "regional_r_squared": (
                    metrics["r_squared"]
                ),
                "regional_mean_bias_mm": (
                    metrics["mean_bias_mm"]
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_fold_metrics(
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        configuration,
        test_fold,
    ), group in ensemble.groupby(
        ["configuration", "test_fold"],
        observed=True,
    ):
        metrics = calculate_metrics(
            group["observed_SM_total_mm"],
            group[
                "ensemble_predicted_SM_total_mm"
            ],
        )

        core = group.loc[
            group["core_region_ge_4_cells"]
        ]

        core_metrics = calculate_metrics(
            core["observed_SM_total_mm"],
            core[
                "ensemble_predicted_SM_total_mm"
            ],
        )

        rows.append(
            {
                "configuration": configuration,
                "test_fold": int(test_fold),
                "test_region_count": int(
                    group["region_id"].nunique()
                ),
                "test_sample_count": int(
                    len(group)
                ),
                "ensemble_fold_rmse_mm": (
                    metrics["rmse_mm"]
                ),
                "ensemble_fold_mae_mm": (
                    metrics["mae_mm"]
                ),
                "ensemble_fold_r_squared": (
                    metrics["r_squared"]
                ),
                "core_ensemble_fold_rmse_mm": (
                    core_metrics["rmse_mm"]
                ),
                "core_ensemble_fold_r_squared": (
                    core_metrics["r_squared"]
                ),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_rmse_difference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    core_only: bool,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    merge_columns = [
        "region_id",
        "month",
    ]

    candidate_columns = [
        *merge_columns,
        "core_region_ge_4_cells",
        "observed_SM_total_mm",
        "ensemble_predicted_SM_total_mm",
    ]

    reference_columns = [
        *merge_columns,
        "observed_SM_total_mm",
        "ensemble_predicted_SM_total_mm",
    ]

    paired = candidate[
        candidate_columns
    ].merge(
        reference[
            reference_columns
        ],
        on=merge_columns,
        how="inner",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )

    target_difference = (
        paired[
            "observed_SM_total_mm_candidate"
        ]
        - paired[
            "observed_SM_total_mm_reference"
        ]
    ).abs()

    if float(target_difference.max()) > 1e-8:
        raise RuntimeError(
            "Candidate and reference targets differ."
        )

    if core_only:
        paired = paired.loc[
            paired["core_region_ge_4_cells"]
        ].copy()

    regions = sorted(
        paired["region_id"].unique()
    )

    if not regions:
        return {
            "region_count": 0,
            "observed_candidate_rmse_mm": float("nan"),
            "observed_reference_rmse_mm": float("nan"),
            "observed_rmse_difference_mm": float("nan"),
            "bootstrap_difference_mean_mm": float("nan"),
            "bootstrap_difference_q025_mm": float("nan"),
            "bootstrap_difference_q50_mm": float("nan"),
            "bootstrap_difference_q975_mm": float("nan"),
            "probability_candidate_lower_rmse": float("nan"),
            "candidate_region_win_count": 0,
            "reference_region_win_count": 0,
            "regional_tie_count": 0,
        }

    candidate_sse = []
    reference_sse = []
    observation_counts = []

    candidate_region_rmse = []
    reference_region_rmse = []

    for region_id in regions:
        region = paired.loc[
            paired["region_id"] == region_id
        ]

        truth = region[
            "observed_SM_total_mm_candidate"
        ].to_numpy(dtype=float)

        candidate_prediction = region[
            "ensemble_predicted_SM_total_mm_candidate"
        ].to_numpy(dtype=float)

        reference_prediction = region[
            "ensemble_predicted_SM_total_mm_reference"
        ].to_numpy(dtype=float)

        candidate_error = (
            truth - candidate_prediction
        )

        reference_error = (
            truth - reference_prediction
        )

        candidate_sse.append(
            float(
                np.sum(candidate_error**2)
            )
        )

        reference_sse.append(
            float(
                np.sum(reference_error**2)
            )
        )

        observation_counts.append(
            int(len(region))
        )

        candidate_region_rmse.append(
            float(
                np.sqrt(
                    np.mean(
                        candidate_error**2
                    )
                )
            )
        )

        reference_region_rmse.append(
            float(
                np.sqrt(
                    np.mean(
                        reference_error**2
                    )
                )
            )
        )

    candidate_sse = np.asarray(
        candidate_sse,
        dtype=float,
    )

    reference_sse = np.asarray(
        reference_sse,
        dtype=float,
    )

    observation_counts = np.asarray(
        observation_counts,
        dtype=float,
    )

    region_count = len(regions)

    sampled_indices = rng.integers(
        low=0,
        high=region_count,
        size=(
            BOOTSTRAP_ITERATIONS,
            region_count,
        ),
    )

    bootstrap_observation_count = (
        observation_counts[
            sampled_indices
        ].sum(axis=1)
    )

    bootstrap_candidate_rmse = np.sqrt(
        candidate_sse[
            sampled_indices
        ].sum(axis=1)
        / bootstrap_observation_count
    )

    bootstrap_reference_rmse = np.sqrt(
        reference_sse[
            sampled_indices
        ].sum(axis=1)
        / bootstrap_observation_count
    )

    differences = (
        bootstrap_candidate_rmse
        - bootstrap_reference_rmse
    )

    observed_candidate_rmse = float(
        np.sqrt(
            candidate_sse.sum()
            / observation_counts.sum()
        )
    )

    observed_reference_rmse = float(
        np.sqrt(
            reference_sse.sum()
            / observation_counts.sum()
        )
    )

    candidate_region_rmse = np.asarray(
        candidate_region_rmse
    )

    reference_region_rmse = np.asarray(
        reference_region_rmse
    )

    return {
        "region_count": int(region_count),
        "observed_candidate_rmse_mm": (
            observed_candidate_rmse
        ),
        "observed_reference_rmse_mm": (
            observed_reference_rmse
        ),
        "observed_rmse_difference_mm": float(
            observed_candidate_rmse
            - observed_reference_rmse
        ),
        "bootstrap_difference_mean_mm": float(
            differences.mean()
        ),
        "bootstrap_difference_q025_mm": float(
            np.quantile(
                differences,
                0.025,
            )
        ),
        "bootstrap_difference_q50_mm": float(
            np.quantile(
                differences,
                0.50,
            )
        ),
        "bootstrap_difference_q975_mm": float(
            np.quantile(
                differences,
                0.975,
            )
        ),
        "probability_candidate_lower_rmse": float(
            np.mean(differences < 0.0)
        ),
        "candidate_region_win_count": int(
            np.sum(
                candidate_region_rmse
                < reference_region_rmse
            )
        ),
        "reference_region_win_count": int(
            np.sum(
                reference_region_rmse
                < candidate_region_rmse
            )
        ),
        "regional_tie_count": int(
            np.sum(
                np.isclose(
                    candidate_region_rmse,
                    reference_region_rmse,
                    atol=1e-10,
                )
            )
        ),
    }


def perform_bootstrap_comparisons(
    ensemble: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    reference = ensemble.loc[
        ensemble["configuration"]
        == REFERENCE_CONFIGURATION
    ].copy()

    rows = []

    for configuration in CONFIGURATION_ORDER:
        candidate = ensemble.loc[
            ensemble["configuration"]
            == configuration
        ].copy()

        for core_only in [False, True]:
            result = bootstrap_rmse_difference(
                candidate,
                reference,
                core_only=core_only,
                rng=rng,
            )

            rows.append(
                {
                    "configuration": (
                        configuration
                    ),
                    "reference_configuration": (
                        REFERENCE_CONFIGURATION
                    ),
                    "population": (
                        "core_regions"
                        if core_only
                        else "all_regions"
                    ),
                    **result,
                }
            )

    return pd.DataFrame(rows)


def make_figure(
    ensemble_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15.0, 5.0),
        constrained_layout=True,
    )

    positions = np.arange(
        len(ensemble_summary)
    )

    labels = ensemble_summary[
        "configuration"
    ].tolist()

    axes[0].bar(
        positions,
        ensemble_summary[
            "ensemble_rmse_mm"
        ],
        edgecolor="black",
        linewidth=0.5,
    )

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        labels,
        rotation=30,
        ha="right",
    )

    axes[0].set_ylabel(
        "Ensemble geographic RMSE (mm)"
    )

    axes[0].set_title(
        "Out-of-region ensemble accuracy"
    )

    comparison = bootstrap.loc[
        (
            bootstrap["population"]
            == "all_regions"
        )
        & (
            bootstrap["configuration"]
            != REFERENCE_CONFIGURATION
        )
    ].copy()

    comparison = comparison.set_index(
        "configuration"
    ).loc[
        [
            configuration
            for configuration
            in CONFIGURATION_ORDER
            if configuration
            != REFERENCE_CONFIGURATION
        ]
    ].reset_index()

    comparison_positions = np.arange(
        len(comparison)
    )

    lower_errors = (
        comparison[
            "observed_rmse_difference_mm"
        ]
        - comparison[
            "bootstrap_difference_q025_mm"
        ]
    )

    upper_errors = (
        comparison[
            "bootstrap_difference_q975_mm"
        ]
        - comparison[
            "observed_rmse_difference_mm"
        ]
    )

    axes[1].errorbar(
        comparison_positions,
        comparison[
            "observed_rmse_difference_mm"
        ],
        yerr=np.vstack(
            [
                lower_errors,
                upper_errors,
            ]
        ),
        marker="o",
        linestyle="none",
        capsize=4,
    )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_xticks(
        comparison_positions
    )

    axes[1].set_xticklabels(
        comparison["configuration"],
        rotation=30,
        ha="right",
    )

    axes[1].set_ylabel(
        "RMSE difference from MLP (mm)"
    )

    axes[1].set_title(
        "Region-block bootstrap, 95% interval"
    )

    width = 0.36

    axes[2].bar(
        positions - width / 2,
        ensemble_summary[
            "median_pooled_test_p_violation"
        ],
        width=width,
        label="Precipitation",
    )

    axes[2].bar(
        positions + width / 2,
        ensemble_summary[
            "median_pooled_test_pet_violation"
        ],
        width=width,
        label="PET",
    )

    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(
        labels,
        rotation=30,
        ha="right",
    )

    axes[2].set_ylabel(
        "Gradient-violation fraction"
    )

    axes[2].set_ylim(0.0, 1.02)

    axes[2].set_title(
        "Physical consistency"
    )

    axes[2].legend(
        frameon=True,
        fontsize=8,
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Ensemble geographic performance and "
        "physics–accuracy trade-off",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig20_spatial_cv_ensemble_selection.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig20_spatial_cv_ensemble_selection.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            PREDICTION_PATH
        )

    if not SPATIAL_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            SPATIAL_SUMMARY_PATH
        )

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    spatial_summary = pd.read_csv(
        SPATIAL_SUMMARY_PATH
    )

    found_configurations = sorted(
        predictions[
            "configuration"
        ].unique().tolist()
    )

    if sorted(CONFIGURATION_ORDER) != (
        found_configurations
    ):
        raise RuntimeError(
            "Unexpected model configurations: "
            f"{found_configurations}"
        )

    ensemble = (
        construct_ensemble_predictions(
            predictions
        )
    )

    ensemble_summary = (
        calculate_ensemble_summary(
            ensemble,
            spatial_summary,
        )
    )

    region_metrics = (
        calculate_region_metrics(
            ensemble
        )
    )

    fold_metrics = (
        calculate_fold_metrics(
            ensemble
        )
    )

    bootstrap = (
        perform_bootstrap_comparisons(
            ensemble
        )
    )

    ensemble_path = (
        TABLE_DIR
        / "spatial_cv_ensemble_predictions.csv.gz"
    )

    summary_path = (
        TABLE_DIR
        / "spatial_cv_ensemble_summary.csv"
    )

    region_metrics_path = (
        TABLE_DIR
        / "spatial_cv_ensemble_region_metrics.csv"
    )

    fold_metrics_path = (
        TABLE_DIR
        / "spatial_cv_ensemble_fold_metrics.csv"
    )

    bootstrap_path = (
        TABLE_DIR
        / "spatial_cv_ensemble_block_bootstrap.csv"
    )

    ensemble.to_csv(
        ensemble_path,
        index=False,
        compression="gzip",
    )

    ensemble_summary.to_csv(
        summary_path,
        index=False,
    )

    region_metrics.to_csv(
        region_metrics_path,
        index=False,
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    make_figure(
        ensemble_summary,
        bootstrap,
    )

    best_configuration = str(
        ensemble_summary.sort_values(
            "ensemble_rmse_mm"
        )["configuration"].iloc[0]
    )

    near_optimal = (
        ensemble_summary.loc[
            ensemble_summary[
                "within_one_percent_of_best"
            ],
            "configuration",
        ].tolist()
    )

    summary_json = {
        "status": "PASS",
        "reference_configuration": (
            REFERENCE_CONFIGURATION
        ),
        "bootstrap_iterations": (
            BOOTSTRAP_ITERATIONS
        ),
        "bootstrap_unit": (
            "Complete regional unit with all 12 months."
        ),
        "region_count": (
            EXPECTED_REGION_COUNT
        ),
        "best_ensemble_configuration_by_rmse": (
            best_configuration
        ),
        "configurations_within_one_percent_of_best": (
            near_optimal
        ),
        "selection_interpretation": (
            "Statistical superiority requires the "
            "region-block 95% interval for RMSE difference "
            "to exclude zero. Otherwise treat models as "
            "predictively indistinguishable and select using "
            "physical consistency and scientific defensibility."
        ),
        "ensemble_summary": str(
            summary_path
        ),
        "bootstrap_comparisons": str(
            bootstrap_path
        ),
        "region_metrics": str(
            region_metrics_path
        ),
        "fold_metrics": str(
            fold_metrics_path
        ),
        "ensemble_predictions": str(
            ensemble_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "spatial_cv_ensemble_selection_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            summary_json,
            stream,
            indent=2,
        )

    print("=" * 78)
    print("SPATIAL-CV ENSEMBLE AND REGION-BLOCK BOOTSTRAP")
    print("=" * 78)

    display_columns = [
        "configuration",
        "ensemble_seed_count",
        "ensemble_rmse_mm",
        "ensemble_mae_mm",
        "ensemble_r_squared",
        "core_ensemble_rmse_mm",
        "core_ensemble_r_squared",
        "mean_prediction_std_across_seeds_mm",
        "median_pooled_test_p_violation",
        "median_pooled_test_pet_violation",
        "accuracy_gap_from_best_percent",
        "within_one_percent_of_best",
    ]

    print(
        ensemble_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("Paired region-block comparison against MLP:")

    bootstrap_display = bootstrap.loc[
        (
            bootstrap["population"]
            == "all_regions"
        )
        & (
            bootstrap["configuration"]
            != REFERENCE_CONFIGURATION
        )
    ][
        [
            "configuration",
            "observed_rmse_difference_mm",
            "bootstrap_difference_q025_mm",
            "bootstrap_difference_q975_mm",
            "probability_candidate_lower_rmse",
            "candidate_region_win_count",
            "reference_region_win_count",
        ]
    ]

    print(
        bootstrap_display.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print(
        f"Best ensemble RMSE configuration : "
        f"{best_configuration}"
    )

    print(
        f"Within 1% of best               : "
        f"{near_optimal}"
    )

    print(f"Ensemble summary                : {summary_path}")
    print(f"Bootstrap comparisons           : {bootstrap_path}")
    print(f"Metadata                        : {metadata_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
