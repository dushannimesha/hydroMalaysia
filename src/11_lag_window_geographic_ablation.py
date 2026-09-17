#!/usr/bin/env python3
"""
Direct geographic lag-window ablation for soil-moisture depth.

Question
--------
Does the addition of older precipitation contributions improve
out-of-region prediction, particularly for deeper soil layers?

Lag windows
-----------
- L0
- L0-L1
- L0-L2
- L0-L3

Evaluation
----------
For outer test fold f:
    validation fold = (f + 1) mod 5
    training folds  = remaining three folds

Ridge alpha is selected using the validation fold. After alpha selection,
the model is refitted on all non-test regions and evaluated on the
completely held-out test fold.

Complete-region bootstrap intervals compare nested lag windows.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "pinn_climatology_features_k4_lag3.parquet"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

TARGETS = {
    "SM_0_10cm_mm": "0–10 cm",
    "SM_10_40cm_mm": "10–40 cm",
    "SM_40_100cm_mm": "40–100 cm",
    "SM_100_200cm_mm": "100–200 cm",
}

TARGET_ORDER = list(TARGETS.keys())

STATIC_FEATURES = [
    "PET_mm",
    "latitude",
    "longitude",
]

LAG_WINDOWS = {
    "L0": 0,
    "L0_L1": 1,
    "L0_L2": 2,
    "L0_L3": 3,
}

WINDOW_ORDER = list(LAG_WINDOWS.keys())

ALPHA_GRID = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
]

FOLD_COUNT = 5

BOOTSTRAP_ITERATIONS = 20_000
BOOTSTRAP_SEED = 20260806

EXPECTED_ROWS = 2_064
EXPECTED_REGIONS = 172
EXPECTED_MONTHS = 12


def precipitation_features(
    maximum_lag: int,
) -> list[str]:
    return [
        f"K4_C{component}_L{lag}_mm"
        for component in range(1, 5)
        for lag in range(maximum_lag + 1)
    ]


def metrics(
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
            "bias_mm": float("nan"),
        }

    return {
        "rmse_mm": float(
            math.sqrt(
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
        "bias_mm": float(
            np.mean(prediction - truth)
        ),
    }


def fit_and_predict(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
) -> np.ndarray:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(
        train[features].to_numpy(
            dtype=float
        )
    )

    y_train = (
        y_scaler.fit_transform(
            train[[target]].to_numpy(
                dtype=float
            )
        )
        .reshape(-1)
    )

    x_evaluation = x_scaler.transform(
        evaluation[features].to_numpy(
            dtype=float
        )
    )

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
    )

    model.fit(
        x_train,
        y_train,
    )

    standardized_prediction = (
        model.predict(
            x_evaluation
        )
    )

    return (
        y_scaler.inverse_transform(
            standardized_prediction.reshape(
                -1,
                1,
            )
        )
        .reshape(-1)
    )


def select_alpha(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[float, pd.DataFrame]:
    rows = []

    best_alpha = None
    best_rmse = float("inf")

    validation_truth = validation[
        target
    ].to_numpy(dtype=float)

    for alpha in ALPHA_GRID:
        prediction = fit_and_predict(
            train,
            validation,
            features,
            target,
            alpha,
        )

        validation_metrics = metrics(
            validation_truth,
            prediction,
        )

        rows.append(
            {
                "alpha": alpha,
                "validation_rmse_mm": (
                    validation_metrics[
                        "rmse_mm"
                    ]
                ),
                "validation_r_squared": (
                    validation_metrics[
                        "r_squared"
                    ]
                ),
            }
        )

        if (
            validation_metrics["rmse_mm"]
            < best_rmse
        ):
            best_rmse = (
                validation_metrics[
                    "rmse_mm"
                ]
            )

            best_alpha = alpha

    if best_alpha is None:
        raise RuntimeError(
            "Ridge alpha selection failed."
        )

    return (
        float(best_alpha),
        pd.DataFrame(rows),
    )


def calculate_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        target,
        window,
    ), group in predictions.groupby(
        ["target", "lag_window"],
        observed=True,
    ):
        result = metrics(
            group["observed_mm"],
            group["predicted_mm"],
        )

        core = group.loc[
            group["core_region_ge_4_cells"]
        ]

        core_result = metrics(
            core["observed_mm"],
            core["predicted_mm"],
        )

        rows.append(
            {
                "target": target,
                "depth_label": (
                    TARGETS[target]
                ),
                "lag_window": window,
                "maximum_lag_months": (
                    LAG_WINDOWS[window]
                ),
                "feature_count": int(
                    group[
                        "feature_count"
                    ].iloc[0]
                ),
                "pooled_rmse_mm": (
                    result["rmse_mm"]
                ),
                "pooled_mae_mm": (
                    result["mae_mm"]
                ),
                "pooled_r_squared": (
                    result["r_squared"]
                ),
                "pooled_bias_mm": (
                    result["bias_mm"]
                ),
                "core_pooled_rmse_mm": (
                    core_result["rmse_mm"]
                ),
                "core_pooled_r_squared": (
                    core_result["r_squared"]
                ),
            }
        )

    summary = pd.DataFrame(rows)

    depth_order = {
        target: index
        for index, target
        in enumerate(TARGET_ORDER)
    }

    window_order = {
        window: index
        for index, window
        in enumerate(WINDOW_ORDER)
    }

    summary["_depth_order"] = (
        summary["target"]
        .map(depth_order)
    )

    summary["_window_order"] = (
        summary["lag_window"]
        .map(window_order)
    )

    return (
        summary.sort_values(
            [
                "_depth_order",
                "_window_order",
            ]
        )
        .drop(
            columns=[
                "_depth_order",
                "_window_order",
            ]
        )
        .reset_index(drop=True)
    )


def bootstrap_difference(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    paired = candidate[
        [
            "region_id",
            "month",
            "observed_mm",
            "predicted_mm",
        ]
    ].merge(
        reference[
            [
                "region_id",
                "month",
                "observed_mm",
                "predicted_mm",
            ]
        ],
        on=["region_id", "month"],
        how="inner",
        suffixes=(
            "_candidate",
            "_reference",
        ),
        validate="one_to_one",
    )

    target_difference = (
        paired["observed_mm_candidate"]
        - paired["observed_mm_reference"]
    ).abs()

    if float(target_difference.max()) > 1e-8:
        raise RuntimeError(
            "Targets differ between lag windows."
        )

    regions = sorted(
        paired["region_id"].unique()
    )

    if len(regions) != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} regions."
        )

    candidate_sse = []
    reference_sse = []
    observation_counts = []

    candidate_regional_rmse = []
    reference_regional_rmse = []

    for region_id in regions:
        region = paired.loc[
            paired["region_id"] == region_id
        ]

        if len(region) != EXPECTED_MONTHS:
            raise RuntimeError(
                f"{region_id} does not have "
                f"{EXPECTED_MONTHS} months."
            )

        truth = region[
            "observed_mm_candidate"
        ].to_numpy(dtype=float)

        candidate_prediction = region[
            "predicted_mm_candidate"
        ].to_numpy(dtype=float)

        reference_prediction = region[
            "predicted_mm_reference"
        ].to_numpy(dtype=float)

        candidate_error = (
            truth - candidate_prediction
        )

        reference_error = (
            truth - reference_prediction
        )

        candidate_region_sse = float(
            np.sum(candidate_error**2)
        )

        reference_region_sse = float(
            np.sum(reference_error**2)
        )

        candidate_sse.append(
            candidate_region_sse
        )

        reference_sse.append(
            reference_region_sse
        )

        observation_counts.append(
            len(region)
        )

        candidate_regional_rmse.append(
            float(
                np.sqrt(
                    candidate_region_sse
                    / len(region)
                )
            )
        )

        reference_regional_rmse.append(
            float(
                np.sqrt(
                    reference_region_sse
                    / len(region)
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

    sampled_regions = rng.integers(
        low=0,
        high=EXPECTED_REGIONS,
        size=(
            BOOTSTRAP_ITERATIONS,
            EXPECTED_REGIONS,
        ),
    )

    sampled_counts = (
        observation_counts[
            sampled_regions
        ].sum(axis=1)
    )

    candidate_bootstrap_rmse = np.sqrt(
        candidate_sse[
            sampled_regions
        ].sum(axis=1)
        / sampled_counts
    )

    reference_bootstrap_rmse = np.sqrt(
        reference_sse[
            sampled_regions
        ].sum(axis=1)
        / sampled_counts
    )

    difference = (
        candidate_bootstrap_rmse
        - reference_bootstrap_rmse
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

    candidate_regional_rmse = np.asarray(
        candidate_regional_rmse
    )

    reference_regional_rmse = np.asarray(
        reference_regional_rmse
    )

    return {
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
        "bootstrap_q025_difference_mm": float(
            np.quantile(
                difference,
                0.025,
            )
        ),
        "bootstrap_q50_difference_mm": float(
            np.quantile(
                difference,
                0.50,
            )
        ),
        "bootstrap_q975_difference_mm": float(
            np.quantile(
                difference,
                0.975,
            )
        ),
        "probability_candidate_lower_rmse": float(
            np.mean(difference < 0.0)
        ),
        "candidate_region_win_count": int(
            np.sum(
                candidate_regional_rmse
                < reference_regional_rmse
            )
        ),
        "reference_region_win_count": int(
            np.sum(
                reference_regional_rmse
                < candidate_regional_rmse
            )
        ),
    }


def perform_bootstrap(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    comparisons = [
        ("L0_L1", "L0", "add_L1"),
        ("L0_L2", "L0_L1", "add_L2"),
        ("L0_L3", "L0_L2", "add_L3"),
        ("L0_L2", "L0", "add_L1_L2"),
        ("L0_L3", "L0", "add_L1_L2_L3"),
    ]

    rows = []

    for target in TARGET_ORDER:
        target_data = predictions.loc[
            predictions["target"] == target
        ]

        for (
            candidate_window,
            reference_window,
            comparison_name,
        ) in comparisons:
            candidate = target_data.loc[
                target_data["lag_window"]
                == candidate_window
            ]

            reference = target_data.loc[
                target_data["lag_window"]
                == reference_window
            ]

            result = bootstrap_difference(
                candidate,
                reference,
                rng,
            )

            rows.append(
                {
                    "target": target,
                    "depth_label": (
                        TARGETS[target]
                    ),
                    "comparison": (
                        comparison_name
                    ),
                    "candidate_window": (
                        candidate_window
                    ),
                    "reference_window": (
                        reference_window
                    ),
                    **result,
                }
            )

    return pd.DataFrame(rows)


def build_decision_table(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for target in TARGET_ORDER:
        target_summary = (
            summary.loc[
                summary["target"] == target
            ]
            .set_index("lag_window")
        )

        l0_rmse = float(
            target_summary.loc[
                "L0",
                "pooled_rmse_mm",
            ]
        )

        full_rmse = float(
            target_summary.loc[
                "L0_L3",
                "pooled_rmse_mm",
            ]
        )

        full_comparison = bootstrap.loc[
            (
                bootstrap["target"]
                == target
            )
            & (
                bootstrap["comparison"]
                == "add_L1_L2_L3"
            )
        ]

        if len(full_comparison) != 1:
            raise RuntimeError(
                f"Missing full-window comparison "
                f"for {target}."
            )

        full_comparison = (
            full_comparison.iloc[0]
        )

        best_window = str(
            target_summary[
                "pooled_rmse_mm"
            ].idxmin()
        )

        rows.append(
            {
                "target": target,
                "depth_label": (
                    TARGETS[target]
                ),
                "l0_rmse_mm": l0_rmse,
                "full_l0_l3_rmse_mm": (
                    full_rmse
                ),
                "full_window_rmse_change_mm": (
                    full_rmse - l0_rmse
                ),
                "full_window_improvement_percent": (
                    100.0
                    * (l0_rmse - full_rmse)
                    / l0_rmse
                ),
                "full_window_bootstrap_q025_mm": float(
                    full_comparison[
                        "bootstrap_q025_difference_mm"
                    ]
                ),
                "full_window_bootstrap_q975_mm": float(
                    full_comparison[
                        "bootstrap_q975_difference_mm"
                    ]
                ),
                "probability_full_window_better": float(
                    full_comparison[
                        "probability_candidate_lower_rmse"
                    ]
                ),
                "best_observed_lag_window": (
                    best_window
                ),
                "best_observed_maximum_lag_months": int(
                    LAG_WINDOWS[
                        best_window
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def make_figure(
    summary: pd.DataFrame,
    decision: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.0, 4.8),
        constrained_layout=True,
    )

    positions = np.arange(
        len(WINDOW_ORDER)
    )

    for target in TARGET_ORDER:
        target_data = (
            summary.loc[
                summary["target"] == target
            ]
            .set_index("lag_window")
            .loc[WINDOW_ORDER]
        )

        axes[0].plot(
            positions,
            target_data[
                "pooled_rmse_mm"
            ],
            marker="o",
            linewidth=1.5,
            label=TARGETS[target],
        )

    axes[0].set_xticks(positions)

    axes[0].set_xticklabels(
        [
            "L0",
            "L0–L1",
            "L0–L2",
            "L0–L3",
        ]
    )

    axes[0].set_xlabel(
        "Included precipitation lag window"
    )

    axes[0].set_ylabel(
        "Out-of-region Ridge RMSE (mm)"
    )

    axes[0].set_title(
        "Predictive benefit of antecedent rainfall"
    )

    axes[0].legend(
        frameon=True,
        fontsize=8,
    )

    depth_positions = np.arange(
        len(TARGET_ORDER)
    )

    decision_ordered = (
        decision.set_index("target")
        .loc[TARGET_ORDER]
    )

    axes[1].bar(
        depth_positions,
        decision_ordered[
            "full_window_improvement_percent"
        ],
        edgecolor="black",
        linewidth=0.4,
    )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_xticks(
        depth_positions
    )

    axes[1].set_xticklabels(
        [
            TARGETS[target]
            for target
            in TARGET_ORDER
        ]
    )

    axes[1].set_xlabel(
        "Soil-moisture depth"
    )

    axes[1].set_ylabel(
        "RMSE improvement from L0 to L0–L3 (%)"
    )

    axes[1].set_title(
        "Does older rainfall become more useful with depth?"
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Geographic lag-window ablation",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig26_lag_window_geographic_ablation.png",
        dpi=200,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig26_lag_window_geographic_ablation.pdf",
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

    data = (
        pd.read_parquet(
            DATA_PATH
        )
        .sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
    )

    if len(data) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(data)}."
        )

    prediction_rows = []
    fold_rows = []
    alpha_rows = []

    print("=" * 78)
    print("GEOGRAPHIC PRECIPITATION LAG-WINDOW ABLATION")
    print("=" * 78)

    for target in TARGET_ORDER:
        print(
            f"Target: {target} "
            f"({TARGETS[target]})"
        )

        for window, maximum_lag in (
            LAG_WINDOWS.items()
        ):
            features = [
                *precipitation_features(
                    maximum_lag
                ),
                *STATIC_FEATURES,
            ]

            for test_fold in range(
                FOLD_COUNT
            ):
                validation_fold = (
                    test_fold + 1
                ) % FOLD_COUNT

                train = data.loc[
                    (
                        data["spatial_fold_5"]
                        != test_fold
                    )
                    & (
                        data["spatial_fold_5"]
                        != validation_fold
                    )
                ].copy()

                validation = data.loc[
                    data["spatial_fold_5"]
                    == validation_fold
                ].copy()

                test = data.loc[
                    data["spatial_fold_5"]
                    == test_fold
                ].copy()

                selected_alpha, alpha_table = (
                    select_alpha(
                        train,
                        validation,
                        features,
                        target,
                    )
                )

                alpha_table.insert(
                    0,
                    "target",
                    target,
                )

                alpha_table.insert(
                    1,
                    "depth_label",
                    TARGETS[target],
                )

                alpha_table.insert(
                    2,
                    "lag_window",
                    window,
                )

                alpha_table.insert(
                    3,
                    "test_fold",
                    test_fold,
                )

                alpha_table.insert(
                    4,
                    "validation_fold",
                    validation_fold,
                )

                alpha_table[
                    "selected_alpha"
                ] = (
                    alpha_table["alpha"]
                    == selected_alpha
                )

                alpha_rows.append(
                    alpha_table
                )

                development = pd.concat(
                    [
                        train,
                        validation,
                    ],
                    ignore_index=True,
                )

                prediction = fit_and_predict(
                    development,
                    test,
                    features,
                    target,
                    selected_alpha,
                )

                truth = test[
                    target
                ].to_numpy(dtype=float)

                fold_metrics = metrics(
                    truth,
                    prediction,
                )

                fold_rows.append(
                    {
                        "target": target,
                        "depth_label": (
                            TARGETS[target]
                        ),
                        "lag_window": window,
                        "maximum_lag_months": (
                            maximum_lag
                        ),
                        "feature_count": (
                            len(features)
                        ),
                        "test_fold": (
                            test_fold
                        ),
                        "validation_fold": (
                            validation_fold
                        ),
                        "selected_alpha": (
                            selected_alpha
                        ),
                        "test_region_count": int(
                            test[
                                "region_id"
                            ].nunique()
                        ),
                        "test_sample_count": int(
                            len(test)
                        ),
                        "test_rmse_mm": (
                            fold_metrics[
                                "rmse_mm"
                            ]
                        ),
                        "test_mae_mm": (
                            fold_metrics[
                                "mae_mm"
                            ]
                        ),
                        "test_r_squared": (
                            fold_metrics[
                                "r_squared"
                            ]
                        ),
                    }
                )

                test = test.reset_index(
                    drop=True
                )

                for index, row in (
                    test.iterrows()
                ):
                    prediction_rows.append(
                        {
                            "target": target,
                            "depth_label": (
                                TARGETS[target]
                            ),
                            "lag_window": (
                                window
                            ),
                            "maximum_lag_months": (
                                maximum_lag
                            ),
                            "feature_count": (
                                len(features)
                            ),
                            "test_fold": (
                                test_fold
                            ),
                            "region_id": (
                                row["region_id"]
                            ),
                            "month": int(
                                row["month"]
                            ),
                            "native_cell_count": int(
                                row[
                                    "native_cell_count"
                                ]
                            ),
                            "core_region_ge_4_cells": bool(
                                row[
                                    "core_region_ge_4_cells"
                                ]
                            ),
                            "observed_mm": float(
                                truth[index]
                            ),
                            "predicted_mm": float(
                                prediction[index]
                            ),
                            "residual_mm": float(
                                truth[index]
                                - prediction[index]
                            ),
                        }
                    )

            print(
                f"  {window}: complete"
            )

    predictions = pd.DataFrame(
        prediction_rows
    )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    alpha_results = pd.concat(
        alpha_rows,
        ignore_index=True,
    )

    summary = calculate_summary(
        predictions
    )

    bootstrap = perform_bootstrap(
        predictions
    )

    decision = build_decision_table(
        summary,
        bootstrap,
    )

    predictions_path = (
        TABLE_DIR
        / "lag_window_geographic_predictions.csv.gz"
    )

    fold_path = (
        TABLE_DIR
        / "lag_window_geographic_fold_metrics.csv"
    )

    alpha_path = (
        TABLE_DIR
        / "lag_window_geographic_alpha_selection.csv"
    )

    summary_path = (
        TABLE_DIR
        / "lag_window_geographic_summary.csv"
    )

    bootstrap_path = (
        TABLE_DIR
        / "lag_window_geographic_block_bootstrap.csv"
    )

    decision_path = (
        TABLE_DIR
        / "lag_window_geographic_decision.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )

    fold_metrics.to_csv(
        fold_path,
        index=False,
    )

    alpha_results.to_csv(
        alpha_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    decision.to_csv(
        decision_path,
        index=False,
    )

    make_figure(
        summary,
        decision,
    )

    metadata = {
        "status": "PASS",
        "lag_windows": LAG_WINDOWS,
        "bootstrap_iterations": (
            BOOTSTRAP_ITERATIONS
        ),
        "bootstrap_unit": (
            "Complete regional unit with all 12 months"
        ),
        "outer_test_policy": (
            "Each geographic fold tested once"
        ),
        "alpha_selection_policy": (
            "Validation fold selects alpha; model then "
            "refitted on all non-test regions"
        ),
        "summary": str(
            summary_path
        ),
        "bootstrap": str(
            bootstrap_path
        ),
        "decision": str(
            decision_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "lag_window_geographic_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("LAG-WINDOW GEOGRAPHIC SUMMARY")
    print("=" * 78)

    print(
        summary[
            [
                "depth_label",
                "lag_window",
                "feature_count",
                "pooled_rmse_mm",
                "pooled_mae_mm",
                "pooled_r_squared",
                "core_pooled_rmse_mm",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Full L0-L3 versus L0:")

    print(
        decision.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(f"Summary    : {summary_path}")
    print(f"Bootstrap  : {bootstrap_path}")
    print(f"Decision   : {decision_path}")
    print(
        f"Figure     : "
        f"{FIGURE_DIR / 'fig26_lag_window_geographic_ablation.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
