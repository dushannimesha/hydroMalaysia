#!/usr/bin/env python3
"""
Grouped precipitation-mode ablation across soil-moisture depths.

Purpose
-------
Measure the predictive contribution of each complete NMF precipitation
mode without interpreting individual correlated lag coefficients.

Depth-specific lag windows
--------------------------
0-10 cm    : L0-L3
10-40 cm   : L0-L3
40-100 cm  : L0-L3
100-200 cm : L0-L1

These windows were fixed from the preceding geographic lag-window
analysis. This is a post-selection explanatory ablation, not a new
independent predictive benchmark.

Configurations
--------------
STATIC_ONLY
FULL
ONLY_C1 ... ONLY_C4
DROP_C1 ... DROP_C4

For every outer geographic test fold, the next fold is used to select
the Ridge alpha. The selected model is then refitted on all non-test
regions and evaluated on the held-out test fold.

Complete-region bootstrapping evaluates:
1. FULL versus STATIC_ONLY.
2. DROP_Ck versus FULL.
3. ONLY_Ck versus STATIC_ONLY.
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

MAXIMUM_LAG_BY_TARGET = {
    "SM_0_10cm_mm": 3,
    "SM_10_40cm_mm": 3,
    "SM_40_100cm_mm": 3,
    "SM_100_200cm_mm": 1,
}

COMPONENTS = [
    "C1",
    "C2",
    "C3",
    "C4",
]

COMPONENT_LABELS = {
    "C1": "January-centred eastern mode",
    "C2": "May–June southwestern mode",
    "C3": "April–October bimodal mode",
    "C4": "November-centred northern mode",
}

STATIC_FEATURES = [
    "PET_mm",
    "latitude",
    "longitude",
]

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


def component_features(
    component: str,
    maximum_lag: int,
) -> list[str]:
    component_number = component.replace(
        "C",
        "",
    )

    return [
        f"K4_C{component_number}_L{lag}_mm"
        for lag in range(maximum_lag + 1)
    ]


def configuration_features(
    configuration: str,
    maximum_lag: int,
) -> list[str]:
    if configuration == "STATIC_ONLY":
        return list(STATIC_FEATURES)

    if configuration == "FULL":
        precipitation = [
            feature
            for component in COMPONENTS
            for feature in component_features(
                component,
                maximum_lag,
            )
        ]

        return [
            *precipitation,
            *STATIC_FEATURES,
        ]

    if configuration.startswith(
        "ONLY_"
    ):
        component = configuration.replace(
            "ONLY_",
            "",
        )

        return [
            *component_features(
                component,
                maximum_lag,
            ),
            *STATIC_FEATURES,
        ]

    if configuration.startswith(
        "DROP_"
    ):
        dropped_component = (
            configuration.replace(
                "DROP_",
                "",
            )
        )

        precipitation = [
            feature
            for component in COMPONENTS
            if component
            != dropped_component
            for feature in component_features(
                component,
                maximum_lag,
            )
        ]

        return [
            *precipitation,
            *STATIC_FEATURES,
        ]

    raise ValueError(
        f"Unknown configuration: {configuration}"
    )


CONFIGURATIONS = [
    "STATIC_ONLY",
    "FULL",
    *[
        f"ONLY_{component}"
        for component in COMPONENTS
    ],
    *[
        f"DROP_{component}"
        for component in COMPONENTS
    ],
]


def regression_metrics(
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
    *,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    target: str,
    alpha: float,
) -> np.ndarray:
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train = feature_scaler.fit_transform(
        train[features].to_numpy(
            dtype=float
        )
    )

    y_train = (
        target_scaler.fit_transform(
            train[[target]].to_numpy(
                dtype=float
            )
        )
        .reshape(-1)
    )

    x_evaluation = (
        feature_scaler.transform(
            evaluation[features].to_numpy(
                dtype=float
            )
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
        model.predict(x_evaluation)
    )

    return (
        target_scaler.inverse_transform(
            standardized_prediction.reshape(
                -1,
                1,
            )
        )
        .reshape(-1)
    )


def select_alpha(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[float, pd.DataFrame]:
    validation_truth = validation[
        target
    ].to_numpy(dtype=float)

    rows = []
    best_alpha = None
    best_rmse = float("inf")

    for alpha in ALPHA_GRID:
        prediction = fit_and_predict(
            train=train,
            evaluation=validation,
            features=features,
            target=target,
            alpha=alpha,
        )

        validation_result = (
            regression_metrics(
                validation_truth,
                prediction,
            )
        )

        rows.append(
            {
                "alpha": alpha,
                "validation_rmse_mm": (
                    validation_result[
                        "rmse_mm"
                    ]
                ),
                "validation_r_squared": (
                    validation_result[
                        "r_squared"
                    ]
                ),
            }
        )

        if (
            validation_result["rmse_mm"]
            < best_rmse
        ):
            best_rmse = (
                validation_result[
                    "rmse_mm"
                ]
            )

            best_alpha = alpha

    if best_alpha is None:
        raise RuntimeError(
            "Alpha selection failed."
        )

    return (
        float(best_alpha),
        pd.DataFrame(rows),
    )


def build_pooled_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        target,
        configuration,
    ), group in predictions.groupby(
        [
            "target",
            "configuration",
        ],
        observed=True,
    ):
        all_result = regression_metrics(
            group["observed_mm"],
            group["predicted_mm"],
        )

        core = group.loc[
            group[
                "core_region_ge_4_cells"
            ]
        ]

        core_result = regression_metrics(
            core["observed_mm"],
            core["predicted_mm"],
        )

        rows.append(
            {
                "target": target,
                "depth_label": (
                    TARGETS[target]
                ),
                "maximum_lag_months": int(
                    group[
                        "maximum_lag_months"
                    ].iloc[0]
                ),
                "configuration": (
                    configuration
                ),
                "feature_count": int(
                    group[
                        "feature_count"
                    ].iloc[0]
                ),
                "pooled_rmse_mm": (
                    all_result["rmse_mm"]
                ),
                "pooled_mae_mm": (
                    all_result["mae_mm"]
                ),
                "pooled_r_squared": (
                    all_result["r_squared"]
                ),
                "pooled_bias_mm": (
                    all_result["bias_mm"]
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

    configuration_order = {
        configuration: index
        for index, configuration
        in enumerate(CONFIGURATIONS)
    }

    summary["_depth_order"] = (
        summary["target"].map(
            depth_order
        )
    )

    summary["_configuration_order"] = (
        summary["configuration"].map(
            configuration_order
        )
    )

    return (
        summary.sort_values(
            [
                "_depth_order",
                "_configuration_order",
            ]
        )
        .drop(
            columns=[
                "_depth_order",
                "_configuration_order",
            ]
        )
        .reset_index(drop=True)
    )


def paired_region_bootstrap(
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
        on=[
            "region_id",
            "month",
        ],
        how="inner",
        suffixes=(
            "_candidate",
            "_reference",
        ),
        validate="one_to_one",
    )

    target_difference = (
        paired[
            "observed_mm_candidate"
        ]
        - paired[
            "observed_mm_reference"
        ]
    ).abs()

    if float(
        target_difference.max()
    ) > 1e-8:
        raise RuntimeError(
            "Candidate and reference "
            "targets differ."
        )

    regions = sorted(
        paired[
            "region_id"
        ].unique()
    )

    if len(regions) != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} regions, "
            f"found {len(regions)}."
        )

    candidate_sse = []
    reference_sse = []
    observation_counts = []

    candidate_regional_rmse = []
    reference_regional_rmse = []

    for region_id in regions:
        region = paired.loc[
            paired["region_id"]
            == region_id
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

    candidate_regional_rmse = (
        np.asarray(
            candidate_regional_rmse,
            dtype=float,
        )
    )

    reference_regional_rmse = (
        np.asarray(
            reference_regional_rmse,
            dtype=float,
        )
    )

    sampled_indices = rng.integers(
        low=0,
        high=EXPECTED_REGIONS,
        size=(
            BOOTSTRAP_ITERATIONS,
            EXPECTED_REGIONS,
        ),
    )

    sampled_counts = (
        observation_counts[
            sampled_indices
        ].sum(axis=1)
    )

    candidate_rmse = np.sqrt(
        candidate_sse[
            sampled_indices
        ].sum(axis=1)
        / sampled_counts
    )

    reference_rmse = np.sqrt(
        reference_sse[
            sampled_indices
        ].sum(axis=1)
        / sampled_counts
    )

    differences = (
        candidate_rmse
        - reference_rmse
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
                differences,
                0.025,
            )
        ),
        "bootstrap_q50_difference_mm": float(
            np.quantile(
                differences,
                0.50,
            )
        ),
        "bootstrap_q975_difference_mm": float(
            np.quantile(
                differences,
                0.975,
            )
        ),
        "probability_candidate_lower_rmse": float(
            np.mean(
                differences < 0.0
            )
        ),
        "probability_candidate_higher_rmse": float(
            np.mean(
                differences > 0.0
            )
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


def run_bootstrap_comparisons(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rows = []

    for target in TARGET_ORDER:
        target_data = predictions.loc[
            predictions["target"]
            == target
        ]

        full = target_data.loc[
            target_data[
                "configuration"
            ]
            == "FULL"
        ]

        static = target_data.loc[
            target_data[
                "configuration"
            ]
            == "STATIC_ONLY"
        ]

        full_result = (
            paired_region_bootstrap(
                full,
                static,
                rng,
            )
        )

        rows.append(
            {
                "target": target,
                "depth_label": (
                    TARGETS[target]
                ),
                "component": "ALL",
                "component_label": (
                    "All precipitation modes"
                ),
                "comparison": (
                    "FULL_vs_STATIC_ONLY"
                ),
                "candidate_configuration": (
                    "FULL"
                ),
                "reference_configuration": (
                    "STATIC_ONLY"
                ),
                **full_result,
            }
        )

        for component in COMPONENTS:
            dropped = target_data.loc[
                target_data[
                    "configuration"
                ]
                == f"DROP_{component}"
            ]

            only = target_data.loc[
                target_data[
                    "configuration"
                ]
                == f"ONLY_{component}"
            ]

            drop_result = (
                paired_region_bootstrap(
                    dropped,
                    full,
                    rng,
                )
            )

            only_result = (
                paired_region_bootstrap(
                    only,
                    static,
                    rng,
                )
            )

            rows.append(
                {
                    "target": target,
                    "depth_label": (
                        TARGETS[target]
                    ),
                    "component": component,
                    "component_label": (
                        COMPONENT_LABELS[
                            component
                        ]
                    ),
                    "comparison": (
                        "DROP_component_vs_FULL"
                    ),
                    "candidate_configuration": (
                        f"DROP_{component}"
                    ),
                    "reference_configuration": (
                        "FULL"
                    ),
                    **drop_result,
                }
            )

            rows.append(
                {
                    "target": target,
                    "depth_label": (
                        TARGETS[target]
                    ),
                    "component": component,
                    "component_label": (
                        COMPONENT_LABELS[
                            component
                        ]
                    ),
                    "comparison": (
                        "ONLY_component_vs_STATIC"
                    ),
                    "candidate_configuration": (
                        f"ONLY_{component}"
                    ),
                    "reference_configuration": (
                        "STATIC_ONLY"
                    ),
                    **only_result,
                }
            )

    return pd.DataFrame(rows)


def build_mode_importance_table(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for target in TARGET_ORDER:
        target_summary = (
            summary.loc[
                summary["target"] == target
            ]
            .set_index("configuration")
        )

        full_rmse = float(
            target_summary.loc[
                "FULL",
                "pooled_rmse_mm",
            ]
        )

        static_rmse = float(
            target_summary.loc[
                "STATIC_ONLY",
                "pooled_rmse_mm",
            ]
        )

        for component in COMPONENTS:
            dropped_rmse = float(
                target_summary.loc[
                    f"DROP_{component}",
                    "pooled_rmse_mm",
                ]
            )

            only_rmse = float(
                target_summary.loc[
                    f"ONLY_{component}",
                    "pooled_rmse_mm",
                ]
            )

            drop_bootstrap = (
                bootstrap.loc[
                    (
                        bootstrap["target"]
                        == target
                    )
                    & (
                        bootstrap[
                            "component"
                        ]
                        == component
                    )
                    & (
                        bootstrap[
                            "comparison"
                        ]
                        == (
                            "DROP_component_"
                            "vs_FULL"
                        )
                    )
                ]
            )

            only_bootstrap = (
                bootstrap.loc[
                    (
                        bootstrap["target"]
                        == target
                    )
                    & (
                        bootstrap[
                            "component"
                        ]
                        == component
                    )
                    & (
                        bootstrap[
                            "comparison"
                        ]
                        == (
                            "ONLY_component_"
                            "vs_STATIC"
                        )
                    )
                ]
            )

            if (
                len(drop_bootstrap) != 1
                or len(only_bootstrap) != 1
            ):
                raise RuntimeError(
                    f"Missing bootstrap result "
                    f"for {target}, {component}."
                )

            drop_bootstrap = (
                drop_bootstrap.iloc[0]
            )

            only_bootstrap = (
                only_bootstrap.iloc[0]
            )

            rows.append(
                {
                    "target": target,
                    "depth_label": (
                        TARGETS[target]
                    ),
                    "maximum_lag_months": (
                        MAXIMUM_LAG_BY_TARGET[
                            target
                        ]
                    ),
                    "component": component,
                    "component_label": (
                        COMPONENT_LABELS[
                            component
                        ]
                    ),
                    "full_rmse_mm": (
                        full_rmse
                    ),
                    "static_only_rmse_mm": (
                        static_rmse
                    ),
                    "drop_component_rmse_mm": (
                        dropped_rmse
                    ),
                    "rmse_increase_when_dropped_mm": (
                        dropped_rmse
                        - full_rmse
                    ),
                    "rmse_increase_when_dropped_percent": (
                        100.0
                        * (
                            dropped_rmse
                            - full_rmse
                        )
                        / full_rmse
                    ),
                    "drop_penalty_q025_mm": float(
                        drop_bootstrap[
                            "bootstrap_q025_difference_mm"
                        ]
                    ),
                    "drop_penalty_q975_mm": float(
                        drop_bootstrap[
                            "bootstrap_q975_difference_mm"
                        ]
                    ),
                    "probability_removal_worsens_rmse": float(
                        drop_bootstrap[
                            "probability_candidate_higher_rmse"
                        ]
                    ),
                    "full_region_win_count_vs_drop": int(
                        drop_bootstrap[
                            "reference_region_win_count"
                        ]
                    ),
                    "drop_region_win_count_vs_full": int(
                        drop_bootstrap[
                            "candidate_region_win_count"
                        ]
                    ),
                    "component_only_rmse_mm": (
                        only_rmse
                    ),
                    "component_only_improvement_over_static_mm": (
                        static_rmse
                        - only_rmse
                    ),
                    "component_only_improvement_percent": (
                        100.0
                        * (
                            static_rmse
                            - only_rmse
                        )
                        / static_rmse
                    ),
                    "probability_component_only_beats_static": float(
                        only_bootstrap[
                            "probability_candidate_lower_rmse"
                        ]
                    ),
                }
            )

    importance = pd.DataFrame(rows)

    importance[
        "importance_rank_by_drop_penalty"
    ] = (
        importance.groupby(
            "target",
            observed=True,
        )[
            "rmse_increase_when_dropped_mm"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    depth_order = {
        target: index
        for index, target
        in enumerate(TARGET_ORDER)
    }

    component_order = {
        component: index
        for index, component
        in enumerate(COMPONENTS)
    }

    importance["_depth_order"] = (
        importance["target"].map(
            depth_order
        )
    )

    importance["_component_order"] = (
        importance["component"].map(
            component_order
        )
    )

    return (
        importance.sort_values(
            [
                "_depth_order",
                "_component_order",
            ]
        )
        .drop(
            columns=[
                "_depth_order",
                "_component_order",
            ]
        )
        .reset_index(drop=True)
    )


def make_importance_figure(
    importance: pd.DataFrame,
) -> None:
    matrix = np.empty(
        (
            len(TARGET_ORDER),
            len(COMPONENTS),
        ),
        dtype=float,
    )

    significant = np.zeros_like(
        matrix,
        dtype=bool,
    )

    for depth_index, target in enumerate(
        TARGET_ORDER
    ):
        target_table = (
            importance.loc[
                importance["target"]
                == target
            ]
            .set_index("component")
            .loc[COMPONENTS]
        )

        matrix[depth_index] = (
            target_table[
                "rmse_increase_when_dropped_percent"
            ].to_numpy(dtype=float)
        )

        significant[depth_index] = (
            target_table[
                "drop_penalty_q025_mm"
            ].to_numpy(dtype=float)
            > 0.0
        )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.8, 5.2),
        constrained_layout=True,
    )

    maximum = float(
        np.max(
            np.abs(matrix)
        )
    )

    image = axes[0].imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-maximum,
        vmax=maximum,
    )

    axes[0].set_xticks(
        range(len(COMPONENTS))
    )

    axes[0].set_xticklabels(
        COMPONENTS
    )

    axes[0].set_yticks(
        range(len(TARGET_ORDER))
    )

    axes[0].set_yticklabels(
        [
            TARGETS[target]
            for target in TARGET_ORDER
        ]
    )

    axes[0].set_xlabel(
        "Removed precipitation mode"
    )

    axes[0].set_ylabel(
        "Soil-moisture depth"
    )

    axes[0].set_title(
        "RMSE increase after removing a mode"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            marker = (
                "*"
                if significant[
                    row,
                    column,
                ]
                else ""
            )

            axes[0].text(
                column,
                row,
                f"{matrix[row, column]:.1f}%{marker}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = figure.colorbar(
        image,
        ax=axes[0],
    )

    colorbar.set_label(
        "Change in pooled RMSE (%)"
    )

    positions = np.arange(
        len(TARGET_ORDER)
    )

    width = 0.18

    for component_index, component in enumerate(
        COMPONENTS
    ):
        component_table = (
            importance.loc[
                importance["component"]
                == component
            ]
            .set_index("target")
            .loc[TARGET_ORDER]
        )

        offset = (
            component_index
            - (len(COMPONENTS) - 1) / 2
        ) * width

        axes[1].bar(
            positions + offset,
            component_table[
                "component_only_improvement_percent"
            ],
            width=width,
            label=component,
        )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_xticks(
        positions
    )

    axes[1].set_xticklabels(
        [
            TARGETS[target]
            for target in TARGET_ORDER
        ]
    )

    axes[1].set_xlabel(
        "Soil-moisture depth"
    )

    axes[1].set_ylabel(
        "Improvement over static-only model (%)"
    )

    axes[1].set_title(
        "Standalone predictive value of each mode"
    )

    axes[1].legend(
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
        "Grouped seasonal precipitation-mode importance",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig27_grouped_precipitation_mode_ablation.png",
        dpi=200,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig27_grouped_precipitation_mode_ablation.pdf",
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
            [
                "region_id",
                "month",
            ]
        )
        .reset_index(drop=True)
    )

    if len(data) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(data)}."
        )

    fold_rows = []
    prediction_rows = []
    alpha_rows = []

    print("=" * 78)
    print("GROUPED PRECIPITATION-MODE GEOGRAPHIC ABLATION")
    print("=" * 78)

    for target in TARGET_ORDER:
        maximum_lag = (
            MAXIMUM_LAG_BY_TARGET[
                target
            ]
        )

        print(
            f"Target: {target} "
            f"({TARGETS[target]}), "
            f"maximum lag={maximum_lag}"
        )

        for configuration in CONFIGURATIONS:
            features = (
                configuration_features(
                    configuration,
                    maximum_lag,
                )
            )

            for test_fold in range(
                FOLD_COUNT
            ):
                validation_fold = (
                    test_fold + 1
                ) % FOLD_COUNT

                train = data.loc[
                    (
                        data[
                            "spatial_fold_5"
                        ]
                        != test_fold
                    )
                    & (
                        data[
                            "spatial_fold_5"
                        ]
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

                (
                    selected_alpha,
                    alpha_table,
                ) = select_alpha(
                    train=train,
                    validation=validation,
                    features=features,
                    target=target,
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
                    "configuration",
                    configuration,
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
                    train=development,
                    evaluation=test,
                    features=features,
                    target=target,
                    alpha=selected_alpha,
                )

                truth = test[
                    target
                ].to_numpy(dtype=float)

                test_result = (
                    regression_metrics(
                        truth,
                        prediction,
                    )
                )

                fold_rows.append(
                    {
                        "target": target,
                        "depth_label": (
                            TARGETS[target]
                        ),
                        "maximum_lag_months": (
                            maximum_lag
                        ),
                        "configuration": (
                            configuration
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
                            test_result["rmse_mm"]
                        ),
                        "test_mae_mm": (
                            test_result["mae_mm"]
                        ),
                        "test_r_squared": (
                            test_result[
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
                            "maximum_lag_months": (
                                maximum_lag
                            ),
                            "configuration": (
                                configuration
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
                f"  {configuration}: complete"
            )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    alpha_results = pd.concat(
        alpha_rows,
        ignore_index=True,
    )

    summary = build_pooled_summary(
        predictions
    )

    bootstrap = (
        run_bootstrap_comparisons(
            predictions
        )
    )

    importance = (
        build_mode_importance_table(
            summary,
            bootstrap,
        )
    )

    fold_path = (
        TABLE_DIR
        / "grouped_mode_ablation_fold_metrics.csv"
    )

    prediction_path = (
        TABLE_DIR
        / "grouped_mode_ablation_predictions.csv.gz"
    )

    alpha_path = (
        TABLE_DIR
        / "grouped_mode_ablation_alpha_selection.csv"
    )

    summary_path = (
        TABLE_DIR
        / "grouped_mode_ablation_summary.csv"
    )

    bootstrap_path = (
        TABLE_DIR
        / "grouped_mode_ablation_block_bootstrap.csv"
    )

    importance_path = (
        TABLE_DIR
        / "grouped_mode_importance_by_depth.csv"
    )

    fold_metrics.to_csv(
        fold_path,
        index=False,
    )

    predictions.to_csv(
        prediction_path,
        index=False,
        compression="gzip",
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

    importance.to_csv(
        importance_path,
        index=False,
    )

    make_importance_figure(
        importance
    )

    metadata = {
        "status": "PASS",
        "depth_specific_maximum_lags": (
            MAXIMUM_LAG_BY_TARGET
        ),
        "component_labels_are_descriptive": (
            True
        ),
        "causal_mode_attribution_supported": (
            False
        ),
        "bootstrap_iterations": (
            BOOTSTRAP_ITERATIONS
        ),
        "bootstrap_unit": (
            "Complete regional unit with all 12 months"
        ),
        "interpretation": (
            "A positive and bootstrap-supported RMSE "
            "increase after dropping a complete mode "
            "indicates that the grouped mode contributes "
            "predictive information beyond the remaining "
            "modes and static predictors."
        ),
        "summary": str(
            summary_path
        ),
        "bootstrap": str(
            bootstrap_path
        ),
        "importance_table": str(
            importance_path
        ),
        "predictions": str(
            prediction_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "grouped_mode_ablation_metadata.json"
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
    print("GROUPED MODE-IMPORTANCE SUMMARY")
    print("=" * 78)

    display_columns = [
        "depth_label",
        "component",
        "rmse_increase_when_dropped_mm",
        "rmse_increase_when_dropped_percent",
        "drop_penalty_q025_mm",
        "drop_penalty_q975_mm",
        "probability_removal_worsens_rmse",
        "component_only_improvement_percent",
        "probability_component_only_beats_static",
        "importance_rank_by_drop_penalty",
    ]

    print(
        importance[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(f"Summary          : {summary_path}")
    print(f"Bootstrap        : {bootstrap_path}")
    print(f"Importance table : {importance_path}")
    print(
        f"Figure           : "
        f"{FIGURE_DIR / 'fig27_grouped_precipitation_mode_ablation.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
