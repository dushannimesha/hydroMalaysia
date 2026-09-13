#!/usr/bin/env python3
"""
Common Ridge attribution analysis across four soil-moisture depths.

Purpose
-------
Prediction models differ by depth, but their native attribution measures
are not directly comparable. This stage therefore fits a common,
standardized Ridge attribution model to all four layers.

Important distinction
---------------------
The full-data Ridge fits in this script are used for interpretation only.
Predictive performance remains defined by the preceding geographic
cross-validation experiments.

Outputs
-------
1. Geographic-CV selection of the Ridge alpha for every depth.
2. Standardized component-lag coefficients.
3. Complete-region bootstrap uncertainty intervals.
4. Component-specific rainfall-memory indices.
5. Multicollinearity diagnostics.
6. Component × lag × depth attribution heatmap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
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

COMPONENTS = [
    "C1",
    "C2",
    "C3",
    "C4",
]

COMPONENT_LABELS = {
    "C1": "January eastern",
    "C2": "May–June southwest",
    "C3": "April–October",
    "C4": "November northern",
}

PRECIPITATION_FEATURES = [
    f"K4_C{component}_L{lag}_mm"
    for component in range(1, 5)
    for lag in range(4)
]

FEATURE_COLUMNS = [
    *PRECIPITATION_FEATURES,
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
    100.0,
]

SPATIAL_FOLD_COUNT = 5

BOOTSTRAP_ITERATIONS = 5_000
BOOTSTRAP_SEED = 20260806

EXPECTED_ROWS = 396
EXPECTED_REGIONS = 33
EXPECTED_MONTHS_PER_REGION = 12

FEATURE_PATTERN = re.compile(
    r"^K4_(C[1-4])_L([0-3])_mm$"
)


def rmse(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            mean_squared_error(
                truth,
                prediction,
            )
        )
    )


def select_ridge_alpha(
    data: pd.DataFrame,
    target: str,
) -> tuple[
    float,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Select a stable Ridge alpha using five geographic folds.

    One-standard-error rule:
    choose the largest alpha whose mean geographic validation RMSE is
    no larger than the best mean RMSE plus the standard error of the
    best-performing alpha.
    """
    fold_rows = []

    for alpha in ALPHA_GRID:
        for validation_fold in range(
            SPATIAL_FOLD_COUNT
        ):
            train = data.loc[
                data["spatial_fold_5"]
                != validation_fold
            ]

            validation = data.loc[
                data["spatial_fold_5"]
                == validation_fold
            ]

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            x_train = x_scaler.fit_transform(
                train[FEATURE_COLUMNS].to_numpy(
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

            x_validation = x_scaler.transform(
                validation[
                    FEATURE_COLUMNS
                ].to_numpy(dtype=float)
            )

            model = Ridge(
                alpha=alpha,
                fit_intercept=True,
            )

            model.fit(
                x_train,
                y_train,
            )

            prediction_standardized = (
                model.predict(
                    x_validation
                )
            )

            prediction = (
                y_scaler.inverse_transform(
                    prediction_standardized.reshape(
                        -1,
                        1,
                    )
                )
                .reshape(-1)
            )

            validation_truth = validation[
                target
            ].to_numpy(dtype=float)

            fold_rows.append(
                {
                    "target": target,
                    "depth_label": (
                        TARGETS[target]
                    ),
                    "alpha": alpha,
                    "validation_fold": (
                        validation_fold
                    ),
                    "validation_region_count": int(
                        validation[
                            "region_id"
                        ].nunique()
                    ),
                    "validation_sample_count": int(
                        len(validation)
                    ),
                    "validation_rmse_mm": rmse(
                        validation_truth,
                        prediction,
                    ),
                }
            )

    fold_table = pd.DataFrame(
        fold_rows
    )

    alpha_summary = (
        fold_table.groupby(
            [
                "target",
                "depth_label",
                "alpha",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_validation_rmse_mm=(
                "validation_rmse_mm",
                "mean",
            ),
            median_validation_rmse_mm=(
                "validation_rmse_mm",
                "median",
            ),
            std_validation_rmse_mm=(
                "validation_rmse_mm",
                "std",
            ),
            minimum_validation_rmse_mm=(
                "validation_rmse_mm",
                "min",
            ),
            maximum_validation_rmse_mm=(
                "validation_rmse_mm",
                "max",
            ),
        )
    )

    alpha_summary[
        "standard_error_validation_rmse_mm"
    ] = (
        alpha_summary[
            "std_validation_rmse_mm"
        ]
        / np.sqrt(SPATIAL_FOLD_COUNT)
    )

    best_index = alpha_summary[
        "mean_validation_rmse_mm"
    ].idxmin()

    best_row = alpha_summary.loc[
        best_index
    ]

    one_standard_error_threshold = float(
        best_row[
            "mean_validation_rmse_mm"
        ]
        + best_row[
            "standard_error_validation_rmse_mm"
        ]
    )

    eligible = alpha_summary.loc[
        alpha_summary[
            "mean_validation_rmse_mm"
        ]
        <= one_standard_error_threshold
    ]

    selected_alpha = float(
        eligible["alpha"].max()
    )

    alpha_summary[
        "best_mean_rmse_alpha"
    ] = (
        alpha_summary["alpha"]
        == float(best_row["alpha"])
    )

    alpha_summary[
        "within_one_standard_error"
    ] = (
        alpha_summary[
            "mean_validation_rmse_mm"
        ]
        <= one_standard_error_threshold
    )

    alpha_summary[
        "selected_alpha"
    ] = (
        alpha_summary["alpha"]
        == selected_alpha
    )

    return (
        selected_alpha,
        fold_table,
        alpha_summary,
    )


def fit_standardized_ridge(
    data: pd.DataFrame,
    target: str,
    alpha: float,
) -> tuple[
    np.ndarray,
    StandardScaler,
    StandardScaler,
    np.ndarray,
    np.ndarray,
]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_standardized = x_scaler.fit_transform(
        data[FEATURE_COLUMNS].to_numpy(
            dtype=float
        )
    )

    y_standardized = (
        y_scaler.fit_transform(
            data[[target]].to_numpy(
                dtype=float
            )
        )
        .reshape(-1)
    )

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
    )

    model.fit(
        x_standardized,
        y_standardized,
    )

    coefficients = np.asarray(
        model.coef_,
        dtype=float,
    ).reshape(-1)

    return (
        coefficients,
        x_scaler,
        y_scaler,
        x_standardized,
        y_standardized,
    )


def bootstrap_coefficients(
    *,
    x_standardized: np.ndarray,
    y_standardized: np.ndarray,
    region_ids: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_regions = sorted(
        np.unique(region_ids).tolist()
    )

    if len(unique_regions) != EXPECTED_REGIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} regions, "
            f"found {len(unique_regions)}."
        )

    region_row_indices = []

    for region_id in unique_regions:
        indices = np.flatnonzero(
            region_ids == region_id
        )

        if len(indices) != (
            EXPECTED_MONTHS_PER_REGION
        ):
            raise RuntimeError(
                f"{region_id} does not have "
                f"{EXPECTED_MONTHS_PER_REGION} months."
            )

        region_row_indices.append(
            indices
        )

    feature_count = (
        x_standardized.shape[1]
    )

    identity = np.eye(
        feature_count,
        dtype=float,
    )

    results = np.empty(
        (
            BOOTSTRAP_ITERATIONS,
            feature_count,
        ),
        dtype=np.float32,
    )

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):
        sampled_region_indices = (
            rng.integers(
                low=0,
                high=EXPECTED_REGIONS,
                size=EXPECTED_REGIONS,
            )
        )

        sampled_rows = np.concatenate(
            [
                region_row_indices[index]
                for index
                in sampled_region_indices
            ]
        )

        x_sample = x_standardized[
            sampled_rows
        ]

        y_sample = y_standardized[
            sampled_rows
        ]

        # Ridge with an intercept is equivalent to centring each
        # bootstrap sample before solving the penalized system.
        x_centered = (
            x_sample
            - x_sample.mean(
                axis=0,
                keepdims=True,
            )
        )

        y_centered = (
            y_sample
            - y_sample.mean()
        )

        system_matrix = (
            x_centered.T
            @ x_centered
            + alpha * identity
        )

        right_hand_side = (
            x_centered.T
            @ y_centered
        )

        try:
            coefficient = np.linalg.solve(
                system_matrix,
                right_hand_side,
            )

        except np.linalg.LinAlgError:
            coefficient = (
                np.linalg.pinv(
                    system_matrix
                )
                @ right_hand_side
            )

        results[iteration] = (
            coefficient.astype(
                np.float32
            )
        )

    return results


def build_coefficient_table(
    *,
    target: str,
    selected_alpha: float,
    full_coefficients: np.ndarray,
    bootstrap_coefficients_array: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for feature_index, feature in enumerate(
        FEATURE_COLUMNS
    ):
        samples = (
            bootstrap_coefficients_array[
                :,
                feature_index,
            ].astype(float)
        )

        match = FEATURE_PATTERN.match(
            feature
        )

        if match:
            component = match.group(1)
            lag = int(match.group(2))
            feature_type = (
                "precipitation_component"
            )

        else:
            component = None
            lag = np.nan
            feature_type = (
                "atmospheric"
                if feature == "PET_mm"
                else "spatial"
            )

        q025 = float(
            np.quantile(
                samples,
                0.025,
            )
        )

        q50 = float(
            np.quantile(
                samples,
                0.50,
            )
        )

        q975 = float(
            np.quantile(
                samples,
                0.975,
            )
        )

        rows.append(
            {
                "target": target,
                "depth_label": (
                    TARGETS[target]
                ),
                "selected_alpha": (
                    selected_alpha
                ),
                "feature": feature,
                "feature_type": (
                    feature_type
                ),
                "component": component,
                "component_label": (
                    COMPONENT_LABELS.get(
                        component,
                        "",
                    )
                ),
                "lag_months": lag,
                "standardized_coefficient": float(
                    full_coefficients[
                        feature_index
                    ]
                ),
                "bootstrap_mean_coefficient": float(
                    samples.mean()
                ),
                "bootstrap_median_coefficient": (
                    q50
                ),
                "bootstrap_q025_coefficient": (
                    q025
                ),
                "bootstrap_q975_coefficient": (
                    q975
                ),
                "probability_positive": float(
                    np.mean(samples > 0.0)
                ),
                "probability_negative": float(
                    np.mean(samples < 0.0)
                ),
                "stable_positive_95": bool(
                    q025 > 0.0
                ),
                "stable_negative_95": bool(
                    q975 < 0.0
                ),
                "absolute_standardized_coefficient": float(
                    abs(
                        full_coefficients[
                            feature_index
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_multicollinearity(
    x_standardized: np.ndarray,
) -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
    correlation_matrix = np.corrcoef(
        x_standardized,
        rowvar=False,
    )

    inverse_correlation = np.linalg.pinv(
        correlation_matrix
    )

    vif_values = np.diag(
        inverse_correlation
    )

    vif_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "variance_inflation_factor": (
                vif_values
            ),
        }
    ).sort_values(
        "variance_inflation_factor",
        ascending=False,
    )

    diagnostics = {
        "feature_correlation_condition_number": float(
            np.linalg.cond(
                correlation_matrix
            )
        ),
        "maximum_absolute_pairwise_correlation": float(
            np.max(
                np.abs(
                    correlation_matrix[
                        np.triu_indices(
                            len(FEATURE_COLUMNS),
                            k=1,
                        )
                    ]
                )
            )
        ),
        "maximum_variance_inflation_factor": float(
            np.max(vif_values)
        ),
        "median_variance_inflation_factor": float(
            np.median(vif_values)
        ),
    }

    return vif_table, diagnostics


def build_memory_table(
    coefficient_table: pd.DataFrame,
) -> pd.DataFrame:
    precipitation = coefficient_table.loc[
        coefficient_table[
            "feature_type"
        ]
        == "precipitation_component"
    ].copy()

    rows = []

    for (
        target,
        component,
    ), group in precipitation.groupby(
        ["target", "component"],
        observed=True,
    ):
        group = group.sort_values(
            "lag_months"
        )

        lags = group[
            "lag_months"
        ].to_numpy(dtype=float)

        coefficients = group[
            "standardized_coefficient"
        ].to_numpy(dtype=float)

        absolute_coefficients = np.abs(
            coefficients
        )

        if absolute_coefficients.sum() > 0:
            absolute_memory_index = float(
                np.sum(
                    lags
                    * absolute_coefficients
                )
                / absolute_coefficients.sum()
            )

        else:
            absolute_memory_index = (
                float("nan")
            )

        positive_coefficients = np.maximum(
            coefficients,
            0.0,
        )

        if positive_coefficients.sum() > 0:
            positive_memory_index = float(
                np.sum(
                    lags
                    * positive_coefficients
                )
                / positive_coefficients.sum()
            )

        else:
            positive_memory_index = (
                float("nan")
            )

        dominant_index = int(
            np.argmax(
                absolute_coefficients
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
                "dominant_lag_by_absolute_effect": int(
                    lags[dominant_index]
                ),
                "dominant_standardized_coefficient": float(
                    coefficients[
                        dominant_index
                    ]
                ),
                "absolute_effect_memory_index_months": (
                    absolute_memory_index
                ),
                "positive_effect_memory_index_months": (
                    positive_memory_index
                ),
                "net_standardized_effect_across_lags": float(
                    coefficients.sum()
                ),
                "total_absolute_effect_across_lags": float(
                    absolute_coefficients.sum()
                ),
                "stable_positive_lag_count": int(
                    group[
                        "stable_positive_95"
                    ].sum()
                ),
                "stable_negative_lag_count": int(
                    group[
                        "stable_negative_95"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def make_coefficient_heatmap(
    coefficient_table: pd.DataFrame,
) -> None:
    precipitation = coefficient_table.loc[
        coefficient_table[
            "feature_type"
        ]
        == "precipitation_component"
    ].copy()

    feature_order = (
        PRECIPITATION_FEATURES
    )

    matrix = np.empty(
        (
            len(TARGET_ORDER),
            len(feature_order),
        ),
        dtype=float,
    )

    significance = np.zeros_like(
        matrix,
        dtype=bool,
    )

    for target_index, target in enumerate(
        TARGET_ORDER
    ):
        target_table = (
            precipitation.loc[
                precipitation["target"]
                == target
            ]
            .set_index("feature")
            .loc[feature_order]
        )

        matrix[target_index] = (
            target_table[
                "standardized_coefficient"
            ].to_numpy(dtype=float)
        )

        significance[target_index] = (
            target_table[
                "stable_positive_95"
            ].to_numpy(dtype=bool)
            | target_table[
                "stable_negative_95"
            ].to_numpy(dtype=bool)
        )

    maximum_absolute_value = float(
        np.max(
            np.abs(matrix)
        )
    )

    figure, axis = plt.subplots(
        figsize=(14.5, 5.3),
        constrained_layout=True,
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-maximum_absolute_value,
        vmax=maximum_absolute_value,
    )

    x_labels = [
        feature.replace(
            "K4_",
            "",
        ).replace(
            "_mm",
            "",
        ).replace(
            "_",
            "–",
        )
        for feature
        in feature_order
    ]

    axis.set_xticks(
        range(len(feature_order))
    )

    axis.set_xticklabels(
        x_labels,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        range(len(TARGET_ORDER))
    )

    axis.set_yticklabels(
        [
            TARGETS[target]
            for target
            in TARGET_ORDER
        ]
    )

    axis.set_xlabel(
        "Precipitation mode and antecedent lag"
    )

    axis.set_ylabel(
        "Soil-moisture depth"
    )

    axis.set_title(
        "Standardized Ridge attribution across "
        "rainfall modes, lags and soil depths"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            marker = (
                "*"
                if significance[row, column]
                else ""
            )

            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}{marker}",
                ha="center",
                va="center",
                fontsize=7.5,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Standardized conditional coefficient"
    )

    figure.text(
        0.01,
        0.01,
        "* complete-region bootstrap 95% interval excludes zero",
        fontsize=8,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig23_component_lag_depth_attribution.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig23_component_lag_depth_attribution.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def make_memory_figure(
    memory_table: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9.2, 5.4),
        constrained_layout=True,
    )

    positions = np.arange(
        len(TARGET_ORDER)
    )

    for component in COMPONENTS:
        component_table = (
            memory_table.loc[
                memory_table["component"]
                == component
            ]
            .set_index("target")
            .loc[TARGET_ORDER]
        )

        axis.plot(
            positions,
            component_table[
                "absolute_effect_memory_index_months"
            ],
            marker="o",
            linewidth=1.5,
            label=(
                f"{component}: "
                f"{COMPONENT_LABELS[component]}"
            ),
        )

    axis.set_xticks(positions)

    axis.set_xticklabels(
        [
            TARGETS[target]
            for target
            in TARGET_ORDER
        ]
    )

    axis.set_ylim(-0.1, 3.1)

    axis.set_xlabel(
        "Soil-moisture depth"
    )

    axis.set_ylabel(
        "Magnitude-weighted rainfall-memory index (months)"
    )

    axis.set_title(
        "Does modelled rainfall memory shift with soil depth?"
    )

    axis.grid(
        linewidth=0.35,
        linestyle=":",
        alpha=0.6,
    )

    axis.legend(
        frameon=True,
        fontsize=8,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig24_component_memory_by_depth.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig24_component_memory_by_depth.pdf",
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

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            DATA_PATH
        )

    data = pd.read_parquet(
        DATA_PATH
    ).sort_values(
        ["region_id", "month"]
    ).reset_index(drop=True)

    if len(data) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(data)}."
        )

    if data[
        FEATURE_COLUMNS + TARGET_ORDER
    ].isna().any().any():
        raise RuntimeError(
            "Attribution features or targets "
            "contain missing values."
        )

    print("=" * 78)
    print("COMPONENT × LAG × DEPTH RIDGE ATTRIBUTION")
    print("=" * 78)
    print(f"Rows                    : {len(data)}")
    print(
        f"Regional units          : "
        f"{data['region_id'].nunique()}"
    )
    print(
        f"Bootstrap iterations    : "
        f"{BOOTSTRAP_ITERATIONS}"
    )
    print()

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    all_fold_tables = []
    all_alpha_summaries = []
    all_coefficient_tables = []

    multicollinearity_table = None
    multicollinearity_summary = None

    selected_alphas = {}

    for target in TARGET_ORDER:
        print(
            f"Target: {target} "
            f"({TARGETS[target]})"
        )

        (
            selected_alpha,
            fold_table,
            alpha_summary,
        ) = select_ridge_alpha(
            data,
            target,
        )

        selected_alphas[target] = (
            selected_alpha
        )

        all_fold_tables.append(
            fold_table
        )

        all_alpha_summaries.append(
            alpha_summary
        )

        (
            full_coefficients,
            x_scaler,
            y_scaler,
            x_standardized,
            y_standardized,
        ) = fit_standardized_ridge(
            data,
            target,
            selected_alpha,
        )

        if multicollinearity_table is None:
            (
                multicollinearity_table,
                multicollinearity_summary,
            ) = calculate_multicollinearity(
                x_standardized
            )

        bootstrap_array = (
            bootstrap_coefficients(
                x_standardized=(
                    x_standardized
                ),
                y_standardized=(
                    y_standardized
                ),
                region_ids=data[
                    "region_id"
                ].to_numpy(),
                alpha=selected_alpha,
                rng=rng,
            )
        )

        coefficient_table = (
            build_coefficient_table(
                target=target,
                selected_alpha=(
                    selected_alpha
                ),
                full_coefficients=(
                    full_coefficients
                ),
                bootstrap_coefficients_array=(
                    bootstrap_array
                ),
            )
        )

        all_coefficient_tables.append(
            coefficient_table
        )

        print(
            f"  selected alpha: "
            f"{selected_alpha:g}"
        )

    fold_table = pd.concat(
        all_fold_tables,
        ignore_index=True,
    )

    alpha_summary = pd.concat(
        all_alpha_summaries,
        ignore_index=True,
    )

    coefficient_table = pd.concat(
        all_coefficient_tables,
        ignore_index=True,
    )

    memory_table = build_memory_table(
        coefficient_table
    )

    fold_path = (
        TABLE_DIR
        / "ridge_attribution_alpha_cv_folds.csv"
    )

    alpha_summary_path = (
        TABLE_DIR
        / "ridge_attribution_alpha_selection.csv"
    )

    coefficient_path = (
        TABLE_DIR
        / "component_lag_depth_standardized_coefficients.csv"
    )

    memory_path = (
        TABLE_DIR
        / "component_lag_depth_memory_indices.csv"
    )

    vif_path = (
        TABLE_DIR
        / "attribution_feature_multicollinearity.csv"
    )

    fold_table.to_csv(
        fold_path,
        index=False,
    )

    alpha_summary.to_csv(
        alpha_summary_path,
        index=False,
    )

    coefficient_table.to_csv(
        coefficient_path,
        index=False,
    )

    memory_table.to_csv(
        memory_path,
        index=False,
    )

    multicollinearity_table.to_csv(
        vif_path,
        index=False,
    )

    make_coefficient_heatmap(
        coefficient_table
    )

    make_memory_figure(
        memory_table
    )

    precipitation_coefficients = (
        coefficient_table.loc[
            coefficient_table[
                "feature_type"
            ]
            == "precipitation_component"
        ]
    )

    metadata = {
        "status": "PASS",
        "interpretation_model": (
            "Common standardized Ridge model "
            "fitted separately at every depth."
        ),
        "predictive_model_selections_unchanged": (
            True
        ),
        "causal_interpretation_supported": (
            False
        ),
        "interpretation_warning": (
            "Coefficients are conditional model associations. "
            "Correlated component-lag predictors can redistribute "
            "coefficient magnitude and sign."
        ),
        "selected_alphas": {
            target: float(alpha)
            for target, alpha
            in selected_alphas.items()
        },
        "bootstrap_iterations": (
            BOOTSTRAP_ITERATIONS
        ),
        "bootstrap_unit": (
            "Complete regional unit with all 12 months"
        ),
        "stable_positive_precipitation_coefficients": int(
            precipitation_coefficients[
                "stable_positive_95"
            ].sum()
        ),
        "stable_negative_precipitation_coefficients": int(
            precipitation_coefficients[
                "stable_negative_95"
            ].sum()
        ),
        "multicollinearity": (
            multicollinearity_summary
        ),
        "coefficient_table": str(
            coefficient_path
        ),
        "memory_table": str(
            memory_path
        ),
        "alpha_selection": str(
            alpha_summary_path
        ),
        "multicollinearity_table": str(
            vif_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "component_lag_depth_attribution_metadata.json"
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
    print("ATTRIBUTION SUMMARY")
    print("=" * 78)

    print("Selected Ridge alphas:")

    for target in TARGET_ORDER:
        print(
            f"  {TARGETS[target]:>10s}: "
            f"{selected_alphas[target]:g}"
        )

    print()
    print(
        "Stable positive precipitation "
        f"coefficients: "
        f"{metadata['stable_positive_precipitation_coefficients']}"
    )

    print(
        "Stable negative precipitation "
        f"coefficients: "
        f"{metadata['stable_negative_precipitation_coefficients']}"
    )

    print()
    print("Component-specific memory indices:")

    print(
        memory_table[
            [
                "depth_label",
                "component",
                "dominant_lag_by_absolute_effect",
                "absolute_effect_memory_index_months",
                "positive_effect_memory_index_months",
                "net_standardized_effect_across_lags",
                "stable_positive_lag_count",
                "stable_negative_lag_count",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Multicollinearity condition number: "
        f"{multicollinearity_summary['feature_correlation_condition_number']:.3f}"
    )

    print(
        "Maximum VIF: "
        f"{multicollinearity_summary['maximum_variance_inflation_factor']:.3f}"
    )

    print()
    print(f"Coefficient table : {coefficient_path}")
    print(f"Memory table      : {memory_path}")
    print(
        f"Heatmap           : "
        f"{FIGURE_DIR / 'fig23_component_lag_depth_attribution.png'}"
    )
    print(
        f"Memory figure     : "
        f"{FIGURE_DIR / 'fig24_component_memory_by_depth.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
