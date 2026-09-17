#!/usr/bin/env python3
"""
Finalize component-lag-depth attribution under two Ridge-alpha policies.

Primary attribution policy
--------------------------
Use the alpha minimizing mean geographic-CV RMSE:
    0-10 cm    : 300
    10-40 cm   : 300
    40-100 cm  : 300
    100-200 cm : 300

Sensitivity policy
------------------
Use the largest alpha within one standard error:
    0-10 cm    : 3000
    10-40 cm   : 10000
    40-100 cm  : 10000
    100-200 cm : 3000

The component-lag-depth conclusion is considered robust only when its
broad memory pattern remains similar under both policies.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STAGE10_PATH = (
    ROOT
    / "src"
    / "10_component_lag_depth_attribution.py"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

ARCHIVE_DIR = (
    FIGURE_DIR / "archive_stage10"
)

BOOTSTRAP_ITERATIONS = 5_000
BASE_RANDOM_SEED = 20260806

PRIMARY_POLICY = "minimum_geographic_cv_rmse"
SENSITIVITY_POLICY = "one_standard_error"

ALPHA_POLICIES = {
    PRIMARY_POLICY: {
        "SM_0_10cm_mm": 300.0,
        "SM_10_40cm_mm": 300.0,
        "SM_40_100cm_mm": 300.0,
        "SM_100_200cm_mm": 300.0,
    },
    SENSITIVITY_POLICY: {
        "SM_0_10cm_mm": 3000.0,
        "SM_10_40cm_mm": 10000.0,
        "SM_40_100cm_mm": 10000.0,
        "SM_100_200cm_mm": 3000.0,
    },
}


def load_stage10():
    specification = (
        importlib.util.spec_from_file_location(
            "stage10_attribution",
            STAGE10_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not load {STAGE10_PATH}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    specification.loader.exec_module(
        module
    )

    return module


def archive_existing_figures() -> None:
    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filenames = [
        "fig23_component_lag_depth_attribution.png",
        "fig23_component_lag_depth_attribution.pdf",
        "fig24_component_memory_by_depth.png",
        "fig24_component_memory_by_depth.pdf",
    ]

    for filename in filenames:
        source = FIGURE_DIR / filename

        if source.exists():
            destination = (
                ARCHIVE_DIR
                / filename.replace(
                    ".png",
                    "_previous.png",
                ).replace(
                    ".pdf",
                    "_previous.pdf",
                )
            )

            shutil.copy2(
                source,
                destination,
            )


def build_policy_comparison(
    coefficient_table: pd.DataFrame,
    memory_table: pd.DataFrame,
    stage10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows = []

    for target in stage10.TARGET_ORDER:
        primary = (
            coefficient_table.loc[
                (
                    coefficient_table[
                        "alpha_policy"
                    ]
                    == PRIMARY_POLICY
                )
                & (
                    coefficient_table["target"]
                    == target
                )
                & (
                    coefficient_table[
                        "feature_type"
                    ]
                    == "precipitation_component"
                )
            ]
            .set_index("feature")
            .loc[
                stage10.PRECIPITATION_FEATURES
            ]
        )

        sensitivity = (
            coefficient_table.loc[
                (
                    coefficient_table[
                        "alpha_policy"
                    ]
                    == SENSITIVITY_POLICY
                )
                & (
                    coefficient_table["target"]
                    == target
                )
                & (
                    coefficient_table[
                        "feature_type"
                    ]
                    == "precipitation_component"
                )
            ]
            .set_index("feature")
            .loc[
                stage10.PRECIPITATION_FEATURES
            ]
        )

        primary_values = primary[
            "standardized_coefficient"
        ].to_numpy(dtype=float)

        sensitivity_values = sensitivity[
            "standardized_coefficient"
        ].to_numpy(dtype=float)

        coefficient_rows.append(
            {
                "target": target,
                "depth_label": (
                    stage10.TARGETS[target]
                ),
                "primary_alpha": float(
                    primary[
                        "selected_alpha"
                    ].iloc[0]
                ),
                "sensitivity_alpha": float(
                    sensitivity[
                        "selected_alpha"
                    ].iloc[0]
                ),
                "coefficient_correlation": float(
                    np.corrcoef(
                        primary_values,
                        sensitivity_values,
                    )[0, 1]
                ),
                "mean_absolute_coefficient_difference": float(
                    np.mean(
                        np.abs(
                            primary_values
                            - sensitivity_values
                        )
                    )
                ),
                "maximum_absolute_coefficient_difference": float(
                    np.max(
                        np.abs(
                            primary_values
                            - sensitivity_values
                        )
                    )
                ),
                "coefficient_sign_agreement_fraction": float(
                    np.mean(
                        np.sign(primary_values)
                        == np.sign(
                            sensitivity_values
                        )
                    )
                ),
                "primary_positive_coefficient_count": int(
                    np.sum(
                        primary_values > 0
                    )
                ),
                "sensitivity_positive_coefficient_count": int(
                    np.sum(
                        sensitivity_values > 0
                    )
                ),
            }
        )

    memory_rows = []

    for target in stage10.TARGET_ORDER:
        for component in stage10.COMPONENTS:
            primary = memory_table.loc[
                (
                    memory_table[
                        "alpha_policy"
                    ]
                    == PRIMARY_POLICY
                )
                & (
                    memory_table["target"]
                    == target
                )
                & (
                    memory_table["component"]
                    == component
                )
            ]

            sensitivity = memory_table.loc[
                (
                    memory_table[
                        "alpha_policy"
                    ]
                    == SENSITIVITY_POLICY
                )
                & (
                    memory_table["target"]
                    == target
                )
                & (
                    memory_table["component"]
                    == component
                )
            ]

            if (
                len(primary) != 1
                or len(sensitivity) != 1
            ):
                raise RuntimeError(
                    f"Missing memory comparison for "
                    f"{target}, {component}."
                )

            primary = primary.iloc[0]
            sensitivity = sensitivity.iloc[0]

            memory_rows.append(
                {
                    "target": target,
                    "depth_label": (
                        stage10.TARGETS[target]
                    ),
                    "component": component,
                    "component_label": (
                        stage10.COMPONENT_LABELS[
                            component
                        ]
                    ),
                    "primary_alpha": float(
                        ALPHA_POLICIES[
                            PRIMARY_POLICY
                        ][target]
                    ),
                    "sensitivity_alpha": float(
                        ALPHA_POLICIES[
                            SENSITIVITY_POLICY
                        ][target]
                    ),
                    "primary_dominant_lag": int(
                        primary[
                            "dominant_lag_by_absolute_effect"
                        ]
                    ),
                    "sensitivity_dominant_lag": int(
                        sensitivity[
                            "dominant_lag_by_absolute_effect"
                        ]
                    ),
                    "dominant_lag_agreement": bool(
                        primary[
                            "dominant_lag_by_absolute_effect"
                        ]
                        == sensitivity[
                            "dominant_lag_by_absolute_effect"
                        ]
                    ),
                    "primary_memory_index_months": float(
                        primary[
                            "absolute_effect_memory_index_months"
                        ]
                    ),
                    "sensitivity_memory_index_months": float(
                        sensitivity[
                            "absolute_effect_memory_index_months"
                        ]
                    ),
                    "memory_index_difference_months": float(
                        sensitivity[
                            "absolute_effect_memory_index_months"
                        ]
                        - primary[
                            "absolute_effect_memory_index_months"
                        ]
                    ),
                    "primary_net_effect": float(
                        primary[
                            "net_standardized_effect_across_lags"
                        ]
                    ),
                    "sensitivity_net_effect": float(
                        sensitivity[
                            "net_standardized_effect_across_lags"
                        ]
                    ),
                }
            )

    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(memory_rows),
    )


def calculate_depth_memory_summary(
    memory_table: pd.DataFrame,
    stage10,
) -> pd.DataFrame:
    summary = (
        memory_table.groupby(
            [
                "alpha_policy",
                "target",
                "depth_label",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_memory_index_months=(
                "absolute_effect_memory_index_months",
                "mean",
            ),
            median_memory_index_months=(
                "absolute_effect_memory_index_months",
                "median",
            ),
            minimum_memory_index_months=(
                "absolute_effect_memory_index_months",
                "min",
            ),
            maximum_memory_index_months=(
                "absolute_effect_memory_index_months",
                "max",
            ),
        )
    )

    depth_order = {
        target: index
        for index, target
        in enumerate(stage10.TARGET_ORDER)
    }

    summary["depth_order"] = (
        summary["target"]
        .map(depth_order)
    )

    return (
        summary.sort_values(
            [
                "alpha_policy",
                "depth_order",
            ]
        )
        .reset_index(drop=True)
    )


def make_sensitivity_figure(
    coefficient_table: pd.DataFrame,
    depth_memory_summary: pd.DataFrame,
    stage10,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.5, 5.0),
        constrained_layout=True,
    )

    positions = np.arange(
        len(stage10.TARGET_ORDER)
    )

    for policy in [
        PRIMARY_POLICY,
        SENSITIVITY_POLICY,
    ]:
        policy_memory = (
            depth_memory_summary.loc[
                depth_memory_summary[
                    "alpha_policy"
                ]
                == policy
            ]
            .set_index("target")
            .loc[
                stage10.TARGET_ORDER
            ]
        )

        axes[0].plot(
            positions,
            policy_memory[
                "mean_memory_index_months"
            ],
            marker="o",
            linewidth=1.6,
            label=policy,
        )

    axes[0].set_xticks(positions)

    axes[0].set_xticklabels(
        [
            stage10.TARGETS[target]
            for target
            in stage10.TARGET_ORDER
        ]
    )

    axes[0].set_xlabel(
        "Soil-moisture depth"
    )

    axes[0].set_ylabel(
        "Mean rainfall-memory index (months)"
    )

    axes[0].set_title(
        "Memory-depth relationship"
    )

    axes[0].legend(
        frameon=True,
        fontsize=8,
    )

    primary = (
        coefficient_table.loc[
            (
                coefficient_table[
                    "alpha_policy"
                ]
                == PRIMARY_POLICY
            )
            & (
                coefficient_table[
                    "feature_type"
                ]
                == "precipitation_component"
            )
        ]
        .sort_values(
            ["target", "feature"]
        )
    )

    sensitivity = (
        coefficient_table.loc[
            (
                coefficient_table[
                    "alpha_policy"
                ]
                == SENSITIVITY_POLICY
            )
            & (
                coefficient_table[
                    "feature_type"
                ]
                == "precipitation_component"
            )
        ]
        .sort_values(
            ["target", "feature"]
        )
    )

    primary_values = primary[
        "standardized_coefficient"
    ].to_numpy(dtype=float)

    sensitivity_values = sensitivity[
        "standardized_coefficient"
    ].to_numpy(dtype=float)

    axes[1].scatter(
        primary_values,
        sensitivity_values,
        s=30,
        edgecolor="black",
        linewidth=0.35,
    )

    minimum = float(
        min(
            primary_values.min(),
            sensitivity_values.min(),
        )
    )

    maximum = float(
        max(
            primary_values.max(),
            sensitivity_values.max(),
        )
    )

    axes[1].plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_xlabel(
        "Minimum-CV-alpha coefficient"
    )

    axes[1].set_ylabel(
        "One-SE-alpha coefficient"
    )

    axes[1].set_title(
        "Coefficient robustness to regularization"
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Regularization sensitivity of "
        "component-lag-depth attribution",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig25_attribution_alpha_sensitivity.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig25_attribution_alpha_sensitivity.pdf",
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

    archive_existing_figures()

    stage10 = load_stage10()

    stage10.BOOTSTRAP_ITERATIONS = (
        BOOTSTRAP_ITERATIONS
    )

    data = (
        pd.read_parquet(
            stage10.DATA_PATH
        )
        .sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
    )

    all_coefficient_tables = []
    all_memory_tables = []

    print("=" * 78)
    print("FINAL ATTRIBUTION ALPHA-SENSITIVITY ANALYSIS")
    print("=" * 78)
    print(
        f"Bootstrap iterations per fit: "
        f"{BOOTSTRAP_ITERATIONS}"
    )
    print()

    for policy_index, (
        policy,
        alpha_dictionary,
    ) in enumerate(
        ALPHA_POLICIES.items()
    ):
        print(f"Policy: {policy}")

        policy_coefficient_tables = []

        for target_index, target in enumerate(
            stage10.TARGET_ORDER
        ):
            alpha = float(
                alpha_dictionary[target]
            )

            (
                full_coefficients,
                x_scaler,
                y_scaler,
                x_standardized,
                y_standardized,
            ) = stage10.fit_standardized_ridge(
                data,
                target,
                alpha,
            )

            rng = np.random.default_rng(
                BASE_RANDOM_SEED
                + policy_index * 10_000
                + target_index * 1_000
            )

            bootstrap_array = (
                stage10.bootstrap_coefficients(
                    x_standardized=(
                        x_standardized
                    ),
                    y_standardized=(
                        y_standardized
                    ),
                    region_ids=data[
                        "region_id"
                    ].to_numpy(),
                    alpha=alpha,
                    rng=rng,
                )
            )

            coefficient_table = (
                stage10.build_coefficient_table(
                    target=target,
                    selected_alpha=alpha,
                    full_coefficients=(
                        full_coefficients
                    ),
                    bootstrap_coefficients_array=(
                        bootstrap_array
                    ),
                )
            )

            coefficient_table.insert(
                0,
                "alpha_policy",
                policy,
            )

            policy_coefficient_tables.append(
                coefficient_table
            )

            print(
                f"  {stage10.TARGETS[target]:>10s}: "
                f"alpha={alpha:g}"
            )

        policy_coefficients = pd.concat(
            policy_coefficient_tables,
            ignore_index=True,
        )

        policy_memory = (
            stage10.build_memory_table(
                policy_coefficients
            )
        )

        policy_memory.insert(
            0,
            "alpha_policy",
            policy,
        )

        all_coefficient_tables.append(
            policy_coefficients
        )

        all_memory_tables.append(
            policy_memory
        )

    coefficient_table = pd.concat(
        all_coefficient_tables,
        ignore_index=True,
    )

    memory_table = pd.concat(
        all_memory_tables,
        ignore_index=True,
    )

    (
        coefficient_comparison,
        memory_comparison,
    ) = build_policy_comparison(
        coefficient_table,
        memory_table,
        stage10,
    )

    depth_memory_summary = (
        calculate_depth_memory_summary(
            memory_table,
            stage10,
        )
    )

    coefficient_path = (
        TABLE_DIR
        / "component_lag_depth_coefficients_alpha_sensitivity.csv"
    )

    memory_path = (
        TABLE_DIR
        / "component_lag_depth_memory_alpha_sensitivity.csv"
    )

    coefficient_comparison_path = (
        TABLE_DIR
        / "attribution_alpha_coefficient_comparison.csv"
    )

    memory_comparison_path = (
        TABLE_DIR
        / "attribution_alpha_memory_comparison.csv"
    )

    depth_summary_path = (
        TABLE_DIR
        / "attribution_alpha_depth_memory_summary.csv"
    )

    coefficient_table.to_csv(
        coefficient_path,
        index=False,
    )

    memory_table.to_csv(
        memory_path,
        index=False,
    )

    coefficient_comparison.to_csv(
        coefficient_comparison_path,
        index=False,
    )

    memory_comparison.to_csv(
        memory_comparison_path,
        index=False,
    )

    depth_memory_summary.to_csv(
        depth_summary_path,
        index=False,
    )

    primary_coefficients = (
        coefficient_table.loc[
            coefficient_table[
                "alpha_policy"
            ]
            == PRIMARY_POLICY
        ]
        .drop(columns="alpha_policy")
        .reset_index(drop=True)
    )

    primary_memory = (
        memory_table.loc[
            memory_table[
                "alpha_policy"
            ]
            == PRIMARY_POLICY
        ]
        .drop(columns="alpha_policy")
        .reset_index(drop=True)
    )

    final_coefficient_path = (
        TABLE_DIR
        / "final_component_lag_depth_coefficients.csv"
    )

    final_memory_path = (
        TABLE_DIR
        / "final_component_lag_depth_memory_indices.csv"
    )

    primary_coefficients.to_csv(
        final_coefficient_path,
        index=False,
    )

    primary_memory.to_csv(
        final_memory_path,
        index=False,
    )

    # Replace the provisional Stage-10 central figures using the
    # finalized minimum-geographic-CV alpha policy.
    stage10.make_coefficient_heatmap(
        primary_coefficients
    )

    stage10.make_memory_figure(
        primary_memory
    )

    make_sensitivity_figure(
        coefficient_table,
        depth_memory_summary,
        stage10,
    )

    monotonic_results = {}

    for policy in [
        PRIMARY_POLICY,
        SENSITIVITY_POLICY,
    ]:
        values = (
            depth_memory_summary.loc[
                depth_memory_summary[
                    "alpha_policy"
                ]
                == policy
            ]
            .set_index("target")
            .loc[
                stage10.TARGET_ORDER,
                "mean_memory_index_months",
            ]
            .to_numpy(dtype=float)
        )

        monotonic_results[policy] = bool(
            np.all(
                np.diff(values) >= 0.0
            )
        )

    minimum_coefficient_correlation = float(
        coefficient_comparison[
            "coefficient_correlation"
        ].min()
    )

    dominant_lag_agreement_fraction = float(
        memory_comparison[
            "dominant_lag_agreement"
        ].mean()
    )

    primary_stable_positive = int(
        primary_coefficients.loc[
            primary_coefficients[
                "feature_type"
            ]
            == "precipitation_component",
            "stable_positive_95",
        ].sum()
    )

    primary_stable_negative = int(
        primary_coefficients.loc[
            primary_coefficients[
                "feature_type"
            ]
            == "precipitation_component",
            "stable_negative_95",
        ].sum()
    )

    metadata = {
        "status": "PASS",
        "primary_attribution_policy": (
            PRIMARY_POLICY
        ),
        "primary_alphas": (
            ALPHA_POLICIES[
                PRIMARY_POLICY
            ]
        ),
        "regularization_sensitivity_policy": (
            SENSITIVITY_POLICY
        ),
        "sensitivity_alphas": (
            ALPHA_POLICIES[
                SENSITIVITY_POLICY
            ]
        ),
        "bootstrap_iterations_per_target_policy": (
            BOOTSTRAP_ITERATIONS
        ),
        "mean_memory_increases_monotonically_with_depth": (
            monotonic_results
        ),
        "minimum_depthwise_coefficient_correlation_between_policies": (
            minimum_coefficient_correlation
        ),
        "dominant_lag_agreement_fraction_between_policies": (
            dominant_lag_agreement_fraction
        ),
        "primary_stable_positive_precipitation_coefficients": (
            primary_stable_positive
        ),
        "primary_stable_negative_precipitation_coefficients": (
            primary_stable_negative
        ),
        "interpretation": (
            "Use the minimum-geographic-CV-alpha results as "
            "the primary common-model attribution. Use the "
            "one-standard-error results only to assess whether "
            "component and lag conclusions are robust to "
            "stronger coefficient shrinkage."
        ),
        "final_coefficient_table": str(
            final_coefficient_path
        ),
        "final_memory_table": str(
            final_memory_path
        ),
        "alpha_sensitivity_coefficients": str(
            coefficient_path
        ),
        "alpha_sensitivity_memory": str(
            memory_path
        ),
        "coefficient_comparison": str(
            coefficient_comparison_path
        ),
        "memory_comparison": str(
            memory_comparison_path
        ),
        "depth_memory_summary": str(
            depth_summary_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "final_attribution_alpha_sensitivity_metadata.json"
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
    print("ALPHA-SENSITIVITY SUMMARY")
    print("=" * 78)

    print("Mean memory index by depth:")

    print(
        depth_memory_summary[
            [
                "alpha_policy",
                "depth_label",
                "mean_memory_index_months",
                "median_memory_index_months",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Coefficient agreement:")

    print(
        coefficient_comparison[
            [
                "depth_label",
                "primary_alpha",
                "sensitivity_alpha",
                "coefficient_correlation",
                "mean_absolute_coefficient_difference",
                "maximum_absolute_coefficient_difference",
                "coefficient_sign_agreement_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Memory agreement:")

    print(
        memory_comparison[
            [
                "depth_label",
                "component",
                "primary_dominant_lag",
                "sensitivity_dominant_lag",
                "dominant_lag_agreement",
                "primary_memory_index_months",
                "sensitivity_memory_index_months",
                "memory_index_difference_months",
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
        "Monotonic mean-memory increase:"
    )

    for policy, result in (
        monotonic_results.items()
    ):
        print(f"  {policy}: {result}")

    print()
    print(
        "Minimum coefficient correlation: "
        f"{minimum_coefficient_correlation:.4f}"
    )

    print(
        "Dominant-lag agreement fraction: "
        f"{dominant_lag_agreement_fraction:.4f}"
    )

    print(
        "Primary stable positive effects: "
        f"{primary_stable_positive}"
    )

    print(
        "Primary stable negative effects: "
        f"{primary_stable_negative}"
    )

    print()
    print(
        f"Final coefficient table : "
        f"{final_coefficient_path}"
    )

    print(
        f"Final memory table      : "
        f"{final_memory_path}"
    )

    print(
        f"Final central heatmap   : "
        f"{FIGURE_DIR / 'fig23_component_lag_depth_attribution.png'}"
    )

    print(
        f"Final memory figure     : "
        f"{FIGURE_DIR / 'fig24_component_memory_by_depth.png'}"
    )

    print(
        f"Alpha-sensitivity figure: "
        f"{FIGURE_DIR / 'fig25_attribution_alpha_sensitivity.png'}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
