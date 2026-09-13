#!/usr/bin/env python3
"""
Paired complete-region bootstrap for soil-moisture layer models.

The bootstrap preserves all twelve months from each regional unit.

Pairwise comparisons
--------------------
1. Ridge versus MLP
2. PINN-P versus MLP
3. PINN-P versus Ridge

Difference convention
---------------------
RMSE difference = candidate RMSE - reference RMSE

Negative values mean that the candidate performs better.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    ROOT
    / "results"
    / "tables"
    / "layer_spatial_cv_ensemble_predictions.csv.gz"
)

SUMMARY_PATH = (
    ROOT
    / "results"
    / "tables"
    / "layer_spatial_cv_summary.csv"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

BOOTSTRAP_ITERATIONS = 20_000
RANDOM_SEED = 20260806

TARGET_ORDER = [
    "SM_0_10cm_mm",
    "SM_10_40cm_mm",
    "SM_40_100cm_mm",
    "SM_100_200cm_mm",
]

DEPTH_LABELS = {
    "SM_0_10cm_mm": "0–10 cm",
    "SM_10_40cm_mm": "10–40 cm",
    "SM_40_100cm_mm": "40–100 cm",
    "SM_100_200cm_mm": "100–200 cm",
}

PAIRWISE_COMPARISONS = [
    ("Ridge", "MLP"),
    ("PINN_P_lambda10", "MLP"),
    ("PINN_P_lambda10", "Ridge"),
]

EXPECTED_REGIONS = 33
EXPECTED_CORE_REGIONS = 29
EXPECTED_MONTHS_PER_REGION = 12


def paired_bootstrap(
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

    candidate_subset = candidate[
        [
            *merge_columns,
            "core_region_ge_4_cells",
            "observed_mm",
            "ensemble_prediction_mm",
        ]
    ].copy()

    reference_subset = reference[
        [
            *merge_columns,
            "observed_mm",
            "ensemble_prediction_mm",
        ]
    ].copy()

    paired = candidate_subset.merge(
        reference_subset,
        on=merge_columns,
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
            "Candidate and reference targets differ."
        )

    if core_only:
        paired = paired.loc[
            paired["core_region_ge_4_cells"]
        ].copy()

    regions = sorted(
        paired["region_id"].unique()
    )

    expected_region_count = (
        EXPECTED_CORE_REGIONS
        if core_only
        else EXPECTED_REGIONS
    )

    if len(regions) != expected_region_count:
        raise RuntimeError(
            f"Expected {expected_region_count} regions, "
            f"found {len(regions)}."
        )

    candidate_sse = []
    reference_sse = []
    observation_counts = []

    candidate_regional_rmse = []
    reference_regional_rmse = []

    for region_id in regions:
        region = paired.loc[
            paired["region_id"] == region_id
        ].sort_values("month")

        if len(region) != EXPECTED_MONTHS_PER_REGION:
            raise RuntimeError(
                f"{region_id} does not have 12 months."
            )

        truth = region[
            "observed_mm_candidate"
        ].to_numpy(dtype=float)

        candidate_prediction = region[
            "ensemble_prediction_mm_candidate"
        ].to_numpy(dtype=float)

        reference_prediction = region[
            "ensemble_prediction_mm_reference"
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

    candidate_regional_rmse = np.asarray(
        candidate_regional_rmse,
        dtype=float,
    )

    reference_regional_rmse = np.asarray(
        reference_regional_rmse,
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

    sampled_observation_count = (
        observation_counts[
            sampled_indices
        ].sum(axis=1)
    )

    sampled_candidate_rmse = np.sqrt(
        candidate_sse[
            sampled_indices
        ].sum(axis=1)
        / sampled_observation_count
    )

    sampled_reference_rmse = np.sqrt(
        reference_sse[
            sampled_indices
        ].sum(axis=1)
        / sampled_observation_count
    )

    sampled_difference = (
        sampled_candidate_rmse
        - sampled_reference_rmse
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
            sampled_difference.mean()
        ),
        "bootstrap_difference_q025_mm": float(
            np.quantile(
                sampled_difference,
                0.025,
            )
        ),
        "bootstrap_difference_q50_mm": float(
            np.quantile(
                sampled_difference,
                0.50,
            )
        ),
        "bootstrap_difference_q975_mm": float(
            np.quantile(
                sampled_difference,
                0.975,
            )
        ),
        "probability_candidate_lower_rmse": float(
            np.mean(
                sampled_difference < 0.0
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
        "regional_tie_count": int(
            np.sum(
                np.isclose(
                    candidate_regional_rmse,
                    reference_regional_rmse,
                    atol=1e-10,
                )
            )
        ),
    }


def build_decision_table(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for target in TARGET_ORDER:
        target_summary = summary.loc[
            summary["target"] == target
        ].set_index("configuration")

        ridge_rmse = float(
            target_summary.loc[
                "Ridge",
                "ensemble_rmse_mm",
            ]
        )

        mlp_rmse = float(
            target_summary.loc[
                "MLP",
                "ensemble_rmse_mm",
            ]
        )

        pinn_rmse = float(
            target_summary.loc[
                "PINN_P_lambda10",
                "ensemble_rmse_mm",
            ]
        )

        comparison = bootstrap.loc[
            (
                bootstrap["target"]
                == target
            )
            & (
                bootstrap["candidate"]
                == "Ridge"
            )
            & (
                bootstrap["reference"]
                == "MLP"
            )
            & (
                bootstrap["population"]
                == "all_regions"
            )
        ]

        if len(comparison) != 1:
            raise RuntimeError(
                f"Missing Ridge-versus-MLP comparison "
                f"for {target}."
            )

        comparison = comparison.iloc[0]

        lower = float(
            comparison[
                "bootstrap_difference_q025_mm"
            ]
        )

        upper = float(
            comparison[
                "bootstrap_difference_q975_mm"
            ]
        )

        relative_gap_percent = (
            100.0
            * abs(ridge_rmse - mlp_rmse)
            / min(ridge_rmse, mlp_rmse)
        )

        if upper < 0.0:
            selected_model = "Ridge"
            decision_strength = (
                "Ridge significantly better"
            )

        elif lower > 0.0:
            selected_model = "MLP"
            decision_strength = (
                "MLP significantly better"
            )

        elif relative_gap_percent <= 1.0:
            selected_model = "Ridge"
            decision_strength = (
                "Statistically indistinguishable; "
                "Ridge selected for parsimony"
            )

        else:
            selected_model = (
                "Ridge"
                if ridge_rmse < mlp_rmse
                else "MLP"
            )

            decision_strength = (
                "Lower observed RMSE selected, "
                "but bootstrap interval includes zero"
            )

        rows.append(
            {
                "target": target,
                "depth_label": (
                    DEPTH_LABELS[target]
                ),
                "ridge_rmse_mm": ridge_rmse,
                "mlp_rmse_mm": mlp_rmse,
                "pinn_rmse_mm": pinn_rmse,
                "ridge_minus_mlp_rmse_mm": (
                    ridge_rmse - mlp_rmse
                ),
                "ridge_minus_mlp_q025_mm": (
                    lower
                ),
                "ridge_minus_mlp_q975_mm": (
                    upper
                ),
                "ridge_mlp_relative_gap_percent": (
                    relative_gap_percent
                ),
                "selected_primary_model": (
                    selected_model
                ),
                "decision_strength": (
                    decision_strength
                ),
                "physics_companion_model": (
                    "PINN_P_lambda10"
                ),
            }
        )

    return pd.DataFrame(rows)


def make_figure(
    decision: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.5, 4.8),
        constrained_layout=True,
    )

    positions = np.arange(
        len(decision)
    )

    depth_labels = decision[
        "depth_label"
    ].tolist()

    width = 0.34

    axes[0].bar(
        positions - width / 2,
        decision["ridge_rmse_mm"],
        width=width,
        label="Ridge",
        edgecolor="black",
        linewidth=0.4,
    )

    axes[0].bar(
        positions + width / 2,
        decision["mlp_rmse_mm"],
        width=width,
        label="MLP",
        edgecolor="black",
        linewidth=0.4,
    )

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        depth_labels
    )

    axes[0].set_xlabel("Soil depth")
    axes[0].set_ylabel(
        "Geographic-CV RMSE (mm)"
    )

    axes[0].set_title(
        "Ridge and MLP predictive performance"
    )

    axes[0].legend(
        frameon=True
    )

    observed = decision[
        "ridge_minus_mlp_rmse_mm"
    ].to_numpy(dtype=float)

    lower_error = (
        observed
        - decision[
            "ridge_minus_mlp_q025_mm"
        ].to_numpy(dtype=float)
    )

    upper_error = (
        decision[
            "ridge_minus_mlp_q975_mm"
        ].to_numpy(dtype=float)
        - observed
    )

    axes[1].errorbar(
        positions,
        observed,
        yerr=np.vstack(
            [lower_error, upper_error]
        ),
        marker="o",
        linestyle="none",
        capsize=4,
        color="black",
    )

    axes[1].axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
    )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        depth_labels
    )

    axes[1].set_xlabel("Soil depth")
    axes[1].set_ylabel(
        "Ridge RMSE − MLP RMSE (mm)"
    )

    axes[1].set_title(
        "Complete-region bootstrap, 95% interval"
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Depth-dependent soil-moisture model selection",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig22_layer_model_block_bootstrap.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig22_layer_model_block_bootstrap.pdf",
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

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    required_configurations = {
        "Mean",
        "Ridge",
        "MLP",
        "PINN_P_lambda10",
    }

    found_configurations = set(
        predictions[
            "configuration"
        ].unique()
    )

    if found_configurations != (
        required_configurations
    ):
        raise RuntimeError(
            "Unexpected configurations: "
            f"{sorted(found_configurations)}"
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    rows = []

    for target in TARGET_ORDER:
        target_data = predictions.loc[
            predictions["target"] == target
        ]

        for (
            candidate_name,
            reference_name,
        ) in PAIRWISE_COMPARISONS:
            candidate = target_data.loc[
                target_data["configuration"]
                == candidate_name
            ]

            reference = target_data.loc[
                target_data["configuration"]
                == reference_name
            ]

            for core_only in [
                False,
                True,
            ]:
                result = paired_bootstrap(
                    candidate,
                    reference,
                    core_only=core_only,
                    rng=rng,
                )

                rows.append(
                    {
                        "target": target,
                        "depth_label": (
                            DEPTH_LABELS[
                                target
                            ]
                        ),
                        "candidate": (
                            candidate_name
                        ),
                        "reference": (
                            reference_name
                        ),
                        "population": (
                            "core_regions"
                            if core_only
                            else "all_regions"
                        ),
                        **result,
                    }
                )

    bootstrap = pd.DataFrame(rows)

    decision = build_decision_table(
        summary,
        bootstrap,
    )

    bootstrap_path = (
        TABLE_DIR
        / "layer_model_block_bootstrap.csv"
    )

    decision_path = (
        TABLE_DIR
        / "final_layer_model_selection.csv"
    )

    bootstrap.to_csv(
        bootstrap_path,
        index=False,
    )

    decision.to_csv(
        decision_path,
        index=False,
    )

    make_figure(decision)

    metadata = {
        "status": "PASS",
        "bootstrap_iterations": (
            BOOTSTRAP_ITERATIONS
        ),
        "bootstrap_unit": (
            "Complete regional unit with all 12 months"
        ),
        "random_seed": RANDOM_SEED,
        "pairwise_comparisons": (
            PAIRWISE_COMPARISONS
        ),
        "selection_policy": (
            "Use a statistically significant Ridge-versus-MLP "
            "difference when present. When indistinguishable "
            "and within 1%, select Ridge for parsimony. "
            "Otherwise select lower observed RMSE while "
            "reporting uncertainty."
        ),
        "bootstrap_table": str(
            bootstrap_path
        ),
        "selection_table": str(
            decision_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "layer_model_block_bootstrap_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("LAYER MODEL COMPLETE-REGION BOOTSTRAP")
    print("=" * 78)

    comparison = bootstrap.loc[
        (
            bootstrap["candidate"]
            == "Ridge"
        )
        & (
            bootstrap["reference"]
            == "MLP"
        )
        & (
            bootstrap["population"]
            == "all_regions"
        )
    ]

    print(
        comparison[
            [
                "depth_label",
                "observed_rmse_difference_mm",
                "bootstrap_difference_q025_mm",
                "bootstrap_difference_q975_mm",
                "probability_candidate_lower_rmse",
                "candidate_region_win_count",
                "reference_region_win_count",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Provisional layer selections:")

    print(
        decision[
            [
                "depth_label",
                "ridge_rmse_mm",
                "mlp_rmse_mm",
                "pinn_rmse_mm",
                "selected_primary_model",
                "decision_strength",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(f"Bootstrap table : {bootstrap_path}")
    print(f"Selection table : {decision_path}")
    print(
        f"Figure          : "
        f"{FIGURE_DIR / 'fig22_layer_model_block_bootstrap.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
