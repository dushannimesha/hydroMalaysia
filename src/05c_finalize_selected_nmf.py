#!/usr/bin/env python3
"""
Finalize and audit the selected K=4 precipitation NMF model.

Primary model
-------------
K = 4, verified using coordinate descent and NNDSVD-based initialization.

Sensitivity model
-----------------
K = 3.

This script:
1. Verifies that the selected best K=4 solution converged.
2. Applies transparent descriptive labels to the four modes.
3. Calculates each component's precipitation contribution for every
   region and month.
4. Audits reconstruction accuracy region by region.
5. Identifies dominant annual and monthly components.
6. Generates final component, dominance, and reconstruction figures.

The labels are descriptive. They are not yet causal atmospheric labels.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    ROOT
    / "results"
    / "nmf_verified"
    / "k_04"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "nmf_selected_k4"
)

VERIFICATION_SUMMARY_PATH = (
    ROOT
    / "results"
    / "tables"
    / "nmf_k3_k4_verification_summary.csv"
)

ALL_RUNS_PATH = (
    ROOT
    / "results"
    / "tables"
    / "nmf_k3_k4_verification_all_runs.csv"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

SELECTED_K = 4
SENSITIVITY_K = 3

EXPECTED_REGIONS = 33
EXPECTED_MONTHS = 12

STABILITY_THRESHOLD = 0.95
DUPLICATION_THRESHOLD = 0.90
MINIMUM_CONVERGENCE_FRACTION = 0.80

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

COMPONENT_LABELS = {
    "K4_C1": {
        "short_label": "January eastern mode",
        "descriptive_label": (
            "January-centred eastern precipitation mode"
        ),
    },
    "K4_C2": {
        "short_label": "May–June southwest mode",
        "descriptive_label": (
            "May–June southwestern precipitation mode"
        ),
    },
    "K4_C3": {
        "short_label": "April–October mode",
        "descriptive_label": (
            "April–October bimodal precipitation mode"
        ),
    },
    "K4_C4": {
        "short_label": "November northern mode",
        "descriptive_label": (
            "November-centred northern precipitation mode"
        ),
    },
}


def safe_r_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    residual_sum_squares = float(
        np.sum((observed - predicted) ** 2)
    )

    total_sum_squares = float(
        np.sum(
            (observed - observed.mean()) ** 2
        )
    )

    if total_sum_squares <= 1e-12:
        return float("nan")

    return float(
        1.0
        - residual_sum_squares
        / total_sum_squares
    )


def load_and_verify_selection() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Verified K=4 directory does not exist: "
            f"{SOURCE_DIR}"
        )

    verification = pd.read_csv(
        VERIFICATION_SUMMARY_PATH
    )

    all_runs = pd.read_csv(
        ALL_RUNS_PATH
    )

    metadata_path = (
        SOURCE_DIR / "model_metadata.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    selected_row = verification.loc[
        verification["component_count"]
        == SELECTED_K
    ]

    if len(selected_row) != 1:
        raise RuntimeError(
            "Expected exactly one K=4 verification row."
        )

    selected_row = selected_row.iloc[0]

    if not bool(metadata["converged"]):
        raise RuntimeError(
            "The best selected K=4 model did not converge."
        )

    mean_stability = float(
        selected_row["mean_component_stability"]
    )

    convergence_fraction = float(
        selected_row["converged_fraction"]
    )

    maximum_similarity = float(
        selected_row[
            "maximum_pairwise_component_similarity"
        ]
    )

    if mean_stability < STABILITY_THRESHOLD:
        raise RuntimeError(
            f"K=4 stability {mean_stability:.4f} is below "
            f"the threshold {STABILITY_THRESHOLD:.2f}."
        )

    if convergence_fraction < MINIMUM_CONVERGENCE_FRACTION:
        raise RuntimeError(
            f"K=4 convergence fraction "
            f"{convergence_fraction:.3f} is below "
            f"{MINIMUM_CONVERGENCE_FRACTION:.2f}."
        )

    if maximum_similarity >= DUPLICATION_THRESHOLD:
        raise RuntimeError(
            f"K=4 maximum component similarity "
            f"{maximum_similarity:.4f} indicates possible "
            "component duplication."
        )

    return verification, all_runs, metadata


def load_selected_model() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    components = pd.read_csv(
        SOURCE_DIR / "seasonal_components.csv"
    )

    loadings = pd.read_csv(
        SOURCE_DIR / "regional_component_loadings.csv"
    )

    reconstruction = pd.read_csv(
        SOURCE_DIR / "precipitation_reconstruction.csv"
    )

    component_names = sorted(
        components["component"].unique().tolist()
    )

    expected_components = [
        "K4_C1",
        "K4_C2",
        "K4_C3",
        "K4_C4",
    ]

    if component_names != expected_components:
        raise RuntimeError(
            f"Expected components {expected_components}, "
            f"found {component_names}."
        )

    if loadings["region_id"].nunique() != EXPECTED_REGIONS:
        raise RuntimeError(
            "Unexpected regional count in loading table."
        )

    if len(reconstruction) != (
        EXPECTED_REGIONS * EXPECTED_MONTHS
    ):
        raise RuntimeError(
            "Unexpected reconstruction-table size."
        )

    if reconstruction[
        ["observed_precipitation_mm",
         "reconstructed_precipitation_mm"]
    ].isna().any().any():
        raise RuntimeError(
            "Reconstruction table contains missing values."
        )

    return components, loadings, reconstruction


def build_component_summary(
    components: pd.DataFrame,
    loadings: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for component_name in sorted(
        components["component"].unique()
    ):
        profile = (
            components.loc[
                components["component"]
                == component_name
            ]
            .sort_values("month")
            .reset_index(drop=True)
        )

        loading_column = (
            f"{component_name}_loading"
        )

        ranked_regions = (
            loadings.sort_values(
                loading_column,
                ascending=False,
            )
            .reset_index(drop=True)
        )

        top_regions = ranked_regions.head(5)

        rows.append(
            {
                "component": component_name,
                "short_label": (
                    COMPONENT_LABELS[
                        component_name
                    ]["short_label"]
                ),
                "descriptive_label": (
                    COMPONENT_LABELS[
                        component_name
                    ]["descriptive_label"]
                ),
                "peak_month": int(
                    profile["peak_month"].iloc[0]
                ),
                "peak_month_name": MONTH_NAMES[
                    int(
                        profile[
                            "peak_month"
                        ].iloc[0]
                    ) - 1
                ],
                "second_peak_month": int(
                    profile[
                        "second_peak_month"
                    ].iloc[0]
                ),
                "second_peak_month_name": (
                    MONTH_NAMES[
                        int(
                            profile[
                                "second_peak_month"
                            ].iloc[0]
                        ) - 1
                    ]
                ),
                "normalized_entropy": float(
                    profile[
                        "normalized_entropy"
                    ].iloc[0]
                ),
                "profile_maximum": float(
                    profile[
                        "profile_maximum"
                    ].iloc[0]
                ),
                "maximum_regional_loading": float(
                    ranked_regions[
                        loading_column
                    ].iloc[0]
                ),
                "top_region_1": str(
                    top_regions[
                        "region_id"
                    ].iloc[0]
                ),
                "top_region_2": str(
                    top_regions[
                        "region_id"
                    ].iloc[1]
                ),
                "top_region_3": str(
                    top_regions[
                        "region_id"
                    ].iloc[2]
                ),
                "top_region_4": str(
                    top_regions[
                        "region_id"
                    ].iloc[3]
                ),
                "top_region_5": str(
                    top_regions[
                        "region_id"
                    ].iloc[4]
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_component_contributions(
    components: pd.DataFrame,
    loadings: pd.DataFrame,
    reconstruction: pd.DataFrame,
) -> pd.DataFrame:
    contribution_rows = []

    component_names = sorted(
        components["component"].unique()
    )

    for loading_row in loadings.itertuples(
        index=False
    ):
        region_id = loading_row.region_id

        for component_name in component_names:
            loading_column = (
                f"{component_name}_loading"
            )

            loading_value = float(
                getattr(
                    loading_row,
                    loading_column,
                )
            )

            profile = (
                components.loc[
                    components["component"]
                    == component_name
                ]
                .sort_values("month")
            )

            for profile_row in profile.itertuples(
                index=False
            ):
                contribution_rows.append(
                    {
                        "region_id": region_id,
                        "native_cell_count": int(
                            loading_row.native_cell_count
                        ),
                        "longitude_mean": float(
                            loading_row.longitude_mean
                        ),
                        "latitude_mean": float(
                            loading_row.latitude_mean
                        ),
                        "region_lon_index": int(
                            loading_row.region_lon_index
                        ),
                        "region_lat_index": int(
                            loading_row.region_lat_index
                        ),
                        "small_region_lt_4_cells": bool(
                            loading_row.small_region_lt_4_cells
                        ),
                        "component": component_name,
                        "component_label": (
                            COMPONENT_LABELS[
                                component_name
                            ]["short_label"]
                        ),
                        "month": int(
                            profile_row.month
                        ),
                        "month_name": str(
                            profile_row.month_name
                        ),
                        "regional_loading_mm": (
                            loading_value
                        ),
                        "normalized_seasonal_profile": float(
                            profile_row.normalized_profile
                        ),
                        "component_contribution_mm": float(
                            loading_value
                            * profile_row.normalized_profile
                        ),
                    }
                )

    contributions = pd.DataFrame(
        contribution_rows
    )

    reconstructed_from_components = (
        contributions.groupby(
            ["region_id", "month"],
            as_index=False,
            observed=True,
        )
        .agg(
            component_sum_mm=(
                "component_contribution_mm",
                "sum",
            )
        )
    )

    audit = reconstructed_from_components.merge(
        reconstruction[
            [
                "region_id",
                "month",
                "observed_precipitation_mm",
                "reconstructed_precipitation_mm",
            ]
        ],
        on=["region_id", "month"],
        how="left",
        validate="one_to_one",
    )

    audit["component_reconstruction_error_mm"] = (
        audit["component_sum_mm"]
        - audit[
            "reconstructed_precipitation_mm"
        ]
    ).abs()

    maximum_error = float(
        audit[
            "component_reconstruction_error_mm"
        ].max()
    )

    if maximum_error > 1e-5:
        raise RuntimeError(
            "Component contributions do not reproduce "
            f"the selected reconstruction. Maximum error: "
            f"{maximum_error:.10f} mm."
        )

    monthly_totals = (
        contributions.groupby(
            ["region_id", "month"],
            as_index=False,
            observed=True,
        )
        .agg(
            reconstructed_total_mm=(
                "component_contribution_mm",
                "sum",
            )
        )
    )

    contributions = contributions.merge(
        monthly_totals,
        on=["region_id", "month"],
        how="left",
        validate="many_to_one",
    )

    contributions[
        "component_fraction"
    ] = np.where(
        contributions[
            "reconstructed_total_mm"
        ] > 1e-12,
        contributions[
            "component_contribution_mm"
        ]
        / contributions[
            "reconstructed_total_mm"
        ],
        np.nan,
    )

    return contributions


def calculate_reconstruction_metrics(
    reconstruction: pd.DataFrame,
    loadings: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for region_id, region_data in reconstruction.groupby(
        "region_id",
        observed=True,
    ):
        region_data = (
            region_data.sort_values("month")
        )

        observed = region_data[
            "observed_precipitation_mm"
        ].to_numpy(dtype=float)

        predicted = region_data[
            "reconstructed_precipitation_mm"
        ].to_numpy(dtype=float)

        residual = observed - predicted

        metadata = loadings.loc[
            loadings["region_id"] == region_id
        ]

        if len(metadata) != 1:
            raise RuntimeError(
                f"Missing metadata for {region_id}."
            )

        metadata = metadata.iloc[0]

        rmse = float(
            np.sqrt(np.mean(residual**2))
        )

        mean_observed = float(
            observed.mean()
        )

        rows.append(
            {
                "region_id": region_id,
                "native_cell_count": int(
                    metadata["native_cell_count"]
                ),
                "longitude_mean": float(
                    metadata["longitude_mean"]
                ),
                "latitude_mean": float(
                    metadata["latitude_mean"]
                ),
                "small_region_lt_4_cells": bool(
                    metadata[
                        "small_region_lt_4_cells"
                    ]
                ),
                "mean_observed_precipitation_mm": (
                    mean_observed
                ),
                "rmse_mm": rmse,
                "normalized_rmse_by_mean": (
                    rmse / mean_observed
                    if mean_observed > 1e-12
                    else float("nan")
                ),
                "mae_mm": float(
                    np.mean(np.abs(residual))
                ),
                "maximum_absolute_error_mm": float(
                    np.max(np.abs(residual))
                ),
                "mean_bias_mm": float(
                    np.mean(predicted - observed)
                ),
                "r_squared": safe_r_squared(
                    observed,
                    predicted,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "normalized_rmse_by_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_dominance_tables(
    contributions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = (
        contributions.groupby(
            [
                "region_id",
                "component",
                "component_label",
                "native_cell_count",
                "longitude_mean",
                "latitude_mean",
                "region_lon_index",
                "region_lat_index",
                "small_region_lt_4_cells",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            annual_component_contribution_mm=(
                "component_contribution_mm",
                "sum",
            )
        )
    )

    annual_total = (
        annual.groupby(
            "region_id",
            as_index=False,
            observed=True,
        )
        .agg(
            reconstructed_annual_precipitation_mm=(
                "annual_component_contribution_mm",
                "sum",
            )
        )
    )

    annual = annual.merge(
        annual_total,
        on="region_id",
        how="left",
        validate="many_to_one",
    )

    annual[
        "annual_component_fraction"
    ] = (
        annual[
            "annual_component_contribution_mm"
        ]
        / annual[
            "reconstructed_annual_precipitation_mm"
        ]
    )

    dominant_annual = (
        annual.sort_values(
            [
                "region_id",
                "annual_component_fraction",
            ],
            ascending=[True, False],
        )
        .groupby(
            "region_id",
            as_index=False,
            observed=True,
        )
        .first()
        .rename(
            columns={
                "component": (
                    "dominant_annual_component"
                ),
                "component_label": (
                    "dominant_annual_component_label"
                ),
                "annual_component_contribution_mm": (
                    "dominant_annual_contribution_mm"
                ),
                "annual_component_fraction": (
                    "dominant_annual_component_fraction"
                ),
            }
        )
    )

    monthly = contributions.copy()

    monthly_dominant = (
        monthly.sort_values(
            [
                "region_id",
                "month",
                "component_fraction",
            ],
            ascending=[True, True, False],
        )
        .groupby(
            ["region_id", "month"],
            as_index=False,
            observed=True,
        )
        .first()
        .rename(
            columns={
                "component": (
                    "dominant_monthly_component"
                ),
                "component_label": (
                    "dominant_monthly_component_label"
                ),
                "component_contribution_mm": (
                    "dominant_monthly_contribution_mm"
                ),
                "component_fraction": (
                    "dominant_monthly_component_fraction"
                ),
            }
        )
    )

    return annual, dominant_annual, monthly_dominant


def plot_component_profiles(
    components: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10.0, 7.5),
        sharex=True,
        constrained_layout=True,
    )

    for axis, component_name in zip(
        axes.flat,
        sorted(
            components["component"].unique()
        ),
    ):
        profile = (
            components.loc[
                components["component"]
                == component_name
            ]
            .sort_values("month")
        )

        axis.plot(
            profile["month"],
            profile["normalized_profile"],
            marker="o",
            linewidth=1.7,
            color="black",
        )

        axis.fill_between(
            profile["month"],
            0,
            profile["normalized_profile"],
            color="0.85",
        )

        axis.set_xticks(range(1, 13))
        axis.set_xticklabels(
            MONTH_NAMES,
            rotation=45,
        )

        axis.set_ylabel(
            "Normalized contribution"
        )

        axis.set_title(
            f"{component_name}: "
            f"{COMPONENT_LABELS[component_name]['short_label']}"
        )

        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Selected four-mode decomposition of "
        "Sri Lankan seasonal precipitation",
        fontsize=14,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig14_selected_k4_component_profiles.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig14_selected_k4_component_profiles.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_dominant_component_map(
    dominant: pd.DataFrame,
) -> None:
    component_names = [
        "K4_C1",
        "K4_C2",
        "K4_C3",
        "K4_C4",
    ]

    cmap = plt.get_cmap(
        "tab10"
    )

    component_colors = {
        component_name: cmap(index)
        for index, component_name
        in enumerate(component_names)
    }

    figure, axis = plt.subplots(
        figsize=(6.5, 8.3),
        constrained_layout=True,
    )

    for row in dominant.itertuples():
        rectangle = Rectangle(
            (
                79.5
                + row.region_lon_index * 0.5,
                5.5
                + row.region_lat_index * 0.5,
            ),
            0.5,
            0.5,
            facecolor=component_colors[
                row.dominant_annual_component
            ],
            edgecolor="black",
            linewidth=0.65,
            hatch=(
                "///"
                if row.small_region_lt_4_cells
                else None
            ),
        )

        axis.add_patch(rectangle)

        axis.text(
            row.longitude_mean,
            row.latitude_mean,
            row.dominant_annual_component.replace(
                "K4_",
                "",
            ),
            ha="center",
            va="center",
            fontsize=7,
        )

    axis.set_xlim(79.4, 82.1)
    axis.set_ylim(5.4, 10.1)

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")

    axis.set_title(
        "Dominant annual precipitation component"
    )

    axis.grid(
        linewidth=0.35,
        linestyle=":",
        alpha=0.5,
    )

    legend_handles = [
        Patch(
            facecolor=component_colors[
                component_name
            ],
            edgecolor="black",
            label=(
                f"{component_name}: "
                f"{COMPONENT_LABELS[component_name]['short_label']}"
            ),
        )
        for component_name in component_names
    ]

    axis.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.5,
        frameon=True,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig15_selected_k4_dominant_component_map.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig15_selected_k4_dominant_component_map.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_reconstruction_quality(
    reconstruction: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.0, 4.8),
        constrained_layout=True,
    )

    observed = reconstruction[
        "observed_precipitation_mm"
    ].to_numpy(dtype=float)

    predicted = reconstruction[
        "reconstructed_precipitation_mm"
    ].to_numpy(dtype=float)

    axes[0].scatter(
        observed,
        predicted,
        s=19,
        facecolor="0.72",
        edgecolor="black",
        linewidth=0.35,
    )

    maximum_value = float(
        max(observed.max(), predicted.max())
    )

    axes[0].plot(
        [0, maximum_value],
        [0, maximum_value],
        linestyle="--",
        linewidth=1.0,
        color="black",
    )

    axes[0].set_xlabel(
        "Observed monthly climatological precipitation (mm)"
    )

    axes[0].set_ylabel(
        "K=4 reconstructed precipitation (mm)"
    )

    axes[0].set_title(
        "All 396 region–month observations"
    )

    ranked = metrics.sort_values(
        "normalized_rmse_by_mean",
        ascending=True,
    )

    axes[1].barh(
        ranked["region_id"],
        ranked["normalized_rmse_by_mean"],
        edgecolor="black",
        linewidth=0.3,
        color="0.72",
    )

    axes[1].set_xlabel(
        "Regional normalized RMSE"
    )

    axes[1].set_ylabel("Regional unit")

    axes[1].set_title(
        "Region-level reconstruction error"
    )

    axes[1].tick_params(
        axis="y",
        labelsize=6.5,
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.55,
        )

    figure.suptitle(
        "Selected K=4 precipitation reconstruction quality",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig16_selected_k4_reconstruction_quality.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig16_selected_k4_reconstruction_quality.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        verification,
        all_runs,
        selected_metadata,
    ) = load_and_verify_selection()

    (
        components,
        loadings,
        reconstruction,
    ) = load_selected_model()

    component_summary = (
        build_component_summary(
            components,
            loadings,
        )
    )

    contributions = (
        calculate_component_contributions(
            components,
            loadings,
            reconstruction,
        )
    )

    reconstruction_metrics = (
        calculate_reconstruction_metrics(
            reconstruction,
            loadings,
        )
    )

    (
        annual_contributions,
        dominant_annual,
        dominant_monthly,
    ) = calculate_dominance_tables(
        contributions
    )

    component_summary_path = (
        TABLE_DIR
        / "selected_k4_component_summary.csv"
    )

    contribution_path = (
        OUTPUT_DIR
        / "regional_monthly_component_contributions.csv"
    )

    reconstruction_metrics_path = (
        TABLE_DIR
        / "selected_k4_regional_reconstruction_metrics.csv"
    )

    annual_contribution_path = (
        TABLE_DIR
        / "selected_k4_annual_component_contributions.csv"
    )

    dominant_annual_path = (
        TABLE_DIR
        / "selected_k4_dominant_annual_component.csv"
    )

    dominant_monthly_path = (
        TABLE_DIR
        / "selected_k4_dominant_monthly_component.csv"
    )

    component_summary.to_csv(
        component_summary_path,
        index=False,
    )

    contributions.to_csv(
        contribution_path,
        index=False,
    )

    reconstruction_metrics.to_csv(
        reconstruction_metrics_path,
        index=False,
    )

    annual_contributions.to_csv(
        annual_contribution_path,
        index=False,
    )

    dominant_annual.to_csv(
        dominant_annual_path,
        index=False,
    )

    dominant_monthly.to_csv(
        dominant_monthly_path,
        index=False,
    )

    # Preserve the exact selected verified model files.
    for filename in [
        "seasonal_components.csv",
        "regional_component_loadings.csv",
        "precipitation_reconstruction.csv",
        "model_metadata.json",
    ]:
        shutil.copy2(
            SOURCE_DIR / filename,
            OUTPUT_DIR / filename,
        )

    selected_verification = (
        verification.loc[
            verification["component_count"]
            == SELECTED_K
        ].iloc[0]
    )

    sensitivity_verification = (
        verification.loc[
            verification["component_count"]
            == SENSITIVITY_K
        ].iloc[0]
    )

    global_observed = reconstruction[
        "observed_precipitation_mm"
    ].to_numpy(dtype=float)

    global_predicted = reconstruction[
        "reconstructed_precipitation_mm"
    ].to_numpy(dtype=float)

    global_residual = (
        global_observed - global_predicted
    )

    converged_selected_runs = all_runs.loc[
        (
            all_runs["component_count"]
            == SELECTED_K
        )
        & all_runs["converged"]
    ]

    summary = {
        "status": "PASS",
        "selected_primary_component_count": (
            SELECTED_K
        ),
        "sensitivity_component_count": (
            SENSITIVITY_K
        ),
        "selected_best_model_converged": bool(
            selected_metadata["converged"]
        ),
        "selected_best_model_iterations": int(
            selected_metadata["iterations"]
        ),
        "selected_model_mean_component_stability": float(
            selected_verification[
                "mean_component_stability"
            ]
        ),
        "selected_model_minimum_run_stability": float(
            selected_verification[
                "minimum_run_stability"
            ]
        ),
        "selected_model_convergence_fraction": float(
            selected_verification[
                "converged_fraction"
            ]
        ),
        "selected_model_converged_run_count": int(
            len(converged_selected_runs)
        ),
        "selected_model_maximum_pairwise_component_similarity": float(
            selected_verification[
                "maximum_pairwise_component_similarity"
            ]
        ),
        "relative_error_reduction_over_k3": float(
            selected_verification[
                "relative_error_reduction_from_k3"
            ]
        ),
        "explained_variance_gain_over_k3": float(
            selected_verification[
                "explained_variance_gain_from_k3"
            ]
        ),
        "selected_explained_variance": float(
            selected_verification[
                "median_explained_variance"
            ]
        ),
        "global_reconstruction_rmse_mm": float(
            np.sqrt(
                np.mean(global_residual**2)
            )
        ),
        "global_reconstruction_mae_mm": float(
            np.mean(
                np.abs(global_residual)
            )
        ),
        "global_reconstruction_r_squared": (
            safe_r_squared(
                global_observed,
                global_predicted,
            )
        ),
        "median_regional_normalized_rmse": float(
            reconstruction_metrics[
                "normalized_rmse_by_mean"
            ].median()
        ),
        "maximum_regional_normalized_rmse": float(
            reconstruction_metrics[
                "normalized_rmse_by_mean"
            ].max()
        ),
        "minimum_regional_r_squared": float(
            reconstruction_metrics[
                "r_squared"
            ].min()
        ),
        "component_labels_are_descriptive": True,
        "atmospheric_attribution_completed": False,
        "component_summary": str(
            component_summary_path
        ),
        "monthly_component_contributions": str(
            contribution_path
        ),
        "regional_reconstruction_metrics": str(
            reconstruction_metrics_path
        ),
        "annual_component_contributions": str(
            annual_contribution_path
        ),
        "dominant_annual_component_table": str(
            dominant_annual_path
        ),
        "dominant_monthly_component_table": str(
            dominant_monthly_path
        ),
    }

    summary_path = (
        TABLE_DIR
        / "selected_k4_nmf_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            summary,
            stream,
            indent=2,
        )

    plot_component_profiles(
        components
    )

    plot_dominant_component_map(
        dominant_annual
    )

    plot_reconstruction_quality(
        reconstruction,
        reconstruction_metrics,
    )

    print("=" * 78)
    print("FINAL SELECTED PRECIPITATION NMF MODEL")
    print("=" * 78)
    print(
        f"Primary component count          : "
        f"{SELECTED_K}"
    )
    print(
        f"Sensitivity component count      : "
        f"{SENSITIVITY_K}"
    )
    print(
        f"Best selected model converged     : "
        f"{summary['selected_best_model_converged']}"
    )
    print(
        f"Best selected model iterations    : "
        f"{summary['selected_best_model_iterations']}"
    )
    print(
        f"Mean component stability          : "
        f"{summary['selected_model_mean_component_stability']:.5f}"
    )
    print(
        f"Convergence fraction              : "
        f"{summary['selected_model_convergence_fraction']:.2%}"
    )
    print(
        f"Converged K=4 runs                : "
        f"{summary['selected_model_converged_run_count']}/100"
    )
    print(
        f"Maximum component similarity      : "
        f"{summary['selected_model_maximum_pairwise_component_similarity']:.5f}"
    )
    print(
        f"Error reduction over K=3          : "
        f"{summary['relative_error_reduction_over_k3']:.2%}"
    )
    print(
        f"Explained variance                : "
        f"{summary['selected_explained_variance']:.5f}"
    )
    print(
        f"Global reconstruction RMSE        : "
        f"{summary['global_reconstruction_rmse_mm']:.3f} mm"
    )
    print(
        f"Global reconstruction R²          : "
        f"{summary['global_reconstruction_r_squared']:.5f}"
    )
    print(
        f"Median regional normalized RMSE   : "
        f"{summary['median_regional_normalized_rmse']:.4f}"
    )
    print(
        f"Maximum regional normalized RMSE  : "
        f"{summary['maximum_regional_normalized_rmse']:.4f}"
    )
    print()
    print("Selected component descriptions:")

    print(
        component_summary[
            [
                "component",
                "descriptive_label",
                "peak_month_name",
                "second_peak_month_name",
                "top_region_1",
                "top_region_2",
                "top_region_3",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Selected-model summary            : {summary_path}")
    print(f"Component contributions           : {contribution_path}")
    print(f"Reconstruction metrics            : {reconstruction_metrics_path}")
    print(f"Selected model directory          : {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
