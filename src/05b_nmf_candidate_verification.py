#!/usr/bin/env python3
"""
Independent verification of the K=3 and K=4 precipitation NMF models.

This stage uses:
- coordinate-descent optimization;
- randomized NNDSVD initialization;
- 100 runs per candidate;
- component matching across runs;
- seasonal-profile diagnostics;
- spatial loading maps;
- component redundancy diagnostics.

The purpose is to distinguish a robust three-component model from a
potentially interpretable four-component model.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]

CLIMATOLOGY_PATH = (
    ROOT
    / "data"
    / "processed"
    / "regional_monthly_climatology_1982_2011_wide.csv"
)

INVENTORY_PATH = (
    ROOT
    / "results"
    / "tables"
    / "region_inventory_0p5degree.csv"
)

OUTPUT_DIR = ROOT / "results" / "nmf_verified"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

K_VALUES = [3, 4]
RUNS_PER_K = 100
BASE_RANDOM_SEED = 20260806

MAX_ITERATIONS = 20_000
TOLERANCE = 1e-6

EXPECTED_REGIONS = 172
EXPECTED_MONTHS = 12

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


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms = np.where(
        norms > 1e-12,
        norms,
        1.0,
    )

    return matrix / norms


def explained_variance(
    observed: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    residual_sum_squares = float(
        np.sum(
            (observed - reconstructed) ** 2
        )
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


def match_components(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """
    Reorder candidate components to match reference components.
    """
    reference_normalized = normalize_rows(reference)
    candidate_normalized = normalize_rows(candidate)

    similarity = (
        reference_normalized
        @ candidate_normalized.T
    )

    reference_indices, candidate_indices = (
        linear_sum_assignment(-similarity)
    )

    order = np.empty(
        len(reference_indices),
        dtype=int,
    )

    matched_similarities = np.empty(
        len(reference_indices),
        dtype=float,
    )

    for reference_index, candidate_index in zip(
        reference_indices,
        candidate_indices,
    ):
        order[reference_index] = candidate_index

        matched_similarities[reference_index] = (
            similarity[
                reference_index,
                candidate_index,
            ]
        )

    return (
        order,
        matched_similarities.tolist(),
    )


def maximum_pairwise_component_similarity(
    components: np.ndarray,
) -> float:
    normalized = normalize_rows(components)

    similarity = normalized @ normalized.T

    upper_triangle = similarity[
        np.triu_indices(
            similarity.shape[0],
            k=1,
        )
    ]

    if len(upper_triangle) == 0:
        return float("nan")

    return float(upper_triangle.max())


def normalized_entropy(
    profile: np.ndarray,
) -> float:
    profile = np.asarray(
        profile,
        dtype=float,
    )

    total = float(profile.sum())

    if total <= 1e-12:
        return float("nan")

    probabilities = profile / total

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -float(
        np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    return entropy / np.log(len(profile))


def load_input() -> tuple[
    np.ndarray,
    list[str],
    pd.DataFrame,
]:
    climatology = pd.read_csv(
        CLIMATOLOGY_PATH
    )

    inventory = pd.read_csv(
        INVENTORY_PATH
    )

    matrix_table = (
        climatology.pivot(
            index="region_id",
            columns="month",
            values="precipitation_mm__mean",
        )
        .sort_index()
        .reindex(columns=range(1, 13))
    )

    if matrix_table.shape != (
        EXPECTED_REGIONS,
        EXPECTED_MONTHS,
    ):
        raise RuntimeError(
            f"Unexpected matrix shape: "
            f"{matrix_table.shape}"
        )

    if matrix_table.isna().any().any():
        raise RuntimeError(
            "Precipitation matrix contains missing values."
        )

    matrix = matrix_table.to_numpy(
        dtype=float
    )

    if np.any(matrix < 0):
        raise RuntimeError(
            "Precipitation matrix contains negative values."
        )

    region_ids = matrix_table.index.tolist()

    inventory = (
        inventory.set_index("region_id")
        .loc[region_ids]
        .reset_index()
    )

    return matrix, region_ids, inventory


def fit_model(
    matrix: np.ndarray,
    component_count: int,
    random_seed: int,
) -> dict[str, object]:
    model = NMF(
        n_components=component_count,
        init="nndsvdar",
        solver="cd",
        max_iter=MAX_ITERATIONS,
        tol=TOLERANCE,
        random_state=random_seed,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
        shuffle=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        weights = model.fit_transform(matrix)
        components = model.components_

    reconstructed = weights @ components

    residual = matrix - reconstructed

    rss = float(
        np.sum(residual**2)
    )

    relative_error = float(
        np.linalg.norm(
            residual,
            ord="fro",
        )
        / np.linalg.norm(
            matrix,
            ord="fro",
        )
    )

    convergence_warning = any(
        issubclass(
            warning.category,
            ConvergenceWarning,
        )
        for warning in caught
    )

    converged = bool(
        model.n_iter_ < MAX_ITERATIONS
        and not convergence_warning
    )

    return {
        "weights": weights,
        "components": components,
        "reconstructed": reconstructed,
        "rss": rss,
        "relative_error": relative_error,
        "explained_variance": explained_variance(
            matrix,
            reconstructed,
        ),
        "iterations": int(model.n_iter_),
        "converged": converged,
        "random_seed": int(random_seed),
    }


def order_components_by_peak(
    weights: np.ndarray,
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    peak_month_indices = np.argmax(
        components,
        axis=1,
    )

    order = np.argsort(
        peak_month_indices,
        kind="stable",
    )

    return (
        weights[:, order],
        components[order, :],
    )


def normalize_factorization(
    weights: np.ndarray,
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    component_sums = components.sum(
        axis=1,
        keepdims=True,
    )

    component_sums = np.where(
        component_sums > 1e-12,
        component_sums,
        1.0,
    )

    normalized_components = (
        components / component_sums
    )

    scaled_weights = (
        weights * component_sums.T
    )

    return (
        scaled_weights,
        normalized_components,
    )


def save_best_model(
    component_count: int,
    result: dict[str, object],
    matrix: np.ndarray,
    region_ids: list[str],
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_directory = (
        OUTPUT_DIR
        / f"k_{component_count:02d}"
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights = np.asarray(
        result["weights"],
        dtype=float,
    )

    components = np.asarray(
        result["components"],
        dtype=float,
    )

    weights, components = (
        order_components_by_peak(
            weights,
            components,
        )
    )

    (
        scaled_weights,
        normalized_components,
    ) = normalize_factorization(
        weights,
        components,
    )

    component_rows = []

    for component_index in range(
        component_count
    ):
        profile = normalized_components[
            component_index
        ]

        peak_month = int(
            np.argmax(profile) + 1
        )

        sorted_month_indices = np.argsort(
            profile
        )[::-1]

        second_peak_month = int(
            sorted_month_indices[1] + 1
        )

        component_name = (
            f"K{component_count}_C"
            f"{component_index + 1}"
        )

        for month in range(1, 13):
            component_rows.append(
                {
                    "component_count": (
                        component_count
                    ),
                    "component": component_name,
                    "component_index": (
                        component_index + 1
                    ),
                    "month": month,
                    "month_name": (
                        MONTH_NAMES[month - 1]
                    ),
                    "normalized_profile": float(
                        profile[month - 1]
                    ),
                    "peak_month": peak_month,
                    "second_peak_month": (
                        second_peak_month
                    ),
                    "normalized_entropy": (
                        normalized_entropy(profile)
                    ),
                    "profile_maximum": float(
                        profile.max()
                    ),
                }
            )

    component_table = pd.DataFrame(
        component_rows
    )

    loading_table = inventory[
        [
            "region_id",
            "native_cell_count",
            "longitude_mean",
            "latitude_mean",
            "region_lon_index",
            "region_lat_index",
            "small_region_lt_4_cells",
        ]
    ].copy()

    for component_index in range(
        component_count
    ):
        component_name = (
            f"K{component_count}_C"
            f"{component_index + 1}"
        )

        loading_table[
            f"{component_name}_loading"
        ] = scaled_weights[
            :,
            component_index,
        ]

        maximum_loading = float(
            scaled_weights[
                :,
                component_index,
            ].max()
        )

        if maximum_loading > 1e-12:
            loading_table[
                f"{component_name}_relative_loading"
            ] = (
                scaled_weights[
                    :,
                    component_index,
                ]
                / maximum_loading
            )
        else:
            loading_table[
                f"{component_name}_relative_loading"
            ] = 0.0

    reconstruction = (
        weights @ components
    )

    reconstruction_rows = []

    for region_index, region_id in enumerate(
        region_ids
    ):
        for month_index in range(12):
            observed = float(
                matrix[
                    region_index,
                    month_index,
                ]
            )

            predicted = float(
                reconstruction[
                    region_index,
                    month_index,
                ]
            )

            reconstruction_rows.append(
                {
                    "region_id": region_id,
                    "month": month_index + 1,
                    "observed_precipitation_mm": (
                        observed
                    ),
                    "reconstructed_precipitation_mm": (
                        predicted
                    ),
                    "residual_mm": (
                        observed - predicted
                    ),
                    "absolute_residual_mm": abs(
                        observed - predicted
                    ),
                }
            )

    reconstruction_table = pd.DataFrame(
        reconstruction_rows
    )

    component_table.to_csv(
        model_directory
        / "seasonal_components.csv",
        index=False,
    )

    loading_table.to_csv(
        model_directory
        / "regional_component_loadings.csv",
        index=False,
    )

    reconstruction_table.to_csv(
        model_directory
        / "precipitation_reconstruction.csv",
        index=False,
    )

    metadata = {
        "component_count": component_count,
        "random_seed": int(
            result["random_seed"]
        ),
        "solver": "coordinate descent",
        "initialization": "nndsvdar",
        "maximum_iterations": MAX_ITERATIONS,
        "tolerance": TOLERANCE,
        "iterations": int(
            result["iterations"]
        ),
        "converged": bool(
            result["converged"]
        ),
        "relative_error": float(
            result["relative_error"]
        ),
        "explained_variance": float(
            result["explained_variance"]
        ),
        "component_ordering": (
            "Components ordered by peak month."
        ),
    }

    with (
        model_directory / "model_metadata.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
        )

    return component_table, loading_table


def make_candidate_figure(
    component_count: int,
    component_table: pd.DataFrame,
    loading_table: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        nrows=2,
        ncols=component_count,
        figsize=(
            4.0 * component_count,
            7.2,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    for component_index in range(
        component_count
    ):
        component_name = (
            f"K{component_count}_C"
            f"{component_index + 1}"
        )

        profile = (
            component_table.loc[
                component_table["component"]
                == component_name
            ]
            .sort_values("month")
        )

        peak_month = int(
            profile["peak_month"].iloc[0]
        )

        entropy = float(
            profile[
                "normalized_entropy"
            ].iloc[0]
        )

        axes[0, component_index].plot(
            profile["month"],
            profile["normalized_profile"],
            marker="o",
            linewidth=1.5,
            color="black",
        )

        axes[0, component_index].set_xticks(
            range(1, 13)
        )

        axes[0, component_index].set_xticklabels(
            MONTH_NAMES,
            rotation=45,
            fontsize=7,
        )

        axes[0, component_index].set_title(
            f"{component_name}\n"
            f"peak={MONTH_NAMES[peak_month - 1]}, "
            f"entropy={entropy:.2f}"
        )

        axes[0, component_index].set_xlabel(
            "Month"
        )

        if component_index == 0:
            axes[0, component_index].set_ylabel(
                "Normalized seasonal profile"
            )

        loading_column = (
            f"{component_name}_relative_loading"
        )

        scatter = axes[1, component_index].scatter(
            loading_table["longitude_mean"],
            loading_table["latitude_mean"],
            c=loading_table[loading_column],
            s=(
                35
                + 2.0
                * loading_table[
                    "native_cell_count"
                ]
            ),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            edgecolor="black",
            linewidth=0.45,
        )

        small_regions = loading_table.loc[
            loading_table[
                "small_region_lt_4_cells"
            ]
        ]

        axes[1, component_index].scatter(
            small_regions["longitude_mean"],
            small_regions["latitude_mean"],
            marker="x",
            s=45,
            color="black",
            linewidth=1.0,
        )

        axes[1, component_index].set_title(
            f"{component_name} spatial loading"
        )

        axes[1, component_index].set_xlabel(
            "Longitude (°E)"
        )

        if component_index == 0:
            axes[1, component_index].set_ylabel(
                "Latitude (°N)"
            )

        axes[1, component_index].set_aspect(
            "equal",
            adjustable="box",
        )

        figure.colorbar(
            scatter,
            ax=axes[1, component_index],
            orientation="horizontal",
            fraction=0.06,
            pad=0.08,
            label="Relative regional loading",
        )

    for axis in axes.flat:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.55,
        )

    figure.suptitle(
        f"Verified K={component_count} precipitation "
        "NMF components and spatial loadings",
        fontsize=14,
    )

    figure.savefig(
        FIGURE_DIR
        / f"fig13_nmf_verified_k{component_count}.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / f"fig13_nmf_verified_k{component_count}.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def build_top_region_table(
    loading_tables: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for component_count, loading_table in (
        loading_tables.items()
    ):
        for component_index in range(
            component_count
        ):
            component_name = (
                f"K{component_count}_C"
                f"{component_index + 1}"
            )

            loading_column = (
                f"{component_name}_loading"
            )

            ranked = (
                loading_table.sort_values(
                    loading_column,
                    ascending=False,
                )
                .head(8)
                .reset_index(drop=True)
            )

            for rank, row in ranked.iterrows():
                rows.append(
                    {
                        "component_count": (
                            component_count
                        ),
                        "component": component_name,
                        "rank": rank + 1,
                        "region_id": row[
                            "region_id"
                        ],
                        "native_cell_count": int(
                            row[
                                "native_cell_count"
                            ]
                        ),
                        "longitude_mean": float(
                            row[
                                "longitude_mean"
                            ]
                        ),
                        "latitude_mean": float(
                            row[
                                "latitude_mean"
                            ]
                        ),
                        "loading": float(
                            row[
                                loading_column
                            ]
                        ),
                    }
                )

    return pd.DataFrame(rows)


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
        matrix,
        region_ids,
        inventory,
    ) = load_input()

    print("=" * 78)
    print("NMF K=3 AND K=4 INDEPENDENT VERIFICATION")
    print("=" * 78)
    print(f"Matrix shape            : {matrix.shape}")
    print(f"Candidate models        : {K_VALUES}")
    print(f"Runs per candidate      : {RUNS_PER_K}")
    print(f"Optimizer               : coordinate descent")
    print(f"Initialization          : randomized NNDSVD")
    print(f"Tolerance               : {TOLERANCE}")
    print(f"Maximum iterations      : {MAX_ITERATIONS}")
    print()

    all_run_rows = []
    candidate_rows = []
    component_tables = {}
    loading_tables = {}

    for component_count in K_VALUES:
        print(
            f"Fitting K={component_count}..."
        )

        results = []

        for run_index in range(
            RUNS_PER_K
        ):
            random_seed = (
                BASE_RANDOM_SEED
                + component_count * 10_000
                + run_index
            )

            result = fit_model(
                matrix,
                component_count,
                random_seed,
            )

            result["run_index"] = run_index
            results.append(result)

        best_result = min(
            results,
            key=lambda result: result["rss"],
        )

        reference_components = np.asarray(
            best_result["components"],
            dtype=float,
        )

        mean_stabilities = []
        minimum_stabilities = []

        for result in results:
            (
                component_order,
                similarities,
            ) = match_components(
                reference_components,
                np.asarray(
                    result["components"],
                    dtype=float,
                ),
            )

            mean_stability = float(
                np.mean(similarities)
            )

            minimum_stability = float(
                np.min(similarities)
            )

            mean_stabilities.append(
                mean_stability
            )

            minimum_stabilities.append(
                minimum_stability
            )

            all_run_rows.append(
                {
                    "component_count": (
                        component_count
                    ),
                    "run_index": int(
                        result["run_index"]
                    ),
                    "random_seed": int(
                        result["random_seed"]
                    ),
                    "rss": float(
                        result["rss"]
                    ),
                    "relative_error": float(
                        result["relative_error"]
                    ),
                    "explained_variance": float(
                        result[
                            "explained_variance"
                        ]
                    ),
                    "iterations": int(
                        result["iterations"]
                    ),
                    "converged": bool(
                        result["converged"]
                    ),
                    "mean_component_stability": (
                        mean_stability
                    ),
                    "minimum_component_stability": (
                        minimum_stability
                    ),
                }
            )

        errors = np.asarray(
            [
                result["relative_error"]
                for result in results
            ],
            dtype=float,
        )

        explained = np.asarray(
            [
                result["explained_variance"]
                for result in results
            ],
            dtype=float,
        )

        converged = np.asarray(
            [
                result["converged"]
                for result in results
            ],
            dtype=bool,
        )

        iterations = np.asarray(
            [
                result["iterations"]
                for result in results
            ],
            dtype=float,
        )

        best_components = np.asarray(
            best_result["components"],
            dtype=float,
        )

        candidate_rows.append(
            {
                "component_count": component_count,
                "best_relative_error": float(
                    errors.min()
                ),
                "median_relative_error": float(
                    np.median(errors)
                ),
                "q25_relative_error": float(
                    np.quantile(errors, 0.25)
                ),
                "q75_relative_error": float(
                    np.quantile(errors, 0.75)
                ),
                "best_explained_variance": float(
                    explained.max()
                ),
                "median_explained_variance": float(
                    np.median(explained)
                ),
                "mean_component_stability": float(
                    np.mean(mean_stabilities)
                ),
                "minimum_run_stability": float(
                    np.min(mean_stabilities)
                ),
                "minimum_component_similarity": float(
                    np.min(minimum_stabilities)
                ),
                "converged_fraction": float(
                    converged.mean()
                ),
                "median_iterations": float(
                    np.median(iterations)
                ),
                "maximum_pairwise_component_similarity": (
                    maximum_pairwise_component_similarity(
                        best_components
                    )
                ),
                "best_random_seed": int(
                    best_result["random_seed"]
                ),
            }
        )

        (
            component_table,
            loading_table,
        ) = save_best_model(
            component_count,
            best_result,
            matrix,
            region_ids,
            inventory,
        )

        component_tables[
            component_count
        ] = component_table

        loading_tables[
            component_count
        ] = loading_table

        make_candidate_figure(
            component_count,
            component_table,
            loading_table,
        )

        print(
            f"  median error="
            f"{np.median(errors):.6f}, "
            f"median EV="
            f"{np.median(explained):.6f}, "
            f"stability="
            f"{np.mean(mean_stabilities):.6f}, "
            f"converged="
            f"{converged.mean():.1%}"
        )

    candidate_summary = (
        pd.DataFrame(candidate_rows)
        .sort_values("component_count")
        .reset_index(drop=True)
    )

    k3_error = float(
        candidate_summary.loc[
            candidate_summary[
                "component_count"
            ] == 3,
            "median_relative_error",
        ].iloc[0]
    )

    k4_error = float(
        candidate_summary.loc[
            candidate_summary[
                "component_count"
            ] == 4,
            "median_relative_error",
        ].iloc[0]
    )

    k3_variance = float(
        candidate_summary.loc[
            candidate_summary[
                "component_count"
            ] == 3,
            "median_explained_variance",
        ].iloc[0]
    )

    k4_variance = float(
        candidate_summary.loc[
            candidate_summary[
                "component_count"
            ] == 4,
            "median_explained_variance",
        ].iloc[0]
    )

    candidate_summary[
        "relative_error_reduction_from_k3"
    ] = np.nan

    candidate_summary[
        "explained_variance_gain_from_k3"
    ] = np.nan

    candidate_summary.loc[
        candidate_summary[
            "component_count"
        ] == 4,
        "relative_error_reduction_from_k3",
    ] = (
        k3_error - k4_error
    ) / k3_error

    candidate_summary.loc[
        candidate_summary[
            "component_count"
        ] == 4,
        "explained_variance_gain_from_k3",
    ] = (
        k4_variance - k3_variance
    )

    all_runs = pd.DataFrame(
        all_run_rows
    )

    all_components = pd.concat(
        component_tables.values(),
        ignore_index=True,
    )

    top_regions = build_top_region_table(
        loading_tables
    )

    summary_path = (
        TABLE_DIR
        / "nmf_k3_k4_verification_summary.csv"
    )

    runs_path = (
        TABLE_DIR
        / "nmf_k3_k4_verification_all_runs.csv"
    )

    components_path = (
        TABLE_DIR
        / "nmf_k3_k4_component_diagnostics.csv"
    )

    top_regions_path = (
        TABLE_DIR
        / "nmf_k3_k4_top_regions.csv"
    )

    candidate_summary.to_csv(
        summary_path,
        index=False,
    )

    all_runs.to_csv(
        runs_path,
        index=False,
    )

    all_components.to_csv(
        components_path,
        index=False,
    )

    top_regions.to_csv(
        top_regions_path,
        index=False,
    )

    metadata = {
        "status": "PASS",
        "primary_candidate": 3,
        "sensitivity_candidate": 4,
        "final_selection_pending": (
            "Inspect seasonal profiles, spatial loadings, "
            "component redundancy and convergence."
        ),
        "candidate_summary": str(
            summary_path
        ),
        "component_diagnostics": str(
            components_path
        ),
        "top_region_table": str(
            top_regions_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "nmf_k3_k4_verification_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
        )

    print()
    print("=" * 78)
    print("VERIFIED CANDIDATE SUMMARY")
    print("=" * 78)

    display_columns = [
        "component_count",
        "median_relative_error",
        "median_explained_variance",
        "mean_component_stability",
        "minimum_run_stability",
        "converged_fraction",
        "median_iterations",
        "maximum_pairwise_component_similarity",
        "relative_error_reduction_from_k3",
        "explained_variance_gain_from_k3",
    ]

    print(
        candidate_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Summary table       : {summary_path}")
    print(f"Component diagnostics: {components_path}")
    print(f"Top-region table    : {top_regions_path}")
    print(f"Verified models     : {OUTPUT_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
