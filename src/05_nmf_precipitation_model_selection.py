#!/usr/bin/env python3
"""
Robust NMF model selection for Malaysian seasonal precipitation.

The input matrix has:
    rows    = 172 regional units
    columns = 12 monthly precipitation climatology values

For each candidate component count K:
1. Fit NMF across 100 random initializations.
2. Record reconstruction error and explained variance.
3. Match components across runs using the Hungarian algorithm.
4. Quantify stability of seasonal component shapes.
5. Save the best factorization and reconstruction for each K.

No final component count is selected automatically.
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

NMF_DIR = ROOT / "results" / "nmf"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

K_VALUES = list(range(2, 9))
RUNS_PER_K = 100
BASE_RANDOM_SEED = 20260805
MAX_ITERATIONS = 5000
TOLERANCE = 1e-8

EXPECTED_REGION_COUNT = 172
EXPECTED_MONTH_COUNT = 12


def normalize_component_rows(
    components: np.ndarray,
) -> np.ndarray:
    """Normalize each component to unit L2 norm."""
    norms = np.linalg.norm(
        components,
        axis=1,
        keepdims=True,
    )

    norms = np.where(norms > 1e-12, norms, 1.0)

    return components / norms


def calculate_explained_variance(
    observed: np.ndarray,
    reconstructed: np.ndarray,
) -> float:
    residual_sum_squares = float(
        np.sum((observed - reconstructed) ** 2)
    )

    total_sum_squares = float(
        np.sum((observed - observed.mean()) ** 2)
    )

    if total_sum_squares <= 1e-12:
        return float("nan")

    return 1.0 - residual_sum_squares / total_sum_squares


def component_stability_against_reference(
    reference_components: np.ndarray,
    candidate_components: np.ndarray,
) -> tuple[float, float, list[float]]:
    """
    Match candidate components to the reference using cosine similarity.

    Returns:
        mean matched similarity,
        minimum matched similarity,
        all matched similarities.
    """
    reference_normalized = normalize_component_rows(
        reference_components
    )

    candidate_normalized = normalize_component_rows(
        candidate_components
    )

    similarity_matrix = (
        reference_normalized
        @ candidate_normalized.T
    )

    reference_indices, candidate_indices = (
        linear_sum_assignment(
            -similarity_matrix
        )
    )

    similarities = similarity_matrix[
        reference_indices,
        candidate_indices,
    ].tolist()

    return (
        float(np.mean(similarities)),
        float(np.min(similarities)),
        [float(value) for value in similarities],
    )


def prepare_input_matrix() -> tuple[
    np.ndarray,
    list[str],
    list[int],
    pd.DataFrame,
]:
    if not CLIMATOLOGY_PATH.exists():
        raise FileNotFoundError(CLIMATOLOGY_PATH)

    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(INVENTORY_PATH)

    climatology = pd.read_csv(CLIMATOLOGY_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    required_column = "precipitation_mm__mean"

    if required_column not in climatology.columns:
        raise RuntimeError(
            f"Missing column: {required_column}"
        )

    precipitation_table = (
        climatology.pivot(
            index="region_id",
            columns="month",
            values=required_column,
        )
        .sort_index()
    )

    precipitation_table = precipitation_table.reindex(
        columns=range(1, 13)
    )

    if precipitation_table.shape != (
        EXPECTED_REGION_COUNT,
        EXPECTED_MONTH_COUNT,
    ):
        raise RuntimeError(
            "Expected precipitation matrix shape "
            f"({EXPECTED_REGION_COUNT}, "
            f"{EXPECTED_MONTH_COUNT}), found "
            f"{precipitation_table.shape}."
        )

    if precipitation_table.isna().any().any():
        raise RuntimeError(
            "Precipitation matrix contains missing values."
        )

    matrix = precipitation_table.to_numpy(
        dtype=float
    )

    if np.any(matrix < 0):
        raise RuntimeError(
            "NMF input contains negative precipitation."
        )

    if not np.isfinite(matrix).all():
        raise RuntimeError(
            "NMF input contains non-finite values."
        )

    region_ids = precipitation_table.index.tolist()
    months = precipitation_table.columns.astype(
        int
    ).tolist()

    inventory_subset = (
        inventory.set_index("region_id")
        .loc[region_ids]
        .reset_index()
    )

    input_long = (
        precipitation_table
        .reset_index()
        .melt(
            id_vars="region_id",
            var_name="month",
            value_name="precipitation_mm",
        )
    )

    input_long.to_csv(
        NMF_DIR / "nmf_input_precipitation_matrix.csv",
        index=False,
    )

    return (
        matrix,
        region_ids,
        months,
        inventory_subset,
    )


def fit_single_nmf(
    matrix: np.ndarray,
    component_count: int,
    random_seed: int,
) -> dict[str, object]:
    model = NMF(
        n_components=component_count,
        init="random",
        solver="mu",
        beta_loss="frobenius",
        max_iter=MAX_ITERATIONS,
        tol=TOLERANCE,
        random_state=random_seed,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        weights = model.fit_transform(matrix)
        components = model.components_

    reconstructed = weights @ components

    reconstruction_norm = float(
        np.linalg.norm(
            matrix - reconstructed,
            ord="fro",
        )
    )

    residual_sum_squares = (
        reconstruction_norm**2
    )

    relative_frobenius_error = float(
        reconstruction_norm
        / np.linalg.norm(matrix, ord="fro")
    )

    explained_variance = (
        calculate_explained_variance(
            matrix,
            reconstructed,
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
        "rss": float(residual_sum_squares),
        "relative_frobenius_error": (
            relative_frobenius_error
        ),
        "explained_variance": float(
            explained_variance
        ),
        "iterations": int(model.n_iter_),
        "converged": converged,
        "convergence_warning": (
            convergence_warning
        ),
        "random_seed": random_seed,
    }


def save_best_factorization(
    component_count: int,
    best_result: dict[str, object],
    matrix: np.ndarray,
    region_ids: list[str],
    months: list[int],
    inventory: pd.DataFrame,
) -> None:
    output_directory = (
        NMF_DIR / f"k_{component_count:02d}"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights = np.asarray(
        best_result["weights"],
        dtype=float,
    )

    components = np.asarray(
        best_result["components"],
        dtype=float,
    )

    reconstructed = np.asarray(
        best_result["reconstructed"],
        dtype=float,
    )

    # Resolve NMF scale ambiguity by normalizing each seasonal
    # component to sum to one and transferring scale into weights.
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

    component_names = [
        f"C{index + 1}"
        for index in range(component_count)
    ]

    component_rows = []

    for component_index, component_name in enumerate(
        component_names
    ):
        for month_index, month in enumerate(months):
            component_rows.append(
                {
                    "component": component_name,
                    "component_index": (
                        component_index + 1
                    ),
                    "month": month,
                    "raw_component_value": float(
                        components[
                            component_index,
                            month_index,
                        ]
                    ),
                    "normalized_seasonal_profile": float(
                        normalized_components[
                            component_index,
                            month_index,
                        ]
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

    for component_index, component_name in enumerate(
        component_names
    ):
        loading_table[
            f"{component_name}_scaled_loading"
        ] = scaled_weights[:, component_index]

        loading_table[
            f"{component_name}_raw_weight"
        ] = weights[:, component_index]

    reconstruction_rows = []

    for region_index, region_id in enumerate(
        region_ids
    ):
        for month_index, month in enumerate(months):
            observed_value = float(
                matrix[region_index, month_index]
            )

            reconstructed_value = float(
                reconstructed[
                    region_index,
                    month_index,
                ]
            )

            reconstruction_rows.append(
                {
                    "region_id": region_id,
                    "month": month,
                    "observed_precipitation_mm": (
                        observed_value
                    ),
                    "reconstructed_precipitation_mm": (
                        reconstructed_value
                    ),
                    "residual_mm": (
                        observed_value
                        - reconstructed_value
                    ),
                    "absolute_residual_mm": abs(
                        observed_value
                        - reconstructed_value
                    ),
                }
            )

    reconstruction_table = pd.DataFrame(
        reconstruction_rows
    )

    component_table.to_csv(
        output_directory
        / "seasonal_components.csv",
        index=False,
    )

    loading_table.to_csv(
        output_directory
        / "regional_component_loadings.csv",
        index=False,
    )

    reconstruction_table.to_csv(
        output_directory
        / "precipitation_reconstruction.csv",
        index=False,
    )

    metadata = {
        "component_count": component_count,
        "best_random_seed": int(
            best_result["random_seed"]
        ),
        "rss": float(best_result["rss"]),
        "relative_frobenius_error": float(
            best_result[
                "relative_frobenius_error"
            ]
        ),
        "explained_variance": float(
            best_result["explained_variance"]
        ),
        "iterations": int(
            best_result["iterations"]
        ),
        "converged": bool(
            best_result["converged"]
        ),
        "factorization_identity": (
            "X approximately equals "
            "scaled_weights multiplied by "
            "normalized_components"
        ),
    }

    with (
        output_directory / "model_metadata.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def make_model_selection_figure(
    summary: pd.DataFrame,
) -> None:
    component_counts = summary[
        "component_count"
    ].to_numpy()

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(13.5, 4.2),
        constrained_layout=True,
    )

    axes[0].plot(
        component_counts,
        summary[
            "median_relative_frobenius_error"
        ],
        marker="o",
        linewidth=1.5,
        color="black",
    )

    axes[0].fill_between(
        component_counts,
        summary[
            "q25_relative_frobenius_error"
        ],
        summary[
            "q75_relative_frobenius_error"
        ],
        color="0.82",
    )

    axes[0].set_xlabel("Number of components, K")
    axes[0].set_ylabel(
        "Relative reconstruction error"
    )
    axes[0].set_title("Reconstruction error")

    axes[1].plot(
        component_counts,
        summary[
            "median_explained_variance"
        ],
        marker="o",
        linewidth=1.5,
        color="black",
    )

    axes[1].fill_between(
        component_counts,
        summary[
            "q25_explained_variance"
        ],
        summary[
            "q75_explained_variance"
        ],
        color="0.82",
    )

    axes[1].set_xlabel("Number of components, K")
    axes[1].set_ylabel("Explained variance")
    axes[1].set_title("Explained precipitation structure")

    axes[2].plot(
        component_counts,
        summary[
            "mean_component_stability"
        ],
        marker="o",
        linewidth=1.5,
        color="black",
        label="Mean stability",
    )

    axes[2].plot(
        component_counts,
        summary[
            "minimum_run_stability"
        ],
        marker="s",
        linewidth=1.2,
        linestyle="--",
        color="0.45",
        label="Minimum run stability",
    )

    axes[2].set_xlabel("Number of components, K")
    axes[2].set_ylabel(
        "Matched cosine similarity"
    )
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_title("Component stability")
    axes[2].legend(
        frameon=True,
        fontsize=8,
    )

    for axis in axes:
        axis.set_xticks(component_counts)
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "NMF model-selection diagnostics for "
        "Malaysian seasonal precipitation",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig12_nmf_model_selection.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig12_nmf_model_selection.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    NMF_DIR.mkdir(
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
        months,
        inventory,
    ) = prepare_input_matrix()

    print("=" * 78)
    print("NMF MODEL SELECTION — SEASONAL PRECIPITATION")
    print("=" * 78)
    print(f"Input matrix shape    : {matrix.shape}")
    print(f"Candidate K values    : {K_VALUES}")
    print(f"Initializations per K : {RUNS_PER_K}")
    print(f"Maximum iterations    : {MAX_ITERATIONS}")
    print()

    all_run_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for component_count in K_VALUES:
        print(
            f"Fitting K={component_count} "
            f"({RUNS_PER_K} initializations)"
        )

        run_results: list[dict[str, object]] = []

        for run_index in range(RUNS_PER_K):
            random_seed = (
                BASE_RANDOM_SEED
                + component_count * 10_000
                + run_index
            )

            result = fit_single_nmf(
                matrix,
                component_count,
                random_seed,
            )

            result["run_index"] = run_index
            run_results.append(result)

        best_result = min(
            run_results,
            key=lambda item: item["rss"],
        )

        reference_components = np.asarray(
            best_result["components"],
            dtype=float,
        )

        run_stabilities = []

        for result in run_results:
            (
                mean_stability,
                minimum_stability,
                matched_similarities,
            ) = component_stability_against_reference(
                reference_components,
                np.asarray(
                    result["components"],
                    dtype=float,
                ),
            )

            result["mean_component_stability"] = (
                mean_stability
            )

            result["minimum_component_stability"] = (
                minimum_stability
            )

            result["matched_component_similarities"] = (
                matched_similarities
            )

            run_stabilities.append(
                mean_stability
            )

            all_run_rows.append(
                {
                    "component_count": component_count,
                    "run_index": int(
                        result["run_index"]
                    ),
                    "random_seed": int(
                        result["random_seed"]
                    ),
                    "rss": float(result["rss"]),
                    "relative_frobenius_error": float(
                        result[
                            "relative_frobenius_error"
                        ]
                    ),
                    "explained_variance": float(
                        result["explained_variance"]
                    ),
                    "iterations": int(
                        result["iterations"]
                    ),
                    "converged": bool(
                        result["converged"]
                    ),
                    "convergence_warning": bool(
                        result[
                            "convergence_warning"
                        ]
                    ),
                    "mean_component_stability": (
                        mean_stability
                    ),
                    "minimum_component_stability": (
                        minimum_stability
                    ),
                }
            )

        rss_values = np.asarray(
            [
                result["rss"]
                for result in run_results
            ],
            dtype=float,
        )

        relative_errors = np.asarray(
            [
                result[
                    "relative_frobenius_error"
                ]
                for result in run_results
            ],
            dtype=float,
        )

        explained_variances = np.asarray(
            [
                result["explained_variance"]
                for result in run_results
            ],
            dtype=float,
        )

        iterations = np.asarray(
            [
                result["iterations"]
                for result in run_results
            ],
            dtype=int,
        )

        convergence_values = np.asarray(
            [
                result["converged"]
                for result in run_results
            ],
            dtype=bool,
        )

        stability_values = np.asarray(
            run_stabilities,
            dtype=float,
        )

        summary_rows.append(
            {
                "component_count": component_count,
                "best_rss": float(
                    rss_values.min()
                ),
                "median_rss": float(
                    np.median(rss_values)
                ),
                "q25_rss": float(
                    np.quantile(rss_values, 0.25)
                ),
                "q75_rss": float(
                    np.quantile(rss_values, 0.75)
                ),
                "best_relative_frobenius_error": float(
                    relative_errors.min()
                ),
                "median_relative_frobenius_error": float(
                    np.median(relative_errors)
                ),
                "q25_relative_frobenius_error": float(
                    np.quantile(
                        relative_errors,
                        0.25,
                    )
                ),
                "q75_relative_frobenius_error": float(
                    np.quantile(
                        relative_errors,
                        0.75,
                    )
                ),
                "best_explained_variance": float(
                    explained_variances.max()
                ),
                "median_explained_variance": float(
                    np.median(
                        explained_variances
                    )
                ),
                "q25_explained_variance": float(
                    np.quantile(
                        explained_variances,
                        0.25,
                    )
                ),
                "q75_explained_variance": float(
                    np.quantile(
                        explained_variances,
                        0.75,
                    )
                ),
                "mean_component_stability": float(
                    stability_values.mean()
                ),
                "median_component_stability": float(
                    np.median(stability_values)
                ),
                "minimum_run_stability": float(
                    stability_values.min()
                ),
                "converged_fraction": float(
                    convergence_values.mean()
                ),
                "median_iterations": float(
                    np.median(iterations)
                ),
                "best_random_seed": int(
                    best_result["random_seed"]
                ),
            }
        )

        save_best_factorization(
            component_count,
            best_result,
            matrix,
            region_ids,
            months,
            inventory,
        )

        print(
            f"  best relative error="
            f"{relative_errors.min():.5f}, "
            f"median explained variance="
            f"{np.median(explained_variances):.5f}, "
            f"mean stability="
            f"{stability_values.mean():.5f}, "
            f"converged="
            f"{convergence_values.mean():.2%}"
        )

    runs_table = pd.DataFrame(
        all_run_rows
    )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values("component_count")

    summary[
        "marginal_median_error_reduction"
    ] = np.nan

    for index in range(1, len(summary)):
        previous_error = float(
            summary.iloc[index - 1][
                "median_relative_frobenius_error"
            ]
        )

        current_error = float(
            summary.iloc[index][
                "median_relative_frobenius_error"
            ]
        )

        summary.loc[
            summary.index[index],
            "marginal_median_error_reduction",
        ] = (
            previous_error - current_error
        ) / previous_error

    runs_path = (
        TABLE_DIR / "nmf_all_initializations.csv"
    )

    summary_path = (
        TABLE_DIR
        / "nmf_model_selection_summary.csv"
    )

    metadata_path = (
        TABLE_DIR
        / "nmf_model_selection_metadata.json"
    )

    runs_table.to_csv(
        runs_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    metadata = {
        "status": "PASS",
        "input_climatology": str(
            CLIMATOLOGY_PATH
        ),
        "matrix_shape": list(matrix.shape),
        "candidate_component_counts": K_VALUES,
        "runs_per_component_count": (
            RUNS_PER_K
        ),
        "base_random_seed": BASE_RANDOM_SEED,
        "maximum_iterations": MAX_ITERATIONS,
        "tolerance": TOLERANCE,
        "solver": "multiplicative update",
        "loss": "Frobenius",
        "normalization": (
            "No normalization was applied to the input "
            "precipitation matrix. Best-factor component "
            "profiles were sum-normalized only for reporting, "
            "with scale transferred into regional loadings."
        ),
        "selection_policy": (
            "Inspect reconstruction elbow, marginal error "
            "reduction, explained variance and component "
            "stability. No K is selected automatically."
        ),
        "all_initializations_table": str(
            runs_path
        ),
        "model_selection_summary": str(
            summary_path
        ),
        "factorizations_directory": str(
            NMF_DIR
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            metadata,
            stream,
            indent=2,
        )

    make_model_selection_figure(summary)

    print()
    print("=" * 78)
    print("NMF MODEL-SELECTION SUMMARY")
    print("=" * 78)

    display_columns = [
        "component_count",
        "median_relative_frobenius_error",
        "marginal_median_error_reduction",
        "median_explained_variance",
        "mean_component_stability",
        "minimum_run_stability",
        "converged_fraction",
    ]

    print(
        summary[display_columns]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )

    print()
    print(f"Summary table       : {summary_path}")
    print(f"All runs            : {runs_path}")
    print(f"Best factorizations : {NMF_DIR}")
    print(
        f"Model-selection plot: "
        f"{FIGURE_DIR / 'fig12_nmf_model_selection.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
