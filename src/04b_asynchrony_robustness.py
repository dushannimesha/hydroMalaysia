#!/usr/bin/env python3
"""
Robustness analysis for precipitation-soil-moisture asynchrony.

For each 0.5-degree region:
1. Recalculate the observed climatological lag.
2. Compare lagged correlation against same-month correlation.
3. Bootstrap the 30 years with replacement.
4. Estimate support for the observed lag and positive asynchrony.
5. Repeat summaries after excluding small coastal regions.
6. Quantify occurrences where AET exceeds PET.

No records are modified.
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

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "malaysia_hydrology_regional_monthly_0p5degree.parquet"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

BOOTSTRAP_ITERATIONS = 1000
MAXIMUM_LAG = 6
RANDOM_SEED = 20260805

EXPECTED_YEARS = list(range(1982, 2012))
EXPECTED_REGIONS = 172


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)

    valid = np.isfinite(first) & np.isfinite(second)

    first = first[valid]
    second = second[valid]

    if len(first) < 3:
        return float("nan")

    if np.std(first) < 1e-12:
        return float("nan")

    if np.std(second) < 1e-12:
        return float("nan")

    return float(np.corrcoef(first, second)[0, 1])


def calculate_lag_correlations(
    precipitation: np.ndarray,
    soil_moisture: np.ndarray,
) -> list[float]:
    correlations = []

    for lag in range(MAXIMUM_LAG + 1):
        antecedent_precipitation = np.roll(
            precipitation,
            lag,
        )

        correlations.append(
            safe_correlation(
                antecedent_precipitation,
                soil_moisture,
            )
        )

    return correlations


def find_best_lag(
    precipitation: np.ndarray,
    soil_moisture: np.ndarray,
) -> tuple[int, float, list[float]]:
    correlations = calculate_lag_correlations(
        precipitation,
        soil_moisture,
    )

    correlation_array = np.asarray(
        correlations,
        dtype=float,
    )

    if np.all(~np.isfinite(correlation_array)):
        return 0, float("nan"), correlations

    best_lag = int(
        np.nanargmax(correlation_array)
    )

    return (
        best_lag,
        float(correlation_array[best_lag]),
        correlations,
    )


def prepare_year_arrays(
    region_data: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    precipitation_by_year = {}
    soil_moisture_by_year = {}

    for year, year_data in region_data.groupby(
        "year",
        observed=True,
    ):
        year_data = (
            year_data
            .sort_values("month")
            .reset_index(drop=True)
        )

        if year_data["month"].tolist() != list(range(1, 13)):
            raise RuntimeError(
                f"Region {region_data['region_id'].iloc[0]}, "
                f"year {year}, does not contain months 1-12."
            )

        precipitation_by_year[int(year)] = (
            year_data["precipitation_mm"]
            .to_numpy(dtype=float)
        )

        soil_moisture_by_year[int(year)] = (
            year_data["soil_moisture_total_mm"]
            .to_numpy(dtype=float)
        )

    return precipitation_by_year, soil_moisture_by_year


def classify_robustness(
    observed_delta_r: float,
    observed_lag_support: float,
    positive_lag_support: float,
) -> str:
    if (
        observed_delta_r >= 0.10
        and observed_lag_support >= 0.80
        and positive_lag_support >= 0.90
    ):
        return "strong"

    if (
        observed_delta_r >= 0.05
        and observed_lag_support >= 0.60
        and positive_lag_support >= 0.75
    ):
        return "moderate"

    return "weak"


def bootstrap_region(
    region_data: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, object]:
    region_id = str(region_data["region_id"].iloc[0])

    (
        precipitation_by_year,
        soil_moisture_by_year,
    ) = prepare_year_arrays(region_data)

    available_years = sorted(
        precipitation_by_year.keys()
    )

    if available_years != EXPECTED_YEARS:
        raise RuntimeError(
            f"{region_id}: unexpected year coverage."
        )

    observed_precipitation = np.mean(
        [
            precipitation_by_year[year]
            for year in available_years
        ],
        axis=0,
    )

    observed_soil_moisture = np.mean(
        [
            soil_moisture_by_year[year]
            for year in available_years
        ],
        axis=0,
    )

    (
        observed_best_lag,
        observed_best_correlation,
        observed_correlations,
    ) = find_best_lag(
        observed_precipitation,
        observed_soil_moisture,
    )

    observed_same_month_correlation = float(
        observed_correlations[0]
    )

    observed_delta_r = float(
        observed_best_correlation
        - observed_same_month_correlation
    )

    bootstrap_best_lags = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=int,
    )

    bootstrap_same_month_correlations = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    bootstrap_best_correlations = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    bootstrap_delta_r = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    number_of_years = len(available_years)

    for iteration in range(BOOTSTRAP_ITERATIONS):
        sampled_years = rng.choice(
            available_years,
            size=number_of_years,
            replace=True,
        )

        bootstrap_precipitation = np.mean(
            [
                precipitation_by_year[int(year)]
                for year in sampled_years
            ],
            axis=0,
        )

        bootstrap_soil_moisture = np.mean(
            [
                soil_moisture_by_year[int(year)]
                for year in sampled_years
            ],
            axis=0,
        )

        (
            bootstrap_best_lag,
            bootstrap_best_correlation,
            bootstrap_correlations,
        ) = find_best_lag(
            bootstrap_precipitation,
            bootstrap_soil_moisture,
        )

        bootstrap_same_month_correlation = float(
            bootstrap_correlations[0]
        )

        bootstrap_best_lags[iteration] = (
            bootstrap_best_lag
        )

        bootstrap_same_month_correlations[iteration] = (
            bootstrap_same_month_correlation
        )

        bootstrap_best_correlations[iteration] = (
            bootstrap_best_correlation
        )

        bootstrap_delta_r[iteration] = (
            bootstrap_best_correlation
            - bootstrap_same_month_correlation
        )

    lag_counts = {
        lag: int(
            np.sum(bootstrap_best_lags == lag)
        )
        for lag in range(MAXIMUM_LAG + 1)
    }

    observed_lag_support = float(
        np.mean(
            bootstrap_best_lags
            == observed_best_lag
        )
    )

    positive_lag_support = float(
        np.mean(bootstrap_best_lags > 0)
    )

    lag_at_least_two_support = float(
        np.mean(bootstrap_best_lags >= 2)
    )

    robustness_class = classify_robustness(
        observed_delta_r,
        observed_lag_support,
        positive_lag_support,
    )

    native_cell_count = int(
        region_data["native_cell_count"].iloc[0]
    )

    output: dict[str, object] = {
        "region_id": region_id,
        "native_cell_count": native_cell_count,
        "longitude_mean": float(
            region_data["longitude_mean"].iloc[0]
        ),
        "latitude_mean": float(
            region_data["latitude_mean"].iloc[0]
        ),
        "small_region_lt_4_cells": (
            native_cell_count < 4
        ),
        "observed_best_lag": observed_best_lag,
        "observed_same_month_correlation": (
            observed_same_month_correlation
        ),
        "observed_best_lag_correlation": (
            observed_best_correlation
        ),
        "observed_delta_r": observed_delta_r,
        "bootstrap_observed_lag_support": (
            observed_lag_support
        ),
        "bootstrap_positive_lag_support": (
            positive_lag_support
        ),
        "bootstrap_lag_at_least_2_support": (
            lag_at_least_two_support
        ),
        "bootstrap_delta_r_median": float(
            np.median(bootstrap_delta_r)
        ),
        "bootstrap_delta_r_q025": float(
            np.quantile(bootstrap_delta_r, 0.025)
        ),
        "bootstrap_delta_r_q975": float(
            np.quantile(bootstrap_delta_r, 0.975)
        ),
        "bootstrap_best_correlation_median": float(
            np.median(bootstrap_best_correlations)
        ),
        "bootstrap_same_month_correlation_median": float(
            np.median(
                bootstrap_same_month_correlations
            )
        ),
        "robustness_class": robustness_class,
    }

    for lag, count in lag_counts.items():
        output[
            f"bootstrap_best_lag_{lag}_count"
        ] = count

        output[
            f"bootstrap_best_lag_{lag}_fraction"
        ] = count / BOOTSTRAP_ITERATIONS

    return output


def calculate_aet_pet_consistency(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()

    working["aet_exceeds_pet"] = (
        working["actual_evapotranspiration_mm"]
        > working["PET_mm"] + 1e-6
    )

    working["aet_minus_pet_mm"] = (
        working["actual_evapotranspiration_mm"]
        - working["PET_mm"]
    )

    by_region = (
        working.groupby(
            "region_id",
            as_index=False,
            observed=True,
        )
        .agg(
            monthly_records=("date", "size"),
            aet_exceeds_pet_records=(
                "aet_exceeds_pet",
                "sum",
            ),
            maximum_aet_minus_pet_mm=(
                "aet_minus_pet_mm",
                "max",
            ),
            mean_aet_minus_pet_mm=(
                "aet_minus_pet_mm",
                "mean",
            ),
        )
    )

    by_region["aet_exceeds_pet_percentage"] = (
        100.0
        * by_region["aet_exceeds_pet_records"]
        / by_region["monthly_records"]
    )

    by_month = (
        working.groupby(
            "month",
            as_index=False,
            observed=True,
        )
        .agg(
            regional_year_records=("date", "size"),
            aet_exceeds_pet_records=(
                "aet_exceeds_pet",
                "sum",
            ),
            maximum_aet_minus_pet_mm=(
                "aet_minus_pet_mm",
                "max",
            ),
            mean_aet_minus_pet_mm=(
                "aet_minus_pet_mm",
                "mean",
            ),
        )
    )

    by_month["aet_exceeds_pet_percentage"] = (
        100.0
        * by_month["aet_exceeds_pet_records"]
        / by_month["regional_year_records"]
    )

    return by_region, by_month


def create_figure(
    bootstrap_metrics: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.0, 4.8),
        constrained_layout=True,
    )

    axes[0].scatter(
        bootstrap_metrics[
            "observed_same_month_correlation"
        ],
        bootstrap_metrics[
            "observed_best_lag_correlation"
        ],
        s=38,
        facecolor="0.65",
        edgecolor="black",
        linewidth=0.6,
    )

    minimum_value = float(
        min(
            bootstrap_metrics[
                "observed_same_month_correlation"
            ].min(),
            bootstrap_metrics[
                "observed_best_lag_correlation"
            ].min(),
        )
    )

    maximum_value = float(
        max(
            bootstrap_metrics[
                "observed_same_month_correlation"
            ].max(),
            bootstrap_metrics[
                "observed_best_lag_correlation"
            ].max(),
        )
    )

    axes[0].plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        linewidth=1.0,
        color="black",
    )

    axes[0].set_xlabel(
        "Same-month precipitation–soil-moisture correlation"
    )

    axes[0].set_ylabel(
        "Best lagged correlation"
    )

    axes[0].set_title(
        "Improvement from antecedent precipitation"
    )

    axes[1].scatter(
        bootstrap_metrics["observed_delta_r"],
        bootstrap_metrics[
            "bootstrap_observed_lag_support"
        ],
        s=38,
        facecolor="0.65",
        edgecolor="black",
        linewidth=0.6,
    )

    two_month_regions = bootstrap_metrics.loc[
        bootstrap_metrics["observed_best_lag"] >= 2
    ]

    for row in two_month_regions.itertuples():
        axes[1].annotate(
            row.region_id,
            xy=(
                row.observed_delta_r,
                row.bootstrap_observed_lag_support,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    axes[1].axvline(
        0.05,
        linestyle=":",
        linewidth=0.9,
        color="black",
    )

    axes[1].axhline(
        0.60,
        linestyle=":",
        linewidth=0.9,
        color="black",
    )

    axes[1].set_xlabel(
        r"Correlation gain, $\Delta r=r_{\mathrm{best}}-r_0$"
    )

    axes[1].set_ylabel(
        "Bootstrap support for observed lag"
    )

    axes[1].set_ylim(-0.02, 1.02)

    axes[1].set_title(
        "Lag magnitude and bootstrap stability"
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.55,
        )

    figure.suptitle(
        "Robustness of regional precipitation–soil-moisture asynchrony",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig11_asynchrony_bootstrap_robustness.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig11_asynchrony_bootstrap_robustness.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def count_classes(
    frame: pd.DataFrame,
) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value
        in frame["robustness_class"]
        .value_counts()
        .to_dict()
        .items()
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    frame = pd.read_parquet(INPUT_PATH)
    frame["date"] = pd.to_datetime(frame["date"])

    if frame["region_id"].nunique() != EXPECTED_REGIONS:
        raise RuntimeError(
            "Unexpected number of regional units."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []

    print("=" * 78)
    print("ASYNCHRONY ROBUSTNESS AND AET–PET CONSISTENCY")
    print("=" * 78)
    print(f"Bootstrap iterations per region : {BOOTSTRAP_ITERATIONS}")
    print(f"Regional units                 : {EXPECTED_REGIONS}")
    print()

    for index, (
        region_id,
        region_data,
    ) in enumerate(
        frame.groupby(
            "region_id",
            observed=True,
            sort=True,
        ),
        start=1,
    ):
        result = bootstrap_region(
            region_data,
            rng,
        )

        rows.append(result)

        print(
            f"[{index:02d}/{EXPECTED_REGIONS:02d}] "
            f"{region_id}: "
            f"lag={result['observed_best_lag']}, "
            f"delta_r={result['observed_delta_r']:.3f}, "
            f"support={result['bootstrap_observed_lag_support']:.3f}, "
            f"class={result['robustness_class']}"
        )

    bootstrap_metrics = pd.DataFrame(rows)

    bootstrap_metrics = bootstrap_metrics.sort_values(
        [
            "observed_best_lag",
            "bootstrap_observed_lag_support",
            "observed_delta_r",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    (
        aet_pet_by_region,
        aet_pet_by_month,
    ) = calculate_aet_pet_consistency(frame)

    bootstrap_path = (
        TABLE_DIR
        / "asynchrony_bootstrap_metrics.csv"
    )

    aet_pet_region_path = (
        TABLE_DIR
        / "aet_pet_consistency_by_region.csv"
    )

    aet_pet_month_path = (
        TABLE_DIR
        / "aet_pet_consistency_by_month.csv"
    )

    bootstrap_metrics.to_csv(
        bootstrap_path,
        index=False,
    )

    aet_pet_by_region.to_csv(
        aet_pet_region_path,
        index=False,
    )

    aet_pet_by_month.to_csv(
        aet_pet_month_path,
        index=False,
    )

    create_figure(bootstrap_metrics)

    core_regions = bootstrap_metrics.loc[
        bootstrap_metrics["native_cell_count"] >= 4
    ]

    well_populated_regions = bootstrap_metrics.loc[
        bootstrap_metrics["native_cell_count"] >= 10
    ]

    total_aet_pet_exceedances = int(
        aet_pet_by_region[
            "aet_exceeds_pet_records"
        ].sum()
    )

    total_records = int(
        aet_pet_by_region[
            "monthly_records"
        ].sum()
    )

    summary = {
        "status": "PASS",
        "bootstrap_iterations_per_region": (
            BOOTSTRAP_ITERATIONS
        ),
        "random_seed": RANDOM_SEED,
        "region_count": int(
            len(bootstrap_metrics)
        ),
        "core_region_count_native_cells_ge_4": int(
            len(core_regions)
        ),
        "well_populated_region_count_native_cells_ge_10": int(
            len(well_populated_regions)
        ),
        "robustness_classes_all_regions": (
            count_classes(bootstrap_metrics)
        ),
        "robustness_classes_native_cells_ge_4": (
            count_classes(core_regions)
        ),
        "robustness_classes_native_cells_ge_10": (
            count_classes(well_populated_regions)
        ),
        "regions_with_positive_lag_support_ge_0_90": int(
            (
                bootstrap_metrics[
                    "bootstrap_positive_lag_support"
                ] >= 0.90
            ).sum()
        ),
        "regions_with_observed_lag_support_ge_0_80": int(
            (
                bootstrap_metrics[
                    "bootstrap_observed_lag_support"
                ] >= 0.80
            ).sum()
        ),
        "regions_with_delta_r_ge_0_10": int(
            (
                bootstrap_metrics[
                    "observed_delta_r"
                ] >= 0.10
            ).sum()
        ),
        "median_observed_delta_r": float(
            bootstrap_metrics[
                "observed_delta_r"
            ].median()
        ),
        "median_bootstrap_observed_lag_support": float(
            bootstrap_metrics[
                "bootstrap_observed_lag_support"
            ].median()
        ),
        "aet_exceeds_pet_records": (
            total_aet_pet_exceedances
        ),
        "total_regional_monthly_records": (
            total_records
        ),
        "aet_exceeds_pet_percentage": float(
            100.0
            * total_aet_pet_exceedances
            / total_records
        ),
        "maximum_regional_aet_minus_pet_mm": float(
            aet_pet_by_region[
                "maximum_aet_minus_pet_mm"
            ].max()
        ),
        "bootstrap_metrics": str(
            bootstrap_path
        ),
        "aet_pet_by_region": str(
            aet_pet_region_path
        ),
        "aet_pet_by_month": str(
            aet_pet_month_path
        ),
    }

    summary_path = (
        TABLE_DIR
        / "asynchrony_robustness_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2)

    print()
    print("=" * 78)
    print("ROBUSTNESS SUMMARY")
    print("=" * 78)
    print(
        "Classes, all regions             : "
        f"{summary['robustness_classes_all_regions']}"
    )
    print(
        "Classes, regions with >=4 cells  : "
        f"{summary['robustness_classes_native_cells_ge_4']}"
    )
    print(
        "Classes, regions with >=10 cells : "
        f"{summary['robustness_classes_native_cells_ge_10']}"
    )
    print(
        "Positive-lag support >=0.90      : "
        f"{summary['regions_with_positive_lag_support_ge_0_90']}"
    )
    print(
        "Observed-lag support >=0.80      : "
        f"{summary['regions_with_observed_lag_support_ge_0_80']}"
    )
    print(
        "Regions with delta_r >=0.10      : "
        f"{summary['regions_with_delta_r_ge_0_10']}"
    )
    print(
        "Median delta_r                   : "
        f"{summary['median_observed_delta_r']:.4f}"
    )
    print(
        "Median observed-lag support      : "
        f"{summary['median_bootstrap_observed_lag_support']:.4f}"
    )
    print(
        "AET > PET records                : "
        f"{total_aet_pet_exceedances:,}/{total_records:,} "
        f"({summary['aet_exceeds_pet_percentage']:.3f}%)"
    )
    print()
    print(f"Summary                           : {summary_path}")
    print(f"Bootstrap metrics                 : {bootstrap_path}")
    print(f"AET–PET month table               : {aet_pet_month_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
