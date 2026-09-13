#!/usr/bin/env python3
"""
Regional seasonal climatology and precipitation-soil-moisture asynchrony.

Inputs
------
Clean 0.5-degree regional monthly hydrology dataset, 1982-2011.

Outputs
-------
1. Regional 30-year monthly climatology.
2. National monthly climatology.
3. Regional seasonal correlation and lag metrics.
4. Ranked asynchrony table.
5. Regional monthly-pattern figures.
6. Regional maps of precipitation-soil-moisture lag and correlation.

Lag convention
--------------
A positive lag means that soil moisture follows earlier precipitation.

For example:
    lag = 2
means that soil moisture in month m is most strongly associated with
precipitation in month m-2.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "srilanka_hydrology_regional_monthly_0p5degree.parquet"
)

INVENTORY_PATH = (
    ROOT
    / "results"
    / "tables"
    / "region_inventory_0p5degree.csv"
)

PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

EXPECTED_REGION_COUNT = 33
EXPECTED_MONTHS_PER_REGION = 360
EXPECTED_YEARS = list(range(1982, 2012))

MONTH_LABELS = [
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

VARIABLES = {
    "precipitation_mm": {
        "short_name": "P",
        "title": "Precipitation",
        "unit": "mm month$^{-1}$",
        "filename": "precipitation",
    },
    "soil_moisture_total_mm": {
        "short_name": "SM",
        "title": "Total soil moisture",
        "unit": "mm",
        "filename": "soil_moisture",
    },
    "total_runoff_mm": {
        "short_name": "R",
        "title": "Total runoff",
        "unit": "mm month$^{-1}$",
        "filename": "runoff",
    },
    "actual_evapotranspiration_mm": {
        "short_name": "AET",
        "title": "Actual evapotranspiration",
        "unit": "mm month$^{-1}$",
        "filename": "actual_evapotranspiration",
    },
    "PET_mm": {
        "short_name": "PET",
        "title": "Potential evapotranspiration",
        "unit": "mm month$^{-1}$",
        "filename": "potential_evapotranspiration",
    },
}

METRIC_VARIABLES = list(VARIABLES.keys())


def pearson_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Calculate Pearson correlation safely."""
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


def shortest_circular_month_difference(
    start_month: int,
    end_month: int,
) -> int:
    """
    Return the shortest signed circular difference.

    Positive values indicate that end_month occurs after start_month.
    """
    difference = (
        (end_month - start_month + 6) % 12
    ) - 6

    return int(difference)


def build_regional_climatology(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate monthly climatology in long and wide formats."""
    aggregation = {}

    for variable in METRIC_VARIABLES:
        aggregation[f"{variable}__mean"] = (
            variable,
            "mean",
        )
        aggregation[f"{variable}__std"] = (
            variable,
            "std",
        )
        aggregation[f"{variable}__q25"] = (
            variable,
            lambda values: values.quantile(0.25),
        )
        aggregation[f"{variable}__median"] = (
            variable,
            "median",
        )
        aggregation[f"{variable}__q75"] = (
            variable,
            lambda values: values.quantile(0.75),
        )

    wide = (
        frame.groupby(
            ["region_id", "month"],
            as_index=False,
            observed=True,
        )
        .agg(**aggregation)
        .sort_values(["region_id", "month"])
        .reset_index(drop=True)
    )

    long_rows: list[dict[str, object]] = []

    for row in wide.itertuples(index=False):
        row_dictionary = row._asdict()

        for variable in METRIC_VARIABLES:
            long_rows.append(
                {
                    "region_id": row_dictionary[
                        "region_id"
                    ],
                    "month": int(
                        row_dictionary["month"]
                    ),
                    "variable": variable,
                    "mean": float(
                        row_dictionary[
                            f"{variable}__mean"
                        ]
                    ),
                    "standard_deviation": float(
                        row_dictionary[
                            f"{variable}__std"
                        ]
                    ),
                    "q25": float(
                        row_dictionary[
                            f"{variable}__q25"
                        ]
                    ),
                    "median": float(
                        row_dictionary[
                            f"{variable}__median"
                        ]
                    ),
                    "q75": float(
                        row_dictionary[
                            f"{variable}__q75"
                        ]
                    ),
                }
            )

    long = pd.DataFrame(long_rows)

    return wide, long


def build_national_climatology(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate area-weighted national monthly climatology.

    Regional area weights are the sums of latitude weights for
    native cells included in each region.
    """
    rows: list[dict[str, object]] = []

    for month, month_data in frame.groupby(
        "month",
        observed=True,
    ):
        row: dict[str, object] = {
            "month": int(month),
        }

        weights = month_data[
            "regional_area_weight_sum"
        ].to_numpy(dtype=float)

        for variable in METRIC_VARIABLES:
            values = month_data[
                variable
            ].to_numpy(dtype=float)

            row[variable] = float(
                np.average(values, weights=weights)
            )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("month")
        .reset_index(drop=True)
    )


def best_antecedent_precipitation_lag(
    precipitation: np.ndarray,
    soil_moisture: np.ndarray,
    maximum_lag: int = 6,
) -> tuple[int, float, list[float]]:
    """
    Find the precipitation lag giving maximum correlation with SM.

    For lag L:
        SM[m] is compared with P[m-L].

    Circular shifting is appropriate because these are repeating
    12-month climatological cycles.
    """
    correlations: list[float] = []

    for lag in range(maximum_lag + 1):
        lagged_precipitation = np.roll(
            precipitation,
            lag,
        )

        correlation = pearson_correlation(
            lagged_precipitation,
            soil_moisture,
        )

        correlations.append(correlation)

    finite_correlations = np.asarray(
        correlations,
        dtype=float,
    )

    if np.all(~np.isfinite(finite_correlations)):
        return 0, float("nan"), correlations

    best_lag = int(
        np.nanargmax(finite_correlations)
    )

    return (
        best_lag,
        float(finite_correlations[best_lag]),
        correlations,
    )


def classify_runoff_control(
    precipitation_runoff_correlation: float,
    soil_moisture_runoff_correlation: float,
    threshold: float = 0.15,
) -> str:
    """Provide a preliminary runoff-control classification."""
    if not np.isfinite(
        precipitation_runoff_correlation
    ):
        return "undetermined"

    if not np.isfinite(
        soil_moisture_runoff_correlation
    ):
        return "undetermined"

    difference = (
        precipitation_runoff_correlation
        - soil_moisture_runoff_correlation
    )

    if difference > threshold:
        return "precipitation-dominant"

    if difference < -threshold:
        return "storage-dominant"

    return "mixed"


def calculate_region_metrics(
    climatology_wide: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate seasonal metrics for every regional unit."""
    rows: list[dict[str, object]] = []

    for region_id, region_data in climatology_wide.groupby(
        "region_id",
        observed=True,
    ):
        region_data = (
            region_data
            .sort_values("month")
            .reset_index(drop=True)
        )

        if len(region_data) != 12:
            raise RuntimeError(
                f"{region_id} does not have 12 "
                "climatological months."
            )

        precipitation = region_data[
            "precipitation_mm__mean"
        ].to_numpy(dtype=float)

        soil_moisture = region_data[
            "soil_moisture_total_mm__mean"
        ].to_numpy(dtype=float)

        runoff = region_data[
            "total_runoff_mm__mean"
        ].to_numpy(dtype=float)

        aet = region_data[
            "actual_evapotranspiration_mm__mean"
        ].to_numpy(dtype=float)

        pet = region_data[
            "PET_mm__mean"
        ].to_numpy(dtype=float)

        precipitation_peak_month = int(
            np.argmax(precipitation) + 1
        )

        soil_moisture_peak_month = int(
            np.argmax(soil_moisture) + 1
        )

        runoff_peak_month = int(
            np.argmax(runoff) + 1
        )

        aet_peak_month = int(
            np.argmax(aet) + 1
        )

        pet_peak_month = int(
            np.argmax(pet) + 1
        )

        peak_forward_lag = int(
            (
                soil_moisture_peak_month
                - precipitation_peak_month
            )
            % 12
        )

        peak_shortest_lag = (
            shortest_circular_month_difference(
                precipitation_peak_month,
                soil_moisture_peak_month,
            )
        )

        (
            best_lag,
            best_lag_correlation,
            lag_correlations,
        ) = best_antecedent_precipitation_lag(
            precipitation,
            soil_moisture,
            maximum_lag=6,
        )

        precipitation_sm_correlation = (
            pearson_correlation(
                precipitation,
                soil_moisture,
            )
        )

        precipitation_runoff_correlation = (
            pearson_correlation(
                precipitation,
                runoff,
            )
        )

        soil_moisture_runoff_correlation = (
            pearson_correlation(
                soil_moisture,
                runoff,
            )
        )

        precipitation_aet_correlation = (
            pearson_correlation(
                precipitation,
                aet,
            )
        )

        pet_aet_correlation = pearson_correlation(
            pet,
            aet,
        )

        metadata = inventory.loc[
            inventory["region_id"] == region_id
        ]

        if len(metadata) != 1:
            raise RuntimeError(
                f"Inventory metadata missing or duplicated "
                f"for {region_id}."
            )

        metadata_row = metadata.iloc[0]

        row: dict[str, object] = {
            "region_id": region_id,
            "native_cell_count": int(
                metadata_row["native_cell_count"]
            ),
            "longitude_mean": float(
                metadata_row["longitude_mean"]
            ),
            "latitude_mean": float(
                metadata_row["latitude_mean"]
            ),
            "region_lon_index": int(
                metadata_row["region_lon_index"]
            ),
            "region_lat_index": int(
                metadata_row["region_lat_index"]
            ),
            "region_lon_min": float(
                metadata_row["region_lon_min"]
            ),
            "region_lon_max": float(
                metadata_row["region_lon_max"]
            ),
            "region_lat_min": float(
                metadata_row["region_lat_min"]
            ),
            "region_lat_max": float(
                metadata_row["region_lat_max"]
            ),
            "small_region_lt_4_cells": bool(
                metadata_row[
                    "small_region_lt_4_cells"
                ]
            ),
            "very_small_region_lt_2_cells": bool(
                metadata_row[
                    "very_small_region_lt_2_cells"
                ]
            ),
            "precipitation_peak_month": (
                precipitation_peak_month
            ),
            "soil_moisture_peak_month": (
                soil_moisture_peak_month
            ),
            "runoff_peak_month": runoff_peak_month,
            "aet_peak_month": aet_peak_month,
            "pet_peak_month": pet_peak_month,
            "peak_month_forward_lag": (
                peak_forward_lag
            ),
            "peak_month_shortest_signed_lag": (
                peak_shortest_lag
            ),
            "best_precipitation_lag_0_to_6": (
                best_lag
            ),
            "best_lag_correlation": (
                best_lag_correlation
            ),
            "corr_precipitation_soil_moisture": (
                precipitation_sm_correlation
            ),
            "corr_precipitation_runoff": (
                precipitation_runoff_correlation
            ),
            "corr_soil_moisture_runoff": (
                soil_moisture_runoff_correlation
            ),
            "corr_precipitation_aet": (
                precipitation_aet_correlation
            ),
            "corr_pet_aet": pet_aet_correlation,
            "precipitation_mean_mm": float(
                precipitation.mean()
            ),
            "precipitation_amplitude_mm": float(
                precipitation.max()
                - precipitation.min()
            ),
            "precipitation_seasonality_ratio": float(
                (
                    precipitation.max()
                    - precipitation.min()
                )
                / precipitation.mean()
            ),
            "soil_moisture_mean_mm": float(
                soil_moisture.mean()
            ),
            "soil_moisture_amplitude_mm": float(
                soil_moisture.max()
                - soil_moisture.min()
            ),
            "soil_moisture_seasonality_ratio": float(
                (
                    soil_moisture.max()
                    - soil_moisture.min()
                )
                / soil_moisture.mean()
            ),
            "runoff_mean_mm": float(
                runoff.mean()
            ),
            "runoff_amplitude_mm": float(
                runoff.max() - runoff.min()
            ),
            "runoff_control_class": (
                classify_runoff_control(
                    precipitation_runoff_correlation,
                    soil_moisture_runoff_correlation,
                )
            ),
        }

        for lag, correlation in enumerate(
            lag_correlations
        ):
            row[
                f"corr_precipitation_lag_{lag}_soil_moisture"
            ] = correlation

        rows.append(row)

    metrics = pd.DataFrame(rows)

    metrics["asynchrony_rank"] = (
        metrics[
            [
                "best_precipitation_lag_0_to_6",
                "peak_month_forward_lag",
            ]
        ]
        .max(axis=1)
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    return (
        metrics.sort_values(
            [
                "region_lat_index",
                "region_lon_index",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def plot_regional_monthly_patterns(
    climatology_long: pd.DataFrame,
    inventory: pd.DataFrame,
    variable: str,
) -> None:
    """Create a geographic matrix of regional monthly curves."""
    variable_metadata = VARIABLES[variable]

    maximum_latitude_index = int(
        inventory["region_lat_index"].max()
    )

    minimum_latitude_index = int(
        inventory["region_lat_index"].min()
    )

    maximum_longitude_index = int(
        inventory["region_lon_index"].max()
    )

    minimum_longitude_index = int(
        inventory["region_lon_index"].min()
    )

    row_count = (
        maximum_latitude_index
        - minimum_latitude_index
        + 1
    )

    column_count = (
        maximum_longitude_index
        - minimum_longitude_index
        + 1
    )

    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=column_count,
        figsize=(15.0, 18.0),
        sharex=True,
        squeeze=False,
    )

    for axis in axes.flat:
        axis.set_visible(False)

    variable_data = climatology_long.loc[
        climatology_long["variable"] == variable
    ].copy()

    for inventory_row in inventory.itertuples():
        row_index = (
            maximum_latitude_index
            - int(inventory_row.region_lat_index)
        )

        column_index = (
            int(inventory_row.region_lon_index)
            - minimum_longitude_index
        )

        axis = axes[row_index, column_index]
        axis.set_visible(True)

        region_data = variable_data.loc[
            variable_data["region_id"]
            == inventory_row.region_id
        ].sort_values("month")

        months = region_data["month"].to_numpy()
        means = region_data["mean"].to_numpy()
        q25 = region_data["q25"].to_numpy()
        q75 = region_data["q75"].to_numpy()

        axis.fill_between(
            months,
            q25,
            q75,
            color="0.85",
            linewidth=0,
            zorder=1,
        )

        axis.plot(
            months,
            means,
            color="black",
            linewidth=1.25,
            marker="o",
            markersize=2.5,
            zorder=2,
        )

        title_suffix = ""

        if inventory_row.native_cell_count < 4:
            title_suffix = " †"

            for spine in axis.spines.values():
                spine.set_linestyle("--")
                spine.set_linewidth(0.8)

        axis.set_title(
            f"{inventory_row.region_id}{title_suffix}\n"
            f"n={inventory_row.native_cell_count}",
            fontsize=7.2,
            pad=2,
        )

        axis.set_xticks([1, 4, 7, 10])
        axis.set_xticklabels(
            ["Jan", "Apr", "Jul", "Oct"],
            fontsize=6,
        )

        axis.tick_params(
            axis="y",
            labelsize=6,
            length=2,
        )

        axis.tick_params(
            axis="x",
            length=2,
        )

        axis.grid(
            linewidth=0.3,
            linestyle=":",
            alpha=0.55,
        )

        axis.margins(x=0.04)

    figure.suptitle(
        f"Regional monthly climatology of "
        f"{variable_metadata['title']}, 1982–2011",
        fontsize=15,
        y=0.995,
    )

    figure.supxlabel(
        "Calendar month",
        fontsize=11,
    )

    figure.supylabel(
        variable_metadata["unit"],
        fontsize=11,
    )

    figure.text(
        0.99,
        0.005,
        "Shaded bands: interannual IQR. "
        "† Region contains fewer than four native cells.",
        ha="right",
        va="bottom",
        fontsize=8,
    )

    figure.tight_layout(
        rect=[0.025, 0.025, 1.0, 0.982]
    )

    filename = variable_metadata["filename"]

    figure.savefig(
        FIGURE_DIR
        / f"fig04_{filename}_regional_climatology.png",
        dpi=350,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / f"fig04_{filename}_regional_climatology.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_asynchrony_maps(
    metrics: pd.DataFrame,
) -> None:
    """Map lag and same-month seasonal correlation."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.6, 8.0),
        constrained_layout=True,
    )

    lag_cmap = plt.get_cmap("viridis", 7)
    lag_boundaries = np.arange(-0.5, 7.5, 1.0)
    lag_norm = BoundaryNorm(
        lag_boundaries,
        lag_cmap.N,
    )

    correlation_cmap = plt.get_cmap("coolwarm")
    correlation_norm = plt.Normalize(
        vmin=-1.0,
        vmax=1.0,
    )

    for row in metrics.itertuples():
        lag_value = int(
            row.best_precipitation_lag_0_to_6
        )

        lag_rectangle = Rectangle(
            (
                row.region_lon_min,
                row.region_lat_min,
            ),
            row.region_lon_max - row.region_lon_min,
            row.region_lat_max - row.region_lat_min,
            facecolor=lag_cmap(
                lag_norm(lag_value)
            ),
            edgecolor="black",
            linewidth=0.55,
            hatch=(
                "///"
                if row.small_region_lt_4_cells
                else None
            ),
        )

        axes[0].add_patch(lag_rectangle)

        axes[0].text(
            row.longitude_mean,
            row.latitude_mean,
            str(lag_value),
            ha="center",
            va="center",
            fontsize=7,
        )

        correlation_value = float(
            row.corr_precipitation_soil_moisture
        )

        correlation_rectangle = Rectangle(
            (
                row.region_lon_min,
                row.region_lat_min,
            ),
            row.region_lon_max - row.region_lon_min,
            row.region_lat_max - row.region_lat_min,
            facecolor=correlation_cmap(
                correlation_norm(correlation_value)
            ),
            edgecolor="black",
            linewidth=0.55,
            hatch=(
                "///"
                if row.small_region_lt_4_cells
                else None
            ),
        )

        axes[1].add_patch(
            correlation_rectangle
        )

        axes[1].text(
            row.longitude_mean,
            row.latitude_mean,
            f"{correlation_value:.2f}",
            ha="center",
            va="center",
            fontsize=6.2,
        )

    longitude_min = float(
        metrics["region_lon_min"].min()
    )

    longitude_max = float(
        metrics["region_lon_max"].max()
    )

    latitude_min = float(
        metrics["region_lat_min"].min()
    )

    latitude_max = float(
        metrics["region_lat_max"].max()
    )

    for axis in axes:
        axis.set_xlim(
            longitude_min - 0.1,
            longitude_max + 0.1,
        )

        axis.set_ylim(
            latitude_min - 0.1,
            latitude_max + 0.1,
        )

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.set_xlabel("Longitude (°E)")
        axis.set_ylabel("Latitude (°N)")

        axis.grid(
            linewidth=0.3,
            linestyle=":",
            alpha=0.5,
        )

    axes[0].set_title(
        "Best antecedent precipitation lag\n"
        "for monthly soil moisture"
    )

    axes[1].set_title(
        "Same-month seasonal correlation\n"
        "between precipitation and soil moisture"
    )

    lag_scalar = plt.cm.ScalarMappable(
        norm=lag_norm,
        cmap=lag_cmap,
    )

    lag_scalar.set_array([])

    lag_colorbar = figure.colorbar(
        lag_scalar,
        ax=axes[0],
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
        ticks=range(0, 7),
    )

    lag_colorbar.set_label(
        "Antecedent precipitation lag (months)"
    )

    correlation_scalar = plt.cm.ScalarMappable(
        norm=correlation_norm,
        cmap=correlation_cmap,
    )

    correlation_scalar.set_array([])

    correlation_colorbar = figure.colorbar(
        correlation_scalar,
        ax=axes[1],
        orientation="horizontal",
        fraction=0.05,
        pad=0.06,
    )

    correlation_colorbar.set_label(
        "Pearson correlation"
    )

    figure.suptitle(
        "Regional precipitation–soil-moisture "
        "asynchrony in Sri Lanka",
        fontsize=14,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig09_precipitation_soil_moisture_asynchrony.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig09_precipitation_soil_moisture_asynchrony.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_national_climatology(
    national: pd.DataFrame,
) -> None:
    """Plot area-weighted national monthly climatology."""
    months = national["month"].to_numpy()

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.4, 8.0),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        months,
        national["precipitation_mm"],
        marker="o",
        linewidth=1.6,
        label="Precipitation",
    )

    axes[0].plot(
        months,
        national["PET_mm"],
        marker="s",
        linewidth=1.4,
        label="PET",
    )

    axes[0].plot(
        months,
        national[
            "actual_evapotranspiration_mm"
        ],
        marker="^",
        linewidth=1.4,
        label="Actual ET",
    )

    axes[0].plot(
        months,
        national["total_runoff_mm"],
        marker="d",
        linewidth=1.4,
        label="Total runoff",
    )

    axes[0].set_ylabel(
        "Monthly flux (mm month$^{-1}$)"
    )

    axes[0].legend(
        frameon=True,
        ncol=2,
    )

    axes[1].plot(
        months,
        national[
            "soil_moisture_total_mm"
        ],
        marker="o",
        linewidth=1.6,
        color="black",
    )

    axes[1].set_ylabel(
        "Total soil moisture (mm)"
    )

    axes[1].set_xlabel("Calendar month")

    axes[1].set_xticks(months)
    axes[1].set_xticklabels(MONTH_LABELS)

    for axis in axes:
        axis.grid(
            linewidth=0.4,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Area-weighted national monthly hydrological "
        "climatology, 1982–2011",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig10_national_monthly_climatology.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig10_national_monthly_climatology.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    PROCESSED_DIR.mkdir(
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

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Regional dataset does not exist: "
            f"{INPUT_PATH}"
        )

    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Region inventory does not exist: "
            f"{INVENTORY_PATH}"
        )

    print("=" * 78)
    print(
        "SRI LANKA HYDROLOGY — "
        "SEASONAL CLIMATOLOGY AND ASYNCHRONY"
    )
    print("=" * 78)

    frame = pd.read_parquet(INPUT_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)

    frame["date"] = pd.to_datetime(
        frame["date"]
    )

    region_count = int(
        frame["region_id"].nunique()
    )

    if region_count != EXPECTED_REGION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_REGION_COUNT} regions, "
            f"found {region_count}."
        )

    months_per_region = (
        frame.groupby(
            "region_id",
            observed=True,
        )["date"]
        .nunique()
    )

    if not (
        months_per_region
        == EXPECTED_MONTHS_PER_REGION
    ).all():
        raise RuntimeError(
            "One or more regional units do not contain "
            f"{EXPECTED_MONTHS_PER_REGION} months."
        )

    years = sorted(
        frame["year"].astype(int).unique().tolist()
    )

    if years != EXPECTED_YEARS:
        raise RuntimeError(
            f"Unexpected year coverage: {years}"
        )

    if frame[
        METRIC_VARIABLES
    ].isna().any().any():
        raise RuntimeError(
            "Hydrological variables contain missing values."
        )

    climatology_wide, climatology_long = (
        build_regional_climatology(frame)
    )

    national_climatology = (
        build_national_climatology(frame)
    )

    metrics = calculate_region_metrics(
        climatology_wide,
        inventory,
    )

    asynchrony_ranking = (
        metrics.sort_values(
            [
                "best_precipitation_lag_0_to_6",
                "best_lag_correlation",
                "soil_moisture_amplitude_mm",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    asynchrony_ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(asynchrony_ranking) + 1,
        ),
    )

    climatology_wide_path = (
        PROCESSED_DIR
        / "regional_monthly_climatology_1982_2011_wide.csv"
    )

    climatology_long_path = (
        PROCESSED_DIR
        / "regional_monthly_climatology_1982_2011_long.csv"
    )

    national_path = (
        TABLE_DIR
        / "national_monthly_climatology_1982_2011.csv"
    )

    metrics_path = (
        TABLE_DIR
        / "regional_seasonality_asynchrony_metrics.csv"
    )

    ranking_path = (
        TABLE_DIR
        / "regional_asynchrony_ranking.csv"
    )

    climatology_wide.to_csv(
        climatology_wide_path,
        index=False,
    )

    climatology_long.to_csv(
        climatology_long_path,
        index=False,
    )

    national_climatology.to_csv(
        national_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    asynchrony_ranking.to_csv(
        ranking_path,
        index=False,
    )

    for variable in METRIC_VARIABLES:
        plot_regional_monthly_patterns(
            climatology_long,
            inventory,
            variable,
        )

    plot_asynchrony_maps(metrics)

    plot_national_climatology(
        national_climatology
    )

    lag_counts = (
        metrics[
            "best_precipitation_lag_0_to_6"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    runoff_class_counts = (
        metrics["runoff_control_class"]
        .value_counts()
        .to_dict()
    )

    regions_with_positive_lag = int(
        (
            metrics[
                "best_precipitation_lag_0_to_6"
            ] > 0
        ).sum()
    )

    regions_with_lag_at_least_2 = int(
        (
            metrics[
                "best_precipitation_lag_0_to_6"
            ] >= 2
        ).sum()
    )

    summary = {
        "status": "PASS",
        "input_dataset": str(INPUT_PATH),
        "region_count": region_count,
        "regional_monthly_rows": int(
            len(frame)
        ),
        "climatology_rows_wide": int(
            len(climatology_wide)
        ),
        "climatology_rows_long": int(
            len(climatology_long)
        ),
        "years": (
            f"{min(years)}-{max(years)}"
        ),
        "regions_with_positive_best_lag": (
            regions_with_positive_lag
        ),
        "regions_with_best_lag_at_least_2_months": (
            regions_with_lag_at_least_2
        ),
        "best_lag_counts": {
            str(key): int(value)
            for key, value in lag_counts.items()
        },
        "median_same_month_precipitation_sm_correlation": float(
            metrics[
                "corr_precipitation_soil_moisture"
            ].median()
        ),
        "minimum_same_month_precipitation_sm_correlation": float(
            metrics[
                "corr_precipitation_soil_moisture"
            ].min()
        ),
        "maximum_same_month_precipitation_sm_correlation": float(
            metrics[
                "corr_precipitation_soil_moisture"
            ].max()
        ),
        "runoff_control_class_counts": {
            str(key): int(value)
            for key, value
            in runoff_class_counts.items()
        },
        "small_regions_lt_4_cells": int(
            inventory[
                "small_region_lt_4_cells"
            ].sum()
        ),
        "regional_climatology_wide": str(
            climatology_wide_path
        ),
        "regional_climatology_long": str(
            climatology_long_path
        ),
        "national_climatology": str(
            national_path
        ),
        "regional_metrics": str(
            metrics_path
        ),
        "asynchrony_ranking": str(
            ranking_path
        ),
    }

    summary_path = (
        TABLE_DIR
        / "seasonal_climatology_asynchrony_summary.json"
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

    print(f"Status                         : {summary['status']}")
    print(f"Regional units                 : {region_count}")
    print(
        f"Regional monthly observations  : "
        f"{len(frame):,}"
    )
    print(
        f"Regional climatology rows      : "
        f"{len(climatology_wide):,}"
    )
    print(
        f"Regions with positive lag      : "
        f"{regions_with_positive_lag}/{region_count}"
    )
    print(
        f"Regions with lag >= 2 months   : "
        f"{regions_with_lag_at_least_2}/{region_count}"
    )
    print(
        f"Best-lag distribution          : "
        f"{lag_counts}"
    )
    print(
        f"Median same-month P-SM corr.   : "
        f"{summary['median_same_month_precipitation_sm_correlation']:.4f}"
    )
    print(
        f"Runoff-control classes         : "
        f"{runoff_class_counts}"
    )

    print()
    print("Top 10 regions by antecedent precipitation lag:")

    display_columns = [
        "rank",
        "region_id",
        "native_cell_count",
        "precipitation_peak_month",
        "soil_moisture_peak_month",
        "best_precipitation_lag_0_to_6",
        "best_lag_correlation",
        "corr_precipitation_soil_moisture",
        "runoff_control_class",
    ]

    print(
        asynchrony_ranking[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print(f"Summary                        : {summary_path}")
    print(f"Regional metrics               : {metrics_path}")
    print(f"Asynchrony ranking             : {ranking_path}")
    print(f"Figures                        : {FIGURE_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
