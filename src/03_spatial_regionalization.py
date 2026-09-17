#!/usr/bin/env python3
"""
Construct 0.5-degree Malaysia hydrological regions.

The script:
1. Maps the 172 retained native 0.5-degree cells.
2. Maps cells containing corrected negative AET values.
3. Assigns native cells to stable 0.5 x 0.5 degree regions.
4. Calculates latitude-area-weighted monthly regional means.
5. Produces regional inventory and quality-control outputs.

Five incomplete Malaysia grids were excluded upstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "malaysia_hydrology_monthly_1982_2011_analysis.parquet"
)

PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

REGION_SIZE_DEG = 0.5

# Fixed origins ensure reproducible region identifiers.
LON_ORIGIN = 99.5
LAT_ORIGIN = 0.5

EXPECTED_NATIVE_GRIDS = 172
EXPECTED_MONTHS_PER_GRID = 360

HYDRO_COLUMNS = [
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm",
    "actual_evapotranspiration_mm_raw",
    "surface_runoff_mm",
    "subsurface_runoff_mm",
    "total_runoff_mm",
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
    "soil_moisture_total_mm",
]


def assign_regions(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign every native cell to a fixed 0.5-degree region."""
    result = frame.copy()

    epsilon = 1e-10

    result["region_lon_index"] = np.floor(
        (
            result["longitude"]
            - LON_ORIGIN
            + epsilon
        )
        / REGION_SIZE_DEG
    ).astype(int)

    result["region_lat_index"] = np.floor(
        (
            result["latitude"]
            - LAT_ORIGIN
            + epsilon
        )
        / REGION_SIZE_DEG
    ).astype(int)

    result["region_id"] = (
        "MYS_"
        + result["region_lat_index"].map(
            lambda value: f"{value:02d}"
        )
        + "_"
        + result["region_lon_index"].map(
            lambda value: f"{value:02d}"
        )
    )

    result["region_lon_min"] = (
        LON_ORIGIN
        + result["region_lon_index"] * REGION_SIZE_DEG
    )

    result["region_lon_max"] = (
        result["region_lon_min"] + REGION_SIZE_DEG
    )

    result["region_lat_min"] = (
        LAT_ORIGIN
        + result["region_lat_index"] * REGION_SIZE_DEG
    )

    result["region_lat_max"] = (
        result["region_lat_min"] + REGION_SIZE_DEG
    )

    result["region_center_lon"] = (
        result["region_lon_min"]
        + REGION_SIZE_DEG / 2.0
    )

    result["region_center_lat"] = (
        result["region_lat_min"]
        + REGION_SIZE_DEG / 2.0
    )

    return result


def make_native_grid_lookup(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create one record per native grid cell."""
    lookup = (
        frame[
            [
                "grid_id",
                "longitude",
                "latitude",
                "region_id",
                "region_lon_index",
                "region_lat_index",
                "region_lon_min",
                "region_lon_max",
                "region_lat_min",
                "region_lat_max",
                "region_center_lon",
                "region_center_lat",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["latitude", "longitude"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    duplicate_grid_ids = int(
        lookup["grid_id"].duplicated().sum()
    )

    if duplicate_grid_ids:
        raise RuntimeError(
            f"Grid lookup contains {duplicate_grid_ids} "
            "duplicate grid IDs."
        )

    if len(lookup) != EXPECTED_NATIVE_GRIDS:
        raise RuntimeError(
            f"Expected {EXPECTED_NATIVE_GRIDS} native grids, "
            f"found {len(lookup)}."
        )

    lookup["latitude_area_weight"] = np.cos(
        np.deg2rad(lookup["latitude"])
    )

    return lookup


def make_aet_grid_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize negative-AET corrections by native grid."""
    summary = (
        frame.groupby(
            ["grid_id", "longitude", "latitude"],
            as_index=False,
            observed=True,
        )
        .agg(
            corrected_aet_records=(
                "aet_negative_corrected",
                "sum",
            ),
            minimum_raw_aet_mm=(
                "actual_evapotranspiration_mm_raw",
                "min",
            ),
            mean_raw_aet_mm=(
                "actual_evapotranspiration_mm_raw",
                "mean",
            ),
        )
    )

    summary["corrected_aet_records"] = (
        summary["corrected_aet_records"].astype(int)
    )

    summary = summary.sort_values(
        [
            "corrected_aet_records",
            "latitude",
            "longitude",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return summary


def make_region_inventory(
    frame: pd.DataFrame,
    grid_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the spatial composition of each region."""
    grid_corrections = (
        frame.groupby(
            ["grid_id", "region_id"],
            as_index=False,
            observed=True,
        )
        .agg(
            corrected_aet_records=(
                "aet_negative_corrected",
                "sum",
            )
        )
    )

    grid_corrections["corrected_aet_records"] = (
        grid_corrections[
            "corrected_aet_records"
        ].astype(int)
    )

    lookup = grid_lookup.merge(
        grid_corrections,
        on=["grid_id", "region_id"],
        how="left",
        validate="one_to_one",
    )

    lookup["corrected_aet_records"] = (
        lookup["corrected_aet_records"]
        .fillna(0)
        .astype(int)
    )

    inventory = (
        lookup.groupby(
            "region_id",
            as_index=False,
            observed=True,
        )
        .agg(
            native_cell_count=("grid_id", "nunique"),
            longitude_mean=("longitude", "mean"),
            latitude_mean=("latitude", "mean"),
            longitude_min_native=("longitude", "min"),
            longitude_max_native=("longitude", "max"),
            latitude_min_native=("latitude", "min"),
            latitude_max_native=("latitude", "max"),
            region_lon_index=("region_lon_index", "first"),
            region_lat_index=("region_lat_index", "first"),
            region_lon_min=("region_lon_min", "first"),
            region_lon_max=("region_lon_max", "first"),
            region_lat_min=("region_lat_min", "first"),
            region_lat_max=("region_lat_max", "first"),
            region_center_lon=("region_center_lon", "first"),
            region_center_lat=("region_center_lat", "first"),
            corrected_aet_records=(
                "corrected_aet_records",
                "sum",
            ),
        )
        .sort_values(
            ["region_lat_index", "region_lon_index"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    # Coastal regions may contain only a few native cells.
    # They are flagged but not removed.
    inventory["small_region_lt_4_cells"] = (
        inventory["native_cell_count"] < 4
    )

    inventory["very_small_region_lt_2_cells"] = (
        inventory["native_cell_count"] < 2
    )

    return inventory


def aggregate_regional_monthly(
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate area-weighted monthly means for each region."""
    working = frame.copy()

    working["_area_weight"] = np.cos(
        np.deg2rad(working["latitude"])
    )

    for column in HYDRO_COLUMNS:
        working[f"_weighted__{column}"] = (
            working[column] * working["_area_weight"]
        )

    keys = [
        "region_id",
        "date",
        "year",
        "month",
    ]

    aggregation = {
        "_area_weight": "sum",
        "grid_id": "nunique",
        "aet_negative_corrected": "sum",
    }

    for column in HYDRO_COLUMNS:
        aggregation[f"_weighted__{column}"] = "sum"

    regional = (
        working.groupby(
            keys,
            as_index=False,
            observed=True,
            sort=True,
        )
        .agg(aggregation)
    )

    regional = regional.rename(
        columns={
            "_area_weight": "regional_area_weight_sum",
            "grid_id": "native_cell_count",
            "aet_negative_corrected": (
                "aet_corrected_native_record_count"
            ),
        }
    )

    for column in HYDRO_COLUMNS:
        weighted_column = f"_weighted__{column}"

        regional[column] = (
            regional[weighted_column]
            / regional["regional_area_weight_sum"]
        )

        regional = regional.drop(
            columns=[weighted_column]
        )

    regional[
        "aet_corrected_native_record_count"
    ] = regional[
        "aet_corrected_native_record_count"
    ].astype(int)

    metadata_columns = [
        "region_id",
        "longitude_mean",
        "latitude_mean",
        "region_center_lon",
        "region_center_lat",
        "region_lon_min",
        "region_lon_max",
        "region_lat_min",
        "region_lat_max",
        "small_region_lt_4_cells",
        "very_small_region_lt_2_cells",
    ]

    regional = regional.merge(
        inventory[metadata_columns],
        on="region_id",
        how="left",
        validate="many_to_one",
    )

    regional = regional.sort_values(
        ["region_id", "date"]
    ).reset_index(drop=True)

    expected_region_months = 30 * 12

    months_per_region = regional.groupby(
        "region_id",
        observed=True,
    )["date"].nunique()

    invalid_regions = months_per_region[
        months_per_region != expected_region_months
    ]

    if not invalid_regions.empty:
        raise RuntimeError(
            f"{len(invalid_regions)} regions do not contain "
            f"{expected_region_months} monthly observations."
        )

    if regional[HYDRO_COLUMNS].isna().any().any():
        raise RuntimeError(
            "Regional dataset contains missing hydrological values."
        )

    return regional


def save_coverage_figure(
    grid_lookup: pd.DataFrame,
    aet_summary: pd.DataFrame,
) -> None:
    """Plot native spatial coverage and corrected-AET cells."""
    corrected = aet_summary[
        aet_summary["corrected_aet_records"] > 0
    ].copy()

    figure, axis = plt.subplots(
        figsize=(6.2, 8.0),
        constrained_layout=True,
    )

    axis.scatter(
        grid_lookup["longitude"],
        grid_lookup["latitude"],
        s=17,
        marker="s",
        facecolor="0.78",
        edgecolor="0.30",
        linewidth=0.25,
        label="Native 0.1° hydrology cells",
        zorder=1,
    )

    if not corrected.empty:
        sizes = (
            35
            + 3.0
            * corrected["corrected_aet_records"]
        )

        axis.scatter(
            corrected["longitude"],
            corrected["latitude"],
            s=sizes,
            marker="x",
            color="black",
            linewidth=1.1,
            label="Cells with corrected AET",
            zorder=3,
        )

        for row in corrected.itertuples():
            if row.corrected_aet_records >= 5:
                axis.annotate(
                    str(row.corrected_aet_records),
                    xy=(row.longitude, row.latitude),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")
    axis.set_title(
        "Malaysia hydrology-grid coverage, 1982–2011"
    )

    axis.set_xlim(
        grid_lookup["longitude"].min() - 0.15,
        grid_lookup["longitude"].max() + 0.15,
    )

    axis.set_ylim(
        grid_lookup["latitude"].min() - 0.15,
        grid_lookup["latitude"].max() + 0.15,
    )

    axis.set_aspect("equal", adjustable="box")
    axis.grid(
        linewidth=0.35,
        linestyle=":",
        alpha=0.6,
    )
    axis.legend(
        loc="lower left",
        frameon=True,
        fontsize=8,
    )

    png_path = (
        FIGURE_DIR / "fig01_native_grid_coverage.png"
    )
    pdf_path = (
        FIGURE_DIR / "fig01_native_grid_coverage.pdf"
    )

    figure.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_regionalization_figure(
    grid_lookup: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    """Plot native cells assigned to 0.5-degree regions."""
    region_ids = inventory["region_id"].tolist()

    region_to_code = {
        region_id: index
        for index, region_id in enumerate(region_ids)
    }

    plot_data = grid_lookup.copy()

    plot_data["region_code"] = (
        plot_data["region_id"].map(region_to_code)
    )

    figure, axis = plt.subplots(
        figsize=(6.5, 8.2),
        constrained_layout=True,
    )

    scatter = axis.scatter(
        plot_data["longitude"],
        plot_data["latitude"],
        c=plot_data["region_code"],
        cmap="tab20",
        s=23,
        marker="s",
        edgecolor="0.25",
        linewidth=0.2,
        zorder=2,
    )

    _ = scatter

    for region in inventory.itertuples():
        rectangle = Rectangle(
            (
                region.region_lon_min,
                region.region_lat_min,
            ),
            REGION_SIZE_DEG,
            REGION_SIZE_DEG,
            fill=False,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )

        axis.add_patch(rectangle)

        axis.text(
            region.longitude_mean,
            region.latitude_mean,
            str(region.native_cell_count),
            ha="center",
            va="center",
            fontsize=6.2,
            color="black",
            zorder=4,
        )

    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Latitude (°N)")

    axis.set_title(
        "Primary 0.5° regionalization\n"
        "Labels show native-cell count"
    )

    axis.set_xlim(
        grid_lookup["longitude"].min() - 0.25,
        grid_lookup["longitude"].max() + 0.25,
    )

    axis.set_ylim(
        grid_lookup["latitude"].min() - 0.25,
        grid_lookup["latitude"].max() + 0.25,
    )

    axis.set_aspect("equal", adjustable="box")

    axis.grid(
        linewidth=0.35,
        linestyle=":",
        alpha=0.5,
    )

    png_path = (
        FIGURE_DIR
        / "fig02_regionalization_0p5degree.png"
    )

    pdf_path = (
        FIGURE_DIR
        / "fig02_regionalization_0p5degree.pdf"
    )

    figure.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"QC dataset does not exist: {INPUT_PATH}"
        )

    print("=" * 78)
    print("MALAYSIA HYDROLOGY — SPATIAL REGIONALIZATION")
    print("=" * 78)

    frame = pd.read_parquet(INPUT_PATH)

    frame["date"] = pd.to_datetime(frame["date"])

    if frame["grid_id"].nunique() != EXPECTED_NATIVE_GRIDS:
        raise RuntimeError(
            "Unexpected native-grid count."
        )

    records_per_grid = frame.groupby(
        "grid_id",
        observed=True,
    ).size()

    if not (
        records_per_grid == EXPECTED_MONTHS_PER_GRID
    ).all():
        raise RuntimeError(
            "One or more native cells do not contain "
            "360 monthly observations."
        )

    frame = assign_regions(frame)

    grid_lookup = make_native_grid_lookup(frame)
    aet_grid_summary = make_aet_grid_summary(frame)

    region_inventory = make_region_inventory(
        frame,
        grid_lookup,
    )

    regional_monthly = aggregate_regional_monthly(
        frame,
        region_inventory,
    )

    grid_lookup_path = (
        PROCESSED_DIR
        / "native_grid_to_region_0p5degree.csv"
    )

    regional_parquet_path = (
        PROCESSED_DIR
        / "malaysia_hydrology_regional_monthly_0p5degree.parquet"
    )

    regional_csv_path = (
        PROCESSED_DIR
        / "malaysia_hydrology_regional_monthly_0p5degree.csv.gz"
    )

    inventory_path = (
        TABLE_DIR / "region_inventory_0p5degree.csv"
    )

    aet_grid_path = (
        TABLE_DIR / "aet_corrections_by_native_grid.csv"
    )

    summary_path = (
        TABLE_DIR / "regionalization_summary_0p5degree.json"
    )

    grid_lookup.to_csv(
        grid_lookup_path,
        index=False,
    )

    regional_monthly.to_parquet(
        regional_parquet_path,
        index=False,
        compression="snappy",
    )

    regional_monthly.to_csv(
        regional_csv_path,
        index=False,
        compression="gzip",
    )

    region_inventory.to_csv(
        inventory_path,
        index=False,
    )

    aet_grid_summary.to_csv(
        aet_grid_path,
        index=False,
    )

    save_coverage_figure(
        grid_lookup,
        aet_grid_summary,
    )

    save_regionalization_figure(
        grid_lookup,
        region_inventory,
    )

    regions_with_aet_correction = int(
        (
            region_inventory[
                "corrected_aet_records"
            ] > 0
        ).sum()
    )

    corrected_native_cells = int(
        (
            aet_grid_summary[
                "corrected_aet_records"
            ] > 0
        ).sum()
    )

    summary = {
        "status": "PASS",
        "input_dataset": str(INPUT_PATH),
        "region_size_degrees": REGION_SIZE_DEG,
        "longitude_origin": LON_ORIGIN,
        "latitude_origin": LAT_ORIGIN,
        "native_grid_count": int(len(grid_lookup)),
        "regional_unit_count": int(
            region_inventory["region_id"].nunique()
        ),
        "regional_monthly_rows": int(
            len(regional_monthly)
        ),
        "months_per_region": 360,
        "minimum_native_cells_per_region": int(
            region_inventory[
                "native_cell_count"
            ].min()
        ),
        "median_native_cells_per_region": float(
            region_inventory[
                "native_cell_count"
            ].median()
        ),
        "maximum_native_cells_per_region": int(
            region_inventory[
                "native_cell_count"
            ].max()
        ),
        "small_regions_lt_4_cells": int(
            region_inventory[
                "small_region_lt_4_cells"
            ].sum()
        ),
        "very_small_regions_lt_2_cells": int(
            region_inventory[
                "very_small_region_lt_2_cells"
            ].sum()
        ),
        "native_cells_with_aet_correction": (
            corrected_native_cells
        ),
        "regions_with_aet_correction": (
            regions_with_aet_correction
        ),
        "total_aet_corrected_records": int(
            frame["aet_negative_corrected"].sum()
        ),
        "native_grid_lookup": str(grid_lookup_path),
        "regional_monthly_parquet": str(
            regional_parquet_path
        ),
        "regional_monthly_csv": str(
            regional_csv_path
        ),
        "region_inventory": str(inventory_path),
        "aet_grid_summary": str(aet_grid_path),
    }

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
    print(
        f"Native grid cells              : "
        f"{summary['native_grid_count']}"
    )
    print(
        f"0.5° regional units            : "
        f"{summary['regional_unit_count']}"
    )
    print(
        f"Regional monthly rows          : "
        f"{summary['regional_monthly_rows']:,}"
    )
    print(
        f"Months per region              : "
        f"{summary['months_per_region']}"
    )
    print(
        f"Native cells per region        : "
        f"min={summary['minimum_native_cells_per_region']}, "
        f"median={summary['median_native_cells_per_region']}, "
        f"max={summary['maximum_native_cells_per_region']}"
    )
    print(
        f"Small regions (<4 cells)       : "
        f"{summary['small_regions_lt_4_cells']}"
    )
    print(
        f"Very small regions (<2 cells)  : "
        f"{summary['very_small_regions_lt_2_cells']}"
    )
    print(
        f"Native cells with AET fixes    : "
        f"{summary['native_cells_with_aet_correction']}"
    )
    print(
        f"Regions with AET fixes         : "
        f"{summary['regions_with_aet_correction']}"
    )
    print(
        f"Total corrected AET records    : "
        f"{summary['total_aet_corrected_records']}"
    )
    print()
    print(
        f"Regional monthly dataset       : "
        f"{regional_parquet_path}"
    )
    print(
        f"Region inventory               : "
        f"{inventory_path}"
    )
    print(
        f"Regionalization summary        : "
        f"{summary_path}"
    )
    print(
        f"Coverage figure                : "
        f"{FIGURE_DIR / 'fig01_native_grid_coverage.png'}"
    )
    print(
        f"Regionalization figure         : "
        f"{FIGURE_DIR / 'fig02_regionalization_0p5degree.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
