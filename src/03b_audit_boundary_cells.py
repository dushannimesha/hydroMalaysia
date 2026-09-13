#!/usr/bin/env python3
"""
Diagnose native cells containing repeated negative-AET corrections.

For each affected cell:
1. Identify its eight nearest valid native neighbours.
2. Compare complete 1982-2011 monthly time series.
3. Compare 30-year monthly climatologies.
4. Calculate correlation, bias and normalized RMSE.
5. Generate diagnostic figures.

No cells are removed by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "srilanka_hydrology_monthly_1982_2011_qc.parquet"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

N_NEIGHBOURS = 8

VARIABLES = [
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm_raw",
    "total_runoff_mm",
    "soil_moisture_total_mm",
]

PLOT_VARIABLES = [
    (
        "precipitation_mm",
        "Precipitation",
        "mm month$^{-1}$",
    ),
    (
        "actual_evapotranspiration_mm_raw",
        "Actual evapotranspiration",
        "mm month$^{-1}$",
    ),
    (
        "total_runoff_mm",
        "Total runoff",
        "mm month$^{-1}$",
    ),
    (
        "soil_moisture_total_mm",
        "Total soil moisture",
        "mm",
    ),
]


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    mask = np.isfinite(first) & np.isfinite(second)

    first_valid = first[mask]
    second_valid = second[mask]

    if len(first_valid) < 3:
        return float("nan")

    if np.std(first_valid) == 0 or np.std(second_valid) == 0:
        return float("nan")

    return float(
        np.corrcoef(first_valid, second_valid)[0, 1]
    )


def calculate_distance_km(
    longitude_1: float,
    latitude_1: float,
    longitude_2: pd.Series,
    latitude_2: pd.Series,
) -> pd.Series:
    """
    Approximate local horizontal distance over Sri Lanka.

    One degree latitude is approximately 111.32 km.
    Longitude distance is adjusted by mean latitude.
    """
    mean_latitude = np.deg2rad(
        (latitude_1 + latitude_2) / 2.0
    )

    delta_x = (
        (longitude_2 - longitude_1)
        * 111.32
        * np.cos(mean_latitude)
    )

    delta_y = (
        latitude_2 - latitude_1
    ) * 111.32

    return np.sqrt(delta_x**2 + delta_y**2)


def find_neighbours(
    grid_lookup: pd.DataFrame,
    suspect_id: str,
) -> pd.DataFrame:
    suspect = grid_lookup.loc[
        grid_lookup["grid_id"] == suspect_id
    ]

    if len(suspect) != 1:
        raise RuntimeError(
            f"Expected one row for {suspect_id}, "
            f"found {len(suspect)}."
        )

    suspect_row = suspect.iloc[0]

    candidates = grid_lookup.loc[
        grid_lookup["grid_id"] != suspect_id
    ].copy()

    candidates["distance_km"] = calculate_distance_km(
        longitude_1=float(suspect_row["longitude"]),
        latitude_1=float(suspect_row["latitude"]),
        longitude_2=candidates["longitude"],
        latitude_2=candidates["latitude"],
    )

    neighbours = (
        candidates
        .sort_values(
            ["distance_km", "grid_id"]
        )
        .head(N_NEIGHBOURS)
        .reset_index(drop=True)
    )

    neighbours.insert(
        0,
        "suspect_grid_id",
        suspect_id,
    )

    neighbours.insert(
        1,
        "neighbour_rank",
        np.arange(1, len(neighbours) + 1),
    )

    return neighbours


def build_neighbour_reference(
    frame: pd.DataFrame,
    neighbour_ids: list[str],
) -> pd.DataFrame:
    neighbour_data = frame.loc[
        frame["grid_id"].isin(neighbour_ids),
        ["date", *VARIABLES],
    ].copy()

    reference = (
        neighbour_data.groupby(
            "date",
            as_index=False,
            observed=True,
        )[VARIABLES]
        .median()
    )

    reference = reference.rename(
        columns={
            variable: f"neighbour_median__{variable}"
            for variable in VARIABLES
        }
    )

    return reference


def calculate_metrics(
    suspect_data: pd.DataFrame,
    reference: pd.DataFrame,
    suspect_id: str,
) -> list[dict[str, object]]:
    combined = suspect_data.merge(
        reference,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != 360:
        raise RuntimeError(
            f"{suspect_id}: expected 360 matched dates, "
            f"found {len(combined)}."
        )

    rows: list[dict[str, object]] = []

    for variable in VARIABLES:
        suspect_values = combined[variable].to_numpy(
            dtype=float
        )

        reference_values = combined[
            f"neighbour_median__{variable}"
        ].to_numpy(dtype=float)

        difference = suspect_values - reference_values

        rmse = float(
            np.sqrt(np.mean(difference**2))
        )

        reference_mean = float(
            np.mean(reference_values)
        )

        reference_range = float(
            np.max(reference_values)
            - np.min(reference_values)
        )

        normalized_rmse_mean = (
            rmse / abs(reference_mean)
            if abs(reference_mean) > 1e-12
            else float("nan")
        )

        normalized_rmse_range = (
            rmse / reference_range
            if reference_range > 1e-12
            else float("nan")
        )

        rows.append(
            {
                "suspect_grid_id": suspect_id,
                "variable": variable,
                "suspect_mean": float(
                    np.mean(suspect_values)
                ),
                "neighbour_median_mean": reference_mean,
                "mean_bias": float(
                    np.mean(difference)
                ),
                "mean_absolute_error": float(
                    np.mean(np.abs(difference))
                ),
                "rmse": rmse,
                "normalized_rmse_by_neighbour_mean": (
                    normalized_rmse_mean
                ),
                "normalized_rmse_by_neighbour_range": (
                    normalized_rmse_range
                ),
                "pearson_correlation": safe_correlation(
                    suspect_values,
                    reference_values,
                ),
                "suspect_minimum": float(
                    np.min(suspect_values)
                ),
                "suspect_maximum": float(
                    np.max(suspect_values)
                ),
                "neighbour_minimum": float(
                    np.min(reference_values)
                ),
                "neighbour_maximum": float(
                    np.max(reference_values)
                ),
            }
        )

    return rows


def calculate_climatology(
    suspect_data: pd.DataFrame,
    reference: pd.DataFrame,
    suspect_id: str,
) -> pd.DataFrame:
    combined = suspect_data.merge(
        reference,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    combined["month"] = combined["date"].dt.month

    climatology_rows = []

    for month, group in combined.groupby(
        "month",
        observed=True,
    ):
        for variable in VARIABLES:
            climatology_rows.append(
                {
                    "suspect_grid_id": suspect_id,
                    "month": int(month),
                    "variable": variable,
                    "suspect_climatology": float(
                        group[variable].mean()
                    ),
                    "neighbour_median_climatology": float(
                        group[
                            f"neighbour_median__{variable}"
                        ].mean()
                    ),
                    "suspect_standard_deviation": float(
                        group[variable].std()
                    ),
                    "neighbour_standard_deviation": float(
                        group[
                            f"neighbour_median__{variable}"
                        ].std()
                    ),
                }
            )

    return pd.DataFrame(climatology_rows)


def make_figure(
    climatology: pd.DataFrame,
    suspect_lookup: pd.DataFrame,
) -> None:
    suspect_ids = suspect_lookup["grid_id"].tolist()

    figure, axes = plt.subplots(
        nrows=len(PLOT_VARIABLES),
        ncols=len(suspect_ids),
        figsize=(10.5, 10.5),
        sharex=True,
        constrained_layout=True,
    )

    if len(suspect_ids) == 1:
        axes = np.asarray(axes).reshape(
            len(PLOT_VARIABLES),
            1,
        )

    for column_index, suspect_id in enumerate(
        suspect_ids
    ):
        coordinate = suspect_lookup.loc[
            suspect_lookup["grid_id"] == suspect_id
        ].iloc[0]

        for row_index, (
            variable,
            title,
            unit,
        ) in enumerate(PLOT_VARIABLES):
            axis = axes[row_index, column_index]

            subset = climatology.loc[
                (
                    climatology["suspect_grid_id"]
                    == suspect_id
                )
                & (
                    climatology["variable"]
                    == variable
                )
            ].sort_values("month")

            axis.plot(
                subset["month"],
                subset["suspect_climatology"],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color="black",
                label="Suspect cell",
            )

            axis.plot(
                subset["month"],
                subset[
                    "neighbour_median_climatology"
                ],
                marker="s",
                linewidth=1.2,
                markersize=3.0,
                linestyle="--",
                color="0.45",
                label="Median of 8 neighbours",
            )

            axis.set_xticks(range(1, 13))
            axis.grid(
                linewidth=0.35,
                linestyle=":",
                alpha=0.6,
            )

            if column_index == 0:
                axis.set_ylabel(
                    f"{title}\n({unit})"
                )

            if row_index == 0:
                axis.set_title(
                    f"{suspect_id}\n"
                    f"({coordinate['longitude']:.2f}°E, "
                    f"{coordinate['latitude']:.2f}°N)"
                )

            if row_index == len(PLOT_VARIABLES) - 1:
                axis.set_xlabel("Month")

            if row_index == 0 and column_index == 0:
                axis.legend(
                    frameon=True,
                    fontsize=8,
                    loc="best",
                )

    figure.suptitle(
        "Monthly climatology of suspect coastal cells "
        "and their nearest neighbours",
        fontsize=12,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig03_boundary_cell_diagnostics.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig03_boundary_cell_diagnostics.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    frame = pd.read_parquet(INPUT_PATH)
    frame["date"] = pd.to_datetime(frame["date"])

    grid_lookup = (
        frame[
            [
                "grid_id",
                "longitude",
                "latitude",
            ]
        ]
        .drop_duplicates()
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    suspect_ids = sorted(
        frame.loc[
            frame["aet_negative_corrected"],
            "grid_id",
        ].unique().tolist()
    )

    if not suspect_ids:
        raise RuntimeError(
            "No cells with corrected negative AET were found."
        )

    print("=" * 78)
    print("BOUNDARY-CELL HYDROLOGICAL DIAGNOSTIC")
    print("=" * 78)
    print(f"Suspect cells found : {len(suspect_ids)}")
    print(f"Suspect IDs         : {suspect_ids}")
    print(f"Neighbours per cell : {N_NEIGHBOURS}")
    print()

    neighbour_tables = []
    metric_rows = []
    climatology_tables = []
    suspect_summary_rows = []

    for suspect_id in suspect_ids:
        suspect_data = (
            frame.loc[
                frame["grid_id"] == suspect_id,
                [
                    "grid_id",
                    "longitude",
                    "latitude",
                    "date",
                    "aet_negative_corrected",
                    *VARIABLES,
                ],
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )

        neighbours = find_neighbours(
            grid_lookup,
            suspect_id,
        )

        neighbour_tables.append(neighbours)

        neighbour_ids = neighbours["grid_id"].tolist()

        reference = build_neighbour_reference(
            frame,
            neighbour_ids,
        )

        metric_rows.extend(
            calculate_metrics(
                suspect_data,
                reference,
                suspect_id,
            )
        )

        climatology_tables.append(
            calculate_climatology(
                suspect_data,
                reference,
                suspect_id,
            )
        )

        suspect_summary_rows.append(
            {
                "grid_id": suspect_id,
                "longitude": float(
                    suspect_data["longitude"].iloc[0]
                ),
                "latitude": float(
                    suspect_data["latitude"].iloc[0]
                ),
                "corrected_aet_records": int(
                    suspect_data[
                        "aet_negative_corrected"
                    ].sum()
                ),
                "corrected_fraction": float(
                    suspect_data[
                        "aet_negative_corrected"
                    ].mean()
                ),
                "minimum_raw_aet_mm": float(
                    suspect_data[
                        "actual_evapotranspiration_mm_raw"
                    ].min()
                ),
                "median_total_soil_moisture_mm": float(
                    suspect_data[
                        "soil_moisture_total_mm"
                    ].median()
                ),
                "median_total_runoff_mm": float(
                    suspect_data[
                        "total_runoff_mm"
                    ].median()
                ),
            }
        )

        print(
            f"{suspect_id}: "
            f"{int(suspect_data['aet_negative_corrected'].sum())} "
            "corrected records"
        )

        print(
            "  nearest neighbours: "
            + ", ".join(neighbour_ids)
        )

    neighbour_table = pd.concat(
        neighbour_tables,
        ignore_index=True,
    )

    metrics = pd.DataFrame(metric_rows)

    climatology = pd.concat(
        climatology_tables,
        ignore_index=True,
    )

    suspect_summary = pd.DataFrame(
        suspect_summary_rows
    )

    neighbour_table.to_csv(
        TABLE_DIR / "boundary_cell_neighbours.csv",
        index=False,
    )

    metrics.to_csv(
        TABLE_DIR / "boundary_cell_metrics.csv",
        index=False,
    )

    climatology.to_csv(
        TABLE_DIR
        / "boundary_cell_monthly_climatology.csv",
        index=False,
    )

    suspect_summary.to_csv(
        TABLE_DIR / "boundary_cell_summary.csv",
        index=False,
    )

    suspect_lookup = grid_lookup.loc[
        grid_lookup["grid_id"].isin(suspect_ids)
    ].copy()

    make_figure(
        climatology,
        suspect_lookup,
    )

    decision_summary = {
        "status": "PASS",
        "suspect_cell_count": len(suspect_ids),
        "suspect_grid_ids": suspect_ids,
        "neighbours_per_suspect_cell": N_NEIGHBOURS,
        "diagnostic_only": True,
        "cells_removed": 0,
        "recommended_next_action": (
            "Review correlation, normalized RMSE and climatology "
            "against neighbouring cells before deciding whether "
            "to exclude the suspect cells from primary analyses."
        ),
    }

    with (
        TABLE_DIR
        / "boundary_cell_diagnostic_summary.json"
    ).open("w", encoding="utf-8") as stream:
        json.dump(
            decision_summary,
            stream,
            indent=2,
        )

    print()
    print("Diagnostic outputs:")
    print(
        f"  {TABLE_DIR / 'boundary_cell_summary.csv'}"
    )
    print(
        f"  {TABLE_DIR / 'boundary_cell_neighbours.csv'}"
    )
    print(
        f"  {TABLE_DIR / 'boundary_cell_metrics.csv'}"
    )
    print(
        f"  {FIGURE_DIR / 'fig03_boundary_cell_diagnostics.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
