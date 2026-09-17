#!/usr/bin/env python3
"""
Create the final analysis-ready Malaysia hydrology dataset.

Five Malaysia grids are excluded because their AET, runoff, and soil
moisture variables are missing for the complete 1982-2011 period.

The QC dataset is preserved unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "malaysia_hydrology_monthly_1982_2011_qc.parquet"
)

OUTPUT_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "malaysia_hydrology_monthly_1982_2011_analysis.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "malaysia_hydrology_monthly_1982_2011_analysis.csv.gz"
)

TABLE_DIR = ROOT / "results" / "tables"

EXCLUDED_GRID_IDS = [
    "MY_0p5_2.2500_104.7500",
    "MY_0p5_2.2500_109.7500",
    "MY_0p5_4.7500_100.2500",
    "MY_0p5_4.7500_113.7500",
    "MY_0p5_5.7500_103.2500",
]

EXPECTED_ORIGINAL_GRIDS = 177
EXPECTED_RETAINED_GRIDS = 172
EXPECTED_RECORDS_PER_GRID = 360
EXPECTED_EXCLUDED_ROWS = 1_800
EXPECTED_RETAINED_ROWS = 61_920

SOIL_LAYER_COLUMNS = [
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input QC dataset does not exist: {INPUT_PATH}"
        )

    frame = pd.read_parquet(INPUT_PATH)
    frame["date"] = pd.to_datetime(frame["date"])

    original_grid_count = int(frame["grid_id"].nunique())

    if original_grid_count != EXPECTED_ORIGINAL_GRIDS:
        raise RuntimeError(
            f"Expected {EXPECTED_ORIGINAL_GRIDS} original grids, "
            f"found {original_grid_count}."
        )

    available_ids = set(frame["grid_id"].unique())
    missing_exclusion_ids = sorted(
        set(EXCLUDED_GRID_IDS) - available_ids
    )

    if missing_exclusion_ids:
        raise RuntimeError(
            f"Excluded grid IDs are missing: {missing_exclusion_ids}"
        )

    excluded = frame.loc[
        frame["grid_id"].isin(EXCLUDED_GRID_IDS)
    ].copy()

    retained = frame.loc[
        ~frame["grid_id"].isin(EXCLUDED_GRID_IDS)
    ].copy()

    excluded_counts = (
        excluded.groupby("grid_id", observed=True)
        .size()
        .sort_index()
    )

    if not (
        excluded_counts == EXPECTED_RECORDS_PER_GRID
    ).all():
        raise RuntimeError(
            "Each excluded grid must contain exactly "
            f"{EXPECTED_RECORDS_PER_GRID} records."
        )

    if len(excluded) != EXPECTED_EXCLUDED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_EXCLUDED_ROWS} excluded rows, "
            f"found {len(excluded)}."
        )

    retained_grid_count = int(
        retained["grid_id"].nunique()
    )

    if retained_grid_count != EXPECTED_RETAINED_GRIDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED_GRIDS} retained grids, "
            f"found {retained_grid_count}."
        )

    if len(retained) != EXPECTED_RETAINED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED_ROWS:,} retained rows, "
            f"found {len(retained):,}."
        )

    retained_counts = (
        retained.groupby("grid_id", observed=True)
        .size()
    )

    if not (
        retained_counts == EXPECTED_RECORDS_PER_GRID
    ).all():
        raise RuntimeError(
            "One or more retained grids do not contain "
            f"{EXPECTED_RECORDS_PER_GRID} records."
        )

    duplicate_count = int(
        retained.duplicated(
            subset=["grid_id", "date"],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        raise RuntimeError(
            f"Retained dataset contains {duplicate_count} "
            "duplicate grid/date rows."
        )

    missing_count = int(retained.isna().sum().sum())

    if missing_count:
        raise RuntimeError(
            f"Retained dataset contains {missing_count} "
            "missing values."
        )

    remaining_negative_aet = int(
        (
            retained["actual_evapotranspiration_mm"]
            < 0
        ).sum()
    )

    remaining_aet_flags = int(
        retained["aet_negative_corrected"].sum()
    )

    if remaining_negative_aet != 0:
        raise RuntimeError(
            "Negative AET values remain in the retained dataset."
        )

    if remaining_aet_flags != 0:
        raise RuntimeError(
            "AET-corrected records remain after excluding the "
            "two invalid boundary cells."
        )

    reconstructed_runoff = (
        retained["surface_runoff_mm"]
        + retained["subsurface_runoff_mm"]
    )

    runoff_error = (
        retained["total_runoff_mm"]
        - reconstructed_runoff
    ).abs()

    reconstructed_soil = retained[
        SOIL_LAYER_COLUMNS
    ].sum(axis=1)

    soil_error = (
        retained["soil_moisture_total_mm"]
        - reconstructed_soil
    ).abs()

    retained = retained.sort_values(
        ["grid_id", "date"],
        kind="mergesort",
    ).reset_index(drop=True)

    excluded = excluded.sort_values(
        ["grid_id", "date"],
        kind="mergesort",
    ).reset_index(drop=True)

    retained.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        compression="snappy",
    )

    retained.to_csv(
        OUTPUT_CSV,
        index=False,
        compression="gzip",
    )

    excluded_path = (
        TABLE_DIR / "excluded_boundary_cell_records.csv.gz"
    )

    excluded.to_csv(
        excluded_path,
        index=False,
        compression="gzip",
    )

    exclusion_manifest = (
        excluded.groupby(
            ["grid_id", "longitude", "latitude"],
            as_index=False,
            observed=True,
        )
        .agg(
            excluded_records=("date", "size"),
            corrected_aet_records=(
                "aet_negative_corrected",
                "sum",
            ),
            precipitation_mean_mm=(
                "precipitation_mm",
                "mean",
            ),
            PET_mean_mm=("PET_mm", "mean"),
            raw_AET_mean_mm=(
                "actual_evapotranspiration_mm_raw",
                "mean",
            ),
            runoff_mean_mm=(
                "total_runoff_mm",
                "mean",
            ),
            soil_moisture_mean_mm=(
                "soil_moisture_total_mm",
                "mean",
            ),
        )
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    manifest_path = (
        TABLE_DIR / "excluded_boundary_cells.csv"
    )

    exclusion_manifest.to_csv(
        manifest_path,
        index=False,
    )

    summary = {
        "status": "PASS",
        "input_dataset": str(INPUT_PATH),
        "output_parquet": str(OUTPUT_PARQUET),
        "output_compressed_csv": str(OUTPUT_CSV),
        "exclusion_manifest": str(manifest_path),
        "excluded_record_archive": str(excluded_path),
        "exclusion_reason": (
            "Hydrologically invalid coastal/boundary cells: "
            "near-zero AET and runoff, nearly constant minimum "
            "soil moisture, and weak agreement with neighbouring "
            "hydrological variables despite valid precipitation "
            "and PET."
        ),
        "excluded_grid_ids": EXCLUDED_GRID_IDS,
        "original_grid_count": original_grid_count,
        "retained_grid_count": retained_grid_count,
        "excluded_grid_count": len(EXCLUDED_GRID_IDS),
        "original_row_count": int(len(frame)),
        "retained_row_count": int(len(retained)),
        "excluded_row_count": int(len(excluded)),
        "records_per_retained_grid": (
            EXPECTED_RECORDS_PER_GRID
        ),
        "missing_values": missing_count,
        "duplicate_grid_date_rows": duplicate_count,
        "remaining_negative_aet_records": (
            remaining_negative_aet
        ),
        "remaining_aet_correction_flags": (
            remaining_aet_flags
        ),
        "maximum_runoff_identity_error_mm": float(
            runoff_error.max()
        ),
        "maximum_soil_identity_error_mm": float(
            soil_error.max()
        ),
    }

    summary_path = (
        TABLE_DIR
        / "analysis_dataset_exclusion_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(summary, stream, indent=2)

    print("=" * 78)
    print("FINAL ANALYSIS-DATASET CONSTRUCTION")
    print("=" * 78)
    print(f"Status                       : {summary['status']}")
    print(
        f"Original grids               : "
        f"{summary['original_grid_count']}"
    )
    print(
        f"Excluded grids               : "
        f"{summary['excluded_grid_count']}"
    )
    print(
        f"Retained grids               : "
        f"{summary['retained_grid_count']}"
    )
    print(
        f"Original rows                : "
        f"{summary['original_row_count']:,}"
    )
    print(
        f"Excluded rows                : "
        f"{summary['excluded_row_count']:,}"
    )
    print(
        f"Retained rows                : "
        f"{summary['retained_row_count']:,}"
    )
    print(
        f"Missing values               : "
        f"{summary['missing_values']}"
    )
    print(
        f"Remaining negative AET       : "
        f"{summary['remaining_negative_aet_records']}"
    )
    print(
        f"Remaining AET correction flags: "
        f"{summary['remaining_aet_correction_flags']}"
    )
    print()
    print(f"Analysis dataset             : {OUTPUT_PARQUET}")
    print(f"Exclusion manifest           : {manifest_path}")
    print(f"Exclusion summary            : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
