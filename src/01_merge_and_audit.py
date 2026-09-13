#!/usr/bin/env python3
"""
Merge and audit the Sri Lanka monthly hydrology dataset.

Outputs
-------
data/processed/srilanka_hydrology_monthly_1982_2011.parquet
data/processed/srilanka_hydrology_monthly_1982_2011.csv.gz
results/tables/annual_file_audit.csv
results/tables/variable_summary.csv
results/tables/grid_coordinates.csv
results/tables/audit_summary.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"

EXPECTED_YEARS = list(range(1982, 2012))

EXPECTED_COLUMNS = [
    "grid_id",
    "longitude",
    "latitude",
    "date",
    "year",
    "month",
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm",
    "surface_runoff_mm",
    "subsurface_runoff_mm",
    "total_runoff_mm",
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
    "soil_moisture_total_mm",
]

FLUX_STATE_COLUMNS = [
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm",
    "surface_runoff_mm",
    "subsurface_runoff_mm",
    "total_runoff_mm",
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
    "soil_moisture_total_mm",
]

SOIL_LAYER_COLUMNS = [
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
]


def stop(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def extract_year(path: Path) -> int:
    match = re.search(r"Hydrology_(\d{4})_M01_12\.csv$", path.name)
    if match is None:
        stop(f"Cannot extract year from filename: {path.name}")
    return int(match.group(1))


def find_annual_files() -> list[Path]:
    files = sorted(
        RAW_DIR.glob("SriLanka_Hydrology_*_M01_12.csv"),
        key=extract_year,
    )

    if not files:
        stop(f"No annual CSV files found in {RAW_DIR}")

    years = [extract_year(path) for path in files]

    missing = sorted(set(EXPECTED_YEARS) - set(years))
    unexpected = sorted(set(years) - set(EXPECTED_YEARS))
    duplicates = sorted(
        year for year in set(years) if years.count(year) > 1
    )

    if missing:
        stop(f"Missing years: {missing}")

    if unexpected:
        stop(f"Unexpected years: {unexpected}")

    if duplicates:
        stop(f"Duplicate years: {duplicates}")

    if len(files) != 30:
        stop(f"Expected 30 annual files but found {len(files)}")

    return files


def read_and_validate_file(
    path: Path,
    reference_grids: set[str] | None,
) -> tuple[pd.DataFrame, dict[str, object], set[str]]:
    file_year = extract_year(path)

    frame = pd.read_csv(path, low_memory=False)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        stop(f"{path.name} is missing columns: {missing_columns}")

    extra_columns = [
        column for column in frame.columns
        if column not in EXPECTED_COLUMNS
    ]

    if extra_columns:
        print(
            f"WARNING: {path.name} contains extra columns: "
            f"{extra_columns}"
        )

    frame = frame[EXPECTED_COLUMNS].copy()

    frame["grid_id"] = frame["grid_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    numeric_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in {"grid_id", "date"}
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_dates = int(frame["date"].isna().sum())
    if invalid_dates:
        stop(f"{path.name} contains {invalid_dates} invalid dates")

    content_years = sorted(
        frame["year"].dropna().astype(int).unique().tolist()
    )

    if content_years != [file_year]:
        stop(
            f"{path.name}: filename year is {file_year}, "
            f"but content contains {content_years}"
        )

    date_years = sorted(frame["date"].dt.year.unique().tolist())

    if date_years != [file_year]:
        stop(
            f"{path.name}: date column contains years {date_years}"
        )

    unique_months = sorted(
        frame["month"].dropna().astype(int).unique().tolist()
    )

    if unique_months != list(range(1, 13)):
        stop(
            f"{path.name}: expected months 1-12, "
            f"found {unique_months}"
        )

    month_date_mismatch = int(
        (
            frame["month"].astype(int)
            != frame["date"].dt.month.astype(int)
        ).sum()
    )

    if month_date_mismatch:
        stop(
            f"{path.name}: {month_date_mismatch} rows have "
            "inconsistent date and month values"
        )

    duplicate_rows = int(
        frame.duplicated(
            subset=["grid_id", "date"],
            keep=False,
        ).sum()
    )

    if duplicate_rows:
        stop(
            f"{path.name}: {duplicate_rows} duplicated "
            "grid_id/date rows"
        )

    grids = set(frame["grid_id"].unique())

    if reference_grids is not None and grids != reference_grids:
        missing_grids = sorted(reference_grids - grids)
        new_grids = sorted(grids - reference_grids)

        stop(
            f"{path.name}: grid coverage differs from the first year. "
            f"Missing grids={len(missing_grids)}, "
            f"new grids={len(new_grids)}"
        )

    records_per_grid = frame.groupby("grid_id").size()

    invalid_grid_counts = records_per_grid[
        records_per_grid != 12
    ]

    if not invalid_grid_counts.empty:
        stop(
            f"{path.name}: {len(invalid_grid_counts)} grids "
            "do not contain exactly 12 records"
        )

    months_per_grid = frame.groupby("grid_id")["month"].nunique()

    invalid_month_counts = months_per_grid[
        months_per_grid != 12
    ]

    if not invalid_month_counts.empty:
        stop(
            f"{path.name}: {len(invalid_month_counts)} grids "
            "do not contain all 12 months"
        )

    missing_values = int(frame.isna().sum().sum())

    negative_counts = {
        column: int((frame[column] < 0).sum())
        for column in FLUX_STATE_COLUMNS
    }

    total_negative_values = int(sum(negative_counts.values()))

    reconstructed_runoff = (
        frame["surface_runoff_mm"]
        + frame["subsurface_runoff_mm"]
    )

    runoff_error = (
        frame["total_runoff_mm"] - reconstructed_runoff
    ).abs()

    runoff_mismatches = int(
        (
            ~np.isclose(
                frame["total_runoff_mm"],
                reconstructed_runoff,
                rtol=1e-5,
                atol=1e-4,
            )
        ).sum()
    )

    reconstructed_soil = frame[SOIL_LAYER_COLUMNS].sum(axis=1)

    soil_error = (
        frame["soil_moisture_total_mm"] - reconstructed_soil
    ).abs()

    soil_mismatches = int(
        (
            ~np.isclose(
                frame["soil_moisture_total_mm"],
                reconstructed_soil,
                rtol=1e-5,
                atol=1e-3,
            )
        ).sum()
    )

    audit = {
        "file": path.name,
        "year": file_year,
        "rows": int(len(frame)),
        "grid_count": int(frame["grid_id"].nunique()),
        "date_min": frame["date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["date"].max().strftime("%Y-%m-%d"),
        "missing_values": missing_values,
        "negative_values": total_negative_values,
        "duplicate_grid_date_rows": duplicate_rows,
        "runoff_identity_mismatches": runoff_mismatches,
        "runoff_identity_max_error_mm": float(runoff_error.max()),
        "soil_identity_mismatches": soil_mismatches,
        "soil_identity_max_error_mm": float(soil_error.max()),
    }

    return frame, audit, grids


def make_variable_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in FLUX_STATE_COLUMNS:
        values = frame[column]

        rows.append(
            {
                "variable": column,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "negative": int((values < 0).sum()),
                "minimum": float(values.min()),
                "q01": float(values.quantile(0.01)),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "q75": float(values.quantile(0.75)),
                "q99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
                "standard_deviation": float(values.std()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SRI LANKA HYDROLOGY — MERGE AND DATA AUDIT")
    print("=" * 78)

    files = find_annual_files()

    frames: list[pd.DataFrame] = []
    annual_audits: list[dict[str, object]] = []

    reference_grids: set[str] | None = None

    for index, path in enumerate(files, start=1):
        frame, audit, grids = read_and_validate_file(
            path,
            reference_grids,
        )

        if reference_grids is None:
            reference_grids = grids

        frames.append(frame)
        annual_audits.append(audit)

        print(
            f"[{index:02d}/30] {path.name} | "
            f"rows={audit['rows']:,} | "
            f"grids={audit['grid_count']} | "
            f"missing={audit['missing_values']} | "
            f"negative={audit['negative_values']}"
        )

    merged = pd.concat(frames, ignore_index=True)

    merged = merged.sort_values(
        ["grid_id", "date"],
        kind="mergesort",
    ).reset_index(drop=True)

    grid_count = int(merged["grid_id"].nunique())
    expected_rows = grid_count * 12 * len(EXPECTED_YEARS)

    if len(merged) != expected_rows:
        stop(
            f"Expected {expected_rows:,} merged rows based on "
            f"{grid_count} grids, but found {len(merged):,}"
        )

    merged_duplicates = int(
        merged.duplicated(
            subset=["grid_id", "date"],
            keep=False,
        ).sum()
    )

    if merged_duplicates:
        stop(
            f"Merged dataset contains {merged_duplicates} "
            "duplicate grid/date rows"
        )

    records_per_grid = merged.groupby("grid_id").size()
    expected_records_per_grid = len(EXPECTED_YEARS) * 12

    bad_grid_history = records_per_grid[
        records_per_grid != expected_records_per_grid
    ]

    if not bad_grid_history.empty:
        stop(
            f"{len(bad_grid_history)} grids do not contain "
            f"{expected_records_per_grid} monthly observations"
        )

    coordinate_ranges = (
        merged.groupby("grid_id")
        .agg(
            longitude_min=("longitude", "min"),
            longitude_max=("longitude", "max"),
            latitude_min=("latitude", "min"),
            latitude_max=("latitude", "max"),
        )
    )

    coordinate_ranges["longitude_difference"] = (
        coordinate_ranges["longitude_max"]
        - coordinate_ranges["longitude_min"]
    )

    coordinate_ranges["latitude_difference"] = (
        coordinate_ranges["latitude_max"]
        - coordinate_ranges["latitude_min"]
    )

    unstable_coordinates = coordinate_ranges[
        (coordinate_ranges["longitude_difference"].abs() > 1e-10)
        | (coordinate_ranges["latitude_difference"].abs() > 1e-10)
    ]

    if not unstable_coordinates.empty:
        stop(
            f"{len(unstable_coordinates)} grid IDs change coordinates "
            "between years"
        )

    reconstructed_runoff = (
        merged["surface_runoff_mm"]
        + merged["subsurface_runoff_mm"]
    )

    reconstructed_soil = merged[SOIL_LAYER_COLUMNS].sum(axis=1)

    runoff_abs_error = (
        merged["total_runoff_mm"] - reconstructed_runoff
    ).abs()

    soil_abs_error = (
        merged["soil_moisture_total_mm"] - reconstructed_soil
    ).abs()

    annual_audit = pd.DataFrame(annual_audits)
    variable_summary = make_variable_summary(merged)

    grid_coordinates = (
        merged[["grid_id", "longitude", "latitude"]]
        .drop_duplicates()
        .sort_values(
            ["latitude", "longitude"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    parquet_path = (
        PROCESSED_DIR
        / "srilanka_hydrology_monthly_1982_2011.parquet"
    )

    compressed_csv_path = (
        PROCESSED_DIR
        / "srilanka_hydrology_monthly_1982_2011.csv.gz"
    )

    annual_audit_path = TABLE_DIR / "annual_file_audit.csv"
    variable_summary_path = TABLE_DIR / "variable_summary.csv"
    coordinate_path = TABLE_DIR / "grid_coordinates.csv"
    summary_path = TABLE_DIR / "audit_summary.json"

    merged.to_parquet(
        parquet_path,
        index=False,
        compression="snappy",
    )

    merged.to_csv(
        compressed_csv_path,
        index=False,
        compression="gzip",
    )

    annual_audit.to_csv(annual_audit_path, index=False)
    variable_summary.to_csv(variable_summary_path, index=False)
    grid_coordinates.to_csv(coordinate_path, index=False)

    summary = {
        "status": "PASS",
        "annual_file_count": len(files),
        "year_start": int(merged["year"].min()),
        "year_end": int(merged["year"].max()),
        "grid_count": grid_count,
        "months_per_grid": expected_records_per_grid,
        "total_rows": int(len(merged)),
        "total_columns": int(len(merged.columns)),
        "missing_values": int(merged.isna().sum().sum()),
        "negative_values": int(
            sum((merged[column] < 0).sum()
                for column in FLUX_STATE_COLUMNS)
        ),
        "duplicate_grid_date_rows": merged_duplicates,
        "longitude_min": float(merged["longitude"].min()),
        "longitude_max": float(merged["longitude"].max()),
        "latitude_min": float(merged["latitude"].min()),
        "latitude_max": float(merged["latitude"].max()),
        "date_start": merged["date"].min().strftime("%Y-%m-%d"),
        "date_end": merged["date"].max().strftime("%Y-%m-%d"),
        "runoff_identity_max_error_mm": float(
            runoff_abs_error.max()
        ),
        "runoff_identity_mean_error_mm": float(
            runoff_abs_error.mean()
        ),
        "soil_identity_max_error_mm": float(
            soil_abs_error.max()
        ),
        "soil_identity_mean_error_mm": float(
            soil_abs_error.mean()
        ),
        "parquet_output": str(parquet_path),
        "compressed_csv_output": str(compressed_csv_path),
    }

    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print()
    print("=" * 78)
    print("FINAL AUDIT SUMMARY")
    print("=" * 78)
    print(f"Status                    : {summary['status']}")
    print(f"Annual files              : {summary['annual_file_count']}")
    print(
        f"Period                    : "
        f"{summary['year_start']}-{summary['year_end']}"
    )
    print(f"Spatial grids             : {summary['grid_count']}")
    print(f"Months per grid           : {summary['months_per_grid']}")
    print(f"Total rows                : {summary['total_rows']:,}")
    print(f"Total columns             : {summary['total_columns']}")
    print(f"Missing values            : {summary['missing_values']}")
    print(f"Negative values           : {summary['negative_values']}")
    print(
        f"Duplicate grid/date rows  : "
        f"{summary['duplicate_grid_date_rows']}"
    )
    print(
        f"Longitude range           : "
        f"{summary['longitude_min']:.4f} to "
        f"{summary['longitude_max']:.4f}"
    )
    print(
        f"Latitude range            : "
        f"{summary['latitude_min']:.4f} to "
        f"{summary['latitude_max']:.4f}"
    )
    print(
        f"Maximum runoff identity error : "
        f"{summary['runoff_identity_max_error_mm']:.10f} mm"
    )
    print(
        f"Maximum soil identity error   : "
        f"{summary['soil_identity_max_error_mm']:.10f} mm"
    )
    print()
    print(f"Master Parquet            : {parquet_path}")
    print(f"Compressed master CSV     : {compressed_csv_path}")
    print(f"Annual audit table        : {annual_audit_path}")
    print(f"Variable summary          : {variable_summary_path}")
    print(f"Grid-coordinate table     : {coordinate_path}")
    print(f"JSON audit summary        : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
