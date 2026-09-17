#!/usr/bin/env python3
"""
Audit runoff-target lineage and extend the climatological modelling table.

This stage verifies whether `runoff_mm` in the Stage-6 PINN dataset
corresponds to total runoff in the regional 1982–2011 dataset.

When surface and subsurface runoff are available, their monthly
climatologies are merged into the 396-row modelling table.

The script also constructs:
- total precipitation reconstructed from the four NMF modes;
- precipitation totals at lags L0–L3;
- one-month antecedent total soil moisture;
- an approximate monthly water-balance residual.

The water-balance residual is diagnostic only. The input products and
monthly climatological differencing may not satisfy strict closure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PINN_PATH = (
    ROOT
    / "data"
    / "processed"
    / "pinn_climatology_features_k4_lag3.parquet"
)

REGIONAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "malaysia_hydrology_regional_monthly_0p5degree.parquet"
)

OUTPUT_PARQUET = (
    ROOT
    / "data"
    / "processed"
    / "runoff_climatology_features_k4_lag3.parquet"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "processed"
    / "runoff_climatology_features_k4_lag3.csv.gz"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

EXPECTED_ROWS = 2064
EXPECTED_REGIONS = 172
EXPECTED_MONTHS = 12

TOTAL_RUNOFF_CANDIDATES = [
    "total_runoff_mm",
    "runoff_mm",
]

SURFACE_RUNOFF_CANDIDATES = [
    "surface_runoff_mm",
    "surface_runoff",
]

SUBSURFACE_RUNOFF_CANDIDATES = [
    "subsurface_runoff_mm",
    "sub_surface_runoff_mm",
    "subsurface_runoff",
]


def first_existing_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    column_set = set(columns)

    for candidate in candidates:
        if candidate in column_set:
            return candidate

    return None


def summary_statistics(
    values: pd.Series,
) -> dict[str, float | int]:
    numeric = values.to_numpy(
        dtype=float
    )

    return {
        "count": int(len(numeric)),
        "minimum": float(
            np.min(numeric)
        ),
        "q01": float(
            np.quantile(
                numeric,
                0.01,
            )
        ),
        "q05": float(
            np.quantile(
                numeric,
                0.05,
            )
        ),
        "q25": float(
            np.quantile(
                numeric,
                0.25,
            )
        ),
        "median": float(
            np.median(numeric)
        ),
        "mean": float(
            np.mean(numeric)
        ),
        "q75": float(
            np.quantile(
                numeric,
                0.75,
            )
        ),
        "q95": float(
            np.quantile(
                numeric,
                0.95,
            )
        ),
        "q99": float(
            np.quantile(
                numeric,
                0.99,
            )
        ),
        "maximum": float(
            np.max(numeric)
        ),
        "standard_deviation": float(
            np.std(
                numeric,
                ddof=0,
            )
        ),
        "zero_count": int(
            np.sum(
                np.isclose(
                    numeric,
                    0.0,
                    atol=1e-10,
                )
            )
        ),
        "negative_count": int(
            np.sum(
                numeric < 0.0
            )
        ),
        "skewness": float(
            values.skew()
        ),
    }


def add_circular_region_lag(
    data: pd.DataFrame,
    source_column: str,
    output_column: str,
    lag: int,
) -> pd.DataFrame:
    result = data.copy()

    result[output_column] = (
        result.groupby(
            "region_id",
            observed=True,
            sort=False,
        )[source_column]
        .transform(
            lambda series: np.roll(
                series.to_numpy(),
                lag,
            )
        )
    )

    return result


def build_monthly_climatology(
    regional: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    required = [
        "region_id",
        "month",
        *columns,
    ]

    missing = [
        column
        for column in required
        if column not in regional.columns
    ]

    if missing:
        raise RuntimeError(
            "Regional dataset is missing: "
            f"{missing}"
        )

    return (
        regional[
            required
        ]
        .groupby(
            [
                "region_id",
                "month",
            ],
            as_index=False,
            observed=True,
        )[columns]
        .mean()
        .sort_values(
            [
                "region_id",
                "month",
            ]
        )
        .reset_index(drop=True)
    )


def make_figure(
    data: pd.DataFrame,
    available_targets: list[str],
) -> None:
    national = (
        data.groupby(
            "month",
            as_index=False,
            observed=True,
        )[
            [
                "precipitation_reconstructed_mm",
                *available_targets,
                "SM_total_mm",
            ]
        ]
        .mean()
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.0, 4.8),
        constrained_layout=True,
    )

    axes[0].plot(
        national["month"],
        national[
            "precipitation_reconstructed_mm"
        ],
        marker="o",
        linewidth=1.4,
        label="Precipitation",
    )

    for target in available_targets:
        axes[0].plot(
            national["month"],
            national[target],
            marker="o",
            linewidth=1.2,
            label=target,
        )

    axes[0].set_xticks(
        range(1, 13)
    )

    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Monthly flux (mm)")
    axes[0].set_title(
        "National climatological precipitation and runoff"
    )

    axes[0].legend(
        frameon=True,
        fontsize=8,
    )

    axes[1].plot(
        national["month"],
        national["SM_total_mm"],
        marker="o",
        linewidth=1.4,
        label="Total soil moisture",
    )

    axes[1].set_xticks(
        range(1, 13)
    )

    axes[1].set_xlabel("Month")
    axes[1].set_ylabel(
        "Total soil moisture (mm)"
    )

    axes[1].set_title(
        "National climatological soil storage"
    )

    axes[1].legend(
        frameon=True,
        fontsize=8,
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Runoff-target audit and seasonal climatology",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig28_runoff_target_audit.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig28_runoff_target_audit.pdf",
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

    if not PINN_PATH.exists():
        raise FileNotFoundError(
            PINN_PATH
        )

    if not REGIONAL_PATH.exists():
        raise FileNotFoundError(
            REGIONAL_PATH
        )

    pinn = (
        pd.read_parquet(
            PINN_PATH
        )
        .sort_values(
            [
                "region_id",
                "month",
            ]
        )
        .reset_index(drop=True)
    )

    regional = pd.read_parquet(
        REGIONAL_PATH
    )

    if len(pinn) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} PINN rows, "
            f"found {len(pinn)}."
        )

    if (
        pinn["region_id"].nunique()
        != EXPECTED_REGIONS
    ):
        raise RuntimeError(
            "Unexpected regional-unit count."
        )

    total_column = (
        first_existing_column(
            regional.columns.tolist(),
            TOTAL_RUNOFF_CANDIDATES,
        )
    )

    surface_column = (
        first_existing_column(
            regional.columns.tolist(),
            SURFACE_RUNOFF_CANDIDATES,
        )
    )

    subsurface_column = (
        first_existing_column(
            regional.columns.tolist(),
            SUBSURFACE_RUNOFF_CANDIDATES,
        )
    )

    source_columns = [
        column
        for column in [
            total_column,
            surface_column,
            subsurface_column,
        ]
        if column is not None
    ]

    if not source_columns:
        raise RuntimeError(
            "No runoff columns were found in "
            "the regional dataset."
        )

    climatology = (
        build_monthly_climatology(
            regional,
            source_columns,
        )
    )

    rename_map = {}

    if total_column is not None:
        rename_map[
            total_column
        ] = "source_total_runoff_mm"

    if surface_column is not None:
        rename_map[
            surface_column
        ] = "surface_runoff_mm"

    if subsurface_column is not None:
        rename_map[
            subsurface_column
        ] = "subsurface_runoff_mm"

    climatology = climatology.rename(
        columns=rename_map
    )

    merged = pinn.merge(
        climatology,
        on=[
            "region_id",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    if merged[
        list(rename_map.values())
    ].isna().any().any():
        raise RuntimeError(
            "Missing values appeared after "
            "merging runoff climatology."
        )

    component_l0_columns = [
        f"K4_C{component}_L0_mm"
        for component in range(1, 5)
    ]

    for lag in range(4):
        lag_columns = [
            f"K4_C{component}_L{lag}_mm"
            for component in range(1, 5)
        ]

        merged[
            f"precipitation_L{lag}_mm"
        ] = merged[
            lag_columns
        ].sum(axis=1)

    merged[
        "precipitation_reconstructed_mm"
    ] = merged[
        component_l0_columns
    ].sum(axis=1)

    merged = add_circular_region_lag(
        merged,
        source_column="SM_total_mm",
        output_column="SM_total_L1_mm",
        lag=1,
    )

    merged = add_circular_region_lag(
        merged,
        source_column="runoff_mm",
        output_column="runoff_L1_mm",
        lag=1,
    )

    lineage_results = {
        "status": "PASS",
        "pinn_dataset": str(
            PINN_PATH
        ),
        "regional_dataset": str(
            REGIONAL_PATH
        ),
        "pinn_runoff_target": (
            "runoff_mm"
        ),
        "regional_total_runoff_column": (
            total_column
        ),
        "regional_surface_runoff_column": (
            surface_column
        ),
        "regional_subsurface_runoff_column": (
            subsurface_column
        ),
        "surface_runoff_recovered": bool(
            surface_column is not None
        ),
        "subsurface_runoff_recovered": bool(
            subsurface_column is not None
        ),
    }

    if total_column is not None:
        total_difference = (
            merged["runoff_mm"]
            - merged[
                "source_total_runoff_mm"
            ]
        )

        lineage_results.update(
            {
                "maximum_absolute_pinn_vs_source_total_difference_mm": float(
                    total_difference.abs().max()
                ),
                "mean_absolute_pinn_vs_source_total_difference_mm": float(
                    total_difference.abs().mean()
                ),
                "pinn_runoff_matches_source_total_within_1e_4_mm": bool(
                    total_difference.abs().max()
                    <= 1e-4
                ),
            }
        )

    if (
        surface_column is not None
        and subsurface_column is not None
    ):
        reconstructed_total = (
            merged["surface_runoff_mm"]
            + merged[
                "subsurface_runoff_mm"
            ]
        )

        if (
            "source_total_runoff_mm"
            in merged.columns
        ):
            identity_reference = merged[
                "source_total_runoff_mm"
            ]

        else:
            identity_reference = merged[
                "runoff_mm"
            ]

        identity_error = (
            reconstructed_total
            - identity_reference
        )

        lineage_results.update(
            {
                "maximum_surface_plus_subsurface_identity_error_mm": float(
                    identity_error.abs().max()
                ),
                "mean_surface_plus_subsurface_identity_error_mm": float(
                    identity_error.abs().mean()
                ),
                "runoff_component_identity_pass_1e_4_mm": bool(
                    identity_error.abs().max()
                    <= 1e-4
                ),
            }
        )

    merged[
        "water_balance_residual_mm"
    ] = (
        merged[
            "precipitation_reconstructed_mm"
        ]
        - merged["AET_mm"]
        - merged["runoff_mm"]
        - merged["delta_SM_total_mm"]
    )

    target_columns = [
        "runoff_mm",
    ]

    if "surface_runoff_mm" in merged.columns:
        target_columns.append(
            "surface_runoff_mm"
        )

    if (
        "subsurface_runoff_mm"
        in merged.columns
    ):
        target_columns.append(
            "subsurface_runoff_mm"
        )

    distribution_rows = []

    for target in target_columns:
        distribution_rows.append(
            {
                "target": target,
                **summary_statistics(
                    merged[target]
                ),
            }
        )

    distribution = pd.DataFrame(
        distribution_rows
    )

    monthly_columns = [
        "precipitation_reconstructed_mm",
        "PET_mm",
        "AET_mm",
        "SM_total_mm",
        "delta_SM_total_mm",
        *target_columns,
    ]

    monthly_climatology = (
        merged.groupby(
            "month",
            as_index=False,
            observed=True,
        )[monthly_columns]
        .mean()
    )

    correlation_predictors = [
        "precipitation_L0_mm",
        "precipitation_L1_mm",
        "precipitation_L2_mm",
        "precipitation_L3_mm",
        "PET_mm",
        "SM_total_mm",
        "SM_total_L1_mm",
        "delta_SM_total_mm",
    ]

    correlation_rows = []

    for target in target_columns:
        for predictor in (
            correlation_predictors
        ):
            correlation_rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "pearson_correlation": float(
                        merged[
                            [
                                target,
                                predictor,
                            ]
                        ]
                        .corr()
                        .iloc[0, 1]
                    ),
                    "spearman_correlation": float(
                        merged[
                            [
                                target,
                                predictor,
                            ]
                        ]
                        .corr(
                            method="spearman"
                        )
                        .iloc[0, 1]
                    ),
                }
            )

    correlations = pd.DataFrame(
        correlation_rows
    )

    residual_summary = pd.DataFrame(
        [
            {
                "diagnostic": (
                    "P - AET - runoff - delta_SM_total"
                ),
                **summary_statistics(
                    merged[
                        "water_balance_residual_mm"
                    ]
                ),
                "root_mean_square_residual_mm": float(
                    np.sqrt(
                        np.mean(
                            merged[
                                "water_balance_residual_mm"
                            ].to_numpy(
                                dtype=float
                            )
                            ** 2
                        )
                    )
                ),
                "mean_absolute_residual_mm": float(
                    merged[
                        "water_balance_residual_mm"
                    ]
                    .abs()
                    .mean()
                ),
            }
        ]
    )

    lineage_path = (
        TABLE_DIR
        / "runoff_target_lineage_audit.json"
    )

    distribution_path = (
        TABLE_DIR
        / "runoff_target_distribution.csv"
    )

    monthly_path = (
        TABLE_DIR
        / "runoff_monthly_climatology.csv"
    )

    correlation_path = (
        TABLE_DIR
        / "runoff_predictor_correlations.csv"
    )

    residual_path = (
        TABLE_DIR
        / "runoff_water_balance_residual_summary.csv"
    )

    lineage_path.write_text(
        json.dumps(
            lineage_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    distribution.to_csv(
        distribution_path,
        index=False,
    )

    monthly_climatology.to_csv(
        monthly_path,
        index=False,
    )

    correlations.to_csv(
        correlation_path,
        index=False,
    )

    residual_summary.to_csv(
        residual_path,
        index=False,
    )

    merged.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    merged.to_csv(
        OUTPUT_CSV,
        index=False,
        compression="gzip",
    )

    make_figure(
        merged,
        target_columns,
    )

    print("=" * 78)
    print("RUNOFF TARGET LINEAGE AND EXTENSION AUDIT")
    print("=" * 78)

    print(
        f"Regional total runoff column     : "
        f"{total_column}"
    )

    print(
        f"Regional surface runoff column   : "
        f"{surface_column}"
    )

    print(
        f"Regional subsurface runoff column: "
        f"{subsurface_column}"
    )

    if total_column is not None:
        print(
            "Maximum PINN–source total difference: "
            f"{lineage_results['maximum_absolute_pinn_vs_source_total_difference_mm']:.8f} mm"
        )

        print(
            "PINN runoff matches total runoff     : "
            f"{lineage_results['pinn_runoff_matches_source_total_within_1e_4_mm']}"
        )

    if (
        surface_column is not None
        and subsurface_column is not None
    ):
        print(
            "Maximum component identity error     : "
            f"{lineage_results['maximum_surface_plus_subsurface_identity_error_mm']:.8f} mm"
        )

        print(
            "Surface + subsurface identity PASS   : "
            f"{lineage_results['runoff_component_identity_pass_1e_4_mm']}"
        )

    print()
    print("Runoff distributions:")

    print(
        distribution.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Approximate water-balance residual:")

    print(
        residual_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Strongest absolute correlations:")

    strongest = (
        correlations.assign(
            absolute_correlation=lambda frame: (
                frame[
                    "pearson_correlation"
                ].abs()
            )
        )
        .sort_values(
            [
                "target",
                "absolute_correlation",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "target",
            as_index=False,
        )
        .head(5)
    )

    print(
        strongest[
            [
                "target",
                "predictor",
                "pearson_correlation",
                "spearman_correlation",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(f"Extended dataset : {OUTPUT_PARQUET}")
    print(f"Lineage audit    : {lineage_path}")
    print(f"Distribution     : {distribution_path}")
    print(f"Correlations     : {correlation_path}")
    print(f"Residual audit   : {residual_path}")
    print(
        f"Figure           : "
        f"{FIGURE_DIR / 'fig28_runoff_target_audit.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
