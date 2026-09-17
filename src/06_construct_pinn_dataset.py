#!/usr/bin/env python3
"""
Construct the modelling dataset for soil-moisture prediction.

Predictors
----------
- Four selected NMF precipitation components.
- Current contribution and three antecedent months: L0-L3.
- PET.
- Latitude and longitude.

Targets
-------
- Total soil moisture.
- Four soil-moisture layers.
- Circular month-to-month change in total soil moisture.

Evaluation assignments
----------------------
1. Replication split:
   For every region, randomly assign 8 months to training,
   2 months to validation and 2 months to testing.

2. Spatial folds:
   Assign complete regions to five geographically coherent
   folds for later spatial cross-validation.

No feature scaling is applied here. Scaling must be fitted using
training data only during the model-training stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]

REGIONAL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "malaysia_hydrology_regional_monthly_0p5degree.parquet"
)

CONTRIBUTION_PATH = (
    ROOT
    / "results"
    / "nmf_selected_k4"
    / "regional_monthly_component_contributions.csv"
)

RECONSTRUCTION_PATH = (
    ROOT
    / "results"
    / "nmf_selected_k4"
    / "precipitation_reconstruction.csv"
)

PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

OUTPUT_PARQUET = (
    PROCESSED_DIR
    / "pinn_climatology_features_k4_lag3.parquet"
)

OUTPUT_CSV = (
    PROCESSED_DIR
    / "pinn_climatology_features_k4_lag3.csv"
)

EXPECTED_REGIONS = 172
EXPECTED_MONTHS = 12
EXPECTED_ROWS = EXPECTED_REGIONS * EXPECTED_MONTHS

MAXIMUM_LAG = 3
COMPONENTS = [
    "K4_C1",
    "K4_C2",
    "K4_C3",
    "K4_C4",
]

REPLICATION_SPLIT_SEED = 20260806
SPATIAL_FOLD_SEED = 20260806
SPATIAL_FOLD_COUNT = 5

SOIL_LAYER_SOURCE_COLUMNS = [
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
]

SOIL_TARGET_COLUMNS = [
    "SM_0_10cm_mm",
    "SM_10_40cm_mm",
    "SM_40_100cm_mm",
    "SM_100_200cm_mm",
    "SM_total_mm",
]


def load_regional_climatology() -> pd.DataFrame:
    if not REGIONAL_PATH.exists():
        raise FileNotFoundError(REGIONAL_PATH)

    frame = pd.read_parquet(REGIONAL_PATH)
    frame["date"] = pd.to_datetime(frame["date"])

    aggregation = {
        "native_cell_count": (
            "native_cell_count",
            "first",
        ),
        "longitude": (
            "longitude_mean",
            "first",
        ),
        "latitude": (
            "latitude_mean",
            "first",
        ),
        "region_lon_index": (
            "region_lon_min",
            lambda values: int(
                round(
                    (
                        float(values.iloc[0])
                        - 99.5
                    )
                    / 0.5
                )
            ),
        ),
        "region_lat_index": (
            "region_lat_min",
            lambda values: int(
                round(
                    (
                        float(values.iloc[0])
                        - 0.5
                    )
                    / 0.5
                )
            ),
        ),
        "small_region_lt_4_cells": (
            "small_region_lt_4_cells",
            "first",
        ),
        "precipitation_observed_mm": (
            "precipitation_mm",
            "mean",
        ),
        "precipitation_interannual_std_mm": (
            "precipitation_mm",
            "std",
        ),
        "PET_mm": (
            "PET_mm",
            "mean",
        ),
        "PET_interannual_std_mm": (
            "PET_mm",
            "std",
        ),
        "AET_mm": (
            "actual_evapotranspiration_mm",
            "mean",
        ),
        "runoff_mm": (
            "total_runoff_mm",
            "mean",
        ),
        "SM_0_10cm_mm": (
            "soil_moisture_0_10cm_mm",
            "mean",
        ),
        "SM_10_40cm_mm": (
            "soil_moisture_10_40cm_mm",
            "mean",
        ),
        "SM_40_100cm_mm": (
            "soil_moisture_40_100cm_mm",
            "mean",
        ),
        "SM_100_200cm_mm": (
            "soil_moisture_100_200cm_mm",
            "mean",
        ),
        "SM_total_mm": (
            "soil_moisture_total_mm",
            "mean",
        ),
        "SM_total_interannual_std_mm": (
            "soil_moisture_total_mm",
            "std",
        ),
    }

    climatology = (
        frame.groupby(
            ["region_id", "month"],
            as_index=False,
            observed=True,
        )
        .agg(**aggregation)
        .sort_values(["region_id", "month"])
        .reset_index(drop=True)
    )

    if len(climatology) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} climatology rows, "
            f"found {len(climatology)}."
        )

    counts = climatology.groupby(
        "region_id",
        observed=True,
    )["month"].nunique()

    if not (counts == EXPECTED_MONTHS).all():
        raise RuntimeError(
            "One or more regions do not contain 12 months."
        )

    return climatology


def load_component_contributions() -> pd.DataFrame:
    if not CONTRIBUTION_PATH.exists():
        raise FileNotFoundError(
            CONTRIBUTION_PATH
        )

    contribution = pd.read_csv(
        CONTRIBUTION_PATH
    )

    found_components = sorted(
        contribution["component"]
        .unique()
        .tolist()
    )

    if found_components != COMPONENTS:
        raise RuntimeError(
            f"Expected components {COMPONENTS}, "
            f"found {found_components}."
        )

    duplicated = int(
        contribution.duplicated(
            subset=[
                "region_id",
                "month",
                "component",
            ],
            keep=False,
        ).sum()
    )

    if duplicated:
        raise RuntimeError(
            f"Component table contains {duplicated} "
            "duplicated region/month/component rows."
        )

    wide = contribution.pivot(
        index=["region_id", "month"],
        columns="component",
        values="component_contribution_mm",
    ).reset_index()

    wide.columns.name = None

    for component in COMPONENTS:
        wide = wide.rename(
            columns={
                component: f"{component}_L0_mm"
            }
        )

    if len(wide) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} contribution rows, "
            f"found {len(wide)}."
        )

    return (
        wide.sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
    )


def attach_lagged_contributions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = (
        frame.sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
        .copy()
    )

    for component in COMPONENTS:
        current_column = (
            f"{component}_L0_mm"
        )

        for lag in range(1, MAXIMUM_LAG + 1):
            lagged_column = (
                f"{component}_L{lag}_mm"
            )

            output[lagged_column] = (
                output.groupby(
                    "region_id",
                    observed=True,
                )[current_column]
                .transform(
                    lambda values, lag=lag: np.roll(
                        values.to_numpy(
                            dtype=float
                        ),
                        lag,
                    )
                )
            )

    return output


def attach_reconstruction(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    if not RECONSTRUCTION_PATH.exists():
        raise FileNotFoundError(
            RECONSTRUCTION_PATH
        )

    reconstruction = pd.read_csv(
        RECONSTRUCTION_PATH
    )

    reconstruction = reconstruction.rename(
        columns={
            "observed_precipitation_mm": (
                "nmf_input_precipitation_mm"
            ),
            "reconstructed_precipitation_mm": (
                "nmf_reconstructed_precipitation_mm"
            ),
            "residual_mm": (
                "nmf_precipitation_residual_mm"
            ),
        }
    )

    selected = reconstruction[
        [
            "region_id",
            "month",
            "nmf_input_precipitation_mm",
            "nmf_reconstructed_precipitation_mm",
            "nmf_precipitation_residual_mm",
        ]
    ].copy()

    output = frame.merge(
        selected,
        on=["region_id", "month"],
        how="left",
        validate="one_to_one",
    )

    input_difference = (
        output["precipitation_observed_mm"]
        - output[
            "nmf_input_precipitation_mm"
        ]
    ).abs()

    component_l0_columns = [
        f"{component}_L0_mm"
        for component in COMPONENTS
    ]

    output[
        "nmf_component_sum_L0_mm"
    ] = output[
        component_l0_columns
    ].sum(axis=1)

    reconstruction_difference = (
        output["nmf_component_sum_L0_mm"]
        - output[
            "nmf_reconstructed_precipitation_mm"
        ]
    ).abs()

    return (
        output,
        float(input_difference.max()),
        float(reconstruction_difference.max()),
    )


def attach_soil_moisture_changes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = (
        frame.sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
        .copy()
    )

    for target in SOIL_TARGET_COLUMNS:
        previous = (
            output.groupby(
                "region_id",
                observed=True,
            )[target]
            .transform(
                lambda values: np.roll(
                    values.to_numpy(
                        dtype=float
                    ),
                    1,
                )
            )
        )

        output[
            f"delta_{target}"
        ] = output[target] - previous

    return output


def attach_replication_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()

    rng = np.random.default_rng(
        REPLICATION_SPLIT_SEED
    )

    assignments = []

    for region_id in sorted(
        output["region_id"].unique()
    ):
        months = np.arange(1, 13)
        permutation = rng.permutation(months)

        train_months = set(
            permutation[:8].tolist()
        )

        validation_months = set(
            permutation[8:10].tolist()
        )

        test_months = set(
            permutation[10:12].tolist()
        )

        for month in months:
            if month in train_months:
                split = "train"
            elif month in validation_months:
                split = "validation"
            elif month in test_months:
                split = "test"
            else:
                raise RuntimeError(
                    "Month was not assigned to a split."
                )

            assignments.append(
                {
                    "region_id": region_id,
                    "month": int(month),
                    "replication_split": split,
                }
            )

    assignment_table = pd.DataFrame(
        assignments
    )

    output = output.merge(
        assignment_table,
        on=["region_id", "month"],
        how="left",
        validate="one_to_one",
    )

    region_split_counts = (
        output.groupby(
            [
                "region_id",
                "replication_split",
            ],
            observed=True,
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    required_counts = {
        "train": 8,
        "validation": 2,
        "test": 2,
    }

    for split, expected_count in (
        required_counts.items()
    ):
        if split not in region_split_counts:
            raise RuntimeError(
                f"Missing split column: {split}"
            )

        if not (
            region_split_counts[split]
            == expected_count
        ).all():
            raise RuntimeError(
                f"Not every region has "
                f"{expected_count} {split} months."
            )

    return output, assignment_table


def attach_spatial_folds(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()

    region_table = (
        output[
            [
                "region_id",
                "longitude",
                "latitude",
                "native_cell_count",
                "small_region_lt_4_cells",
            ]
        ]
        .drop_duplicates()
        .sort_values("region_id")
        .reset_index(drop=True)
    )

    coordinates = region_table[
        ["longitude", "latitude"]
    ].to_numpy(dtype=float)

    model = KMeans(
        n_clusters=SPATIAL_FOLD_COUNT,
        n_init=50,
        random_state=SPATIAL_FOLD_SEED,
    )

    raw_labels = model.fit_predict(
        coordinates
    )

    region_table[
        "_raw_spatial_cluster"
    ] = raw_labels

    centroid_table = (
        region_table.groupby(
            "_raw_spatial_cluster",
            as_index=False,
            observed=True,
        )
        .agg(
            centroid_longitude=(
                "longitude",
                "mean",
            ),
            centroid_latitude=(
                "latitude",
                "mean",
            ),
        )
        .sort_values(
            [
                "centroid_latitude",
                "centroid_longitude",
            ]
        )
        .reset_index(drop=True)
    )

    label_mapping = {
        int(row._raw_spatial_cluster):
        int(index)
        for index, row
        in centroid_table.iterrows()
    }

    region_table["spatial_fold_5"] = (
        region_table[
            "_raw_spatial_cluster"
        ]
        .map(label_mapping)
        .astype(int)
    )

    region_table = region_table.drop(
        columns=["_raw_spatial_cluster"]
    )

    output = output.merge(
        region_table[
            [
                "region_id",
                "spatial_fold_5",
            ]
        ],
        on="region_id",
        how="left",
        validate="many_to_one",
    )

    return output, region_table


def calculate_component_dominance(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    current_columns = [
        f"{component}_L0_mm"
        for component in COMPONENTS
    ]

    working = frame[
        [
            "region_id",
            "month",
            *current_columns,
        ]
    ].copy()

    working[
        "dominant_component"
    ] = (
        working[current_columns]
        .idxmax(axis=1)
        .str.replace(
            "_L0_mm",
            "",
            regex=False,
        )
    )

    counts = (
        working.groupby(
            [
                "month",
                "dominant_component",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            regional_count=(
                "region_id",
                "nunique",
            )
        )
    )

    complete_index = pd.MultiIndex.from_product(
        [
            range(1, 13),
            COMPONENTS,
        ],
        names=[
            "month",
            "dominant_component",
        ],
    )

    counts = (
        counts.set_index(
            [
                "month",
                "dominant_component",
            ]
        )
        .reindex(
            complete_index,
            fill_value=0,
        )
        .reset_index()
    )

    return counts


def calculate_feature_correlations(
    frame: pd.DataFrame,
    precipitation_features: list[str],
) -> pd.DataFrame:
    predictor_columns = [
        *precipitation_features,
        "PET_mm",
        "latitude",
        "longitude",
    ]

    rows = []

    for predictor in predictor_columns:
        for target in SOIL_TARGET_COLUMNS:
            correlation = float(
                frame[
                    [predictor, target]
                ]
                .corr()
                .iloc[0, 1]
            )

            rows.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "pearson_correlation": (
                        correlation
                    ),
                }
            )

    return pd.DataFrame(rows)


def plot_total_sm_correlation_matrix(
    correlation_table: pd.DataFrame,
) -> None:
    selected = correlation_table.loc[
        correlation_table["target"]
        == "SM_total_mm"
    ].copy()

    matrix = np.full(
        (
            len(COMPONENTS),
            MAXIMUM_LAG + 1,
        ),
        np.nan,
        dtype=float,
    )

    for component_index, component in enumerate(
        COMPONENTS
    ):
        for lag in range(
            MAXIMUM_LAG + 1
        ):
            predictor = (
                f"{component}_L{lag}_mm"
            )

            value = selected.loc[
                selected["predictor"]
                == predictor,
                "pearson_correlation",
            ]

            if len(value) != 1:
                raise RuntimeError(
                    f"Missing correlation for "
                    f"{predictor}."
                )

            matrix[
                component_index,
                lag,
            ] = float(value.iloc[0])

    figure, axis = plt.subplots(
        figsize=(7.3, 5.3),
        constrained_layout=True,
    )

    image = axis.imshow(
        matrix,
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
    )

    axis.set_xticks(
        range(MAXIMUM_LAG + 1)
    )

    axis.set_xticklabels(
        [
            f"L{lag}"
            for lag in range(
                MAXIMUM_LAG + 1
            )
        ]
    )

    axis.set_yticks(
        range(len(COMPONENTS))
    )

    axis.set_yticklabels(
        COMPONENTS
    )

    axis.set_xlabel(
        "Precipitation contribution lag"
    )

    axis.set_ylabel(
        "Selected NMF component"
    )

    axis.set_title(
        "Univariate correlations between "
        "component-lag predictors and total soil moisture"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Pearson correlation"
    )

    figure.savefig(
        FIGURE_DIR
        / "fig17_component_lag_sm_correlations.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig17_component_lag_sm_correlations.pdf",
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

    print("=" * 78)
    print(
        "PINN FEATURE DATASET — "
        "K=4 PRECIPITATION MODES, L0-L3"
    )
    print("=" * 78)

    climatology = (
        load_regional_climatology()
    )

    component_table = (
        load_component_contributions()
    )

    dataset = climatology.merge(
        component_table,
        on=["region_id", "month"],
        how="left",
        validate="one_to_one",
    )

    dataset = attach_lagged_contributions(
        dataset
    )

    (
        dataset,
        maximum_nmf_input_difference,
        maximum_component_sum_difference,
    ) = attach_reconstruction(dataset)

    dataset = attach_soil_moisture_changes(
        dataset
    )

    (
        dataset,
        replication_assignments,
    ) = attach_replication_split(
        dataset
    )

    (
        dataset,
        spatial_folds,
    ) = attach_spatial_folds(
        dataset
    )

    dataset["core_region_ge_4_cells"] = (
        dataset["native_cell_count"] >= 4
    )

    precipitation_features = [
        f"{component}_L{lag}_mm"
        for component in COMPONENTS
        for lag in range(
            MAXIMUM_LAG + 1
        )
    ]

    input_features = [
        *precipitation_features,
        "PET_mm",
        "latitude",
        "longitude",
    ]

    required_columns = [
        "region_id",
        "month",
        *input_features,
        *SOIL_TARGET_COLUMNS,
        "delta_SM_total_mm",
        "replication_split",
        "spatial_fold_5",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing modelling columns: "
            f"{missing_columns}"
        )

    missing_values = int(
        dataset[
            input_features
            + SOIL_TARGET_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:
        raise RuntimeError(
            f"Modelling dataset contains "
            f"{missing_values} missing values."
        )

    layer_sum = dataset[
        [
            "SM_0_10cm_mm",
            "SM_10_40cm_mm",
            "SM_40_100cm_mm",
            "SM_100_200cm_mm",
        ]
    ].sum(axis=1)

    maximum_layer_sum_error = float(
        (
            dataset["SM_total_mm"]
            - layer_sum
        )
        .abs()
        .max()
    )

    if maximum_layer_sum_error > 1e-3:
        raise RuntimeError(
            "Layer-wise soil moisture does not "
            "reproduce total soil moisture."
        )

    component_dominance = (
        calculate_component_dominance(
            dataset
        )
    )

    correlation_table = (
        calculate_feature_correlations(
            dataset,
            precipitation_features,
        )
    )

    split_month_counts = (
        dataset.groupby(
            [
                "replication_split",
                "month",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            sample_count=(
                "region_id",
                "size",
            )
        )
    )

    split_region_counts = (
        dataset.groupby(
            [
                "region_id",
                "replication_split",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            month_count=(
                "month",
                "size",
            )
        )
    )

    dataset = dataset.sort_values(
        ["region_id", "month"]
    ).reset_index(drop=True)

    ordered_columns = [
        "region_id",
        "month",
        "native_cell_count",
        "core_region_ge_4_cells",
        "small_region_lt_4_cells",
        "longitude",
        "latitude",
        "region_lon_index",
        "region_lat_index",
        "replication_split",
        "spatial_fold_5",
        *precipitation_features,
        "PET_mm",
        "precipitation_observed_mm",
        "nmf_input_precipitation_mm",
        "nmf_reconstructed_precipitation_mm",
        "nmf_precipitation_residual_mm",
        "nmf_component_sum_L0_mm",
        "AET_mm",
        "runoff_mm",
        *SOIL_TARGET_COLUMNS,
        "delta_SM_0_10cm_mm",
        "delta_SM_10_40cm_mm",
        "delta_SM_40_100cm_mm",
        "delta_SM_100_200cm_mm",
        "delta_SM_total_mm",
        "precipitation_interannual_std_mm",
        "PET_interannual_std_mm",
        "SM_total_interannual_std_mm",
    ]

    dataset = dataset[
        ordered_columns
    ]

    dataset.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        compression="snappy",
    )

    dataset.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    replication_assignment_path = (
        TABLE_DIR
        / "replication_month_split_assignments.csv"
    )

    spatial_fold_path = (
        TABLE_DIR
        / "spatial_cv_region_folds.csv"
    )

    dominance_path = (
        TABLE_DIR
        / "selected_k4_monthly_dominance_counts.csv"
    )

    correlation_path = (
        TABLE_DIR
        / "pinn_predictor_target_correlations.csv"
    )

    split_month_path = (
        TABLE_DIR
        / "replication_split_counts_by_month.csv"
    )

    split_region_path = (
        TABLE_DIR
        / "replication_split_counts_by_region.csv"
    )

    replication_assignments.to_csv(
        replication_assignment_path,
        index=False,
    )

    spatial_folds.to_csv(
        spatial_fold_path,
        index=False,
    )

    component_dominance.to_csv(
        dominance_path,
        index=False,
    )

    correlation_table.to_csv(
        correlation_path,
        index=False,
    )

    split_month_counts.to_csv(
        split_month_path,
        index=False,
    )

    split_region_counts.to_csv(
        split_region_path,
        index=False,
    )

    plot_total_sm_correlation_matrix(
        correlation_table
    )

    split_counts = (
        dataset[
            "replication_split"
        ]
        .value_counts()
        .to_dict()
    )

    fold_counts = (
        spatial_folds[
            "spatial_fold_5"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    monthly_dominance_totals = (
        component_dominance.groupby(
            "dominant_component",
            observed=True,
        )["regional_count"]
        .sum()
        .to_dict()
    )

    feature_manifest = {
        "dataset": str(OUTPUT_PARQUET),
        "row_count": int(len(dataset)),
        "region_count": int(
            dataset["region_id"].nunique()
        ),
        "month_count_per_region": 12,
        "maximum_precipitation_lag": (
            MAXIMUM_LAG
        ),
        "precipitation_component_count": (
            len(COMPONENTS)
        ),
        "precipitation_lag_feature_count": (
            len(precipitation_features)
        ),
        "input_feature_count": (
            len(input_features)
        ),
        "input_features": input_features,
        "monotonic_increasing_features": (
            precipitation_features
        ),
        "candidate_monotonic_decreasing_features": [
            "PET_mm"
        ],
        "static_features": [
            "latitude",
            "longitude",
        ],
        "primary_target": "SM_total_mm",
        "layer_targets": [
            "SM_0_10cm_mm",
            "SM_10_40cm_mm",
            "SM_40_100cm_mm",
            "SM_100_200cm_mm",
        ],
        "auxiliary_target": (
            "delta_SM_total_mm"
        ),
        "region_one_hot_encoding_used": False,
        "feature_scaling_applied": False,
        "scaling_policy": (
            "Fit all scalers using training observations "
            "only during model training."
        ),
        "replication_split": (
            "8 train, 2 validation and 2 test "
            "months per region."
        ),
        "spatial_evaluation": (
            "Five complete-region geographic folds."
        ),
    }

    manifest_path = (
        TABLE_DIR
        / "pinn_feature_manifest_k4_lag3.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            feature_manifest,
            stream,
            indent=2,
        )

    summary = {
        "status": "PASS",
        "output_parquet": str(
            OUTPUT_PARQUET
        ),
        "output_csv": str(
            OUTPUT_CSV
        ),
        "rows": int(len(dataset)),
        "regions": int(
            dataset["region_id"].nunique()
        ),
        "months_per_region": 12,
        "input_feature_count": (
            len(input_features)
        ),
        "precipitation_lag_feature_count": (
            len(precipitation_features)
        ),
        "soil_moisture_target_count": (
            len(SOIL_TARGET_COLUMNS)
        ),
        "missing_modelling_values": (
            missing_values
        ),
        "maximum_climatology_nmf_input_difference_mm": (
            maximum_nmf_input_difference
        ),
        "maximum_component_sum_reconstruction_difference_mm": (
            maximum_component_sum_difference
        ),
        "maximum_soil_layer_sum_error_mm": (
            maximum_layer_sum_error
        ),
        "replication_split_counts": {
            str(key): int(value)
            for key, value
            in split_counts.items()
        },
        "spatial_fold_region_counts": {
            str(key): int(value)
            for key, value
            in fold_counts.items()
        },
        "core_regions_ge_4_cells": int(
            spatial_folds[
                "native_cell_count"
            ].ge(4).sum()
        ),
        "small_regions_lt_4_cells": int(
            spatial_folds[
                "small_region_lt_4_cells"
            ].sum()
        ),
        "monthly_dominance_totals": {
            str(key): int(value)
            for key, value
            in monthly_dominance_totals.items()
        },
        "feature_manifest": str(
            manifest_path
        ),
        "replication_split_assignments": str(
            replication_assignment_path
        ),
        "spatial_fold_assignments": str(
            spatial_fold_path
        ),
        "monthly_component_dominance": str(
            dominance_path
        ),
        "predictor_target_correlations": str(
            correlation_path
        ),
    }

    summary_path = (
        TABLE_DIR
        / "pinn_dataset_audit_summary.json"
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

    print(f"Status                          : {summary['status']}")
    print(f"Dataset rows                    : {summary['rows']}")
    print(f"Regional units                  : {summary['regions']}")
    print(f"Months per region               : {summary['months_per_region']}")
    print(f"Input features                  : {summary['input_feature_count']}")
    print(
        f"Component-lag features          : "
        f"{summary['precipitation_lag_feature_count']}"
    )
    print(
        f"Soil-moisture targets           : "
        f"{summary['soil_moisture_target_count']}"
    )
    print(
        f"Missing modelling values        : "
        f"{summary['missing_modelling_values']}"
    )
    print(
        f"Maximum NMF input difference    : "
        f"{maximum_nmf_input_difference:.10f} mm"
    )
    print(
        f"Maximum component-sum error     : "
        f"{maximum_component_sum_difference:.10f} mm"
    )
    print(
        f"Maximum layer-sum error         : "
        f"{maximum_layer_sum_error:.10f} mm"
    )
    print(
        f"Replication split counts        : "
        f"{summary['replication_split_counts']}"
    )
    print(
        f"Spatial-fold region counts      : "
        f"{summary['spatial_fold_region_counts']}"
    )
    print(
        f"Monthly dominance totals        : "
        f"{summary['monthly_dominance_totals']}"
    )
    print()
    print(f"PINN-ready Parquet              : {OUTPUT_PARQUET}")
    print(f"Feature manifest                : {manifest_path}")
    print(f"Audit summary                   : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
