#!/usr/bin/env python3
"""
Five-fold geographic evaluation of total soil-moisture models.

Outer evaluation
----------------
For outer test fold f:
    test fold       = f
    validation fold = (f + 1) mod 5
    training folds  = the remaining three folds

Complete regions are held out. Therefore, no region appearing in the
outer test fold is used during model fitting, feature scaling, model
selection, or early stopping.

Fixed neural candidates
-----------------------
1. MLP
2. PINN-P, lambda_P = 10
3. PINN-P-PET, lambda_P = lambda_PET = 1
4. Strict PINN-P-PET, lambda_P = lambda_PET = 10

Additional baselines
--------------------
- Mean predictor
- Ridge regression with alpha selected from the validation fold

All reported outer-test predictions are out-of-region predictions
conditional on the previously fitted precipitation NMF representation.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "pinn_climatology_features_k4_lag3.parquet"
)

MANIFEST_PATH = (
    ROOT
    / "results"
    / "tables"
    / "pinn_feature_manifest_k4_lag3.json"
)

MODEL_DIR = (
    ROOT
    / "results"
    / "models"
    / "stage08_spatial_cv"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

TARGET_COLUMN = "SM_total_mm"

PRECIPITATION_FEATURES = [
    f"K4_C{component}_L{lag}_mm"
    for component in range(1, 5)
    for lag in range(4)
]

FEATURE_COLUMNS = [
    *PRECIPITATION_FEATURES,
    "PET_mm",
    "latitude",
    "longitude",
]

PET_FEATURE = "PET_mm"

SPATIAL_FOLD_COUNT = 5
EXPECTED_ROWS = 396
EXPECTED_REGIONS = 33

RIDGE_ALPHA_GRID = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
]

NEURAL_SEEDS = [
    20260806,
    20260807,
    20260808,
    20260809,
    20260810,
]

NEURAL_CONFIGURATIONS = [
    {
        "configuration": "MLP",
        "model_family": "MLP",
        "lambda_precipitation": 0.0,
        "lambda_pet": 0.0,
    },
    {
        "configuration": "PINN_P_lambda10",
        "model_family": "PINN_P",
        "lambda_precipitation": 10.0,
        "lambda_pet": 0.0,
    },
    {
        "configuration": "PINN_P_PET_lambda1",
        "model_family": "PINN_P_PET",
        "lambda_precipitation": 1.0,
        "lambda_pet": 1.0,
    },
    {
        "configuration": "PINN_P_PET_lambda10",
        "model_family": "PINN_P_PET",
        "lambda_precipitation": 10.0,
        "lambda_pet": 10.0,
    },
]

HIDDEN_DIMENSIONS = [64, 64]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAXIMUM_EPOCHS = 2500
EARLY_STOPPING_PATIENCE = 200

GRADIENT_VIOLATION_TOLERANCE = 1e-5


class SoilMoistureMLP(nn.Module):
    """Smooth MLP supporting input-gradient penalties."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int],
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_dimension = input_dimension

        for hidden_dimension in hidden_dimensions:
            layers.extend(
                [
                    nn.Linear(
                        previous_dimension,
                        hidden_dimension,
                    ),
                    nn.Tanh(),
                ]
            )

            previous_dimension = hidden_dimension

        layers.append(
            nn.Linear(
                previous_dimension,
                1,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def regression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    truth = np.asarray(
        truth,
        dtype=float,
    ).reshape(-1)

    prediction = np.asarray(
        prediction,
        dtype=float,
    ).reshape(-1)

    return {
        "rmse_mm": float(
            math.sqrt(
                mean_squared_error(
                    truth,
                    prediction,
                )
            )
        ),
        "mae_mm": float(
            mean_absolute_error(
                truth,
                prediction,
            )
        ),
        "r_squared": float(
            r2_score(
                truth,
                prediction,
            )
        ),
        "mean_bias_mm": float(
            np.mean(prediction - truth)
        ),
    }


def inverse_target(
    standardized_values: np.ndarray,
    target_scaler: StandardScaler,
) -> np.ndarray:
    return (
        target_scaler.inverse_transform(
            np.asarray(
                standardized_values,
            ).reshape(-1, 1)
        )
        .reshape(-1)
    )


def calculate_gradient_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
    precipitation_indices: list[int],
    pet_index: int,
) -> dict[str, float]:
    model.eval()

    diagnostic_inputs = (
        inputs.detach()
        .clone()
        .requires_grad_(True)
    )

    predictions = model(diagnostic_inputs)

    gradients = torch.autograd.grad(
        outputs=predictions.sum(),
        inputs=diagnostic_inputs,
        create_graph=False,
        retain_graph=False,
    )[0]

    precipitation_gradients = (
        gradients[
            :,
            precipitation_indices,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    pet_gradients = (
        gradients[:, pet_index]
        .detach()
        .cpu()
        .numpy()
    )

    precipitation_violations = (
        precipitation_gradients
        < -GRADIENT_VIOLATION_TOLERANCE
    )

    pet_violations = (
        pet_gradients
        > GRADIENT_VIOLATION_TOLERANCE
    )

    return {
        "precipitation_violation_fraction": float(
            precipitation_violations.mean()
        ),
        "precipitation_negative_gradient_mean": float(
            np.maximum(
                -precipitation_gradients,
                0.0,
            ).mean()
        ),
        "precipitation_negative_gradient_maximum": float(
            np.maximum(
                -precipitation_gradients,
                0.0,
            ).max()
        ),
        "pet_violation_fraction": float(
            pet_violations.mean()
        ),
        "pet_positive_gradient_mean": float(
            np.maximum(
                pet_gradients,
                0.0,
            ).mean()
        ),
        "pet_positive_gradient_maximum": float(
            np.maximum(
                pet_gradients,
                0.0,
            ).max()
        ),
    }


def train_neural_model(
    *,
    model_family: str,
    lambda_precipitation: float,
    lambda_pet: float,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    precipitation_indices: list[int],
    pet_index: int,
    device: torch.device,
) -> tuple[
    SoilMoistureMLP,
    dict[str, float | int],
]:
    set_random_seed(seed)

    model = SoilMoistureMLP(
        input_dimension=x_train.shape[1],
        hidden_dimensions=HIDDEN_DIMENSIONS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    mse_loss = nn.MSELoss()

    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if model_family == "MLP":
            training_prediction = model(
                x_train
            )

            data_loss = mse_loss(
                training_prediction,
                y_train,
            )

            precipitation_penalty = torch.zeros(
                (),
                device=device,
            )

            pet_penalty = torch.zeros(
                (),
                device=device,
            )

        else:
            physics_inputs = (
                x_train.detach()
                .clone()
                .requires_grad_(True)
            )

            training_prediction = model(
                physics_inputs
            )

            data_loss = mse_loss(
                training_prediction,
                y_train,
            )

            input_gradients = torch.autograd.grad(
                outputs=training_prediction.sum(),
                inputs=physics_inputs,
                create_graph=True,
                retain_graph=True,
            )[0]

            precipitation_gradients = (
                input_gradients[
                    :,
                    precipitation_indices,
                ]
            )

            precipitation_penalty = (
                torch.relu(
                    -precipitation_gradients
                )
                .pow(2)
                .mean()
            )

            pet_gradients = (
                input_gradients[:, pet_index]
            )

            pet_penalty = (
                torch.relu(
                    pet_gradients
                )
                .pow(2)
                .mean()
            )

        total_loss = (
            data_loss
            + lambda_precipitation
            * precipitation_penalty
            + lambda_pet
            * pet_penalty
        )

        total_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_prediction = model(
                x_validation
            )

            validation_loss = mse_loss(
                validation_prediction,
                y_validation,
            ).item()

        if (
            validation_loss
            < best_validation_loss - 1e-8
        ):
            best_validation_loss = (
                validation_loss
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "Neural training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)
    model.eval()

    return (
        model,
        {
            "best_epoch": int(best_epoch),
            "epochs_completed": int(epoch),
            "best_validation_mse_standardized": float(
                best_validation_loss
            ),
            "training_seconds": float(
                time.time() - start_time
            ),
        },
    )


def predict_neural(
    model: nn.Module,
    inputs: torch.Tensor,
    target_scaler: StandardScaler,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        standardized_prediction = (
            model(inputs)
            .detach()
            .cpu()
            .numpy()
        )

    return inverse_target(
        standardized_prediction,
        target_scaler,
    )


def append_predictions(
    rows: list[dict[str, object]],
    *,
    metadata: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
    configuration: str,
    model_family: str,
    lambda_precipitation: float,
    lambda_pet: float,
    seed: int,
    test_fold: int,
    validation_fold: int,
) -> None:
    metadata = metadata.reset_index(
        drop=True
    )

    for index, row in metadata.iterrows():
        rows.append(
            {
                "configuration": configuration,
                "model_family": model_family,
                "lambda_precipitation": (
                    lambda_precipitation
                ),
                "lambda_pet": lambda_pet,
                "seed": seed,
                "test_fold": test_fold,
                "validation_fold": validation_fold,
                "region_id": row["region_id"],
                "month": int(row["month"]),
                "native_cell_count": int(
                    row["native_cell_count"]
                ),
                "core_region_ge_4_cells": bool(
                    row["core_region_ge_4_cells"]
                ),
                "observed_SM_total_mm": float(
                    truth[index]
                ),
                "predicted_SM_total_mm": float(
                    prediction[index]
                ),
                "residual_mm": float(
                    truth[index]
                    - prediction[index]
                ),
            }
        )


def calculate_pooled_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    grouping_columns = [
        "configuration",
        "model_family",
        "lambda_precipitation",
        "lambda_pet",
        "seed",
    ]

    for group_values, group in predictions.groupby(
        grouping_columns,
        observed=True,
        dropna=False,
    ):
        (
            configuration,
            model_family,
            lambda_precipitation,
            lambda_pet,
            seed,
        ) = group_values

        if len(group) != EXPECTED_ROWS:
            raise RuntimeError(
                f"{configuration}, seed {seed}: "
                f"expected {EXPECTED_ROWS} pooled test rows, "
                f"found {len(group)}."
            )

        metrics = regression_metrics(
            group["observed_SM_total_mm"],
            group["predicted_SM_total_mm"],
        )

        core = group.loc[
            group["core_region_ge_4_cells"]
        ]

        core_metrics = regression_metrics(
            core["observed_SM_total_mm"],
            core["predicted_SM_total_mm"],
        )

        corresponding_folds = fold_metrics.loc[
            (
                fold_metrics["configuration"]
                == configuration
            )
            & (
                fold_metrics["seed"]
                == seed
            )
        ].copy()

        sample_weights = corresponding_folds[
            "test_sample_count"
        ].to_numpy(dtype=float)

        precipitation_violation = np.average(
            corresponding_folds[
                "test_precipitation_violation_fraction"
            ],
            weights=sample_weights,
        )

        pet_violation = np.average(
            corresponding_folds[
                "test_pet_violation_fraction"
            ],
            weights=sample_weights,
        )

        rows.append(
            {
                "configuration": configuration,
                "model_family": model_family,
                "lambda_precipitation": float(
                    lambda_precipitation
                ),
                "lambda_pet": float(
                    lambda_pet
                ),
                "seed": int(seed),
                "pooled_test_rmse_mm": (
                    metrics["rmse_mm"]
                ),
                "pooled_test_mae_mm": (
                    metrics["mae_mm"]
                ),
                "pooled_test_r_squared": (
                    metrics["r_squared"]
                ),
                "pooled_test_mean_bias_mm": (
                    metrics["mean_bias_mm"]
                ),
                "pooled_core_test_rmse_mm": (
                    core_metrics["rmse_mm"]
                ),
                "pooled_core_test_mae_mm": (
                    core_metrics["mae_mm"]
                ),
                "pooled_core_test_r_squared": (
                    core_metrics["r_squared"]
                ),
                "pooled_test_precipitation_violation_fraction": float(
                    precipitation_violation
                ),
                "pooled_test_pet_violation_fraction": float(
                    pet_violation
                ),
            }
        )

    return pd.DataFrame(rows)


def make_summary_figure(
    summary: pd.DataFrame,
) -> None:
    plotting = summary.copy()

    labels = plotting[
        "configuration"
    ].tolist()

    positions = np.arange(len(labels))

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.0, 5.0),
        constrained_layout=True,
    )

    lower_error = (
        plotting["median_pooled_test_rmse_mm"]
        - plotting["q25_pooled_test_rmse_mm"]
    )

    upper_error = (
        plotting["q75_pooled_test_rmse_mm"]
        - plotting["median_pooled_test_rmse_mm"]
    )

    axes[0].errorbar(
        positions,
        plotting[
            "median_pooled_test_rmse_mm"
        ],
        yerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        marker="o",
        linestyle="none",
        capsize=4,
    )

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    axes[0].set_ylabel(
        "Geographic-CV pooled RMSE (mm)"
    )

    axes[0].set_title(
        "Out-of-region predictive accuracy"
    )

    width = 0.36

    axes[1].bar(
        positions - width / 2,
        plotting[
            "median_pooled_test_p_violation"
        ],
        width=width,
        label="Precipitation violation",
    )

    axes[1].bar(
        positions + width / 2,
        plotting[
            "median_pooled_test_pet_violation"
        ],
        width=width,
        label="PET violation",
    )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    axes[1].set_ylabel(
        "Gradient-violation fraction"
    )

    axes[1].set_ylim(0.0, 1.02)

    axes[1].set_title(
        "Out-of-region physical consistency"
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
        "Five-fold geographic evaluation of "
        "total soil-moisture models",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig19_spatial_cv_total_sm_models.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig19_spatial_cv_total_sm_models.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    MODEL_DIR.mkdir(
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

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as stream:
        manifest = json.load(stream)

    if manifest["input_features"] != FEATURE_COLUMNS:
        raise RuntimeError(
            "Feature order differs from the feature manifest."
        )

    data = pd.read_parquet(DATA_PATH)

    if len(data) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(data)}."
        )

    if data["region_id"].nunique() != EXPECTED_REGIONS:
        raise RuntimeError(
            "Unexpected number of regional units."
        )

    if data[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].isna().any().any():
        raise RuntimeError(
            "Input features or target contain missing values."
        )

    folds = sorted(
        data["spatial_fold_5"]
        .astype(int)
        .unique()
        .tolist()
    )

    if folds != list(range(SPATIAL_FOLD_COUNT)):
        raise RuntimeError(
            f"Expected folds 0-4, found {folds}."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.set_num_threads(
        min(
            8,
            max(1, torch.get_num_threads()),
        )
    )

    precipitation_indices = [
        FEATURE_COLUMNS.index(feature)
        for feature
        in PRECIPITATION_FEATURES
    ]

    pet_index = FEATURE_COLUMNS.index(
        PET_FEATURE
    )

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    print("=" * 78)
    print("FIVE-FOLD GEOGRAPHIC TOTAL SOIL-MOISTURE EVALUATION")
    print("=" * 78)
    print(f"Device                  : {device}")
    print(f"Regional units          : {EXPECTED_REGIONS}")
    print(f"Region-month rows       : {EXPECTED_ROWS}")
    print(f"Spatial folds           : {folds}")
    print(f"Neural seeds            : {NEURAL_SEEDS}")
    print()

    for test_fold in folds:
        validation_fold = (
            test_fold + 1
        ) % SPATIAL_FOLD_COUNT

        train_mask = (
            (data["spatial_fold_5"] != test_fold)
            & (
                data["spatial_fold_5"]
                != validation_fold
            )
        )

        validation_mask = (
            data["spatial_fold_5"]
            == validation_fold
        )

        test_mask = (
            data["spatial_fold_5"]
            == test_fold
        )

        split_data = {
            "train": (
                data.loc[train_mask]
                .sort_values(
                    ["region_id", "month"]
                )
                .reset_index(drop=True)
            ),
            "validation": (
                data.loc[validation_mask]
                .sort_values(
                    ["region_id", "month"]
                )
                .reset_index(drop=True)
            ),
            "test": (
                data.loc[test_mask]
                .sort_values(
                    ["region_id", "month"]
                )
                .reset_index(drop=True)
            ),
        }

        train_regions = int(
            split_data["train"][
                "region_id"
            ].nunique()
        )

        validation_regions = int(
            split_data["validation"][
                "region_id"
            ].nunique()
        )

        test_regions = int(
            split_data["test"][
                "region_id"
            ].nunique()
        )

        print(
            f"Outer fold {test_fold}: "
            f"train={train_regions} regions, "
            f"validation={validation_regions}, "
            f"test={test_regions}"
        )

        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        x_train_raw = split_data[
            "train"
        ][FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        y_train_raw = split_data[
            "train"
        ][TARGET_COLUMN].to_numpy(
            dtype=np.float32
        )

        feature_scaler.fit(
            x_train_raw
        )

        target_scaler.fit(
            y_train_raw.reshape(-1, 1)
        )

        split_arrays = {}
        tensor_splits = {}

        for split_name, frame in (
            split_data.items()
        ):
            x_raw = frame[
                FEATURE_COLUMNS
            ].to_numpy(dtype=np.float32)

            y_raw = frame[
                TARGET_COLUMN
            ].to_numpy(dtype=np.float32)

            x_scaled = (
                feature_scaler.transform(
                    x_raw
                )
                .astype(np.float32)
            )

            y_scaled = (
                target_scaler.transform(
                    y_raw.reshape(-1, 1)
                )
                .reshape(-1)
                .astype(np.float32)
            )

            split_arrays[split_name] = {
                "x_raw": x_raw,
                "x_scaled": x_scaled,
                "y_raw": y_raw,
                "y_scaled": y_scaled,
                "metadata": frame[
                    [
                        "region_id",
                        "month",
                        "native_cell_count",
                        "core_region_ge_4_cells",
                    ]
                ].reset_index(drop=True),
            }

            tensor_splits[split_name] = {
                "x": torch.tensor(
                    x_scaled,
                    dtype=torch.float32,
                    device=device,
                ),
                "y": torch.tensor(
                    y_scaled,
                    dtype=torch.float32,
                    device=device,
                ),
            }

        joblib.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "target_column": TARGET_COLUMN,
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
                "test_fold": test_fold,
                "validation_fold": validation_fold,
            },
            MODEL_DIR
            / f"fold_{test_fold}_scalers.joblib",
        )

        # --------------------------------------------------------------
        # Mean baseline
        # --------------------------------------------------------------
        mean_model = DummyRegressor(
            strategy="mean"
        )

        mean_model.fit(
            split_arrays["train"]["x_raw"],
            split_arrays["train"]["y_raw"],
        )

        mean_test_prediction = mean_model.predict(
            split_arrays["test"]["x_raw"]
        )

        mean_test_metrics = regression_metrics(
            split_arrays["test"]["y_raw"],
            mean_test_prediction,
        )

        mean_core_mask = split_arrays[
            "test"
        ]["metadata"][
            "core_region_ge_4_cells"
        ].to_numpy(dtype=bool)

        mean_core_metrics = regression_metrics(
            split_arrays["test"]["y_raw"][
                mean_core_mask
            ],
            mean_test_prediction[
                mean_core_mask
            ],
        )

        fold_rows.append(
            {
                "configuration": "Mean",
                "model_family": "Mean",
                "lambda_precipitation": 0.0,
                "lambda_pet": 0.0,
                "seed": -1,
                "test_fold": test_fold,
                "validation_fold": (
                    validation_fold
                ),
                "train_region_count": (
                    train_regions
                ),
                "validation_region_count": (
                    validation_regions
                ),
                "test_region_count": (
                    test_regions
                ),
                "test_sample_count": int(
                    len(split_data["test"])
                ),
                "selected_ridge_alpha": np.nan,
                "best_epoch": 0,
                "epochs_completed": 0,
                "training_seconds": 0.0,
                "validation_rmse_mm": np.nan,
                "test_rmse_mm": (
                    mean_test_metrics["rmse_mm"]
                ),
                "test_mae_mm": (
                    mean_test_metrics["mae_mm"]
                ),
                "test_r_squared": (
                    mean_test_metrics["r_squared"]
                ),
                "test_mean_bias_mm": (
                    mean_test_metrics[
                        "mean_bias_mm"
                    ]
                ),
                "core_test_rmse_mm": (
                    mean_core_metrics["rmse_mm"]
                ),
                "core_test_r_squared": (
                    mean_core_metrics["r_squared"]
                ),
                "test_precipitation_violation_fraction": np.nan,
                "test_pet_violation_fraction": np.nan,
            }
        )

        append_predictions(
            prediction_rows,
            metadata=split_arrays[
                "test"
            ]["metadata"],
            truth=split_arrays[
                "test"
            ]["y_raw"],
            prediction=mean_test_prediction,
            configuration="Mean",
            model_family="Mean",
            lambda_precipitation=0.0,
            lambda_pet=0.0,
            seed=-1,
            test_fold=test_fold,
            validation_fold=validation_fold,
        )

        # --------------------------------------------------------------
        # Ridge baseline
        # --------------------------------------------------------------
        best_ridge = None
        best_ridge_alpha = None
        best_validation_rmse = (
            float("inf")
        )

        for alpha in RIDGE_ALPHA_GRID:
            ridge = Ridge(
                alpha=alpha,
                fit_intercept=True,
            )

            ridge.fit(
                split_arrays["train"][
                    "x_scaled"
                ],
                split_arrays["train"][
                    "y_scaled"
                ],
            )

            validation_prediction = (
                inverse_target(
                    ridge.predict(
                        split_arrays[
                            "validation"
                        ]["x_scaled"]
                    ),
                    target_scaler,
                )
            )

            validation_rmse = (
                regression_metrics(
                    split_arrays[
                        "validation"
                    ]["y_raw"],
                    validation_prediction,
                )["rmse_mm"]
            )

            if (
                validation_rmse
                < best_validation_rmse
            ):
                best_validation_rmse = (
                    validation_rmse
                )

                best_ridge_alpha = alpha
                best_ridge = copy.deepcopy(
                    ridge
                )

        if best_ridge is None:
            raise RuntimeError(
                "Ridge model selection failed."
            )

        ridge_test_prediction = inverse_target(
            best_ridge.predict(
                split_arrays["test"][
                    "x_scaled"
                ]
            ),
            target_scaler,
        )

        ridge_test_metrics = regression_metrics(
            split_arrays["test"]["y_raw"],
            ridge_test_prediction,
        )

        ridge_core_metrics = regression_metrics(
            split_arrays["test"]["y_raw"][
                mean_core_mask
            ],
            ridge_test_prediction[
                mean_core_mask
            ],
        )

        ridge_coefficients = np.asarray(
            best_ridge.coef_,
            dtype=float,
        ).reshape(-1)

        ridge_p_violation = float(
            np.mean(
                ridge_coefficients[
                    precipitation_indices
                ]
                < -GRADIENT_VIOLATION_TOLERANCE
            )
        )

        ridge_pet_violation = float(
            ridge_coefficients[pet_index]
            > GRADIENT_VIOLATION_TOLERANCE
        )

        fold_rows.append(
            {
                "configuration": "Ridge",
                "model_family": "Ridge",
                "lambda_precipitation": 0.0,
                "lambda_pet": 0.0,
                "seed": -1,
                "test_fold": test_fold,
                "validation_fold": (
                    validation_fold
                ),
                "train_region_count": (
                    train_regions
                ),
                "validation_region_count": (
                    validation_regions
                ),
                "test_region_count": (
                    test_regions
                ),
                "test_sample_count": int(
                    len(split_data["test"])
                ),
                "selected_ridge_alpha": float(
                    best_ridge_alpha
                ),
                "best_epoch": 0,
                "epochs_completed": 0,
                "training_seconds": 0.0,
                "validation_rmse_mm": float(
                    best_validation_rmse
                ),
                "test_rmse_mm": (
                    ridge_test_metrics["rmse_mm"]
                ),
                "test_mae_mm": (
                    ridge_test_metrics["mae_mm"]
                ),
                "test_r_squared": (
                    ridge_test_metrics["r_squared"]
                ),
                "test_mean_bias_mm": (
                    ridge_test_metrics[
                        "mean_bias_mm"
                    ]
                ),
                "core_test_rmse_mm": (
                    ridge_core_metrics["rmse_mm"]
                ),
                "core_test_r_squared": (
                    ridge_core_metrics["r_squared"]
                ),
                "test_precipitation_violation_fraction": (
                    ridge_p_violation
                ),
                "test_pet_violation_fraction": (
                    ridge_pet_violation
                ),
            }
        )

        append_predictions(
            prediction_rows,
            metadata=split_arrays[
                "test"
            ]["metadata"],
            truth=split_arrays[
                "test"
            ]["y_raw"],
            prediction=ridge_test_prediction,
            configuration="Ridge",
            model_family="Ridge",
            lambda_precipitation=0.0,
            lambda_pet=0.0,
            seed=-1,
            test_fold=test_fold,
            validation_fold=validation_fold,
        )

        joblib.dump(
            {
                "model": best_ridge,
                "alpha": best_ridge_alpha,
            },
            MODEL_DIR
            / f"fold_{test_fold}_ridge.joblib",
        )

        # --------------------------------------------------------------
        # Neural configurations
        # --------------------------------------------------------------
        for configuration in (
            NEURAL_CONFIGURATIONS
        ):
            configuration_name = (
                configuration[
                    "configuration"
                ]
            )

            model_family = configuration[
                "model_family"
            ]

            lambda_precipitation = float(
                configuration[
                    "lambda_precipitation"
                ]
            )

            lambda_pet = float(
                configuration["lambda_pet"]
            )

            for seed in NEURAL_SEEDS:
                print(
                    f"  {configuration_name:24s} "
                    f"| seed={seed}"
                )

                (
                    model,
                    training_information,
                ) = train_neural_model(
                    model_family=model_family,
                    lambda_precipitation=(
                        lambda_precipitation
                    ),
                    lambda_pet=lambda_pet,
                    seed=seed,
                    x_train=tensor_splits[
                        "train"
                    ]["x"],
                    y_train=tensor_splits[
                        "train"
                    ]["y"],
                    x_validation=tensor_splits[
                        "validation"
                    ]["x"],
                    y_validation=tensor_splits[
                        "validation"
                    ]["y"],
                    precipitation_indices=(
                        precipitation_indices
                    ),
                    pet_index=pet_index,
                    device=device,
                )

                validation_prediction = (
                    predict_neural(
                        model,
                        tensor_splits[
                            "validation"
                        ]["x"],
                        target_scaler,
                    )
                )

                test_prediction = (
                    predict_neural(
                        model,
                        tensor_splits[
                            "test"
                        ]["x"],
                        target_scaler,
                    )
                )

                validation_metrics = (
                    regression_metrics(
                        split_arrays[
                            "validation"
                        ]["y_raw"],
                        validation_prediction,
                    )
                )

                test_metrics = (
                    regression_metrics(
                        split_arrays[
                            "test"
                        ]["y_raw"],
                        test_prediction,
                    )
                )

                core_test_metrics = (
                    regression_metrics(
                        split_arrays[
                            "test"
                        ]["y_raw"][
                            mean_core_mask
                        ],
                        test_prediction[
                            mean_core_mask
                        ],
                    )
                )

                test_gradient_metrics = (
                    calculate_gradient_diagnostics(
                        model,
                        tensor_splits[
                            "test"
                        ]["x"],
                        precipitation_indices,
                        pet_index,
                    )
                )

                fold_rows.append(
                    {
                        "configuration": (
                            configuration_name
                        ),
                        "model_family": (
                            model_family
                        ),
                        "lambda_precipitation": (
                            lambda_precipitation
                        ),
                        "lambda_pet": (
                            lambda_pet
                        ),
                        "seed": seed,
                        "test_fold": test_fold,
                        "validation_fold": (
                            validation_fold
                        ),
                        "train_region_count": (
                            train_regions
                        ),
                        "validation_region_count": (
                            validation_regions
                        ),
                        "test_region_count": (
                            test_regions
                        ),
                        "test_sample_count": int(
                            len(
                                split_data["test"]
                            )
                        ),
                        "selected_ridge_alpha": (
                            np.nan
                        ),
                        "best_epoch": (
                            training_information[
                                "best_epoch"
                            ]
                        ),
                        "epochs_completed": (
                            training_information[
                                "epochs_completed"
                            ]
                        ),
                        "training_seconds": (
                            training_information[
                                "training_seconds"
                            ]
                        ),
                        "validation_rmse_mm": (
                            validation_metrics[
                                "rmse_mm"
                            ]
                        ),
                        "test_rmse_mm": (
                            test_metrics["rmse_mm"]
                        ),
                        "test_mae_mm": (
                            test_metrics["mae_mm"]
                        ),
                        "test_r_squared": (
                            test_metrics["r_squared"]
                        ),
                        "test_mean_bias_mm": (
                            test_metrics[
                                "mean_bias_mm"
                            ]
                        ),
                        "core_test_rmse_mm": (
                            core_test_metrics[
                                "rmse_mm"
                            ]
                        ),
                        "core_test_r_squared": (
                            core_test_metrics[
                                "r_squared"
                            ]
                        ),
                        "test_precipitation_violation_fraction": (
                            test_gradient_metrics[
                                "precipitation_violation_fraction"
                            ]
                        ),
                        "test_pet_violation_fraction": (
                            test_gradient_metrics[
                                "pet_violation_fraction"
                            ]
                        ),
                        "test_precipitation_negative_gradient_mean": (
                            test_gradient_metrics[
                                "precipitation_negative_gradient_mean"
                            ]
                        ),
                        "test_pet_positive_gradient_mean": (
                            test_gradient_metrics[
                                "pet_positive_gradient_mean"
                            ]
                        ),
                    }
                )

                append_predictions(
                    prediction_rows,
                    metadata=split_arrays[
                        "test"
                    ]["metadata"],
                    truth=split_arrays[
                        "test"
                    ]["y_raw"],
                    prediction=test_prediction,
                    configuration=(
                        configuration_name
                    ),
                    model_family=model_family,
                    lambda_precipitation=(
                        lambda_precipitation
                    ),
                    lambda_pet=lambda_pet,
                    seed=seed,
                    test_fold=test_fold,
                    validation_fold=(
                        validation_fold
                    ),
                )

                model_path = (
                    MODEL_DIR
                    / (
                        f"fold_{test_fold}_"
                        f"{configuration_name}_"
                        f"seed_{seed}.pt"
                    )
                )

                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value
                            in model.state_dict().items()
                        },
                        "configuration": (
                            configuration_name
                        ),
                        "model_family": (
                            model_family
                        ),
                        "lambda_precipitation": (
                            lambda_precipitation
                        ),
                        "lambda_pet": (
                            lambda_pet
                        ),
                        "seed": seed,
                        "test_fold": (
                            test_fold
                        ),
                        "validation_fold": (
                            validation_fold
                        ),
                        "feature_columns": (
                            FEATURE_COLUMNS
                        ),
                        "hidden_dimensions": (
                            HIDDEN_DIMENSIONS
                        ),
                        "training_information": (
                            training_information
                        ),
                    },
                    model_path,
                )

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    pooled_seed_metrics = (
        calculate_pooled_metrics(
            predictions,
            fold_metrics,
        )
    )

    fold_metrics_path = (
        TABLE_DIR
        / "spatial_cv_total_sm_fold_metrics.csv"
    )

    predictions_path = (
        TABLE_DIR
        / "spatial_cv_total_sm_predictions.csv.gz"
    )

    pooled_seed_path = (
        TABLE_DIR
        / "spatial_cv_total_sm_pooled_seed_metrics.csv"
    )

    fold_metrics.to_csv(
        fold_metrics_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )

    pooled_seed_metrics.to_csv(
        pooled_seed_path,
        index=False,
    )

    summary = (
        pooled_seed_metrics.groupby(
            [
                "configuration",
                "model_family",
                "lambda_precipitation",
                "lambda_pet",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            seed_count=("seed", "size"),
            median_pooled_test_rmse_mm=(
                "pooled_test_rmse_mm",
                "median",
            ),
            mean_pooled_test_rmse_mm=(
                "pooled_test_rmse_mm",
                "mean",
            ),
            std_pooled_test_rmse_mm=(
                "pooled_test_rmse_mm",
                "std",
            ),
            q25_pooled_test_rmse_mm=(
                "pooled_test_rmse_mm",
                lambda values: values.quantile(
                    0.25
                ),
            ),
            q75_pooled_test_rmse_mm=(
                "pooled_test_rmse_mm",
                lambda values: values.quantile(
                    0.75
                ),
            ),
            median_pooled_test_mae_mm=(
                "pooled_test_mae_mm",
                "median",
            ),
            median_pooled_test_r_squared=(
                "pooled_test_r_squared",
                "median",
            ),
            median_pooled_core_test_rmse_mm=(
                "pooled_core_test_rmse_mm",
                "median",
            ),
            median_pooled_core_test_r_squared=(
                "pooled_core_test_r_squared",
                "median",
            ),
            median_pooled_test_p_violation=(
                "pooled_test_precipitation_violation_fraction",
                "median",
            ),
            median_pooled_test_pet_violation=(
                "pooled_test_pet_violation_fraction",
                "median",
            ),
        )
    )

    summary[
        "std_pooled_test_rmse_mm"
    ] = summary[
        "std_pooled_test_rmse_mm"
    ].fillna(0.0)

    order = {
        "Mean": 0,
        "Ridge": 1,
        "MLP": 2,
        "PINN_P_lambda10": 3,
        "PINN_P_PET_lambda1": 4,
        "PINN_P_PET_lambda10": 5,
    }

    summary["_order"] = (
        summary["configuration"]
        .map(order)
    )

    summary = (
        summary.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    summary_path = (
        TABLE_DIR
        / "spatial_cv_total_sm_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    make_summary_figure(summary)

    metadata = {
        "status": "PASS",
        "dataset": str(DATA_PATH),
        "device": str(device),
        "outer_spatial_fold_count": (
            SPATIAL_FOLD_COUNT
        ),
        "outer_test_policy": (
            "Each complete geographic fold is tested once."
        ),
        "validation_policy": (
            "Validation fold is the next geographic fold "
            "cyclically; remaining three folds are training."
        ),
        "conditional_generalization_note": (
            "The precipitation NMF representation was fitted "
            "before geographic cross-validation. Results therefore "
            "measure soil-moisture-model generalization conditional "
            "on the fixed selected NMF precipitation representation."
        ),
        "neural_seeds": NEURAL_SEEDS,
        "neural_configurations": (
            NEURAL_CONFIGURATIONS
        ),
        "ridge_alpha_grid": (
            RIDGE_ALPHA_GRID
        ),
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "fold_metrics": str(
            fold_metrics_path
        ),
        "pooled_seed_metrics": str(
            pooled_seed_path
        ),
        "summary_table": str(
            summary_path
        ),
        "predictions": str(
            predictions_path
        ),
        "model_directory": str(
            MODEL_DIR
        ),
        "final_model_selection": (
            "PENDING_REVIEW"
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "spatial_cv_total_sm_metadata.json"
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
    print("GEOGRAPHIC-CV SUMMARY")
    print("=" * 78)

    display_columns = [
        "configuration",
        "seed_count",
        "median_pooled_test_rmse_mm",
        "std_pooled_test_rmse_mm",
        "median_pooled_test_mae_mm",
        "median_pooled_test_r_squared",
        "median_pooled_core_test_rmse_mm",
        "median_pooled_core_test_r_squared",
        "median_pooled_test_p_violation",
        "median_pooled_test_pet_violation",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print(f"Fold metrics         : {fold_metrics_path}")
    print(f"Pooled seed metrics  : {pooled_seed_path}")
    print(f"Summary              : {summary_path}")
    print(f"Predictions          : {predictions_path}")
    print(
        f"Comparison figure    : "
        f"{FIGURE_DIR / 'fig19_spatial_cv_total_sm_models.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
