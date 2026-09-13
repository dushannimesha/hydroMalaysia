#!/usr/bin/env python3
"""
Compare statistical, neural and physics-informed models for total soil moisture.

Models
------
1. Mean baseline
2. Ridge regression
3. Ordinary multilayer perceptron
4. PINN-P:
       dSM/dP_component_lag >= 0
5. PINN-P-PET:
       dSM/dP_component_lag >= 0
       dSM/dPET <= 0

The existing 8/2/2 month split is used:
    train      = 264 samples
    validation = 66 samples
    test       = 66 samples

All feature and target scaling is fitted using training data only.

This stage performs a physics-weight sweep. Final model selection is
intentionally deferred until prediction accuracy and physical consistency
are inspected together.
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
    / "stage07"
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

EXPECTED_ROWS = 396
EXPECTED_TRAIN = 264
EXPECTED_VALIDATION = 66
EXPECTED_TEST = 66

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

PHYSICS_LAMBDAS = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
]

NEURAL_SEEDS = [
    20260806,
    20260807,
    20260808,
    20260809,
    20260810,
]

HIDDEN_DIMENSIONS = [64, 64]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAXIMUM_EPOCHS = 2500
EARLY_STOPPING_PATIENCE = 200

GRADIENT_VIOLATION_TOLERANCE = 1e-5


class SoilMoistureMLP(nn.Module):
    """Smooth MLP suitable for input-gradient constraints."""

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

    rmse = math.sqrt(
        mean_squared_error(
            truth,
            prediction,
        )
    )

    return {
        "rmse_mm": float(rmse),
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
    values_standardized: np.ndarray,
    target_scaler: StandardScaler,
) -> np.ndarray:
    return (
        target_scaler.inverse_transform(
            np.asarray(
                values_standardized
            ).reshape(-1, 1)
        )
        .reshape(-1)
    )


def format_lambda(value: float) -> str:
    return (
        f"{value:.6g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def calculate_gradient_diagnostics(
    model: nn.Module,
    inputs: torch.Tensor,
    precipitation_indices: list[int],
    pet_index: int,
) -> dict[str, float]:
    model.eval()

    evaluation_inputs = (
        inputs.detach()
        .clone()
        .requires_grad_(True)
    )

    predictions = model(
        evaluation_inputs
    )

    gradients = torch.autograd.grad(
        outputs=predictions.sum(),
        inputs=evaluation_inputs,
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

    precipitation_violation_mask = (
        precipitation_gradients
        < -GRADIENT_VIOLATION_TOLERANCE
    )

    pet_violation_mask = (
        pet_gradients
        > GRADIENT_VIOLATION_TOLERANCE
    )

    precipitation_negative_magnitude = (
        np.maximum(
            -precipitation_gradients,
            0.0,
        )
    )

    pet_positive_magnitude = np.maximum(
        pet_gradients,
        0.0,
    )

    return {
        "precipitation_violation_fraction": float(
            precipitation_violation_mask.mean()
        ),
        "precipitation_negative_gradient_mean": float(
            precipitation_negative_magnitude.mean()
        ),
        "precipitation_negative_gradient_maximum": float(
            precipitation_negative_magnitude.max()
        ),
        "pet_violation_fraction": float(
            pet_violation_mask.mean()
        ),
        "pet_positive_gradient_mean": float(
            pet_positive_magnitude.mean()
        ),
        "pet_positive_gradient_maximum": float(
            pet_positive_magnitude.max()
        ),
    }


def train_neural_model(
    *,
    family: str,
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

        if family == "MLP":
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
            "Neural training did not produce a valid state."
        )

    model.load_state_dict(best_state)
    model.eval()

    elapsed_seconds = time.time() - start_time

    training_information = {
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epoch),
        "best_validation_mse_standardized": float(
            best_validation_loss
        ),
        "training_seconds": float(
            elapsed_seconds
        ),
    }

    return model, training_information


def predict_neural(
    model: nn.Module,
    inputs: torch.Tensor,
    target_scaler: StandardScaler,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        prediction_standardized = (
            model(inputs)
            .detach()
            .cpu()
            .numpy()
        )

    return inverse_target(
        prediction_standardized,
        target_scaler,
    )


def append_prediction_rows(
    rows: list[dict[str, object]],
    *,
    metadata: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
    model_family: str,
    lambda_precipitation: float,
    lambda_pet: float,
    seed: int,
    split_name: str,
) -> None:
    for index, metadata_row in (
        metadata.reset_index(drop=True)
        .iterrows()
    ):
        rows.append(
            {
                "model_family": model_family,
                "lambda_precipitation": (
                    lambda_precipitation
                ),
                "lambda_pet": lambda_pet,
                "seed": seed,
                "split": split_name,
                "region_id": metadata_row[
                    "region_id"
                ],
                "month": int(
                    metadata_row["month"]
                ),
                "native_cell_count": int(
                    metadata_row[
                        "native_cell_count"
                    ]
                ),
                "core_region_ge_4_cells": bool(
                    metadata_row[
                        "core_region_ge_4_cells"
                    ]
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

    if (
        manifest["input_features"]
        != FEATURE_COLUMNS
    ):
        raise RuntimeError(
            "Feature order differs from the saved manifest."
        )

    data = pd.read_parquet(DATA_PATH)

    if len(data) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(data)}."
        )

    if data[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].isna().any().any():
        raise RuntimeError(
            "Model inputs or target contain missing values."
        )

    split_masks = {
        split: (
            data["replication_split"] == split
        )
        for split in [
            "train",
            "validation",
            "test",
        ]
    }

    split_sizes = {
        split: int(mask.sum())
        for split, mask
        in split_masks.items()
    }

    expected_sizes = {
        "train": EXPECTED_TRAIN,
        "validation": EXPECTED_VALIDATION,
        "test": EXPECTED_TEST,
    }

    if split_sizes != expected_sizes:
        raise RuntimeError(
            f"Unexpected split sizes: {split_sizes}"
        )

    x_train_raw = data.loc[
        split_masks["train"],
        FEATURE_COLUMNS,
    ].to_numpy(dtype=np.float32)

    y_train_raw = data.loc[
        split_masks["train"],
        TARGET_COLUMN,
    ].to_numpy(dtype=np.float32)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train_scaled = feature_scaler.fit_transform(
        x_train_raw
    ).astype(np.float32)

    y_train_scaled = (
        target_scaler.fit_transform(
            y_train_raw.reshape(-1, 1)
        )
        .reshape(-1)
        .astype(np.float32)
    )

    split_arrays: dict[
        str,
        dict[str, object],
    ] = {}

    for split_name, split_mask in (
        split_masks.items()
    ):
        raw_features = data.loc[
            split_mask,
            FEATURE_COLUMNS,
        ].to_numpy(dtype=np.float32)

        truth = data.loc[
            split_mask,
            TARGET_COLUMN,
        ].to_numpy(dtype=np.float32)

        scaled_features = (
            feature_scaler.transform(
                raw_features
            )
            .astype(np.float32)
        )

        scaled_target = (
            target_scaler.transform(
                truth.reshape(-1, 1)
            )
            .reshape(-1)
            .astype(np.float32)
        )

        metadata = data.loc[
            split_mask,
            [
                "region_id",
                "month",
                "native_cell_count",
                "core_region_ge_4_cells",
            ],
        ].reset_index(drop=True)

        split_arrays[split_name] = {
            "x_raw": raw_features,
            "x_scaled": scaled_features,
            "y_raw": truth,
            "y_scaled": scaled_target,
            "metadata": metadata,
        }

    scaler_path = (
        MODEL_DIR / "training_scalers.joblib"
    )

    joblib.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
        },
        scaler_path,
    )

    precipitation_indices = [
        FEATURE_COLUMNS.index(feature)
        for feature
        in PRECIPITATION_FEATURES
    ]

    pet_index = FEATURE_COLUMNS.index(
        PET_FEATURE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.set_num_threads(
        min(8, max(1, torch.get_num_threads()))
    )

    tensor_splits = {}

    for split_name, arrays in (
        split_arrays.items()
    ):
        tensor_splits[split_name] = {
            "x": torch.tensor(
                arrays["x_scaled"],
                dtype=torch.float32,
                device=device,
            ),
            "y": torch.tensor(
                arrays["y_scaled"],
                dtype=torch.float32,
                device=device,
            ),
        }

    print("=" * 78)
    print("TOTAL SOIL-MOISTURE MODEL SWEEP")
    print("=" * 78)
    print(f"Device                         : {device}")
    print(f"Features                       : {len(FEATURE_COLUMNS)}")
    print(f"Precipitation features         : {len(PRECIPITATION_FEATURES)}")
    print(f"Train/validation/test          : {split_sizes}")
    print(f"Neural seeds                   : {NEURAL_SEEDS}")
    print(f"Physics lambdas                : {PHYSICS_LAMBDAS}")
    print()

    run_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Mean baseline
    # ------------------------------------------------------------------
    dummy = DummyRegressor(
        strategy="mean"
    )

    dummy.fit(
        x_train_raw,
        y_train_raw,
    )

    for split_name, arrays in (
        split_arrays.items()
    ):
        prediction = dummy.predict(
            arrays["x_raw"]
        )

        metrics = regression_metrics(
            arrays["y_raw"],
            prediction,
        )

        core_mask = arrays[
            "metadata"
        ]["core_region_ge_4_cells"].to_numpy(
            dtype=bool
        )

        core_metrics = regression_metrics(
            arrays["y_raw"][core_mask],
            prediction[core_mask],
        )

        if split_name == "test":
            run_rows.append(
                {
                    "model_family": "Mean",
                    "lambda_precipitation": 0.0,
                    "lambda_pet": 0.0,
                    "seed": -1,
                    "selected_ridge_alpha": np.nan,
                    "best_epoch": 0,
                    "epochs_completed": 0,
                    "training_seconds": 0.0,
                    "validation_rmse_mm": np.nan,
                    "test_rmse_mm": metrics["rmse_mm"],
                    "test_mae_mm": metrics["mae_mm"],
                    "test_r_squared": metrics["r_squared"],
                    "test_mean_bias_mm": metrics[
                        "mean_bias_mm"
                    ],
                    "core_test_rmse_mm": core_metrics[
                        "rmse_mm"
                    ],
                    "core_test_mae_mm": core_metrics[
                        "mae_mm"
                    ],
                    "core_test_r_squared": core_metrics[
                        "r_squared"
                    ],
                    "validation_precipitation_violation_fraction": np.nan,
                    "test_precipitation_violation_fraction": np.nan,
                    "validation_pet_violation_fraction": np.nan,
                    "test_pet_violation_fraction": np.nan,
                }
            )

        append_prediction_rows(
            prediction_rows,
            metadata=arrays["metadata"],
            truth=arrays["y_raw"],
            prediction=prediction,
            model_family="Mean",
            lambda_precipitation=0.0,
            lambda_pet=0.0,
            seed=-1,
            split_name=split_name,
        )

    # ------------------------------------------------------------------
    # Ridge selection using validation RMSE
    # ------------------------------------------------------------------
    best_ridge = None
    best_ridge_alpha = None
    best_ridge_validation_rmse = float("inf")

    for alpha in RIDGE_ALPHA_GRID:
        ridge = Ridge(
            alpha=alpha,
            fit_intercept=True,
        )

        ridge.fit(
            x_train_scaled,
            y_train_scaled,
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
            < best_ridge_validation_rmse
        ):
            best_ridge_validation_rmse = (
                validation_rmse
            )

            best_ridge_alpha = alpha
            best_ridge = copy.deepcopy(
                ridge
            )

    if best_ridge is None:
        raise RuntimeError(
            "Ridge selection failed."
        )

    ridge_coefficients = (
        np.asarray(
            best_ridge.coef_,
            dtype=float,
        ).reshape(-1)
    )

    ridge_precipitation_violation = float(
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

    ridge_split_metrics = {}

    for split_name, arrays in (
        split_arrays.items()
    ):
        prediction = inverse_target(
            best_ridge.predict(
                arrays["x_scaled"]
            ),
            target_scaler,
        )

        metrics = regression_metrics(
            arrays["y_raw"],
            prediction,
        )

        ridge_split_metrics[
            split_name
        ] = metrics

        append_prediction_rows(
            prediction_rows,
            metadata=arrays["metadata"],
            truth=arrays["y_raw"],
            prediction=prediction,
            model_family="Ridge",
            lambda_precipitation=0.0,
            lambda_pet=0.0,
            seed=-1,
            split_name=split_name,
        )

    ridge_test_arrays = split_arrays["test"]
    ridge_test_prediction = inverse_target(
        best_ridge.predict(
            ridge_test_arrays["x_scaled"]
        ),
        target_scaler,
    )

    ridge_core_mask = ridge_test_arrays[
        "metadata"
    ]["core_region_ge_4_cells"].to_numpy(
        dtype=bool
    )

    ridge_core_metrics = regression_metrics(
        ridge_test_arrays["y_raw"][
            ridge_core_mask
        ],
        ridge_test_prediction[
            ridge_core_mask
        ],
    )

    run_rows.append(
        {
            "model_family": "Ridge",
            "lambda_precipitation": 0.0,
            "lambda_pet": 0.0,
            "seed": -1,
            "selected_ridge_alpha": (
                best_ridge_alpha
            ),
            "best_epoch": 0,
            "epochs_completed": 0,
            "training_seconds": 0.0,
            "validation_rmse_mm": (
                ridge_split_metrics[
                    "validation"
                ]["rmse_mm"]
            ),
            "test_rmse_mm": (
                ridge_split_metrics[
                    "test"
                ]["rmse_mm"]
            ),
            "test_mae_mm": (
                ridge_split_metrics[
                    "test"
                ]["mae_mm"]
            ),
            "test_r_squared": (
                ridge_split_metrics[
                    "test"
                ]["r_squared"]
            ),
            "test_mean_bias_mm": (
                ridge_split_metrics[
                    "test"
                ]["mean_bias_mm"]
            ),
            "core_test_rmse_mm": (
                ridge_core_metrics["rmse_mm"]
            ),
            "core_test_mae_mm": (
                ridge_core_metrics["mae_mm"]
            ),
            "core_test_r_squared": (
                ridge_core_metrics["r_squared"]
            ),
            "validation_precipitation_violation_fraction": (
                ridge_precipitation_violation
            ),
            "test_precipitation_violation_fraction": (
                ridge_precipitation_violation
            ),
            "validation_pet_violation_fraction": (
                ridge_pet_violation
            ),
            "test_pet_violation_fraction": (
                ridge_pet_violation
            ),
        }
    )

    joblib.dump(
        {
            "model": best_ridge,
            "alpha": best_ridge_alpha,
        },
        MODEL_DIR / "ridge_model.joblib",
    )

    # ------------------------------------------------------------------
    # Neural and physics-informed models
    # ------------------------------------------------------------------
    neural_configurations = [
        {
            "family": "MLP",
            "lambda_precipitation": 0.0,
            "lambda_pet": 0.0,
        }
    ]

    for physics_lambda in PHYSICS_LAMBDAS:
        neural_configurations.append(
            {
                "family": "PINN_P",
                "lambda_precipitation": (
                    physics_lambda
                ),
                "lambda_pet": 0.0,
            }
        )

        neural_configurations.append(
            {
                "family": "PINN_P_PET",
                "lambda_precipitation": (
                    physics_lambda
                ),
                "lambda_pet": physics_lambda,
            }
        )

    for configuration in neural_configurations:
        family = configuration["family"]
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
                f"{family:10s} | "
                f"lambda_P={lambda_precipitation:g} | "
                f"lambda_PET={lambda_pet:g} | "
                f"seed={seed}"
            )

            (
                model,
                training_information,
            ) = train_neural_model(
                family=family,
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

            split_predictions = {}
            split_metrics = {}
            split_gradient_metrics = {}

            for split_name in [
                "train",
                "validation",
                "test",
            ]:
                prediction = predict_neural(
                    model,
                    tensor_splits[
                        split_name
                    ]["x"],
                    target_scaler,
                )

                split_predictions[
                    split_name
                ] = prediction

                split_metrics[
                    split_name
                ] = regression_metrics(
                    split_arrays[
                        split_name
                    ]["y_raw"],
                    prediction,
                )

                split_gradient_metrics[
                    split_name
                ] = (
                    calculate_gradient_diagnostics(
                        model,
                        tensor_splits[
                            split_name
                        ]["x"],
                        precipitation_indices,
                        pet_index,
                    )
                )

                append_prediction_rows(
                    prediction_rows,
                    metadata=split_arrays[
                        split_name
                    ]["metadata"],
                    truth=split_arrays[
                        split_name
                    ]["y_raw"],
                    prediction=prediction,
                    model_family=family,
                    lambda_precipitation=(
                        lambda_precipitation
                    ),
                    lambda_pet=lambda_pet,
                    seed=seed,
                    split_name=split_name,
                )

            test_metadata = split_arrays[
                "test"
            ]["metadata"]

            core_test_mask = test_metadata[
                "core_region_ge_4_cells"
            ].to_numpy(dtype=bool)

            core_test_metrics = (
                regression_metrics(
                    split_arrays[
                        "test"
                    ]["y_raw"][
                        core_test_mask
                    ],
                    split_predictions[
                        "test"
                    ][
                        core_test_mask
                    ],
                )
            )

            run_rows.append(
                {
                    "model_family": family,
                    "lambda_precipitation": (
                        lambda_precipitation
                    ),
                    "lambda_pet": lambda_pet,
                    "seed": seed,
                    "selected_ridge_alpha": np.nan,
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
                        split_metrics[
                            "validation"
                        ]["rmse_mm"]
                    ),
                    "test_rmse_mm": (
                        split_metrics[
                            "test"
                        ]["rmse_mm"]
                    ),
                    "test_mae_mm": (
                        split_metrics[
                            "test"
                        ]["mae_mm"]
                    ),
                    "test_r_squared": (
                        split_metrics[
                            "test"
                        ]["r_squared"]
                    ),
                    "test_mean_bias_mm": (
                        split_metrics[
                            "test"
                        ]["mean_bias_mm"]
                    ),
                    "core_test_rmse_mm": (
                        core_test_metrics[
                            "rmse_mm"
                        ]
                    ),
                    "core_test_mae_mm": (
                        core_test_metrics[
                            "mae_mm"
                        ]
                    ),
                    "core_test_r_squared": (
                        core_test_metrics[
                            "r_squared"
                        ]
                    ),
                    "validation_precipitation_violation_fraction": (
                        split_gradient_metrics[
                            "validation"
                        ][
                            "precipitation_violation_fraction"
                        ]
                    ),
                    "test_precipitation_violation_fraction": (
                        split_gradient_metrics[
                            "test"
                        ][
                            "precipitation_violation_fraction"
                        ]
                    ),
                    "validation_pet_violation_fraction": (
                        split_gradient_metrics[
                            "validation"
                        ][
                            "pet_violation_fraction"
                        ]
                    ),
                    "test_pet_violation_fraction": (
                        split_gradient_metrics[
                            "test"
                        ][
                            "pet_violation_fraction"
                        ]
                    ),
                    "test_precipitation_negative_gradient_mean": (
                        split_gradient_metrics[
                            "test"
                        ][
                            "precipitation_negative_gradient_mean"
                        ]
                    ),
                    "test_pet_positive_gradient_mean": (
                        split_gradient_metrics[
                            "test"
                        ][
                            "pet_positive_gradient_mean"
                        ]
                    ),
                }
            )

            model_filename = (
                f"{family}"
                f"_lp_{format_lambda(lambda_precipitation)}"
                f"_lpet_{format_lambda(lambda_pet)}"
                f"_seed_{seed}.pt"
            )

            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value
                        in model.state_dict().items()
                    },
                    "family": family,
                    "lambda_precipitation": (
                        lambda_precipitation
                    ),
                    "lambda_pet": lambda_pet,
                    "seed": seed,
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
                MODEL_DIR / model_filename,
            )

    runs = pd.DataFrame(run_rows)

    predictions = pd.DataFrame(
        prediction_rows
    )

    runs_path = (
        TABLE_DIR
        / "total_sm_model_sweep_all_runs.csv"
    )

    predictions_path = (
        TABLE_DIR
        / "total_sm_model_sweep_predictions.csv.gz"
    )

    runs.to_csv(
        runs_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )

    summary = (
        runs.groupby(
            [
                "model_family",
                "lambda_precipitation",
                "lambda_pet",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            run_count=("seed", "size"),
            median_validation_rmse_mm=(
                "validation_rmse_mm",
                "median",
            ),
            mean_validation_rmse_mm=(
                "validation_rmse_mm",
                "mean",
            ),
            std_validation_rmse_mm=(
                "validation_rmse_mm",
                "std",
            ),
            median_test_rmse_mm=(
                "test_rmse_mm",
                "median",
            ),
            mean_test_rmse_mm=(
                "test_rmse_mm",
                "mean",
            ),
            std_test_rmse_mm=(
                "test_rmse_mm",
                "std",
            ),
            median_test_mae_mm=(
                "test_mae_mm",
                "median",
            ),
            median_test_r_squared=(
                "test_r_squared",
                "median",
            ),
            median_core_test_rmse_mm=(
                "core_test_rmse_mm",
                "median",
            ),
            median_core_test_r_squared=(
                "core_test_r_squared",
                "median",
            ),
            median_validation_p_violation=(
                "validation_precipitation_violation_fraction",
                "median",
            ),
            median_test_p_violation=(
                "test_precipitation_violation_fraction",
                "median",
            ),
            median_validation_pet_violation=(
                "validation_pet_violation_fraction",
                "median",
            ),
            median_test_pet_violation=(
                "test_pet_violation_fraction",
                "median",
            ),
            median_best_epoch=(
                "best_epoch",
                "median",
            ),
            median_training_seconds=(
                "training_seconds",
                "median",
            ),
        )
    )

    summary[
        "std_validation_rmse_mm"
    ] = summary[
        "std_validation_rmse_mm"
    ].fillna(0.0)

    summary[
        "std_test_rmse_mm"
    ] = summary[
        "std_test_rmse_mm"
    ].fillna(0.0)

    family_order = {
        "Mean": 0,
        "Ridge": 1,
        "MLP": 2,
        "PINN_P": 3,
        "PINN_P_PET": 4,
    }

    summary["_family_order"] = (
        summary["model_family"]
        .map(family_order)
    )

    summary = (
        summary.sort_values(
            [
                "_family_order",
                "lambda_precipitation",
                "lambda_pet",
            ]
        )
        .drop(columns=["_family_order"])
        .reset_index(drop=True)
    )

    summary_path = (
        TABLE_DIR
        / "total_sm_model_sweep_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # Validation trade-off plot.
    neural_summary = summary.loc[
        summary["model_family"].isin(
            ["PINN_P", "PINN_P_PET"]
        )
    ].copy()

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.0, 4.5),
        constrained_layout=True,
    )

    for family in [
        "PINN_P",
        "PINN_P_PET",
    ]:
        family_data = (
            neural_summary.loc[
                neural_summary[
                    "model_family"
                ] == family
            ]
            .sort_values(
                "lambda_precipitation"
            )
        )

        axes[0].plot(
            family_data[
                "lambda_precipitation"
            ],
            family_data[
                "median_validation_rmse_mm"
            ],
            marker="o",
            linewidth=1.5,
            label=family,
        )

        axes[1].plot(
            family_data[
                "lambda_precipitation"
            ],
            family_data[
                "median_validation_p_violation"
            ],
            marker="o",
            linewidth=1.5,
            label=f"{family}: precipitation",
        )

        if family == "PINN_P_PET":
            axes[1].plot(
                family_data[
                    "lambda_precipitation"
                ],
                family_data[
                    "median_validation_pet_violation"
                ],
                marker="s",
                linewidth=1.3,
                linestyle="--",
                label=f"{family}: PET",
            )

    axes[0].set_xscale("log")
    axes[0].set_xlabel(
        "Physics penalty weight"
    )
    axes[0].set_ylabel(
        "Median validation RMSE (mm)"
    )
    axes[0].set_title(
        "Predictive accuracy"
    )

    axes[1].set_xscale("log")
    axes[1].set_xlabel(
        "Physics penalty weight"
    )
    axes[1].set_ylabel(
        "Median gradient-violation fraction"
    )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title(
        "Physical consistency"
    )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )
        axis.legend(
            frameon=True,
            fontsize=8,
        )

    figure.suptitle(
        "Accuracy–physics trade-off for total "
        "soil-moisture prediction",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig18_total_sm_physics_weight_sweep.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig18_total_sm_physics_weight_sweep.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)

    summary_json = {
        "status": "PASS",
        "device": str(device),
        "dataset": str(DATA_PATH),
        "target": TARGET_COLUMN,
        "feature_count": len(FEATURE_COLUMNS),
        "precipitation_feature_count": (
            len(PRECIPITATION_FEATURES)
        ),
        "split_sizes": split_sizes,
        "ridge_alpha_grid": (
            RIDGE_ALPHA_GRID
        ),
        "selected_ridge_alpha": float(
            best_ridge_alpha
        ),
        "neural_seeds": NEURAL_SEEDS,
        "physics_lambdas": PHYSICS_LAMBDAS,
        "hidden_dimensions": (
            HIDDEN_DIMENSIONS
        ),
        "maximum_epochs": (
            MAXIMUM_EPOCHS
        ),
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "selection_status": (
            "PENDING_REVIEW_OF_VALIDATION_ACCURACY_"
            "AND_GRADIENT_VIOLATIONS"
        ),
        "all_runs": str(runs_path),
        "summary_table": str(
            summary_path
        ),
        "predictions": str(
            predictions_path
        ),
        "model_directory": str(
            MODEL_DIR
        ),
        "scalers": str(scaler_path),
    }

    summary_json_path = (
        TABLE_DIR
        / "total_sm_model_sweep_metadata.json"
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            summary_json,
            stream,
            indent=2,
        )

    print()
    print("=" * 78)
    print("MODEL-SWEEP SUMMARY")
    print("=" * 78)

    display_columns = [
        "model_family",
        "lambda_precipitation",
        "lambda_pet",
        "run_count",
        "median_validation_rmse_mm",
        "median_test_rmse_mm",
        "median_test_mae_mm",
        "median_test_r_squared",
        "median_core_test_rmse_mm",
        "median_validation_p_violation",
        "median_test_p_violation",
        "median_validation_pet_violation",
        "median_test_pet_violation",
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
    print(f"Selected Ridge alpha : {best_ridge_alpha}")
    print(f"All-run table        : {runs_path}")
    print(f"Summary table        : {summary_path}")
    print(f"Predictions          : {predictions_path}")
    print(f"Model directory      : {MODEL_DIR}")
    print(
        f"Trade-off figure     : "
        f"{FIGURE_DIR / 'fig18_total_sm_physics_weight_sweep.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
