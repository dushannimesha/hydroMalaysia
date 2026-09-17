#!/usr/bin/env python3
"""
Geographic cross-validation for four soil-moisture layers.

Targets
-------
- 0-10 cm
- 10-40 cm
- 40-100 cm
- 100-200 cm

Models
------
- Mean baseline
- Ridge regression
- Five-seed MLP
- Five-seed precipitation-constrained PINN

For outer test fold f:
    test       = f
    validation = (f + 1) mod 5
    training   = remaining three geographic folds

Feature and target scalers are fitted using training data only.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

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

MODEL_DIR = (
    ROOT
    / "results"
    / "models"
    / "stage09_layers"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

TARGETS = {
    "SM_0_10cm_mm": "0–10 cm",
    "SM_10_40cm_mm": "10–40 cm",
    "SM_40_100cm_mm": "40–100 cm",
    "SM_100_200cm_mm": "100–200 cm",
}

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

FOLD_COUNT = 5

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
        "family": "MLP",
        "lambda_precipitation": 0.0,
    },
    {
        "configuration": "PINN_P_lambda10",
        "family": "PINN_P",
        "lambda_precipitation": 10.0,
    },
]

HIDDEN_DIMENSIONS = [64, 64]
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAXIMUM_EPOCHS = 2500
EARLY_STOPPING_PATIENCE = 200

GRADIENT_TOLERANCE = 1e-5


class SoilMoistureMLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dimension,
                HIDDEN_DIMENSIONS[0],
            ),
            nn.Tanh(),
            nn.Linear(
                HIDDEN_DIMENSIONS[0],
                HIDDEN_DIMENSIONS[1],
            ),
            nn.Tanh(),
            nn.Linear(
                HIDDEN_DIMENSIONS[1],
                1,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(
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

    if truth.size == 0 or prediction.size == 0:
        return {
            "rmse_mm": float("nan"),
            "mae_mm": float("nan"),
            "r_squared": float("nan"),
            "bias_mm": float("nan"),
        }

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
        "bias_mm": float(
            np.mean(prediction - truth)
        ),
    }


def inverse_target(
    values: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    return (
        scaler.inverse_transform(
            np.asarray(values).reshape(-1, 1)
        )
        .reshape(-1)
    )


def train_neural_model(
    *,
    family: str,
    lambda_precipitation: float,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    precipitation_indices: list[int],
    device: torch.device,
) -> tuple[nn.Module, dict[str, float | int]]:
    set_seed(seed)

    model = SoilMoistureMLP(
        x_train.shape[1]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.MSELoss()

    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
    ):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if family == "MLP":
            prediction = model(x_train)

            data_loss = loss_function(
                prediction,
                y_train,
            )

            physics_penalty = torch.zeros(
                (),
                device=device,
            )

        else:
            physics_inputs = (
                x_train.detach()
                .clone()
                .requires_grad_(True)
            )

            prediction = model(
                physics_inputs
            )

            data_loss = loss_function(
                prediction,
                y_train,
            )

            gradients = torch.autograd.grad(
                outputs=prediction.sum(),
                inputs=physics_inputs,
                create_graph=True,
                retain_graph=True,
            )[0]

            precipitation_gradients = (
                gradients[
                    :,
                    precipitation_indices,
                ]
            )

            physics_penalty = (
                torch.relu(
                    -precipitation_gradients
                )
                .pow(2)
                .mean()
            )

        total_loss = (
            data_loss
            + lambda_precipitation
            * physics_penalty
        )

        total_loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_prediction = model(
                x_validation
            )

            validation_loss = (
                loss_function(
                    validation_prediction,
                    y_validation,
                ).item()
            )

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
            "No valid neural checkpoint was produced."
        )

    model.load_state_dict(best_state)
    model.eval()

    return (
        model,
        {
            "best_epoch": int(best_epoch),
            "epochs_completed": int(epoch),
            "training_seconds": float(
                time.time() - start_time
            ),
        },
    )


def neural_prediction(
    model: nn.Module,
    inputs: torch.Tensor,
    target_scaler: StandardScaler,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        standardized = (
            model(inputs)
            .detach()
            .cpu()
            .numpy()
        )

    return inverse_target(
        standardized,
        target_scaler,
    )


def precipitation_violation_fraction(
    model: nn.Module,
    inputs: torch.Tensor,
    precipitation_indices: list[int],
) -> float:
    model.eval()

    diagnostic_inputs = (
        inputs.detach()
        .clone()
        .requires_grad_(True)
    )

    predictions = model(
        diagnostic_inputs
    )

    gradients = torch.autograd.grad(
        outputs=predictions.sum(),
        inputs=diagnostic_inputs,
        create_graph=False,
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

    return float(
        np.mean(
            precipitation_gradients
            < -GRADIENT_TOLERANCE
        )
    )


def append_predictions(
    rows: list[dict[str, object]],
    *,
    frame: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
    target: str,
    configuration: str,
    family: str,
    seed: int,
    test_fold: int,
) -> None:
    frame = frame.reset_index(drop=True)

    for index, row in frame.iterrows():
        rows.append(
            {
                "target": target,
                "depth_label": TARGETS[target],
                "configuration": configuration,
                "model_family": family,
                "seed": seed,
                "test_fold": test_fold,
                "region_id": row["region_id"],
                "month": int(row["month"]),
                "native_cell_count": int(
                    row["native_cell_count"]
                ),
                "core_region_ge_4_cells": bool(
                    row[
                        "core_region_ge_4_cells"
                    ]
                ),
                "observed_mm": float(
                    truth[index]
                ),
                "predicted_mm": float(
                    prediction[index]
                ),
            }
        )


def build_ensemble_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "target",
        "depth_label",
        "configuration",
        "model_family",
        "test_fold",
        "region_id",
        "month",
        "native_cell_count",
        "core_region_ge_4_cells",
    ]

    ensemble = (
        predictions.groupby(
            group_columns,
            as_index=False,
            observed=True,
        )
        .agg(
            observed_mm=(
                "observed_mm",
                "first",
            ),
            ensemble_prediction_mm=(
                "predicted_mm",
                "mean",
            ),
            prediction_std_mm=(
                "predicted_mm",
                "std",
            ),
            seed_count=(
                "seed",
                "nunique",
            ),
        )
    )

    ensemble[
        "prediction_std_mm"
    ] = ensemble[
        "prediction_std_mm"
    ].fillna(0.0)

    ensemble["residual_mm"] = (
        ensemble["observed_mm"]
        - ensemble[
            "ensemble_prediction_mm"
        ]
    )

    return ensemble


def build_summary(
    ensemble: pd.DataFrame,
    violation_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        target,
        configuration,
    ), group in ensemble.groupby(
        ["target", "configuration"],
        observed=True,
    ):
        all_metrics = metrics(
            group["observed_mm"],
            group["ensemble_prediction_mm"],
        )

        core = group.loc[
            group["core_region_ge_4_cells"]
        ]

        core_metrics = metrics(
            core["observed_mm"],
            core["ensemble_prediction_mm"],
        )

        target_mean = float(
            group["observed_mm"].mean()
        )

        target_standard_deviation = float(
            group["observed_mm"].std(
                ddof=0
            )
        )

        violation = violation_table.loc[
            (
                violation_table["target"]
                == target
            )
            & (
                violation_table[
                    "configuration"
                ]
                == configuration
            ),
            "median_test_p_violation",
        ]

        violation_value = (
            float(violation.iloc[0])
            if len(violation) == 1
            else float("nan")
        )

        rows.append(
            {
                "target": target,
                "depth_label": TARGETS[target],
                "configuration": configuration,
                "ensemble_seed_count": int(
                    group["seed_count"].iloc[0]
                ),
                "ensemble_rmse_mm": (
                    all_metrics["rmse_mm"]
                ),
                "ensemble_mae_mm": (
                    all_metrics["mae_mm"]
                ),
                "ensemble_r_squared": (
                    all_metrics["r_squared"]
                ),
                "ensemble_bias_mm": (
                    all_metrics["bias_mm"]
                ),
                "rmse_divided_by_target_mean": (
                    all_metrics["rmse_mm"]
                    / target_mean
                ),
                "rmse_divided_by_target_std": (
                    all_metrics["rmse_mm"]
                    / target_standard_deviation
                ),
                "core_ensemble_rmse_mm": (
                    core_metrics["rmse_mm"]
                ),
                "core_ensemble_r_squared": (
                    core_metrics["r_squared"]
                ),
                "mean_prediction_std_mm": float(
                    group[
                        "prediction_std_mm"
                    ].mean()
                ),
                "median_test_p_violation": (
                    violation_value
                ),
            }
        )

    summary = pd.DataFrame(rows)

    best_by_target = (
        summary.groupby(
            "target",
            observed=True,
        )["ensemble_rmse_mm"]
        .transform("min")
    )

    summary[
        "accuracy_gap_from_best_percent"
    ] = (
        100.0
        * (
            summary["ensemble_rmse_mm"]
            - best_by_target
        )
        / best_by_target
    )

    order = {
        "Mean": 0,
        "Ridge": 1,
        "MLP": 2,
        "PINN_P_lambda10": 3,
    }

    summary["_order"] = (
        summary["configuration"]
        .map(order)
    )

    return (
        summary.sort_values(
            ["target", "_order"]
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def make_figure(
    summary: pd.DataFrame,
) -> None:
    plotting = summary.loc[
        summary["configuration"].isin(
            [
                "Ridge",
                "MLP",
                "PINN_P_lambda10",
            ]
        )
    ].copy()

    targets = list(TARGETS.keys())

    configurations = [
        "Ridge",
        "MLP",
        "PINN_P_lambda10",
    ]

    positions = np.arange(
        len(targets)
    )

    width = 0.25

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.5, 5.0),
        constrained_layout=True,
    )

    for index, configuration in enumerate(
        configurations
    ):
        configuration_data = (
            plotting.loc[
                plotting["configuration"]
                == configuration
            ]
            .set_index("target")
            .loc[targets]
        )

        offset = (
            index
            - (len(configurations) - 1) / 2
        ) * width

        axes[0].bar(
            positions + offset,
            configuration_data[
                "ensemble_rmse_mm"
            ],
            width=width,
            label=configuration,
        )

        axes[1].bar(
            positions + offset,
            configuration_data[
                "ensemble_r_squared"
            ],
            width=width,
            label=configuration,
        )

    depth_labels = [
        TARGETS[target]
        for target in targets
    ]

    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(
        depth_labels
    )

    axes[0].set_ylabel(
        "Geographic-CV ensemble RMSE (mm)"
    )

    axes[0].set_xlabel(
        "Soil depth"
    )

    axes[0].set_title(
        "Layer prediction error"
    )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        depth_labels
    )

    axes[1].set_ylabel(
        r"Geographic-CV ensemble $R^2$"
    )

    axes[1].set_xlabel(
        "Soil depth"
    )

    axes[1].set_title(
        "Layer explained variance"
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
        "Out-of-region soil-moisture prediction by depth",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig21_layer_spatial_cv_models.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig21_layer_spatial_cv_models.pdf",
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

    data = pd.read_parquet(
        DATA_PATH
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    precipitation_indices = [
        FEATURE_COLUMNS.index(feature)
        for feature in PRECIPITATION_FEATURES
    ]

    run_rows = []
    prediction_rows = []

    print("=" * 78)
    print("SOIL-MOISTURE LAYER GEOGRAPHIC CROSS-VALIDATION")
    print("=" * 78)
    print(f"Device            : {device}")
    print(f"Targets           : {list(TARGETS.values())}")
    print(f"Neural seeds      : {NEURAL_SEEDS}")
    print()

    for target, depth_label in TARGETS.items():
        print(f"Target: {target} ({depth_label})")

        for test_fold in range(FOLD_COUNT):
            validation_fold = (
                test_fold + 1
            ) % FOLD_COUNT

            train_mask = (
                (
                    data["spatial_fold_5"]
                    != test_fold
                )
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

            splits = {
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

            feature_scaler = StandardScaler()
            target_scaler = StandardScaler()

            feature_scaler.fit(
                splits["train"][
                    FEATURE_COLUMNS
                ].to_numpy(dtype=np.float32)
            )

            target_scaler.fit(
                splits["train"][
                    [target]
                ].to_numpy(dtype=np.float32)
            )

            arrays = {}
            tensors = {}

            for split_name, frame in splits.items():
                x_raw = frame[
                    FEATURE_COLUMNS
                ].to_numpy(
                    dtype=np.float32
                )

                y_raw = frame[
                    target
                ].to_numpy(
                    dtype=np.float32
                )

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

                arrays[split_name] = {
                    "x_raw": x_raw,
                    "x_scaled": x_scaled,
                    "y_raw": y_raw,
                    "y_scaled": y_scaled,
                }

                tensors[split_name] = {
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

            # Mean baseline
            mean_model = DummyRegressor(
                strategy="mean"
            )

            mean_model.fit(
                arrays["train"]["x_raw"],
                arrays["train"]["y_raw"],
            )

            mean_prediction = (
                mean_model.predict(
                    arrays["test"]["x_raw"]
                )
            )

            mean_metrics = metrics(
                arrays["test"]["y_raw"],
                mean_prediction,
            )

            run_rows.append(
                {
                    "target": target,
                    "depth_label": depth_label,
                    "configuration": "Mean",
                    "model_family": "Mean",
                    "seed": -1,
                    "test_fold": test_fold,
                    "validation_fold": (
                        validation_fold
                    ),
                    "selected_ridge_alpha": np.nan,
                    "test_rmse_mm": (
                        mean_metrics["rmse_mm"]
                    ),
                    "test_r_squared": (
                        mean_metrics["r_squared"]
                    ),
                    "test_p_violation": np.nan,
                }
            )

            append_predictions(
                prediction_rows,
                frame=splits["test"],
                truth=arrays["test"]["y_raw"],
                prediction=mean_prediction,
                target=target,
                configuration="Mean",
                family="Mean",
                seed=-1,
                test_fold=test_fold,
            )

            # Ridge
            best_ridge = None
            best_alpha = None
            best_validation_rmse = float(
                "inf"
            )

            for alpha in RIDGE_ALPHA_GRID:
                ridge = Ridge(
                    alpha=alpha
                )

                ridge.fit(
                    arrays["train"][
                        "x_scaled"
                    ],
                    arrays["train"][
                        "y_scaled"
                    ],
                )

                validation_prediction = (
                    inverse_target(
                        ridge.predict(
                            arrays[
                                "validation"
                            ]["x_scaled"]
                        ),
                        target_scaler,
                    )
                )

                validation_rmse = metrics(
                    arrays["validation"][
                        "y_raw"
                    ],
                    validation_prediction,
                )["rmse_mm"]

                if (
                    validation_rmse
                    < best_validation_rmse
                ):
                    best_validation_rmse = (
                        validation_rmse
                    )

                    best_alpha = alpha
                    best_ridge = copy.deepcopy(
                        ridge
                    )

            ridge_prediction = inverse_target(
                best_ridge.predict(
                    arrays["test"][
                        "x_scaled"
                    ]
                ),
                target_scaler,
            )

            ridge_metrics = metrics(
                arrays["test"]["y_raw"],
                ridge_prediction,
            )

            ridge_coefficients = np.asarray(
                best_ridge.coef_
            ).reshape(-1)

            ridge_violation = float(
                np.mean(
                    ridge_coefficients[
                        precipitation_indices
                    ]
                    < -GRADIENT_TOLERANCE
                )
            )

            run_rows.append(
                {
                    "target": target,
                    "depth_label": depth_label,
                    "configuration": "Ridge",
                    "model_family": "Ridge",
                    "seed": -1,
                    "test_fold": test_fold,
                    "validation_fold": (
                        validation_fold
                    ),
                    "selected_ridge_alpha": (
                        best_alpha
                    ),
                    "test_rmse_mm": (
                        ridge_metrics["rmse_mm"]
                    ),
                    "test_r_squared": (
                        ridge_metrics[
                            "r_squared"
                        ]
                    ),
                    "test_p_violation": (
                        ridge_violation
                    ),
                }
            )

            append_predictions(
                prediction_rows,
                frame=splits["test"],
                truth=arrays["test"]["y_raw"],
                prediction=ridge_prediction,
                target=target,
                configuration="Ridge",
                family="Ridge",
                seed=-1,
                test_fold=test_fold,
            )

            # Neural models
            for configuration in (
                NEURAL_CONFIGURATIONS
            ):
                for seed in NEURAL_SEEDS:
                    (
                        model,
                        training_information,
                    ) = train_neural_model(
                        family=configuration[
                            "family"
                        ],
                        lambda_precipitation=(
                            configuration[
                                "lambda_precipitation"
                            ]
                        ),
                        seed=seed,
                        x_train=tensors[
                            "train"
                        ]["x"],
                        y_train=tensors[
                            "train"
                        ]["y"],
                        x_validation=tensors[
                            "validation"
                        ]["x"],
                        y_validation=tensors[
                            "validation"
                        ]["y"],
                        precipitation_indices=(
                            precipitation_indices
                        ),
                        device=device,
                    )

                    test_prediction = (
                        neural_prediction(
                            model,
                            tensors["test"]["x"],
                            target_scaler,
                        )
                    )

                    test_metrics = metrics(
                        arrays["test"]["y_raw"],
                        test_prediction,
                    )

                    violation = (
                        precipitation_violation_fraction(
                            model,
                            tensors["test"]["x"],
                            precipitation_indices,
                        )
                    )

                    run_rows.append(
                        {
                            "target": target,
                            "depth_label": depth_label,
                            "configuration": (
                                configuration[
                                    "configuration"
                                ]
                            ),
                            "model_family": (
                                configuration[
                                    "family"
                                ]
                            ),
                            "seed": seed,
                            "test_fold": test_fold,
                            "validation_fold": (
                                validation_fold
                            ),
                            "selected_ridge_alpha": (
                                np.nan
                            ),
                            "test_rmse_mm": (
                                test_metrics[
                                    "rmse_mm"
                                ]
                            ),
                            "test_r_squared": (
                                test_metrics[
                                    "r_squared"
                                ]
                            ),
                            "test_p_violation": (
                                violation
                            ),
                            "best_epoch": (
                                training_information[
                                    "best_epoch"
                                ]
                            ),
                            "training_seconds": (
                                training_information[
                                    "training_seconds"
                                ]
                            ),
                        }
                    )

                    append_predictions(
                        prediction_rows,
                        frame=splits["test"],
                        truth=arrays[
                            "test"
                        ]["y_raw"],
                        prediction=(
                            test_prediction
                        ),
                        target=target,
                        configuration=(
                            configuration[
                                "configuration"
                            ]
                        ),
                        family=configuration[
                            "family"
                        ],
                        seed=seed,
                        test_fold=test_fold,
                    )

            print(
                f"  fold {test_fold}: complete"
            )

    runs = pd.DataFrame(run_rows)
    predictions = pd.DataFrame(
        prediction_rows
    )

    violation_table = (
        runs.groupby(
            [
                "target",
                "configuration",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            median_test_p_violation=(
                "test_p_violation",
                "median",
            )
        )
    )

    ensemble = build_ensemble_predictions(
        predictions
    )

    summary = build_summary(
        ensemble,
        violation_table,
    )

    fold_rows = []

    for (
        target,
        configuration,
        test_fold,
    ), group in ensemble.groupby(
        [
            "target",
            "configuration",
            "test_fold",
        ],
        observed=True,
    ):
        fold_metrics = metrics(
            group["observed_mm"],
            group[
                "ensemble_prediction_mm"
            ],
        )

        fold_rows.append(
            {
                "target": target,
                "depth_label": (
                    TARGETS[target]
                ),
                "configuration": (
                    configuration
                ),
                "test_fold": int(
                    test_fold
                ),
                "test_region_count": int(
                    group[
                        "region_id"
                    ].nunique()
                ),
                "test_sample_count": int(
                    len(group)
                ),
                "ensemble_fold_rmse_mm": (
                    fold_metrics["rmse_mm"]
                ),
                "ensemble_fold_r_squared": (
                    fold_metrics[
                        "r_squared"
                    ]
                ),
            }
        )

    fold_summary = pd.DataFrame(
        fold_rows
    )

    runs_path = (
        TABLE_DIR
        / "layer_spatial_cv_all_runs.csv"
    )

    predictions_path = (
        TABLE_DIR
        / "layer_spatial_cv_predictions.csv.gz"
    )

    ensemble_path = (
        TABLE_DIR
        / "layer_spatial_cv_ensemble_predictions.csv.gz"
    )

    summary_path = (
        TABLE_DIR
        / "layer_spatial_cv_summary.csv"
    )

    fold_path = (
        TABLE_DIR
        / "layer_spatial_cv_fold_metrics.csv"
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

    ensemble.to_csv(
        ensemble_path,
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    fold_summary.to_csv(
        fold_path,
        index=False,
    )

    make_figure(summary)

    metadata = {
        "status": "PASS",
        "targets": TARGETS,
        "features": FEATURE_COLUMNS,
        "fold_count": FOLD_COUNT,
        "neural_seeds": NEURAL_SEEDS,
        "neural_configurations": (
            NEURAL_CONFIGURATIONS
        ),
        "primary_comparison": (
            "Five-seed MLP ensemble versus "
            "five-seed precipitation-constrained "
            "PINN ensemble."
        ),
        "summary": str(
            summary_path
        ),
        "fold_metrics": str(
            fold_path
        ),
        "ensemble_predictions": str(
            ensemble_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "layer_spatial_cv_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("LAYER GEOGRAPHIC-CV SUMMARY")
    print("=" * 78)

    display_columns = [
        "depth_label",
        "configuration",
        "ensemble_seed_count",
        "ensemble_rmse_mm",
        "ensemble_mae_mm",
        "ensemble_r_squared",
        "rmse_divided_by_target_mean",
        "rmse_divided_by_target_std",
        "core_ensemble_rmse_mm",
        "median_test_p_violation",
        "accuracy_gap_from_best_percent",
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
    print(f"Summary table    : {summary_path}")
    print(f"Fold metrics     : {fold_path}")
    print(f"Predictions      : {ensemble_path}")
    print(
        f"Layer figure     : "
        f"{FIGURE_DIR / 'fig21_layer_spatial_cv_models.png'}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
