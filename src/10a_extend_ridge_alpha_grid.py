#!/usr/bin/env python3
"""
Check whether the Stage-10 Ridge alpha selection was truncated at alpha=100.

This loads the existing Stage-10 functions and extends the regularization
grid without repeating the coefficient bootstrap.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCRIPT = (
    ROOT
    / "src"
    / "10_component_lag_depth_attribution.py"
)

TABLE_DIR = ROOT / "results" / "tables"

EXTENDED_ALPHA_GRID = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
    300000.0,
]


def load_stage10_module():
    specification = (
        importlib.util.spec_from_file_location(
            "stage10_attribution",
            SOURCE_SCRIPT,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not load {SOURCE_SCRIPT}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    specification.loader.exec_module(
        module
    )

    return module


def main() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage10 = load_stage10_module()

    stage10.ALPHA_GRID = (
        EXTENDED_ALPHA_GRID
    )

    data = (
        pd.read_parquet(
            stage10.DATA_PATH
        )
        .sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
    )

    all_fold_tables = []
    all_alpha_tables = []
    decision_rows = []

    print("=" * 78)
    print("EXTENDED RIDGE ALPHA BOUNDARY CHECK")
    print("=" * 78)
    print(
        f"Extended alpha grid: "
        f"{EXTENDED_ALPHA_GRID}"
    )
    print()

    for target in stage10.TARGET_ORDER:
        (
            selected_alpha,
            fold_table,
            alpha_table,
        ) = stage10.select_ridge_alpha(
            data,
            target,
        )

        all_fold_tables.append(
            fold_table
        )

        all_alpha_tables.append(
            alpha_table
        )

        best_index = (
            alpha_table[
                "mean_validation_rmse_mm"
            ].idxmin()
        )

        best_row = alpha_table.loc[
            best_index
        ]

        selected_row = (
            alpha_table.loc[
                alpha_table["alpha"]
                == selected_alpha
            ]
            .iloc[0]
        )

        maximum_alpha = max(
            EXTENDED_ALPHA_GRID
        )

        boundary_hit = bool(
            selected_alpha
            == maximum_alpha
        )

        decision_rows.append(
            {
                "target": target,
                "depth_label": (
                    stage10.TARGETS[target]
                ),
                "minimum_mean_rmse_alpha": float(
                    best_row["alpha"]
                ),
                "minimum_mean_validation_rmse_mm": float(
                    best_row[
                        "mean_validation_rmse_mm"
                    ]
                ),
                "selected_one_standard_error_alpha": float(
                    selected_alpha
                ),
                "selected_alpha_mean_validation_rmse_mm": float(
                    selected_row[
                        "mean_validation_rmse_mm"
                    ]
                ),
                "selected_alpha_median_validation_rmse_mm": float(
                    selected_row[
                        "median_validation_rmse_mm"
                    ]
                ),
                "selected_alpha_at_upper_boundary": (
                    boundary_hit
                ),
            }
        )

        print(
            f"{stage10.TARGETS[target]:>10s}: "
            f"best-mean alpha="
            f"{float(best_row['alpha']):g}, "
            f"one-SE alpha="
            f"{selected_alpha:g}, "
            f"boundary={boundary_hit}"
        )

    fold_output = pd.concat(
        all_fold_tables,
        ignore_index=True,
    )

    alpha_output = pd.concat(
        all_alpha_tables,
        ignore_index=True,
    )

    decision_output = pd.DataFrame(
        decision_rows
    )

    fold_path = (
        TABLE_DIR
        / "ridge_alpha_extended_cv_folds.csv"
    )

    alpha_path = (
        TABLE_DIR
        / "ridge_alpha_extended_selection.csv"
    )

    decision_path = (
        TABLE_DIR
        / "ridge_alpha_extended_decision.csv"
    )

    metadata_path = (
        TABLE_DIR
        / "ridge_alpha_extended_metadata.json"
    )

    fold_output.to_csv(
        fold_path,
        index=False,
    )

    alpha_output.to_csv(
        alpha_path,
        index=False,
    )

    decision_output.to_csv(
        decision_path,
        index=False,
    )

    boundary_targets = (
        decision_output.loc[
            decision_output[
                "selected_alpha_at_upper_boundary"
            ],
            "target",
        ].tolist()
    )

    metadata = {
        "status": (
            "BOUNDARY_REMAINS"
            if boundary_targets
            else "PASS"
        ),
        "extended_alpha_grid": (
            EXTENDED_ALPHA_GRID
        ),
        "targets_still_at_upper_boundary": (
            boundary_targets
        ),
        "decision_table": str(
            decision_path
        ),
        "alpha_selection_table": str(
            alpha_path
        ),
        "fold_table": str(
            fold_path
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Extended alpha decisions:")
    print(
        decision_output.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.5f}"
            ),
        )
    )

    print()
    print(f"Status         : {metadata['status']}")
    print(f"Decision table : {decision_path}")
    print(f"Metadata       : {metadata_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
