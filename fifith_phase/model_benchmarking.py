"""
Point 5 - Regression Benchmarking for baseRent

What this script does:
1. Loads Dataset/immo_data_curated.csv
2. Splits data into train/test (holdout)
3. Benchmarks multiple regression models with 5-fold CV on train only
4. Evaluates each model on test set
5. Repeats the benchmark with log1p(target) to compare robustness
6. Saves report-ready artifacts:
   - second_phase/benchmark_outputs/benchmark_raw_target.csv
   - second_phase/benchmark_outputs/benchmark_log_target.csv
   - second_phase/benchmark_outputs/benchmark_top_models.csv
   - second_phase/benchmark_outputs/benchmark_mae_comparison.png
   - second_phase/benchmark_outputs/residuals_top2.png

Important note:
- The curated dataset may already include target encoding/scaling computed before split.
  For strict leakage-safe research, fit preprocessing inside CV pipelines.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
TARGET = "baseRent"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "immo_data_curated.csv")
OUT_DIR = os.path.join(BASE_DIR, "second_phase", "benchmark_outputs")


# ============================================================
# MODEL ZOO
# ============================================================


def build_models() -> Dict[str, object]:
    """Create a diverse model set for fair benchmarking."""
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "DummyMedian": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=5000),
        "ElasticNet": ElasticNet(
            alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def sanitize_for_modeling(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """Make benchmark input fully finite without dropping feature columns."""
    X = X.copy()
    y = y.copy()

    # Diagnostics before sanitization
    nan_before = int(X.isna().sum().sum())
    numeric_before = X.select_dtypes(include=[np.number])
    inf_before = int(np.isinf(numeric_before.values).sum()) if not numeric_before.empty else 0

    # Target must be finite numeric
    y = pd.to_numeric(y, errors="coerce")
    valid_target_mask = np.isfinite(y.values)
    dropped_target_rows = int((~valid_target_mask).sum())
    X = X.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask].copy()

    # Replace inf in numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X.loc[:, numeric_cols] = X.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Convert bools to int
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        for col in bool_cols:
            X[col] = X[col].astype(int)

    # Keep all columns: encode non-numeric columns using stable category codes
    object_like_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in object_like_cols:
        as_str = X[col].astype("string").fillna("__MISSING__")
        X[col] = pd.Categorical(as_str).codes.astype(float)

    # Final numeric coercion without dropping columns
    X = X.apply(pd.to_numeric, errors="coerce")

    # Median imputation for any remaining missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    nan_after = int(X.isna().sum().sum())
    inf_after = int(np.isinf(X.values).sum())

    if nan_after > 0 or inf_after > 0:
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Sanitization failed: non-finite values remain. "
            f"NaN={nan_after}, Inf={inf_after}, columns={bad_cols[:10]}"
        )

    stats = {
        "nan_before": nan_before,
        "inf_before": inf_before,
        "nan_after": nan_after,
        "inf_after": inf_after,
        "target_rows_removed": dropped_target_rows,
        "encoded_object_cols": len(object_like_cols),
    }
    return X, y, stats


def to_strict_finite_matrix(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sanitized pandas objects to strict float64 arrays with finite values only."""
    Xv = X.to_numpy(dtype=np.float64, copy=True)
    yv = y.to_numpy(dtype=np.float64, copy=True)

    # Repair any unexpected non-finite feature values defensively
    if not np.isfinite(Xv).all():
        col_medians = np.nanmedian(np.where(np.isfinite(Xv), Xv, np.nan), axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)

        nan_rows, nan_cols = np.where(~np.isfinite(Xv))
        for r, c in zip(nan_rows, nan_cols):
            Xv[r, c] = col_medians[c]

    # Repair any unexpected non-finite target values defensively
    if not np.isfinite(yv).all():
        finite_mask = np.isfinite(yv)
        y_median = float(np.median(yv[finite_mask])) if finite_mask.any() else 0.0
        yv = np.where(np.isfinite(yv), yv, y_median)

    if not np.isfinite(Xv).all() or not np.isfinite(yv).all():
        raise ValueError("Strict finite conversion failed: X or y still contains non-finite values.")

    return Xv, yv


@dataclass
class BenchmarkResult:
    model: str
    target_mode: str
    cv_mae_mean: float
    cv_mae_std: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_medae_mean: float
    cv_r2_mean: float
    test_mae: float
    test_rmse: float
    test_medae: float
    test_r2: float


@dataclass
class BenchmarkError:
    model: str
    target_mode: str
    error_type: str
    error_message: str


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics in a single place for consistency."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def run_single_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_mode: str,
) -> BenchmarkResult:
    """Run CV + holdout test for one model."""
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "medae": "neg_median_absolute_error",
        "r2": "r2",
    }

    cv_res = cross_validate(
        clone(model),
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )

    cv_mae = -cv_res["test_mae"]
    cv_rmse = -cv_res["test_rmse"]
    cv_medae = -cv_res["test_medae"]
    cv_r2 = cv_res["test_r2"]

    fitted = clone(model)
    fitted.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)

    if target_mode == "log1p":
        y_pred = np.expm1(y_pred)
        y_test_eval = np.expm1(y_test)
    else:
        y_test_eval = y_test

    test_metrics = regression_scores(y_test_eval, y_pred)

    return BenchmarkResult(
        model=model_name,
        target_mode=target_mode,
        cv_mae_mean=float(np.mean(cv_mae)),
        cv_mae_std=float(np.std(cv_mae)),
        cv_rmse_mean=float(np.mean(cv_rmse)),
        cv_rmse_std=float(np.std(cv_rmse)),
        cv_medae_mean=float(np.mean(cv_medae)),
        cv_r2_mean=float(np.mean(cv_r2)),
        test_mae=float(test_metrics["mae"]),
        test_rmse=float(test_metrics["rmse"]),
        test_medae=float(test_metrics["medae"]),
        test_r2=float(test_metrics["r2"]),
    )


def benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all models and return a sorted benchmark table."""
    models = build_models()
    rows: List[BenchmarkResult] = []
    errors: List[BenchmarkError] = []

    print(f"\nBenchmark target mode: {target_mode}")
    print("-" * 70)

    for name, model in models.items():
        try:
            result = run_single_model(name, model, X_train, y_train, X_test, y_test, target_mode)
            rows.append(result)
            print(
                f"{name:20s} | CV MAE: {result.cv_mae_mean:8.2f} | "
                f"CV RMSE: {result.cv_rmse_mean:8.2f} | Test MAE: {result.test_mae:8.2f}"
            )
        except Exception as exc:
            errors.append(
                BenchmarkError(
                    model=name,
                    target_mode=target_mode,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            msg = str(exc).replace("\n", " ")
            print(f"{name:20s} | FAILED ({type(exc).__name__}): {msg[:160]}")

    if rows:
        df_res = pd.DataFrame([r.__dict__ for r in rows])
        df_res = df_res.sort_values(["cv_mae_mean", "cv_rmse_mean"], ascending=True)
        df_res["rank"] = np.arange(1, len(df_res) + 1)
    else:
        df_res = pd.DataFrame(
            columns=[
                "model", "target_mode", "cv_mae_mean", "cv_mae_std", "cv_rmse_mean", "cv_rmse_std",
                "cv_medae_mean", "cv_r2_mean", "test_mae", "test_rmse", "test_medae", "test_r2", "rank"
            ]
        )

    df_err = pd.DataFrame([e.__dict__ for e in errors]) if errors else pd.DataFrame(
        columns=["model", "target_mode", "error_type", "error_message"]
    )
    return df_res, df_err


def plot_mae_comparison(df_raw: pd.DataFrame, df_log: pd.DataFrame, out_path: str) -> None:
    """Plot CV MAE comparison between raw target and log target approaches."""
    merged = df_raw[["model", "cv_mae_mean"]].merge(
        df_log[["model", "cv_mae_mean"]], on="model", suffixes=("_raw", "_log")
    )
    merged = merged.sort_values("cv_mae_mean_raw")

    plt.figure(figsize=(13, 7))
    x = np.arange(len(merged))
    width = 0.4
    plt.bar(x - width / 2, merged["cv_mae_mean_raw"], width=width, label="Raw target")
    plt.bar(x + width / 2, merged["cv_mae_mean_log"], width=width, label="log1p target")
    plt.xticks(x, merged["model"], rotation=35, ha="right")
    plt.ylabel("CV MAE (EUR)")
    plt.title("Benchmark Comparison - CV MAE by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_residuals_top2(
    top_models: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train_raw: pd.Series,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    out_path: str,
) -> None:
    """Create residual plots for the top 2 models from raw-target ranking."""
    models = build_models()
    top2 = top_models.head(2)["model"].tolist()

    if len(top2) == 0:
        return

    n_plots = len(top2)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for i, model_name in enumerate(top2):
        model = clone(models[model_name])
        model.fit(X_train, y_train_raw)
        pred = model.predict(X_test)
        residuals = y_test_raw - pred

        sns.scatterplot(x=pred, y=residuals, s=12, alpha=0.5, ax=axes[i])
        axes[i].axhline(0.0, color="red", linestyle="--", linewidth=1)
        axes[i].set_title(f"Residuals - {model_name}")
        axes[i].set_xlabel("Predicted baseRent")
        axes[i].set_ylabel("Residual (actual - predicted)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("POINT 5 - MODEL BENCHMARKING")
    print("Dataset: immo_data_curated.csv | Target: baseRent")
    print("=" * 72)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Curated dataset not found at: {DATA_PATH}\n"
            "Run second_phase/data_cleansing_pipeline.py first."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in curated dataset.")

    X = df.drop(columns=[TARGET])
    y_raw = df[TARGET].astype(float)

    X, y_raw, sanitize_stats = sanitize_for_modeling(X, y_raw)
    X_values, y_values = to_strict_finite_matrix(X, y_raw)

    if y_values.min() <= -1:
        raise ValueError("baseRent has values <= -1, cannot apply log1p benchmark.")

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X_values,
        y_values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Rows: {len(df):,} | Features: {X_values.shape[1]}")
    print(f"Train/Test split: {len(X_train):,} / {len(X_test):,}")
    print(
        "Sanitization: "
        f"NaN {sanitize_stats['nan_before']}->{sanitize_stats['nan_after']}, "
        f"Inf {sanitize_stats['inf_before']}->{sanitize_stats['inf_after']}, "
        f"target rows removed={sanitize_stats['target_rows_removed']}, "
        f"object cols encoded={sanitize_stats['encoded_object_cols']}"
    )

    # Benchmark with raw target
    raw_results, raw_errors = benchmark_models(
        X_train=X_train,
        y_train=y_train_raw,
        X_test=X_test,
        y_test=y_test_raw,
        target_mode="raw",
    )

    # Benchmark with log1p target
    y_train_log = np.log1p(y_train_raw)
    y_test_log = np.log1p(y_test_raw)
    log_results, log_errors = benchmark_models(
        X_train=X_train,
        y_train=y_train_log,
        X_test=X_test,
        y_test=y_test_log,
        target_mode="log1p",
    )

    # Save tables
    raw_csv = os.path.join(OUT_DIR, "benchmark_raw_target.csv")
    log_csv = os.path.join(OUT_DIR, "benchmark_log_target.csv")
    top_csv = os.path.join(OUT_DIR, "benchmark_top_models.csv")
    err_csv = os.path.join(OUT_DIR, "benchmark_model_errors.csv")

    raw_results.to_csv(raw_csv, index=False)
    log_results.to_csv(log_csv, index=False)
    pd.concat([raw_errors, log_errors], ignore_index=True).to_csv(err_csv, index=False)

    top_summary = raw_results[["rank", "model", "cv_mae_mean", "cv_rmse_mean", "test_mae", "test_rmse", "test_r2"]].head(5).copy()
    top_summary.to_csv(top_csv, index=False)

    # Save plots
    mae_plot = os.path.join(OUT_DIR, "benchmark_mae_comparison.png")
    residual_plot = os.path.join(OUT_DIR, "residuals_top2.png")

    plot_mae_comparison(raw_results, log_results, mae_plot)
    plot_residuals_top2(raw_results, X_train, y_train_raw, X_test, y_test_raw, residual_plot)

    print("\n" + "=" * 72)
    print("BENCHMARK FINISHED")
    print("=" * 72)
    print(f"Saved: {raw_csv}")
    print(f"Saved: {log_csv}")
    print(f"Saved: {top_csv}")
    print(f"Saved: {err_csv}")
    print(f"Saved: {mae_plot}")
    print(f"Saved: {residual_plot}")

    print("\nTop 5 models (raw target, ranked by CV MAE):")
    print(top_summary.to_string(index=False))


if __name__ == "__main__":
    main()
