"""
Data Engineering & Cleansing Pipeline
Dataset: Apartment Rental Offers in Germany (immo_data.csv)
Target : baseRent (regression)

Pipeline:
  1. Load raw data
  2. Drop columns (>70% missing + user-specified)
  3. Filter extreme / suspicious values
  4. Remove duplicates
  5. Impute missing values (median for numeric, 'Unknown' for categorical)
  6. Apply skewness transformations (Yeo-Johnson, Box-Cox, log1p)
  7. Encode categorical variables (One-Hot, Ordinal, Target Encoding)
  8. Scale numeric features (RobustScaler, StandardScaler)
  9. Save curated dataset

Note: Target Encoding and Scaling are fit on the full dataset here.
      For production ML, fit these within cross-validation to prevent data leakage.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Dataset", "immo_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Dataset", "immo_data_curated.csv")

TARGET = "baseRent"

# --- Columns to drop --------------------------------------------------------

COLS_TO_DROP = [
    # >70% missing
    "telekomHybridUploadSpeed",   # 83.25% missing
    "electricityBasePrice",       # 82.58% missing
    "electricityKwhPrice",        # 82.58% missing
    "energyEfficiencyClass",      # 71.07% missing
    # User-specified removals (mateus recomendou)
    "facilities",                 # Free-text, 189k unique, 19.69% missing
    "scoutId",                    # Row identifier — not predictive
    "houseNumber",                # High cardinality (5.5k), 26.42% missing
    "description",                # Free-text, 212k unique
    "baseRentRange",              # Derived from target — leakage risk
    "pricetrend",                 # User-specified removal
    "priceTrend",                 # Handle alternate casing
    "totalRent",                  # Highly correlated with target — leakage risk
]

# --- Skewness transformations ------------------------------------------------

YJ_COLS = [                       # Yeo-Johnson (handles zeros and negatives)
    "livingSpace", "serviceCharge", "noParkSpaces",
    "floor", "numberOfFloors", "heatingCosts", "picturecount",
]
BC_COLS = [                       # Box-Cox (strictly positive)
    "noRooms", "lastRefurbish", "thermalChar", "yearConstructed",
]
LOG1P_COLS = [                    # log1p (non-negative, moderate skew)
    "yearConstructedRange", "livingSpaceRange", "telekomUploadSpeed",
]

# --- Encoding groups ---------------------------------------------------------

OHE_COLS = [                      # One-Hot — low cardinality (≤10)
    "condition", "typeOfFlat", "date",
    "interiorQual", "telekomTvOffer", "petsAllowed",
]
ORDINAL_COLS = [                  # Ordinal — medium cardinality (11-30)
    "regio1", "geo_bln", "heatingType",
]
TARGET_ENC_COLS = [               # Target Encoding — high cardinality (>30)
    "streetPlain", "street", "regio3",
    "geo_krs", "regio2", "firingTypes",
]
TARGET_ENC_SMOOTHING = 10

# --- Scaling groups ----------------------------------------------------------

ROBUST_COLS = [                   # RobustScaler — variables with many outliers
    "serviceCharge", "picturecount", "telekomUploadSpeed",
    "yearConstructed", "noParkSpaces", "livingSpace", "noRooms",
    "thermalChar", "floor", "numberOfFloors", "heatingCosts",
    "lastRefurbish",
    # Target-encoded cols live on the baseRent scale → robust scaling
    "streetPlain", "street", "regio3", "geo_krs", "regio2", "firingTypes",
]
STANDARD_COLS = [                 # StandardScaler — approx. normal distributions
    "yearConstructedRange", "geo_plz", "noRoomsRange", "livingSpaceRange",
    # Ordinal-encoded cols — small integer range
    "regio1", "geo_bln", "heatingType",
]


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================


def load_data(path: str) -> pd.DataFrame:
    """Step 1: Load raw dataset."""
    df = pd.read_csv(path)
    print(f"[1/9] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Remove high-missing and user-specified columns."""
    existing = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[2/9] Dropped {len(existing)} columns -> {df.shape[1]} remaining")
    return df


def filter_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Remove rows with impossible values; null out placeholders."""
    n_before = len(df)

    # ---- Row-level filters (clearly invalid records) ----
    df = df[(df[TARGET] > 0) & (df[TARGET] < 9_999_999)]
    df = df[(df["livingSpace"] > 0) & (df["livingSpace"] < 1_000)]
    df = df[df["noRooms"] < 100]

    if "serviceCharge" in df.columns:
        df = df[(df["serviceCharge"] < 10_000) | df["serviceCharge"].isna()]
    if "heatingCosts" in df.columns:
        df = df[(df["heatingCosts"] < 5_000) | df["heatingCosts"].isna()]
    if "noParkSpaces" in df.columns:
        df = df[(df["noParkSpaces"] < 100) | df["noParkSpaces"].isna()]

    # ---- Sentinel / placeholder values -> NaN (keep the row) ----
    if "floor" in df.columns:
        df.loc[df["floor"] == 999, "floor"] = np.nan
    if "numberOfFloors" in df.columns:
        df.loc[df["numberOfFloors"] == 999, "numberOfFloors"] = np.nan
    if "yearConstructed" in df.columns:
        df.loc[df["yearConstructed"] < 1800, "yearConstructed"] = np.nan
    if "lastRefurbish" in df.columns:
        df.loc[
            (df["lastRefurbish"] < 1900) | (df["lastRefurbish"] > 2026),
            "lastRefurbish",
        ] = np.nan
    if "street" in df.columns:
        df.loc[df["street"] == "no_information", "street"] = np.nan

    print(
        f"[3/9] Filtered extremes: removed {n_before - len(df):,} rows"
        f" -> {len(df):,} remaining"
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Drop duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"[4/9] Removed {n_before - len(df):,} duplicates -> {len(df):,} remaining")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5: Median for numeric features, 'Unknown' for categorical."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if col != TARGET and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    print(f"[5/9] Imputed missing -> {df.isna().sum().sum()} NaN remaining")
    return df


def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Step 6: Correct skewness with Yeo-Johnson, Box-Cox, and log1p."""
    applied = []

    # Yeo-Johnson — handles zeros and negatives
    for col in YJ_COLS:
        if col not in df.columns or not df[col].notna().all():
            continue
        try:
            # Validate data before transformation
            values = df[col].values
            if np.any(np.isinf(values)):
                print(f"  ⚠️  {col}: Contains infinities, skipping YJ")
                continue
            df[col], _ = yeojohnson(values)
            applied.append(f"{col}(YJ)")
        except Exception as e:
            print(f"  ⚠️  {col}: YJ failed ({type(e).__name__}), skipping")

    # Box-Cox — requires strictly positive values; fallback to YJ
    for col in BC_COLS:
        if col not in df.columns:
            continue
        try:
            if (df[col] > 0).all():
                values = df[col].values
                if np.any(np.isinf(values)):
                    print(f"  ⚠️  {col}: Contains infinities, skipping BC")
                    continue
                df[col], _ = boxcox(values)
                applied.append(f"{col}(BC)")
            else:
                # Fallback to YJ if non-positive values exist
                values = df[col].values
                if np.any(np.isinf(values)):
                    print(f"  ⚠️  {col}: Contains infinities, skipping YJ*")
                    continue
                df[col], _ = yeojohnson(values)
                applied.append(f"{col}(YJ*)")
        except Exception as e:
            print(f"  ⚠️  {col}: Transformation failed ({type(e).__name__}), skipping")

    # log1p — non-negative, moderate skew
    for col in LOG1P_COLS:
        if col not in df.columns or not (df[col] >= 0).all():
            continue
        try:
            values = df[col].values
            if np.any(np.isinf(values)):
                print(f"  ⚠️  {col}: Contains infinities, skipping log1p")
                continue
            df[col] = np.log1p(df[col].values)
            applied.append(f"{col}(log1p)")
        except Exception as e:
            print(f"  ⚠️  {col}: log1p failed ({type(e).__name__}), skipping")

    print(f"[6/9] Transformed {len(applied)} columns: {', '.join(applied)}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Step 7: One-Hot, Ordinal, and Target-encode categorical features."""
    # ---- One-Hot (low cardinality <= 10) ----
    ohe = [c for c in OHE_COLS if c in df.columns]
    if ohe:
        df = pd.get_dummies(df, columns=ohe, drop_first=True, dtype=int)

    # ---- Ordinal (medium cardinality 11-30) — sorted alphabetically ----
    for col in ORDINAL_COLS:
        if col in df.columns:
            cats = sorted(df[col].dropna().unique())
            df[col] = df[col].map({c: i for i, c in enumerate(cats)})

    # ---- Target Encoding with Bayesian smoothing (high cardinality) ----
    global_mean = df[TARGET].mean()
    for col in TARGET_ENC_COLS:
        if col not in df.columns:
            continue
        agg = df.groupby(col)[TARGET].agg(["mean", "count"])
        smoothed = (
            agg["count"] * agg["mean"] + TARGET_ENC_SMOOTHING * global_mean
        ) / (agg["count"] + TARGET_ENC_SMOOTHING)
        df[col] = df[col].map(smoothed)

    n_ord = sum(1 for c in ORDINAL_COLS if c in df.columns)
    n_te = sum(1 for c in TARGET_ENC_COLS if c in df.columns)
    print(
        f"[7/9] Encoded categoricals -> {df.shape[1]} columns "
        f"(OHE: {len(ohe)}, Ordinal: {n_ord}, TargetEnc: {n_te})"
    )
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 8: RobustScaler for outlier-heavy vars; StandardScaler otherwise."""
    n_robust = n_standard = 0

    # RobustScaler: (x - median) / IQR
    for col in ROBUST_COLS:
        if col not in df.columns:
            continue
        q1, median, q3 = df[col].quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            df[col] = (df[col] - median) / iqr
            n_robust += 1

    # StandardScaler: (x - mean) / std
    for col in STANDARD_COLS:
        if col not in df.columns:
            continue
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
            n_standard += 1

    print(f"[8/9] Scaled: RobustScaler({n_robust}), StandardScaler({n_standard})")
    return df


def save_curated(df: pd.DataFrame, path: str) -> None:
    """Step 9: Save curated dataset and verify project constraints."""
    # Final quality guardrail: enforce finite values before writing output
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)

    if df[numeric_cols].isna().sum().sum() > 0:
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(df.loc[:, numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0 and df[categorical_cols].isna().sum().sum() > 0:
        df.loc[:, categorical_cols] = df.loc[:, categorical_cols].fillna("Unknown")

    final_nan = int(df.isna().sum().sum())
    final_inf = int(np.isinf(df.select_dtypes(include=[np.number]).values).sum())
    if final_nan > 0 or final_inf > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            "Curated dataset still has non-finite values. "
            f"NaN={final_nan}, Inf={final_inf}, columns={nan_cols[:10]}"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\n[9/9] Saved -> {path}")
    print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  NaN    : {df.isna().sum().sum()}")
    print(
        f"  Cols>=30 : {'PASS' if df.shape[1] >= 30 else 'FAIL (' + str(df.shape[1]) + ')'}"
    )
    print(
        f"  Rows>=500: {'PASS' if df.shape[0] >= 500 else 'FAIL (' + str(df.shape[0]) + ')'}"
    )


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 70)
    print("DATA ENGINEERING & CLEANSING PIPELINE")
    print("Dataset: immo_data.csv  |  Target: baseRent (regression)")
    print("=" * 70 + "\n")

    df = load_data(INPUT_PATH)
    df = drop_columns(df)
    df = filter_extremes(df)
    df = remove_duplicates(df)
    df = impute_missing(df)
    df = apply_transformations(df)
    df = encode_categoricals(df)
    df = scale_features(df)
    save_curated(df, OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
