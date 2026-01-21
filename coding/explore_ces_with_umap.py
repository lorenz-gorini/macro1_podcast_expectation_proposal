# %% [markdown]
# # CES UMAP Analysis: Goal-Based Feature Sets (Strict Pipeline)
#
# A strict pipeline that:
# 1. Downloads ECB Consumer Expectations Survey (CES) data (fails fast on errors)
# 2. Uses exact column lists for three analysis goals (A=Expectations, B=Behavior, C=Broad)
# 3. Analyzes missingness patterns with strict thresholds
# 4. Runs UMAP dimensionality reduction with proper preprocessing
# 5. Visualizes embeddings over time using weights for marker size

# %% [markdown]
# ## 1. Imports + Constants

# %%
"""CES UMAP Analysis Pipeline with Strict Feature Selection."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import umap
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
DROP_THRESHOLD = 0.70  # Drop columns with >70% missing
IMPUTE_NUMERIC = False  # If True, use median imputation; else complete-case
YEARS = list(range(2021, 2025))  # Years to download
SAVE_DIR = "./ces_data"
RESULTS_DIR = "./results"
MIN_ROWS_FOR_UMAP = 500  # Minimum rows required after cleaning

# UMAP parameters
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# Marker size range for weight-based sizing
MARKER_SIZE_MIN = 4
MARKER_SIZE_MAX = 40

# Weight column (required)
WEIGHT_COL = "wgt"

# Wave column for time inference (required)
WAVE_COL = "a0030"

# Create directories
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

print("Configuration loaded:")
print(f"  DROP_THRESHOLD: {DROP_THRESHOLD}")
print(f"  IMPUTE_NUMERIC: {IMPUTE_NUMERIC}")
print(f"  YEARS: {YEARS}")
print(f"  WEIGHT_COL: {WEIGHT_COL}")
print(f"  WAVE_COL: {WAVE_COL}")

# %% [markdown]
# ## 2. Column Definitions (Exact Lists)

# %%
# =============================================================================
# EXACT COLUMN LISTS FROM USER'S DATASET
# =============================================================================


def expand_range(prefix: str, start: int, end: int) -> List[str]:
    """Generate column names like prefix_1, prefix_2, ..., prefix_end."""
    return [f"{prefix}_{i}" for i in range(start, end + 1)]


# All columns present in the combined dataset
COLUMNS_PRESENT = [
    "a0010",
    "a0020",
    "a0030",
    "a1010_age_prec",
    "b7040_quintile",
    "wgt",
    "c1010",
    "c1020",
    "c1110",
    "c1120",
    *expand_range("c1150", 1, 8),
    *expand_range("c1152", 1, 10),
    "c1210",
    "c1220",
    "c2110",
    "c2120",
    *expand_range("c2150", 1, 8),
    *expand_range("c2151", 1, 10),
    "c3010",
    "c3110",
    "c3210",
    "c3220",
    *expand_range("c3250", 1, 8),
    *expand_range("c3251", 1, 10),
    "c4010",
    "c4020",
    "c4030",
    "c4031",
    "c4032",
    "c5111",
    "c5113",
    "c6010",
    "c6020",
    "c6030",
    "c6110",
    "c6120",
    "c6130",
    "c7010",
    "c7110",
    "c7111",
    "c7120",
    "c7121",
    *expand_range("c8010", 1, 4),
    *expand_range("c8011", 1, 5),
    "e2010",
    "e2020",
    "emp_status",
    *expand_range("h2020", 1, 7),
    *expand_range("p1410", 1, 5),
    "t3010",
    "t3020_1",
    "t3020_2",
    "w1112_prec",
    *expand_range("x1040", 1, 9),
    *expand_range("x6020", 1, 9),
    "x8020",
    "x8110",
    "x8120",
]

# -----------------------------------------------------------------------------
# GOAL A: Expectations / Beliefs
# -----------------------------------------------------------------------------
GOAL_A_FEATURES = [
    # Inflation expectations/perceptions
    "c1010",
    "c1020",
    "c1110",
    "c1120",
    "c1210",
    "c1220",
    *expand_range("c1150", 1, 8),
    *expand_range("c1152", 1, 10),
    "e2010",
    "e2020",
    # House price expectations
    "c2110",
    "c2120",
    *expand_range("c2150", 1, 8),
    *expand_range("c2151", 1, 10),
    # Income/financial expectations
    "c3010",
    "c3110",
    "c3210",
    "c3220",
    *expand_range("c3250", 1, 8),
    *expand_range("c3251", 1, 10),
    # Unemployment/macro expectations
    "c4010",
    "c4020",
    "c4030",
    "c4031",
    "c4032",
    # Interest rate expectations
    "c5111",
    "c5113",
]

# -----------------------------------------------------------------------------
# GOAL B: Behavior / Constraints
# -----------------------------------------------------------------------------
GOAL_B_FEATURES = [
    # Spending behavior
    "c6010",
    "c6020",
    "c6030",
    "c6110",
    "c6120",
    "c6130",
    # Liquidity
    "c7010",
    # Credit access
    "c7110",
    "c7111",
    "c7120",
    "c7121",
    # Behavior-related (financial literacy/behavior)
    *expand_range("x6020", 1, 9),
]

# -----------------------------------------------------------------------------
# GOAL C: Broad Segmentation (A + B + demographics + trust)
# -----------------------------------------------------------------------------
GOAL_C_DEMOGRAPHICS = [
    "a1010_age_prec",
    "b7040_quintile",
    "emp_status",
    "w1112_prec",
]

GOAL_C_TRUST = expand_range("c8011", 1, 5)

GOAL_C_FEATURES = GOAL_A_FEATURES + GOAL_B_FEATURES + GOAL_C_DEMOGRAPHICS + GOAL_C_TRUST

# Deduplicate while preserving order
GOAL_C_FEATURES = list(dict.fromkeys(GOAL_C_FEATURES))

print(f"\nFeature counts:")
print(f"  Goal A (Expectations): {len(GOAL_A_FEATURES)} features")
print(f"  Goal B (Behavior): {len(GOAL_B_FEATURES)} features")
print(f"  Goal C (Broad): {len(GOAL_C_FEATURES)} features")


# %% [markdown]
# ## 3. Download/Cache CES Files (Strict)


# %%
def download_ces_data_cached(
    year: int, frequency: str = "monthly", save_dir: str = SAVE_DIR
) -> pd.DataFrame:
    """Download CES data with caching - fails if download fails.

    Args:
        year: Year of data to download.
        frequency: 'monthly' or 'quarterly'.
        save_dir: Directory to cache files.

    Returns:
        DataFrame with CES data.

    Raises:
        RuntimeError: If download fails or file is invalid.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(save_dir) / f"CES_data_{year}_{frequency}.csv"

    # Check cache first
    if filename.exists() and filename.stat().st_size > 0:
        print(f"  [CACHE] Loading {filename}")
        try:
            df = pd.read_csv(filename, low_memory=False)
            # Lowercase all column names
            df.columns = df.columns.str.lower().str.strip()
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to read cached file {filename}: {e}")

    # Download
    url = (
        "https://www.ecb.europa.eu/stats/ecb_surveys/consumer_exp_survey/shared/pdf/"
        f"ecb.CES_data_{year}_{frequency}.en.csv"
    )

    print(f"  [DOWNLOAD] Fetching {year}...")
    response = requests.get(url, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download CES data for {year}: HTTP {response.status_code}. "
            f"URL: {url}"
        )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"  [OK] Downloaded: {filename}")
    df = pd.read_csv(filename, low_memory=False)
    # Lowercase all column names
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_ces_data(
    years: List[int],
    frequency: str = "monthly",
    save_dir: str = SAVE_DIR,
) -> pd.DataFrame:
    """Download and concatenate CES datasets for multiple years.

    Args:
        years: List of years to download.
        frequency: 'monthly' or 'quarterly'.
        save_dir: Directory to cache files.

    Returns:
        Concatenated DataFrame with source_year column.

    Raises:
        RuntimeError: If any year fails to download.
    """
    print(f"\nLoading CES data for years: {years}")
    frames = []

    for year in years:
        df = download_ces_data_cached(year, frequency=frequency, save_dir=save_dir)
        df = df.copy()
        df["source_year"] = year
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n✓ Loaded {len(combined):,} rows, {len(combined.columns)} columns")
    return combined


# %%
# Load data
ces_df = load_ces_data(YEARS, frequency="monthly")
print(f"\nDataFrame shape: {ces_df.shape}")
print(f"Columns (first 20): {list(ces_df.columns[:20])}")


# %% [markdown]
# ## 4. Time Inference from a0030 (Strict)


# %%
def compute_month_ts_strict(df: pd.DataFrame, wave_col: str = WAVE_COL) -> pd.Series:
    """Compute month_ts from wave column a0030 using strict mapping.

    Wave 4 = 2020-04-01, each subsequent wave is +1 month.

    Args:
        df: CES DataFrame with wave column.
        wave_col: Name of wave column (must be 'a0030').

    Returns:
        Series of Period[M] timestamps.

    Raises:
        KeyError: If wave column is missing.
        ValueError: If wave values are invalid.
    """
    if wave_col not in df.columns:
        raise KeyError(
            f"Required wave column '{wave_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns[:30])}..."
        )

    wave_series = df[wave_col].copy()

    # Validate numeric
    if not pd.api.types.is_numeric_dtype(wave_series):
        raise ValueError(
            f"Wave column '{wave_col}' must be numeric, got {wave_series.dtype}"
        )

    # Check for NaN
    nan_count = wave_series.isna().sum()
    if nan_count > 0:
        print(
            f"  [WARN] {nan_count:,} NaN values in {wave_col}, will propagate to month_ts"
        )

    # Validate integer-valued (no fractional part)
    non_nan = wave_series.dropna()
    if len(non_nan) > 0:
        fractional = (non_nan != non_nan.round()).sum()
        if fractional > 0:
            raise ValueError(
                f"Wave column '{wave_col}' contains {fractional} non-integer values. "
                f"Sample: {non_nan[non_nan != non_nan.round()].head()}"
            )

    # Convert to integer (keeping NaN as NaT after date conversion)
    wave_int = wave_series.round().astype("Int64")  # Nullable integer

    # Canonical mapping: wave 4 = 2020-04-01
    # month_ts = 2020-04-01 + (wave - 4) months
    base_date = pd.Timestamp("2020-04-01")

    def wave_to_month(w):
        if pd.isna(w):
            return pd.NaT
        return base_date + pd.DateOffset(months=int(w - 4))

    month_ts = wave_int.apply(wave_to_month)
    month_ts = pd.to_datetime(month_ts).dt.to_period("M")

    # Validation
    unique_months = month_ts.dropna().nunique()
    min_wave = wave_int.min()
    max_wave = wave_int.max()
    min_month = month_ts.dropna().min()
    max_month = month_ts.dropna().max()
    n_years = df["source_year"].nunique() if "source_year" in df.columns else 1

    print(f"\nTime inference from '{wave_col}':")
    print(f"  Wave range: {min_wave} to {max_wave}")
    print(f"  Month range: {min_month} to {max_month}")
    print(f"  Unique months: {unique_months}")
    print(f"  Number of years: {n_years}")

    # Strict assertions
    if unique_months < 12:
        raise ValueError(
            f"Too few unique months ({unique_months}). Expected >=12 for multi-year data. "
            f"Check wave column '{wave_col}' mapping."
        )

    if unique_months <= n_years:
        raise ValueError(
            f"Unique months ({unique_months}) should be > number of years ({n_years}). "
            f"This suggests the time inference is wrong."
        )

    return month_ts


# %%
# Compute month_ts strictly from a0030
ces_df["month_ts"] = compute_month_ts_strict(ces_df, wave_col=WAVE_COL)
print(f"\n✓ month_ts computed successfully")


# %% [markdown]
# ## 5. Trust Variable Harmonization (c8010 → c8011)


# %%
def harmonize_trust_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize trust variables: fill c8011_k from c8010_k where needed.

    Canonical trust set is c8011_1..c8011_5.
    For k=1..4, if c8011_k is NA but c8010_k exists, fill from c8010_k.
    c8011_5 has no old counterpart.

    Args:
        df: CES DataFrame.

    Returns:
        DataFrame with harmonized trust columns.
    """
    df = df.copy()
    print("\nHarmonizing trust variables (c8010 → c8011):")

    for k in range(1, 5):  # k = 1, 2, 3, 4
        old_col = f"c8010_{k}"
        new_col = f"c8011_{k}"

        if old_col not in df.columns:
            print(f"  {old_col}: not present, skipping")
            continue

        if new_col not in df.columns:
            # Create new column from old
            df[new_col] = df[old_col]
            filled = df[new_col].notna().sum()
            print(f"  {old_col} → {new_col}: created ({filled:,} values)")
        else:
            # Fill NA in new from old
            na_before = df[new_col].isna().sum()
            df[new_col] = df[new_col].fillna(df[old_col])
            na_after = df[new_col].isna().sum()
            filled = na_before - na_after
            print(f"  {old_col} → {new_col}: filled {filled:,} NA values")

    # Check c8011_5 (no old counterpart)
    if "c8011_5" in df.columns:
        print(f"  c8011_5: present ({df['c8011_5'].notna().sum():,} non-NA values)")
    else:
        print("  c8011_5: not present")

    return df


# %%
# Apply trust harmonization
ces_df = harmonize_trust_variables(ces_df)


# %% [markdown]
# ## 6. Validate Required Columns


# %%
def validate_required_columns(
    df: pd.DataFrame, required: List[str], context: str = ""
) -> List[str]:
    """Validate that required columns exist, return list of present columns.

    Args:
        df: DataFrame to check.
        required: List of required column names.
        context: Context string for error message.

    Returns:
        List of columns that are present.

    Raises:
        KeyError: If any required columns are missing.
    """
    present = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise KeyError(
            f"{context}: Missing {len(missing)} required columns: {missing[:20]}"
            + ("..." if len(missing) > 20 else "")
        )

    return present


def validate_weight_column(df: pd.DataFrame, weight_col: str = WEIGHT_COL) -> None:
    """Validate weight column exists and has valid values.

    Args:
        df: DataFrame to check.
        weight_col: Name of weight column.

    Raises:
        KeyError: If weight column is missing.
        ValueError: If weight column has no valid values.
    """
    if weight_col not in df.columns:
        raise KeyError(
            f"Required weight column '{weight_col}' not found. "
            f"Available columns: {[c for c in df.columns if 'wgt' in c.lower()]}"
        )

    valid_count = df[weight_col].notna().sum()
    if valid_count == 0:
        raise ValueError(f"Weight column '{weight_col}' has no valid (non-NA) values")

    print(f"\n✓ Weight column '{weight_col}': {valid_count:,} valid values")


# %%
# Validate weight column
validate_weight_column(ces_df, WEIGHT_COL)


# %% [markdown]
# ## 7. GoalSpec and CESUMAPPipeline Classes


# %%
@dataclass
class GoalSpec:
    """Specification for a UMAP analysis goal."""

    name: str  # Short name: A, B, C
    title: str  # Descriptive title
    feature_cols: List[str]  # Exact list of feature columns
    required_cols: List[str] = field(default_factory=list)  # Columns that MUST exist


# Define goal specifications
GOAL_SPEC_A = GoalSpec(
    name="A",
    title="Expectations / Beliefs",
    feature_cols=GOAL_A_FEATURES,
    required_cols=["c1010", "c1020", "c3010"],  # Key columns that must exist
)

GOAL_SPEC_B = GoalSpec(
    name="B",
    title="Behavior / Constraints",
    feature_cols=GOAL_B_FEATURES,
    required_cols=["c6010", "c7010"],
)

GOAL_SPEC_C = GoalSpec(
    name="C",
    title="Broad Segmentation",
    feature_cols=GOAL_C_FEATURES,
    required_cols=["c1010", "c6010", "a1010_age_prec"],
)


class CESUMAPPipeline:
    """Pipeline for running UMAP on CES data with a specific goal."""

    def __init__(
        self,
        df: pd.DataFrame,
        spec: GoalSpec,
        weight_col: str = WEIGHT_COL,
        drop_threshold: float = DROP_THRESHOLD,
        impute: bool = IMPUTE_NUMERIC,
        min_rows: int = MIN_ROWS_FOR_UMAP,
    ):
        """Initialize pipeline.

        Args:
            df: Full CES DataFrame.
            spec: GoalSpec defining features.
            weight_col: Weight column name.
            drop_threshold: Drop columns with missing > threshold.
            impute: Whether to median-impute (False = complete-case).
            min_rows: Minimum rows required for UMAP.
        """
        self.df = df
        self.spec = spec
        self.weight_col = weight_col
        self.drop_threshold = drop_threshold
        self.impute = impute
        self.min_rows = min_rows

        # State
        self.X_raw: Optional[pd.DataFrame] = None
        self.X_clean: Optional[pd.DataFrame] = None
        self.X_scaled: Optional[np.ndarray] = None
        self.valid_idx: Optional[pd.Index] = None
        self.embedding: Optional[np.ndarray] = None
        self.diagnostics: Dict = {}

    def select_features(self) -> pd.DataFrame:
        """Select features based on GoalSpec.

        Returns:
            DataFrame with selected features.

        Raises:
            KeyError: If required columns are missing.
            RuntimeError: If too few features available.
        """
        print(f"\n{'='*60}")
        print(f"GOAL {self.spec.name}: {self.spec.title}")
        print(f"{'='*60}")

        # Check required columns
        if self.spec.required_cols:
            missing_required = [
                c for c in self.spec.required_cols if c not in self.df.columns
            ]
            if missing_required:
                raise KeyError(
                    f"Goal {self.spec.name}: Missing required columns: {missing_required}"
                )

        # Select available feature columns
        available = [c for c in self.spec.feature_cols if c in self.df.columns]
        missing = [c for c in self.spec.feature_cols if c not in self.df.columns]

        print(f"  Requested features: {len(self.spec.feature_cols)}")
        print(f"  Available features: {len(available)}")

        if missing:
            print(f"  Missing features ({len(missing)}): {missing[:10]}...")

        if len(available) < 5:
            raise RuntimeError(
                f"Goal {self.spec.name}: Only {len(available)} features available, need >=5"
            )

        self.X_raw = self.df[available].copy()
        return self.X_raw

    def analyze_missingness(self, show_plots: bool = True) -> pd.DataFrame:
        """Analyze and handle missing values.

        Args:
            show_plots: Whether to display diagnostic plots.

        Returns:
            Cleaned DataFrame.

        Raises:
            RuntimeError: If too few rows remain.
        """
        if self.X_raw is None:
            raise RuntimeError("Must call select_features() first")

        X = self.X_raw
        n_rows, n_cols = X.shape
        print(f"\nMissingness Analysis:")
        print(f"  Initial shape: {n_rows:,} rows x {n_cols} columns")

        # 1. Calculate missing share per column
        missing_share = X.isna().mean().sort_values(ascending=False)

        # Plot 1: Bar chart of top missing columns
        if show_plots:
            n_plot = min(60, len(missing_share))
            top_missing = missing_share.head(n_plot)

            fig, ax = plt.subplots(figsize=(10, max(6, n_plot * 0.15)))
            ax.barh(range(len(top_missing)), top_missing.values[::-1])
            ax.set_yticks(range(len(top_missing)))
            ax.set_yticklabels(top_missing.index[::-1], fontsize=7)
            ax.axvline(
                self.drop_threshold,
                color="red",
                linestyle="--",
                label=f"Threshold ({self.drop_threshold:.0%})",
            )
            ax.set_xlabel("Missing Share")
            ax.set_title(f"Goal {self.spec.name}: Column Missingness (Top {n_plot})")
            ax.legend()
            plt.tight_layout()
            plt.show()

        # 2. Drop columns above threshold
        cols_to_drop = missing_share[missing_share > self.drop_threshold].index.tolist()
        cols_to_keep = [c for c in X.columns if c not in cols_to_drop]

        print(
            f"\n  Columns with >{self.drop_threshold:.0%} missing: {len(cols_to_drop)}"
        )
        if cols_to_drop:
            print(
                f"    Dropping: {cols_to_drop[:10]}{'...' if len(cols_to_drop) > 10 else ''}"
            )

        self.X_clean = X[cols_to_keep].copy()

        # 3. Row-level missingness
        row_missing = self.X_clean.isna().sum(axis=1)
        rows_with_nan = (row_missing > 0).sum()
        rows_complete = n_rows - rows_with_nan

        print(f"\n  After dropping {len(cols_to_drop)} columns:")
        print(f"    Columns remaining: {len(cols_to_keep)}")
        print(f"    Rows with any NaN: {rows_with_nan:,} ({rows_with_nan/n_rows:.1%})")
        print(f"    Complete-case rows: {rows_complete:,} ({rows_complete/n_rows:.1%})")

        # Plot 2: Histograms of row missing counts
        if show_plots and rows_with_nan > 0:
            fig, ax = plt.subplots(figsize=(10, 5))

            dropped_counts = row_missing[row_missing > 0]
            if len(dropped_counts) > 0 and dropped_counts.max() > 0:
                ax.hist(
                    dropped_counts,
                    bins=min(30, int(dropped_counts.max())),
                    edgecolor="black",
                )
                ax.set_xlabel("Missing Values per Row")
                ax.set_ylabel("Number of Rows")
                ax.set_title(
                    f"Goal {self.spec.name}: Missing Count (Rows with >=1 NaN)"
                )

            plt.tight_layout()
            plt.show()

        # Plot 3: Column contribution to row drops
        if show_plots and rows_with_nan > 0:
            mask = row_missing > 0
            col_contrib = (
                self.X_clean.loc[mask].isna().sum().sort_values(ascending=False)
            )
            # Only columns with >0 missing in dropped rows
            top_contrib = col_contrib[col_contrib > 0]

            if top_contrib.sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(range(len(top_contrib)), top_contrib.values[::-1])
                ax.set_yticks(range(len(top_contrib)))
                ax.set_yticklabels(top_contrib.index[::-1], fontsize=8)
                ax.set_xlabel("NaN Count in by Column")
                ax.set_title(f"Goal {self.spec.name}: Column Contribution to Row Drops")
                plt.tight_layout()
                plt.show()

                # Print table of missing values per column
                missing_by_col = self.X_clean.isna().sum().sort_values(ascending=False)
                missing_by_col = missing_by_col[missing_by_col > 0]

                print(f"\n  Columns with missing values (in rows to be dropped):")
                print(
                    pd.DataFrame(
                        {
                            "Column": missing_by_col.index,
                            "Missing Count": missing_by_col.values,
                            "Missing %": (
                                missing_by_col.values / len(self.X_clean) * 100
                            ).round(2),
                        }
                    )
                )

        self.diagnostics = {
            "initial_rows": n_rows,
            "initial_cols": n_cols,
            "cols_dropped": len(cols_to_drop),
            "cols_remaining": len(cols_to_keep),
            "rows_with_nan": rows_with_nan,
            "rows_complete": rows_complete,
        }

        return self.X_clean

    def prepare_matrix(self) -> Tuple[np.ndarray, pd.Index]:
        """Prepare feature matrix for UMAP: encode + scale.

        Returns:
            Tuple of (scaled array, valid row indices).

        Raises:
            RuntimeError: If too few rows remain.
        """
        if self.X_clean is None:
            raise RuntimeError("Must call analyze_missingness() first")

        X = self.X_clean.copy()
        print(f"\nPreparing matrix (impute={self.impute})...")

        # Handle categorical columns (one-hot encode low-cardinality)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            nunique = X[col].nunique()
            if nunique <= 50:
                dummies = pd.get_dummies(X[col], prefix=col, dummy_na=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            else:
                X = X.drop(columns=[col])
                print(f"    Dropped high-cardinality: {col} ({nunique} unique)")

        if self.impute:
            # Median imputation
            for col in X.columns:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
            valid_idx = X.index
        else:
            # Complete-case analysis
            valid_mask = ~X.isna().any(axis=1)
            X = X.loc[valid_mask]
            valid_idx = X.index

        if len(valid_idx) < self.min_rows:
            raise RuntimeError(
                f"Goal {self.spec.name}: Only {len(valid_idx)} rows after cleaning, "
                f"need >= {self.min_rows}. Consider:\n"
                f"  - Increasing DROP_THRESHOLD (currently {self.drop_threshold})\n"
                f"  - Setting IMPUTE_NUMERIC=True\n"
                f"  - Removing problematic features"
            )

        # StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Verify no NaNs
        if np.isnan(X_scaled).any():
            raise RuntimeError("NaNs in scaled matrix - this should not happen!")

        self.X_scaled = X_scaled
        self.valid_idx = valid_idx

        print(
            f"  Final matrix: {X_scaled.shape[0]:,} rows x {X_scaled.shape[1]} features"
        )
        return X_scaled, valid_idx

    def fit_umap(
        self,
        n_neighbors: int = UMAP_N_NEIGHBORS,
        min_dist: float = UMAP_MIN_DIST,
        random_state: int = UMAP_RANDOM_STATE,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Fit UMAP to prepared data.

        Returns:
            2D embedding array.
        """
        if self.X_scaled is None:
            raise RuntimeError("Must call prepare_matrix() first")

        print(f"\nRunning UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.embedding = reducer.fit_transform(self.X_scaled)
        print(f"  ✓ UMAP complete: {self.embedding.shape}")
        return self.embedding

    def _get_weights_and_months(self) -> Tuple[np.ndarray, pd.Series]:
        """Get weights and month_ts for valid rows."""
        weights = self.df.loc[self.valid_idx, self.weight_col].fillna(1.0).values
        months = self.df.loc[self.valid_idx, "month_ts"]
        return weights, months

    def _scale_weights_for_markers(self, weights: np.ndarray) -> np.ndarray:
        """Scale weights to marker size range with quantile clipping."""
        w = np.array(weights, dtype=float)
        valid = ~np.isnan(w)
        if not valid.any():
            return np.full_like(w, (MARKER_SIZE_MIN + MARKER_SIZE_MAX) / 2)

        # Clip at 1% and 99% percentiles
        lo, hi = np.nanpercentile(w, [1, 99])
        w = np.clip(w, lo, hi)

        # Min-max scale
        w_min, w_max = np.nanmin(w), np.nanmax(w)
        if w_max - w_min > 1e-9:
            w_scaled = (w - w_min) / (w_max - w_min)
        else:
            w_scaled = np.full_like(w, 0.5)

        return MARKER_SIZE_MIN + w_scaled * (MARKER_SIZE_MAX - MARKER_SIZE_MIN)

    def plot_umap(self) -> None:
        """Plot UMAP embedding with weight-based marker size and time coloring."""
        if self.embedding is None:
            raise RuntimeError("Must call fit_umap() first")

        weights, months = self._get_weights_and_months()
        sizes = self._scale_weights_for_markers(weights)
        month_ordinal = months.astype(str).rank(method="dense")

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            c=month_ordinal,
            s=sizes,
            alpha=0.5,
            cmap="viridis",
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        unique_months = sorted(months.dropna().unique())
        if len(unique_months) > 1:
            n_ticks = min(8, len(unique_months))
            tick_positions = np.linspace(
                month_ordinal.min(), month_ordinal.max(), n_ticks
            )
            tick_indices = np.linspace(0, len(unique_months) - 1, n_ticks).astype(int)
            tick_labels = [str(unique_months[i]) for i in tick_indices]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
        cbar.set_label("Month")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(
            f"Goal {self.spec.name}: {self.spec.title}\n(marker size = weight)"
        )
        plt.tight_layout()
        plt.show()

    def _compute_weighted_centroids(self) -> pd.DataFrame:
        """Compute weighted centroids per month."""
        weights, months = self._get_weights_and_months()

        df = pd.DataFrame(
            {
                "umap_1": self.embedding[:, 0],
                "umap_2": self.embedding[:, 1],
                "month": months.values,
                "weight": weights,
            }
        )

        def weighted_mean(g):
            w = g["weight"].values
            if np.nansum(w) == 0:
                w = np.ones_like(w)
            return pd.Series(
                {
                    "umap_1": np.average(g["umap_1"], weights=w),
                    "umap_2": np.average(g["umap_2"], weights=w),
                }
            )

        centroids = (
            df.groupby("month", observed=True).apply(weighted_mean).reset_index()
        )
        return centroids.sort_values("month")

    def plot_centroid_trajectory(self) -> None:
        """Plot UMAP with monthly centroid trajectory."""
        if self.embedding is None:
            raise RuntimeError("Must call fit_umap() first")

        centroids = self._compute_weighted_centroids()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Background scatter
        ax.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            s=5,
            alpha=0.15,
            c="gray",
        )

        # Centroid trajectory
        ax.plot(
            centroids["umap_1"].values,
            centroids["umap_2"].values,
            "o-",
            linewidth=2,
            markersize=6,
            color="red",
        )

        # Annotate start and end
        if not centroids.empty:
            ax.annotate(
                f"Start: {centroids['month'].iloc[0]}",
                (centroids["umap_1"].iloc[0], centroids["umap_2"].iloc[0]),
                fontsize=9,
                color="darkred",
            )
            ax.annotate(
                f"End: {centroids['month'].iloc[-1]}",
                (centroids["umap_1"].iloc[-1], centroids["umap_2"].iloc[-1]),
                fontsize=9,
                color="darkred",
            )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(
            f"Goal {self.spec.name}: Centroid Trajectory\n(weighted by {self.weight_col})"
        )
        plt.tight_layout()
        plt.show()

    def run(self, show_plots: bool = True, n_jobs: int = -1) -> pd.DataFrame:
        """Run full pipeline: select → analyze → prepare → UMAP → plot.

        Args:
            show_plots: Whether to show diagnostic and result plots.

        Returns:
            DataFrame with embedding and metadata.
        """
        # Run UMAP pipeline
        self.fit_umap(n_jobs=n_jobs)

        # Visualize
        if show_plots:
            self.plot_umap()
            self.plot_centroid_trajectory()

        # Build result DataFrame
        weights, months = self._get_weights_and_months()
        result = pd.DataFrame(
            {
                "umap_1": self.embedding[:, 0],
                "umap_2": self.embedding[:, 1],
                "source_year": self.df.loc[self.valid_idx, "source_year"].values,
                "month_ts": months.astype(str).values,
                "weight": weights,
            }
        )

        print(
            f"\n✓ Goal {self.spec.name} complete: {len(result):,} observations embedded"
        )
        return result


# %% [markdown]
# ## 8. Run Goal A: Expectations/Beliefs

# %%
pipeline_A = CESUMAPPipeline(
    ces_df,
    GOAL_SPEC_A,
    weight_col=WEIGHT_COL,
    drop_threshold=DROP_THRESHOLD,
    impute=IMPUTE_NUMERIC,
)
selected_feat_A = pipeline_A.select_features()
no_nan_A = pipeline_A.analyze_missingness(show_plots=True)
X_scaled_A, valid_idx_A = pipeline_A.prepare_matrix()

# %%
result_A = pipeline_A.run(show_plots=True)

# %% [markdown]
# ## 9. Run Goal B: Behavior/Constraints

# %%
pipeline_B = CESUMAPPipeline(
    ces_df,
    GOAL_SPEC_B,
    weight_col=WEIGHT_COL,
    drop_threshold=DROP_THRESHOLD,
    impute=IMPUTE_NUMERIC,
)
selected_feat_B = pipeline_B.select_features()
no_nan_B = pipeline_B.analyze_missingness(show_plots=True)
X_scaled_B, valid_idx_B = pipeline_B.prepare_matrix()
result_B = pipeline_B.run(show_plots=True)

# %% [markdown]
# ## 10. Run Goal C: Broad Segmentation

# %%
pipeline_C = CESUMAPPipeline(
    ces_df,
    GOAL_SPEC_C,
    weight_col=WEIGHT_COL,
    drop_threshold=DROP_THRESHOLD,
    impute=IMPUTE_NUMERIC,
)
selected_feat_C = pipeline_C.select_features()
no_nan_C = pipeline_C.analyze_missingness(show_plots=True)
X_scaled_C, valid_idx_C = pipeline_C.prepare_matrix()
result_C = pipeline_C.run(show_plots=True)

# %% [markdown]
# ## 11. Save Embeddings


# %%
def save_embeddings(
    results: Dict[str, Optional[pd.DataFrame]], output_dir: str = RESULTS_DIR
) -> None:
    """Save embedding results to parquet files.

    Args:
        results: Dict mapping goal name to result DataFrame.
        output_dir: Output directory.

    Raises:
        RuntimeError: If any goal has no results.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for goal_name, df in results.items():
        if df is None:
            raise RuntimeError(f"Goal {goal_name} has no results to save!")

        filename = Path(output_dir) / f"umap_goal_{goal_name}.parquet"
        df.to_parquet(filename, index=False)
        print(f"Saved: {filename} ({len(df):,} rows)")


# Save all results
save_embeddings(
    {
        "A": result_A,
        "B": result_B,
        "C": result_C,
    }
)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
