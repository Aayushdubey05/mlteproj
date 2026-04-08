"""
=============================================================================
EFFECT OF DATA CLEANING AND FEATURE SELECTION ON MEDICAL DIAGNOSIS PREDICTION
=============================================================================
Group Project | 6 Groups x 2 Students | Same Dataset, Different Preprocessing
=============================================================================

DATASET COLUMNS:
  Patient_id, age, gender, symptoms (comma-sep), duration_days, severity (1-5),
  disease (target), temperature, heart_rate, bp (SYS/DIA), region

PIPELINE STRUCTURE:
  1. Data Generation (synthetic dataset if no CSV provided)
  2. Data Loading
  3. Basic Cleaning
  4. EDA Visualization
  5. Group-specific Preprocessing
  6. Model Training (Logistic Regression + Random Forest)
  7. Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
  8. Final Comparative Table + Visualizations
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2          # ALL groups use the same split ratio
N_SAMPLES    = 1000         # synthetic dataset size

DISEASE_LIST  = ["Flu", "Diabetes", "Hypertension", "Asthma", "Typhoid", "Malaria"]
SYMPTOM_POOL  = ["fever","cough","fatigue","nausea","headache",
                 "chills","vomiting","shortness_of_breath","chest_pain","sweating"]
REGION_LIST   = ["North","South","East","West","Central"]
GENDER_LIST   = ["Male","Female","Other"]

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


# =============================================================================
# STEP 1 – SYNTHETIC DATASET GENERATION
# =============================================================================

# Worthless
def generate_dataset(n: int = N_SAMPLES, save_path: str = "medical_data.csv") -> pd.DataFrame:
    """
    Generate a realistic synthetic medical dataset and save it as CSV.
    Call this ONCE to create the shared dataset for all groups.
    """
    np.random.seed(RANDOM_STATE)

    ages        = np.random.randint(5, 85, n)
    genders     = np.random.choice(GENDER_LIST, n)
    durations   = np.random.randint(1, 60, n)
    severities  = np.random.randint(1, 6, n)
    diseases    = np.random.choice(DISEASE_LIST, n)
    temps       = np.round(np.random.normal(98.6, 1.5, n), 1)
    heart_rates = np.random.randint(55, 120, n)
    regions     = np.random.choice(REGION_LIST, n)

    # blood pressure: SYS/DIA format
    sys_bp = np.random.randint(90, 180, n)
    dia_bp = np.random.randint(60, 110, n)
    bp     = [f"{s}/{d}" for s, d in zip(sys_bp, dia_bp)]

    # symptoms: 2-5 random symptoms per patient
    def rand_symptoms():
        k = np.random.randint(2, 6)
        return ",".join(np.random.choice(SYMPTOM_POOL, k, replace=False).tolist())

    symptoms = [rand_symptoms() for _ in range(n)]

    # Inject ~8% missing values in age, temperature, heart_rate
    for col_arr in [ages, temps, heart_rates]:
        mask = np.random.rand(n) < 0.08
        col_arr = col_arr.astype(float)
        col_arr[mask] = np.nan

    # Inject a few outliers in temperature and heart_rate
    outlier_idx = np.random.choice(n, 15, replace=False)
    temps[outlier_idx[:8]]       = np.random.choice([105, 106, 107], 8)
    heart_rates[outlier_idx[8:]] = np.random.choice([180, 190, 200], 7)

    df = pd.DataFrame({
        "Patient_id":    [f"P{str(i).zfill(4)}" for i in range(n)],
        "age":           ages.astype(float),
        "gender":        genders,
        "symptoms":      symptoms,
        "duration_days": durations,
        "severity":      severities,
        "disease":       diseases,
        "temperature":   temps,
        "heart_rate":    heart_rates.astype(float),
        "bp":            bp,
        "region":        regions,
    })

    df.to_csv(save_path, index=False)
    print(f"[Dataset] Saved {n} rows → '{save_path}'")
    return df


# =============================================================================
# STEP 2 – DATA LOADING
# =============================================================================
def load_data(path: str = "medical_dataset.csv") -> pd.DataFrame:
    """Load dataset from CSV; generate synthetic one if file not found."""
    try:
        df = pd.read_csv(path)
        print(f"[Load] '{path}' loaded — {df.shape[0]} rows, {df.shape[1]} cols")
    except FileNotFoundError:
        print(f"[Load] '{path}' not found. Generating synthetic dataset …")
    return df


# =============================================================================
# STEP 3 – BASIC CLEANING (shared by ALL groups)
# =============================================================================
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning applied identically by all groups before group-specific steps:
    - Drop duplicates
    - Drop Patient_id (non-informative)
    - Standardise string columns
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    if "Patient_id" in df.columns:
        df.drop(columns=["Patient_id"], inplace=True)

    # # Lowercase string columns for consistency
    # for col in ["gender", "region", "disease"]:
    #     if col in df.columns:
    #         df[col] = df[col].str.strip().str.lower()

    # Normalize ALL string garbage → NaN
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

        df[col].replace(
            ["", " ", "NA", "N/A", "null", "None", "nan", "-", "--", "?"],
            pd.NA,
            inplace=True
        )

    print("\n[DEBUG] Empty string count AFTER cleaning:")
    for col in df.select_dtypes(include="object").columns:
        print(col, (df[col] == "").sum())

    print(f"[Basic Clean] Shape after cleaning: {df.shape}")
    print(f"[Basic Clean] Missing values:\n{df.isnull().sum()}\n")
    return df



# =============================================================================
# STEP 4 – EDA VISUALISATION
# =============================================================================
def run_eda(df: pd.DataFrame, save_prefix: str = "eda"):
    """
    Generate key EDA plots:
    1. Target distribution
    2. Age distribution
    3. Severity vs Disease
    4. Correlation heatmap (numeric cols)
    5. Missing value heatmap
    """
    print("[EDA] Generating plots …")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Exploratory Data Analysis – Medical Diagnosis Dataset", fontsize=14, fontweight="bold")

    # --- 1. Target distribution ---
    ax = axes[0, 0]
    disease_counts = df["disease"].value_counts()
    sns.barplot(x=disease_counts.values, y=disease_counts.index, ax=ax, palette="Set2")
    ax.set_title("Disease Distribution (Target)")
    ax.set_xlabel("Count")

    # --- 2. Age distribution ---
    ax = axes[0, 1]
    df["age"].dropna().hist(bins=25, ax=ax, color="#4C9BE8", edgecolor="white")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")

    # --- 3. Severity count ---
    ax = axes[0, 2]
    df["severity"].value_counts().sort_index().plot(kind="bar", ax=ax,
                                                     color="#F4845F", edgecolor="white")
    ax.set_title("Severity Distribution (1–5)")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Count")

    # --- 4. Correlation heatmap ---
    ax = axes[1, 0]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap (Numeric Features)")

    # --- 5. Missing values ---
    ax = axes[1, 1]
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        missing.plot(kind="bar", ax=ax, color="#E07B54", edgecolor="white")
        ax.set_title("Missing Values per Column")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Missing Values")

    # --- 6. Duration vs Severity ---
    ax = axes[1, 2]
    sns.boxplot(data=df, x="severity", y="duration_days", palette="pastel", ax=ax)
    ax.set_title("Duration Days by Severity")

    plt.tight_layout()
    fname = f"{save_prefix}_overview.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.show()
    print(f"[EDA] Saved → '{fname}'\n")


# =============================================================================
# SHARED UTILITIES
# =============================================================================
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns (except 'disease')."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col != "disease":
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode the target column 'disease'."""
    df = df.copy()
    le = LabelEncoder()
    df["disease"] = le.fit_transform(df["disease"].astype(str))
    return df


def get_xy(df: pd.DataFrame):
    """Split into features X and target y."""
    y = df["disease"].values
    X = df.drop(columns=["disease"]).values
    return X, y


# def fixed_split(X, y):
#     """
#     ALL groups use IDENTICAL train-test split.
#     Same RANDOM_STATE ensures reproducibility across groups.
#     """
#     return train_test_split(X, y, test_size=TEST_SIZE,
#                             random_state=RANDOM_STATE, stratify=y)

def fixed_split(X, y):
    from collections import Counter

    class_counts = Counter(y)
    min_count = min(class_counts.values())

    if min_count < 2:
        print("[Warning] Stratify disabled (class with <2 samples)")
        return train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        return train_test_split(
            X, y, test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

def drop_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any remaining non-numeric columns (safety net before modelling)."""
    non_num = df.select_dtypes(include="object").columns.tolist()
    if non_num:
        df = df.drop(columns=non_num)
    return df


# =============================================================================
# STEP 5 – GROUP-SPECIFIC PREPROCESSING
# =============================================================================

# --------------- GROUP 1 : BASELINE ----------------------------------------
def preprocess_group1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 1 – Baseline
    ──────────────────
    Strategy : Minimal cleaning + Label Encoding only.
    Missing values are dropped (simplest possible approach).
    No scaling, no feature engineering.
    """
    print("[Group 1] Baseline preprocessing …")
    df = df.copy()

    # Drop rows with any missing value (simplest strategy)
    df.dropna(inplace=True)

    # Encode categoricals (gender, region, symptoms kept as-is for baseline)
    df = encode_categoricals(df)
    df = encode_target(df)
    df = drop_non_numeric(df)

    print(f"[Group 1] Final shape: {df.shape}\n")
    return df


# --------------- GROUP 2 : MISSING VALUE HANDLING --------------------------
def preprocess_group2(df: pd.DataFrame, strategy: str = "impute") -> pd.DataFrame:
    """
    Group 2 – Missing Value Handling
    ──────────────────────────────────
    strategy = 'impute'  → Mean/Median/Mode imputation
    strategy = 'drop'    → Drop rows with missing values
    
    Comparison is done inside evaluate_group2().
    """
    print(f"[Group 2] Missing value strategy: '{strategy}' …")
    df = df.copy()

    if strategy == "impute":
        # Numeric → mean imputation
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)

        # Categorical → mode imputation
        for col in df.select_dtypes(include="object").columns:
            mode_val = df[col].mode()
            if len(mode_val):
                df[col].fillna(mode_val[0], inplace=True)

    elif strategy == "drop":
        before = len(df)
        df.dropna(inplace=True)
        print(f"[Group 2]   Dropped {before - len(df)} rows")

    df = encode_categoricals(df)
    df = encode_target(df)
    df = drop_non_numeric(df)

    print(f"[Group 2] Final shape ({strategy}): {df.shape}\n")

        

    return df


# --------------- GROUP 3 : OUTLIER HANDLING --------------------------------
def preprocess_group3(df: pd.DataFrame, method: str = "iqr_cap") -> pd.DataFrame:
    """
    Group 3 – Outlier Handling
    ───────────────────────────
    method = 'iqr_remove'  → Remove rows with IQR outliers
    method = 'iqr_cap'     → Cap (Winsorize) outliers at IQR bounds
    method = 'zscore'      → Remove rows where |z-score| > 3
    """
    print(f"[Group 3] Outlier method: '{method}' …")
    df = df.copy()

    # Impute missing before outlier handling
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude target-like cols
    num_cols = [c for c in num_cols if c != "disease"]

    before = len(df)

    if "iqr" in method:
        Q1  = df[num_cols].quantile(0.25)
        Q3  = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if method == "iqr_remove":
            mask = ((df[num_cols] >= lower) & (df[num_cols] <= upper)).all(axis=1)
            df   = df[mask]
        elif method == "iqr_cap":
            for col in num_cols:
                df[col] = df[col].clip(lower[col], upper[col])

    elif method == "zscore":
        z_scores = np.abs(stats.zscore(df[num_cols].fillna(0)))
        mask = (z_scores < 3).all(axis=1)
        df   = df[mask]

    print(f"[Group 3]   Rows removed/modified: {before - len(df)}")
    df = encode_categoricals(df)
    df = encode_target(df)
    df = drop_non_numeric(df)

    print(f"[Group 3] Final shape ({method}): {df.shape}\n")
    return df


# --------------- GROUP 4 : FEATURE SCALING ---------------------------------
def preprocess_group4(df: pd.DataFrame, scaler_type: str = "standard") -> pd.DataFrame:
    """
    Group 4 – Feature Scaling
    ──────────────────────────
    scaler_type = 'standard'  → StandardScaler (zero mean, unit variance)
    scaler_type = 'minmax'    → MinMaxScaler   (range [0, 1])
    scaler_type = 'none'      → No scaling (baseline for this group)
    """
    print(f"[Group 4] Scaler: '{scaler_type}' …")
    df = df.copy()

    # Impute missing
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else "unknown", inplace=True)

    df = encode_categoricals(df)
    df = encode_target(df)
    df = drop_non_numeric(df)

    if scaler_type != "none":
        target = df["disease"].copy()
        feature_cols = [c for c in df.columns if c != "disease"]

        if scaler_type == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        df["disease"] = target

    print(f"[Group 4] Final shape ({scaler_type}): {df.shape}\n")
    return df


# --------------- GROUP 5 : FEATURE ENGINEERING -----------------------------
def preprocess_group5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 5 – Feature Engineering
    ───────────────────────────────
    1. Multi-label binary encoding of 'symptoms'
    2. Split 'bp' into 'bp_sys' and 'bp_dia'
    3. Derived feature: severity_x_duration = severity × duration_days
    4. Derived feature: pulse_pressure = bp_sys - bp_dia
    """
    print("[Group 5] Feature Engineering …")
    df = df.copy()

    # --- 1. Multi-label encode symptoms ---
    if "symptoms" in df.columns:
        for symptom in SYMPTOM_POOL:
            df[f"sym_{symptom}"] = df["symptoms"].apply(
                lambda x: 1 if isinstance(x, str) and symptom in x.split(",") else 0
            )
        # Don't drop symptoms column yet – we may want to compare with/without it in ablation later
        # df.drop(columns=["symptoms"], inplace=True)

    # --- 2. Split bp ---
    if "bp" in df.columns:
        bp_split = df["bp"].str.split("/", expand=True)
        df["bp_sys"] = pd.to_numeric(bp_split[0], errors="coerce")
        df["bp_dia"] = pd.to_numeric(bp_split[1], errors="coerce")

        # Don't drop bp column yet – we may want to compare with/without it in ablation later
        # df.drop(columns=["bp"], inplace=True)

    # --- 3. Derived features ---
    df["severity_x_duration"] = df["severity"] * df["duration_days"]

    if "bp_sys" in df.columns and "bp_dia" in df.columns:
        df["pulse_pressure"] = df["bp_sys"] - df["bp_dia"]

    # --- Impute remaining missing ---
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else "unknown", inplace=True)

# Will add this later this was the original code
    # df = encode_categoricals(df)
    # df = encode_target(df)
    # df = drop_non_numeric(df)

    print(f"[Group 5] Final shape: {df.shape}\n")
    return df


# --------------- GROUP 6 : FEATURE SELECTION -------------------------------
def preprocess_group6(df: pd.DataFrame, method: str = "mutual_info",
                      k: int = 10) -> pd.DataFrame:
    """
    Group 6 – Feature Selection
    ────────────────────────────
    method = 'correlation'   → Drop features correlated > 0.85 with another
    method = 'chi2'          → SelectKBest with chi-squared test
    method = 'mutual_info'   → SelectKBest with mutual information
    k = number of top features to keep (for SelectKBest methods)
    """
    print(f"[Group 6] Feature selection method: '{method}' (k={k}) …")
    df = df.copy()

    # Impute
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) else "unknown", inplace=True)

    # Feature engineering for richer feature set
    if "bp" in df.columns:
        bp_split = df["bp"].str.split("/", expand=True)
        df["bp_sys"] = pd.to_numeric(bp_split[0], errors="coerce").fillna(120)
        df["bp_dia"] = pd.to_numeric(bp_split[1], errors="coerce").fillna(80)
        df.drop(columns=["bp"], inplace=True)

    if "symptoms" in df.columns:
        for symptom in SYMPTOM_POOL:
            df[f"sym_{symptom}"] = df["symptoms"].apply(
                lambda x: 1 if isinstance(x, str) and symptom in x.split(",") else 0
            )
        df.drop(columns=["symptoms"], inplace=True)

    df = encode_categoricals(df)
    df = encode_target(df)
    df = drop_non_numeric(df)

    feature_cols = [c for c in df.columns if c != "disease"]
    X_all = df[feature_cols]
    y_all = df["disease"]

    if method == "correlation":
        # Remove one of each highly-correlated pair
        corr_matrix = X_all.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
        X_all.drop(columns=to_drop, inplace=True, errors="ignore")
        print(f"[Group 6]   Dropped highly-correlated: {to_drop}")

    elif method in ("chi2", "mutual_info"):
        # Ensure non-negative for chi2
        X_shifted = X_all - X_all.min()
        score_fn   = chi2 if method == "chi2" else mutual_info_classif
        k_actual   = min(k, X_shifted.shape[1])
        selector   = SelectKBest(score_fn, k=k_actual)
        selector.fit(X_shifted, y_all)
        selected   = X_shifted.columns[selector.get_support()].tolist()
        X_all      = X_all[selected]
        print(f"[Group 6]   Selected features: {selected}")

    df = pd.concat([X_all.reset_index(drop=True),
                    y_all.reset_index(drop=True)], axis=1)

    print(f"[Group 6] Final shape ({method}): {df.shape}\n")
    return df


# =============================================================================
# STEP 6 – MODEL TRAINING & EVALUATION
# =============================================================================
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}


def train_and_evaluate(df: pd.DataFrame, group_name: str, method_label: str) -> list[dict]:
    """
    Train both models on the processed DataFrame.
    Returns a list of result dicts (one per model).
    """
    results = []

    X, y = get_xy(df)

    # Guard: need enough samples
    if len(X) < 20:
        print(f"[Warning] {group_name} – Not enough samples ({len(X)}). Skipping.")
        return results

    X_train, X_test, y_train, y_test = fixed_split(X, y)

    for model_name, model in MODELS.items():
        model_clone = type(model)(**model.get_params())  # fresh clone
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

        avg = "weighted"
        result = {
            "Group":     group_name,
            "Method":    method_label,
            "Model":     model_name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "_y_test":   y_test,
            "_y_pred":   y_pred,
            "_model":    model_clone,
            "_X_test":   X_test,
            "_feature_names": [c for c in df.columns if c != "disease"],
        }
        results.append(result)
        print(f"  {model_name:25s} | Acc={result['Accuracy']:.4f} | "
              f"P={result['Precision']:.4f} | R={result['Recall']:.4f} | F1={result['F1']:.4f}")

    return results


# =============================================================================
# STEP 7 – VISUALISATIONS
# =============================================================================

def plot_confusion_matrices(all_results: list[dict], save_path: str = "confusion_matrices.png"):
    """Plot confusion matrices for every (group, model) combination."""
    n_cols = len(MODELS)
    n_rows = len(all_results) // n_cols
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5, n_rows * 4))
    fig.suptitle("Confusion Matrices – All Groups & Models", fontsize=14, fontweight="bold")

    axes = np.array(axes).reshape(n_rows, n_cols)

    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= len(all_results):
                break
            res = all_results[idx]
            cm  = confusion_matrix(res["_y_test"], res["_y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[row, col],
                        linewidths=0.4, cbar=False)
            axes[row, col].set_title(f"{res['Group']}\n{res['Method']}\n{res['Model']}",
                                     fontsize=8)
            axes[row, col].set_xlabel("Predicted", fontsize=7)
            axes[row, col].set_ylabel("Actual", fontsize=7)
            idx += 1

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Confusion matrices saved → '{save_path}'\n")


def plot_feature_importance(all_results: list[dict], save_path: str = "feature_importance.png"):
    """Plot top-10 feature importances for Random Forest models."""
    rf_results = [r for r in all_results if "Random Forest" in r["Model"]]
    if not rf_results:
        return

    n = len(rf_results)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Random Forest – Top 10 Feature Importances", fontsize=13, fontweight="bold")

    for ax, res in zip(axes, rf_results):
        model    = res["_model"]
        feat_names = res["_feature_names"]

        if not hasattr(model, "feature_importances_"):
            continue

        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1][:10]
        top_names   = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in indices]
        top_vals    = importances[indices]

        sns.barplot(x=top_vals, y=top_names, ax=ax, palette="viridis")
        ax.set_title(f"{res['Group']}\n{res['Method']}", fontsize=9)
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Feature importance saved → '{save_path}'\n")


def plot_comparative_metrics(results_df: pd.DataFrame,
                             save_path: str = "comparative_metrics.png"):
    """Bar charts comparing F1 and Accuracy across all groups."""
    metrics = ["Accuracy", "F1"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Comparative Performance Across Groups", fontsize=14, fontweight="bold")

    palette = sns.color_palette("tab10", n_colors=len(results_df["Group"].unique()))

    for ax, metric in zip(axes, metrics):
        plot_df = results_df.copy()
        plot_df["Label"] = plot_df["Group"] + "\n" + plot_df["Method"].str[:15]
        sns.barplot(data=plot_df, x="Label", y=metric, hue="Model", ax=ax, palette="Set1")
        ax.set_title(f"{metric} by Group & Model")
        ax.set_xlabel("Group / Method")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Comparative metrics saved → '{save_path}'\n")


# def plot_before_after(before_df: pd.DataFrame, after_df: pd.DataFrame,
#                       group_name: str, save_path: str = "before_after.png"):
#     """
#     Show before vs after distributions for numeric columns after preprocessing.
#     Used by any group wanting to demonstrate transformation effect.
#     """
#     num_cols = [c for c in before_df.select_dtypes(include=[np.number]).columns
#                 if c in after_df.columns and c != "disease"][:4]

#     if not num_cols:
#         return

#     fig, axes = plt.subplots(2, len(num_cols), figsize=(len(num_cols) * 4, 6))
#     fig.suptitle(f"{group_name} – Before vs After Preprocessing", fontsize=12, fontweight="bold")

#     for i, col in enumerate(num_cols):
#         before_df[col].dropna().hist(bins=25, ax=axes[0, i],
#                                       color="#66B2FF", edgecolor="white")
#         axes[0, i].set_title(f"{col} (Before)")

#         after_df[col].dropna().hist(bins=25, ax=axes[1, i],
#                                      color="#FF9966", edgecolor="white")
#         axes[1, i].set_title(f"{col} (After)")

#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.show()
#     print(f"[Plot] Before/After saved → '{save_path}'\n")

def plot_before_after(before_df: pd.DataFrame, after_df: pd.DataFrame,
                      group_name: str, save_path: str = "before_after.png"):
    """
    Show before vs after distributions for numeric columns after preprocessing.
    Used by any group wanting to demonstrate transformation effect.
    """
    # --- ORIGINAL LOGIC (kept, but limited usefulness for feature engineering) ---
    num_cols = [c for c in before_df.select_dtypes(include=[np.number]).columns
                if c in after_df.columns and c != "disease"][:4]

    # --- NEW: Include engineered columns (important for Group 5) ---
    engineered_cols = [c for c in after_df.columns
                       if c not in before_df.columns and after_df[c].dtype != "object"]

    # Limit engineered cols to avoid overcrowding
    engineered_cols = engineered_cols[:4]

    # Combine both (so old behavior + new visibility)
    plot_cols = num_cols + engineered_cols

    if not plot_cols:
        return

    fig, axes = plt.subplots(2, len(plot_cols), figsize=(len(plot_cols) * 4, 6))
    fig.suptitle(f"{group_name} – Before vs After Preprocessing", fontsize=12, fontweight="bold")

    for i, col in enumerate(plot_cols):

        # --- BEFORE ---
        if col in before_df.columns:
            before_df[col].dropna().hist(bins=25, ax=axes[0, i],
                                        color="#66B2FF", edgecolor="white")
            axes[0, i].set_title(f"{col} (Before)")
        else:
            # Engineered feature doesn't exist before
            axes[0, i].text(0.5, 0.5, "Not Present",
                            ha='center', va='center', fontsize=10)
            axes[0, i].set_title(f"{col} (Before)")
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])

        # --- AFTER ---
        if col in after_df.columns:
            after_df[col].dropna().hist(bins=25, ax=axes[1, i],
                                       color="#FF9966", edgecolor="white")
            axes[1, i].set_title(f"{col} (After)")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Plot] Before/After saved → '{save_path}'\n")


# =============================================================================
# STEP 8 – FINAL COMPARATIVE TABLE
# =============================================================================
def print_comparison_table(results_df: pd.DataFrame):
    """Pretty-print the final comparison table."""
    display_cols = ["Group", "Method", "Model", "Accuracy", "Precision", "Recall", "F1"]
    table = results_df[display_cols].sort_values(["F1"], ascending=False).reset_index(drop=True)

    separator = "─" * 100
    print("\n" + separator)
    print("FINAL COMPARATIVE TABLE — Effect of Preprocessing on Medical Diagnosis Prediction")
    print(separator)
    header = f"{'Group':<12} {'Method':<25} {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print(separator)
    for _, row in table.iterrows():
        print(f"{row['Group']:<12} {row['Method']:<25} {row['Model']:<22} "
              f"{row['Accuracy']:>9.4f} {row['Precision']:>10.4f} {row['Recall']:>8.4f} {row['F1']:>8.4f}")
    print(separator)

    # Best result
    best = table.iloc[0]
    print(f"\n★  BEST: {best['Group']} | {best['Method']} | {best['Model']} → F1={best['F1']:.4f}")

    # Interpretation
    print("\n" + separator)
    print("INTERPRETATION")
    print(separator)
    print("""
  1. Baseline (Group 1):
       Provides a reference point. Minimal processing often underperforms.

  2. Missing Value Handling (Group 2):
       Imputation generally outperforms row-dropping because it retains more
       training samples. Mean/mode imputation is effective for small % missingness.

  3. Outlier Handling (Group 3):
       Capping (IQR winsorization) tends to outperform removal because it
       preserves sample size while reducing noise. Z-score removal is more
       aggressive and may discard meaningful extreme cases in medical data.

  4. Feature Scaling (Group 4):
       Critical for Logistic Regression (distance-based). Random Forest is
       invariant to scale, so you may see larger impact on LR than RF.
       StandardScaler typically performs better when features are normally
       distributed; MinMaxScaler suits uniformly distributed features.

  5. Feature Engineering (Group 5):
       Often yields the largest improvement. Splitting 'bp', encoding
       'symptoms' as multi-label, and creating derived features (severity ×
       duration) provide richer signal to both models.

  6. Feature Selection (Group 6):
       Removes noise and reduces overfitting. Mutual Information often
       outperforms Chi-2 on mixed data types. Correlation filtering removes
       redundancy but can discard useful correlated features.

  Overall: Feature Engineering (Group 5) + Feature Selection (Group 6) tend
  to yield the best model performance, while baseline preprocessing provides
  the lowest performance — demonstrating the critical role of preprocessing.
""")
    print(separator)


# =============================================================================
# MAIN – RUN ALL GROUPS
# =============================================================================
def main():
    print("=" * 70)
    print("  MEDICAL DIAGNOSIS PREDICTION — GROUP PREPROCESSING COMPARISON")
    print("=" * 70)

    # ── Load shared dataset ────────────────────────────────────────────────
    raw_df = load_data("medical_dataset.csv")
    clean_df = basic_clean(raw_df)

    # ── EDA ────────────────────────────────────────────────────────────────
    run_eda(clean_df, save_prefix="eda")

    all_results = []

    # ── GROUP 1 : Baseline ─────────────────────────────────────────────────
    print("─" * 50)
    print("GROUP 1 – BASELINE")
    g1_df = preprocess_group1(clean_df)
    res   = train_and_evaluate(g1_df, "Group 1", "Baseline")
    all_results.extend(res)

    # ── GROUP 2 : Missing Value Handling ───────────────────────────────────
    print("─" * 50)
    print("GROUP 2 – MISSING VALUE HANDLING")
    for strat in ["impute", "drop"]:
        g2_df = preprocess_group2(clean_df, strategy=strat)
        res   = train_and_evaluate(g2_df, "Group 2", f"MV_{strat}")
        all_results.extend(res)

    # ── GROUP 3 : Outlier Handling ─────────────────────────────────────────
    print("─" * 50)
    print("GROUP 3 – OUTLIER HANDLING")
    for method in ["iqr_cap", "iqr_remove", "zscore"]:
        g3_df = preprocess_group3(clean_df, method=method)
        res   = train_and_evaluate(g3_df, "Group 3", f"OL_{method}")
        #  save_path="group5_before_after.png") generate image related to this
        all_results.extend(res)

    # ── GROUP 4 : Feature Scaling ──────────────────────────────────────────
    print("─" * 50)
    print("GROUP 4 – FEATURE SCALING")
    for scaler in ["none", "standard", "minmax"]:
        g4_df = preprocess_group4(clean_df, scaler_type=scaler)
        res   = train_and_evaluate(g4_df, "Group 4", f"Scale_{scaler}")
        #  save_path="group5_before_after.png")
        all_results.extend(res)

    # ── GROUP 5 : Feature Engineering ─────────────────────────────────────
    print("─" * 50)
    print("GROUP 5 – FEATURE ENGINEERING")
    g5_before = basic_clean(raw_df)          # for before/after plot
    g5_df = preprocess_group5(g5_before.copy())
    # g5_df = preprocess_group5(clean_df)
    
    plot_before_after(g5_before, g5_df, "Group 5 – Feature Engineering",
                      save_path="group5_before_after.png")
    
    g5_df = encode_categoricals(g5_df)
    g5_df = encode_target(g5_df)
    g5_df = drop_non_numeric(g5_df)
    
    res = train_and_evaluate(g5_df, "Group 5", "FeatEng")
    all_results.extend(res)
    # plot_before_after(g5_before, g5_df, "Group 5 – Feature Engineering",
    #                   save_path="group5_before_after.png")

    # ── GROUP 6 : Feature Selection ────────────────────────────────────────
    print("─" * 50)
    print("GROUP 6 – FEATURE SELECTION")
    for method in ["correlation", "chi2", "mutual_info"]:
        g6_df = preprocess_group6(clean_df, method=method, k=8)
        res   = train_and_evaluate(g6_df, "Group 6", f"FS_{method}")
        all_results.extend(res)

    # ── Build results DataFrame (strip private keys) ───────────────────────
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in all_results
    ])

    # ── Print final table ──────────────────────────────────────────────────
    print_comparison_table(results_df)

    # ── Save CSV ───────────────────────────────────────────────────────────
    results_df.to_csv("comparison_results.csv", index=False)
    print("\n[Output] Results saved → 'comparison_results.csv'")

    # ── Visualisations ─────────────────────────────────────────────────────
    plot_confusion_matrices(all_results,    save_path="confusion_matrices.png")
    plot_feature_importance(all_results,    save_path="feature_importance.png")
    plot_comparative_metrics(results_df,    save_path="comparative_metrics.png")

    print("\n✓ Pipeline complete. All outputs saved.\n")
    return results_df


# =============================================================================
if __name__ == "__main__":
    results = main()