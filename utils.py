# utils.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import (
    recall_score, fbeta_score, roc_auc_score, precision_score, accuracy_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from imblearn.over_sampling import SMOTE

# Coba pakai XGBoost; fallback ke RandomForest jika tidak tersedia
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGB = False

def make_onehot_encoder():
    """
    Return OneHotEncoder yang kompatibel lintas versi scikit-learn.
    - scikit-learn >= 1.2: gunakan sparse_output=False
    - scikit-learn < 1.2 : gunakan sparse=False
    """
    try:
        # Versi baru: parameter 'sparse_output'
        return OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        # Versi lama: fallback 'sparse'
        return OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

# ====== Konstanta & Direktori ======
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_COST_PER_CHURN = 4_000_000  # Rp 4 juta, sesuai permintaan

# ====== Helper umum ======
def safe_div(numer, denom):
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.true_divide(numer, denom)
        out[~np.isfinite(out)] = np.nan
    return out

# ====== Feature Engineering ======
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["overtime_ratio"] = safe_div(df["overtime_hours_per_week"], df["working_hours_per_week"])
    df["tenure_per_age"] = safe_div(df["company_tenure_years"], df["age"])
    df["performance_gap"] = 1 - df["target_achievement"]
    df["salary_per_hour"] = safe_div(df["salary"], df["working_hours_per_week"])
    df["satisfaction_support_mean"] = (df["job_satisfaction"] + df["manager_support_score"]) / 2
    df["commuting_work_ratio"] = safe_div(df["distance_to_office_km"], df["working_hours_per_week"])
    df["salary_commission_rate"] = df["salary"] * df["target_achievement"] * df["commission_rate"]
    return df

# ====== Preprocess bundle ======
@dataclass
class PreprocessBundle:
    cat_onehot: List[str]
    cat_label: List[str]
    label_encoders: Dict[str, Any]
    onehot: OneHotEncoder
    scaler: StandardScaler
    base_numeric_cols: List[str]
    final_columns_: List[str]

# ====== Fit Preprocessing ======
def fit_preprocess(df_raw: pd.DataFrame, y_col: str = "churn") -> Tuple[pd.DataFrame, pd.Series, PreprocessBundle]:
    df = df_raw.copy()

    # Drop ID & leakage jika ada
    drop_cols = [c for c in ["employee_id", "churn_period"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Feature engineering
    df = feature_engineering(df)

    # Target
    y = df[y_col].astype(int)
    X = df.drop(columns=[y_col])

    # Categorical configs sesuai skrip awal
    cat_onehot = ["gender", "marital_status"]  # OHE (drop first)
    cat_label = ["education", "work_location"] # LabelEncoder
    label_encoders: Dict[str, LabelEncoder] = {}

    # Label encode
    X_le = X.copy()
    for col in cat_label:
        le = LabelEncoder()
        X_le[col] = le.fit_transform(X_le[col])
        label_encoders[col] = le

    # One-hot encode
    # gunakan 'sparse=False' agar kompatibel dengan scikit-learn <1.2
    onehot = make_onehot_encoder()
    ohe_arr = onehot.fit_transform(X_le[cat_onehot])
    ohe_df = pd.DataFrame(ohe_arr, columns=onehot.get_feature_names_out(cat_onehot), index=X_le.index)

    # Numeric (selain cat_onehot & cat_label)
    num_cols = [c for c in X_le.columns if c not in (cat_onehot + cat_label)]
    X_num = X_le[num_cols]

    # Gabungkan
    X_enc = pd.concat([X_num, X_le[cat_label], ohe_df], axis=1)

    # Scaling: skala semua kolom numerik di X_enc (termasuk hasil label-encode dan OHE)
    scaler = StandardScaler()
    num_cols_after = X_enc.select_dtypes(include=["int64", "float64"]).columns
    X_enc[num_cols_after] = scaler.fit_transform(X_enc[num_cols_after])

    bundle = PreprocessBundle(
        cat_onehot=cat_onehot,
        cat_label=cat_label,
        label_encoders=label_encoders,
        onehot=onehot,
        scaler=scaler,
        base_numeric_cols=num_cols,
        final_columns_=X_enc.columns.tolist(),
    )
    return X_enc, y, bundle

# ====== Transform data baru ======
def transform_new(df_raw: pd.DataFrame, bundle: PreprocessBundle, y_col: str | None = None) -> pd.DataFrame:
    df = df_raw.copy()
    # drop kolom yang tidak digunakan model
    drop_cols = [c for c in ["employee_id", "churn_period"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # FE
    df = feature_engineering(df)

    # hilangkan target jika ada
    if y_col and y_col in df.columns:
        df = df.drop(columns=[y_col])

    # Label-encode kolom bertingkat (handle unseen -> -1)
    for col in bundle.cat_label:
        le = bundle.label_encoders[col]
        df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # One-hot encode kolom nominal
    ohe_arr = bundle.onehot.transform(df[bundle.cat_onehot])
    ohe_df = pd.DataFrame(ohe_arr, columns=bundle.onehot.get_feature_names_out(bundle.cat_onehot), index=df.index)

    # Numeric lainnya
    num_cols = [c for c in df.columns if c not in (bundle.cat_onehot + bundle.cat_label)]
    X_enc = pd.concat([df[num_cols], df[bundle.cat_label], ohe_df], axis=1)

    # Align kolom (isi kolom hilang -> 0)
    for c in bundle.final_columns_:
        if c not in X_enc.columns:
            X_enc[c] = 0.0
    X_enc = X_enc[bundle.final_columns_]

    # Scaling
    num_cols_after = X_enc.select_dtypes(include=["int64", "float64"]).columns
    X_enc[num_cols_after] = bundle.scaler.transform(X_enc[num_cols_after])
    return X_enc

# ====== Training model ======
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)

    if HAS_XGB:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1
        )
        model.fit(X_sm, y_sm)
    else:
        # Fallback: RandomForest (agar app tetap berjalan jika xgboost belum terpasang)
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
        model.fit(X_sm, y_sm)
    return model

def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "recall_test": recall_score(y_test, y_pred),
        "f2_test": fbeta_score(y_test, y_pred, beta=2),
        "precision_test": precision_score(y_test, y_pred),
        "accuracy_test": accuracy_score(y_test, y_pred),
        "roc_auc_test": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_prob

# ====== Load atau Train ======
def load_or_train(data_path: str):
    model_path = ARTIFACTS_DIR / "best_model.joblib"
    prep_path = ARTIFACTS_DIR / "preprocess.joblib"
    cols_path = ARTIFACTS_DIR / "columns.json"

    if model_path.exists() and prep_path.exists() and cols_path.exists():
        model = joblib.load(model_path)
        preprocess = joblib.load(prep_path)
        # evaluasi cepat untuk halaman Analisis Biaya
        df = pd.read_csv(data_path)
        X_all, y_all, _ = fit_preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        metrics, y_prob_test = evaluate(model, X_test, y_test)
        eval_info = dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            y_prob_test=y_prob_test, metrics=metrics
        )
        return {"model": model, "preprocess": preprocess, "eval_info": eval_info}

    # else -> train dari awal
    df = pd.read_csv(data_path)
    X_all, y_all, preprocess = fit_preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

    model = train_model(X_train, y_train)
    metrics, y_prob_test = evaluate(model, X_test, y_test)

    # simpan artifacts
    joblib.dump(model, model_path)
    joblib.dump(preprocess, prep_path)
    with open(cols_path, "w") as f:
        json.dump({"final_columns": preprocess.final_columns_}, f, indent=2)

    eval_info = dict(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        y_prob_test=y_prob_test, metrics=metrics
    )
    return {"model": model, "preprocess": preprocess, "eval_info": eval_info}

# ====== Inference ======
def predict_one(raw: Dict[str, Any], preprocess: PreprocessBundle, model, threshold: float = 0.5):
    df = pd.DataFrame([raw])
    X = transform_new(df, preprocess)
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= threshold)
    return proba, label

def predict_batch(df_input: pd.DataFrame, preprocess: PreprocessBundle, model, threshold: float = 0.5):
    cols_needed = [
        "age","gender","education","experience_years","monthly_target","target_achievement",
        "working_hours_per_week","overtime_hours_per_week","salary","commission_rate","job_satisfaction",
        "work_location","manager_support_score","company_tenure_years","marital_status","distance_to_office_km"
    ]
    missing = [c for c in cols_needed if c not in df_input.columns]
    if missing:
        raise ValueError(f"Kolom wajib belum lengkap: {missing}")

    X = transform_new(df_input, preprocess, y_col=None)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    out = df_input.copy()
    out["churn_proba"] = prob
    out["churn_pred"] = pred
    return out

# ====== Plot helpers (Analisis Biaya & umum) ======
def plot_confusion_matrix_fig(y_true, y_prob, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    return fig

def expected_savings_curve(y_true, y_prob, cost_per_churn: int, intervention_cost: int, effectiveness: float, n_points: int = 100):
    thresholds = np.linspace(0.05, 0.95, n_points)
    net = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        n_intervened = (y_pred == 1).sum()
        prevented = tp * effectiveness
        gross = prevented * cost_per_churn
        total_interv = n_intervened * intervention_cost
        net.append(gross - total_interv)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, net, color="#008080")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Net Saving (Rp)")
    ax.set_title("Expected Net Saving vs Threshold")
    fig.tight_layout()
    return fig

# ====== EDA Utilities ======
def get_schema():
    """
    Skema kolom mengikuti dataset yang kamu kirim.
    - numeric_cols: fitur numerik untuk EDA & korelasi (excl. ID/leakage/target)
    - cat_ordered: kategori bertingkat
    - cat_nominal: kategori nominal
    - target_col: kolom target
    """
    numeric_cols = [
        "age", "experience_years", "monthly_target", "target_achievement",
        "working_hours_per_week", "overtime_hours_per_week", "salary", "commission_rate",
        "company_tenure_years", "job_satisfaction", "manager_support_score",
        "distance_to_office_km"
    ]
    cat_ordered = ["education", "churn_period", "work_location"]
    cat_nominal = ["gender", "marital_status"]
    target_col = "churn"
    return numeric_cols, cat_ordered, cat_nominal, target_col

def plot_missing_bar(df: pd.DataFrame):
    miss = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.25*len(miss)), 3.5))
    sns.barplot(x=miss.index, y=miss.values, ax=ax, color="#b0c4de")
    ax.set_title("Jumlah Missing per Kolom")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig

def plot_target_counts(df: pd.DataFrame, target_col: str = "churn"):
    if target_col not in df.columns:
        return plt.figure()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=df, x=target_col, ax=ax, palette="Set2")
    for c in ax.containers:
        ax.bar_label(c)
    ax.set_title("Distribusi Target (churn)")
    ax.set_xlabel("Churn (0=Stay, 1=Churn)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

def plot_churn_period_counts(df: pd.DataFrame, churn_col: str = "churn"):
    if "churn_period" not in df.columns or churn_col not in df.columns:
        return plt.figure()
    d = df.loc[df[churn_col] == 1]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    order = d["churn_period"].value_counts().index
    sns.countplot(data=d, x="churn_period", order=order, ax=ax, palette="pastel")
    for c in ax.containers:
        ax.bar_label(c)
    ax.set_title("Distribusi Churn Period (hanya churn=1)")
    ax.set_xlabel("Churn Period")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig

def facet_numeric_hist(df: pd.DataFrame, numeric_cols: List[str], hue: str | None = None, max_cols: int = 4):
    cols = [c for c in numeric_cols if c in df.columns]
    n = len(cols)
    if n == 0:
        return plt.figure()
    ncols = min(max_cols, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        if hue and hue in df.columns:
            sns.histplot(data=df, x=col, hue=hue, kde=True, ax=ax, element="step", stat="density")
        else:
            sns.histplot(data=df, x=col, kde=True, ax=ax, color="#66CDAA", stat="density")
        ax.set_title(col)

    # Kosongkan sisa axes jika tidak terpakai
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribusi Fitur Numerik", y=1.02, fontsize=12, weight="bold")
    fig.tight_layout()
    return fig

def cat_vs_target(df: pd.DataFrame, cat_cols: List[str], target: str | None = "churn", wrap: int = 3):
    cols = [c for c in cat_cols if c in df.columns]
    n = len(cols)
    if n == 0:
        return plt.figure()
    ncols = min(wrap, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        if target and target in df.columns:
            sns.countplot(data=df, x=col, hue=target, ax=ax)
            for cont in ax.containers:
                ax.bar_label(cont)
            ax.legend(title=target)
        else:
            sns.countplot(data=df, x=col, ax=ax, color="#87CEFA")
            for cont in ax.containers:
                ax.bar_label(cont)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=0)

    # Kosongkan sisa axes jika tidak terpakai
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribusi Fitur Kategorikal", y=1.02, fontsize=12, weight="bold")
    fig.tight_layout()
    return fig

def corr_heatmap(df: pd.DataFrame, numeric_cols: List[str]):
    cols = [c for c in numeric_cols if c in df.columns]
    if not cols:
        return plt.figure()
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(min(1.2*len(cols), 14), min(1.2*len(cols), 14)))
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", ax=ax, cbar_kws={"shrink": .75})
    ax.set_title("Korelasi Fitur Numerik")
    fig.tight_layout()
    return fig
