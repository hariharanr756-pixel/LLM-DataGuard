import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# LLM DATAGUARD
# DATA POISONING DETECTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POISONED_FILE = BASE_DIR / "data" / "poisoned" / "train_poisoned.csv"
INDEX_FILE = BASE_DIR / "data" / "poisoned" / "poison_indices.csv"
RESULTS_FILE = BASE_DIR / "results" / "results.json"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LLM DATAGUARD - POISONING DETECTION")
print("=" * 60)

print("\nLoading poisoned dataset...")

if not POISONED_FILE.exists():
    raise FileNotFoundError(
        f"Poisoned dataset not found:\n{POISONED_FILE}"
    )

if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f"Poison index file not found:\n{INDEX_FILE}"
    )

df = pd.read_csv(POISONED_FILE)
poison_indices = pd.read_csv(INDEX_FILE)

print("Dataset loaded successfully.")
print("Total samples:", len(df))


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "text",
    "label",
    "original_label",
    "is_poisoned",
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# GROUND TRUTH
# ============================================================

y_true = df["is_poisoned"].astype(int).to_numpy()

actual_poisoned = int(y_true.sum())
actual_clean = int(len(y_true) - actual_poisoned)

print("\nGround-truth poisoning:")
print("Poisoned samples:", actual_poisoned)
print("Clean samples:", actual_clean)


# ============================================================
# TEXT FEATURE EXTRACTION
# ============================================================

print("\nBuilding TF-IDF representation...")

vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=2,
    max_df=0.95,
    stop_words="english",
)

X = vectorizer.fit_transform(
    df["text"].fillna("").astype(str)
)

print("Feature matrix:", X.shape)


# ============================================================
# ISOLATION FOREST
# ============================================================

print("\nRunning Isolation Forest anomaly detection...")

poison_rate = (
    actual_poisoned / len(df)
    if len(df) > 0
    else 0.0
)

detector = IsolationForest(
    n_estimators=200,
    contamination=max(
        min(poison_rate, 0.49),
        0.001,
    ),
    random_state=42,
    n_jobs=-1,
)

detector.fit(X)


# ============================================================
# PREDICTION
# ============================================================

raw_predictions = detector.predict(X)

# Isolation Forest:
#  1  = normal
# -1  = anomaly
#
# Convert:
# anomaly -> 1
# normal  -> 0

y_pred = (raw_predictions == -1).astype(int)

flagged_samples = int(y_pred.sum())

print("Detection completed.")
print("Samples flagged:", flagged_samples)


# ============================================================
# METRICS
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

detection_f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

flag_rate = (
    flagged_samples / len(df)
    if len(df) > 0
    else 0.0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()


# ============================================================
# RESULTS
# ============================================================

detection_results = {
    "detector": "Isolation Forest",
    "feature_extraction": "TF-IDF",
    "random_state": 42,
    "total_samples": int(len(df)),
    "actual_poisoned": actual_poisoned,
    "actual_clean": actual_clean,
    "flagged_samples": flagged_samples,
    "flag_rate": float(flag_rate),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(detection_f1),
    "confusion_matrix": {
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    },
}


# ============================================================
# SAVE DETECTION CSV
# ============================================================

detection_output = df.copy()

detection_output["detector_flag"] = y_pred
detection_output["anomaly_score"] = -detector.decision_function(X)

DETECTION_FILE = (
    BASE_DIR
    / "results"
    / "detection_results.csv"
)

detection_output.to_csv(
    DETECTION_FILE,
    index=False,
)


# ============================================================
# UPDATE results.json
# ============================================================

if RESULTS_FILE.exists():

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

else:
    results = {}


results["detection"] = detection_results


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("POISONING DETECTION COMPLETE")
print("=" * 60)

print("\nDetector:")
print("Isolation Forest")

print("\nFeature extraction:")
print("TF-IDF")

print("\nDetection Metrics")
print("-" * 40)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {detection_f1:.4f}")
print(f"Flag Rate : {flag_rate:.4%}")

print("\nConfusion Matrix")
print("-" * 40)

print("True Positive :", tp)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Negative :", tn)

print("\nFiles created:")
print(DETECTION_FILE)
print(RESULTS_FILE)

print("\n" + "=" * 60)
print("NEXT STAGE: REFRESH THE DASHBOARD")
print("=" * 60)