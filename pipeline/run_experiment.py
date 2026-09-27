import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLEAN_FILE = ROOT / "data" / "processed" / "train.csv"
POISONED_FILE = ROOT / "data" / "poisoned" / "train_poisoned.csv"
TEST_FILE = ROOT / "data" / "processed" / "test.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "results.json"

RANDOM_STATE = 42
MAX_FEATURES = 20000


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LLM DATAGUARD - MODEL EXPERIMENT")
print("=" * 60)

print("\nLoading datasets...")

clean_df = pd.read_csv(CLEAN_FILE)
poisoned_df = pd.read_csv(POISONED_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Clean training samples    : {len(clean_df)}")
print(f"Poisoned training samples : {len(poisoned_df)}")
print(f"Test samples              : {len(test_df)}")


# ============================================================
# TEXT VECTORISATION
# ============================================================

print("\nCreating TF-IDF representations...")

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_clean = vectorizer.fit_transform(clean_df["text"].astype(str))
X_poisoned = vectorizer.transform(poisoned_df["text"].astype(str))
X_test = vectorizer.transform(test_df["text"].astype(str))

y_clean = clean_df["label"].astype(int)
y_poisoned = poisoned_df["label"].astype(int)
y_test = test_df["label"].astype(int)

print(f"Number of TF-IDF features: {X_clean.shape[1]}")


# ============================================================
# CLEAN MODEL
# ============================================================

print("\n" + "-" * 60)
print("TRAINING CLEAN MODEL")
print("-" * 60)

clean_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)

clean_model.fit(X_clean, y_clean)

clean_predictions = clean_model.predict(X_test)

clean_accuracy = accuracy_score(y_test, clean_predictions)
clean_f1 = f1_score(y_test, clean_predictions)

print(f"Clean Accuracy : {clean_accuracy:.4f}")
print(f"Clean F1       : {clean_f1:.4f}")


# ============================================================
# POISONED MODEL
# ============================================================

print("\n" + "-" * 60)
print("TRAINING POISONED MODEL")
print("-" * 60)

poisoned_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)

poisoned_model.fit(X_poisoned, y_poisoned)

poisoned_predictions = poisoned_model.predict(X_test)

poisoned_accuracy = accuracy_score(y_test, poisoned_predictions)
poisoned_f1 = f1_score(y_test, poisoned_predictions)

print(f"Poisoned Accuracy : {poisoned_accuracy:.4f}")
print(f"Poisoned F1       : {poisoned_f1:.4f}")


# ============================================================
# IMPACT ANALYSIS
# ============================================================

accuracy_drop = clean_accuracy - poisoned_accuracy
f1_drop = clean_f1 - poisoned_f1

print("\n" + "=" * 60)
print("POISONING IMPACT")
print("=" * 60)

print(f"Accuracy drop : {accuracy_drop:.4f}")
print(f"F1 drop       : {f1_drop:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "config": {
        "dataset_name": "IMDB",
        "model_name": "TF-IDF + Logistic Regression",
        "train_samples": int(len(clean_df)),
        "test_samples": int(len(test_df)),
        "poisoned_samples": int(
            poisoned_df["is_poisoned"].sum()
            if "is_poisoned" in poisoned_df.columns
            else 0
        ),
        "random_state": RANDOM_STATE,
        "max_features": MAX_FEATURES
    },

    "clean_model": {
        "accuracy": float(clean_accuracy),
        "f1": float(clean_f1)
    },

    "poisoned_model": {
        "accuracy": float(poisoned_accuracy),
        "f1": float(poisoned_f1)
    },

    "impact": {
        "accuracy_drop": float(accuracy_drop),
        "f1_drop": float(f1_drop)
    }
}


with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

print(f"Results saved to:")
print(RESULTS_FILE)

print("\nNext stage: poisoning detection")