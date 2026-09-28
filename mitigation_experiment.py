import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# LLM DATAGUARD — MITIGATION EXPERIMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CLEAN_TRAIN = BASE_DIR / "data" / "processed" / "train.csv"
POISONED_TRAIN = BASE_DIR / "data" / "poisoned" / "train_poisoned.csv"
MITIGATED_TRAIN = BASE_DIR / "data" / "processed" / "train_mitigated.csv"
TEST_DATA = BASE_DIR / "data" / "processed" / "test.csv"

OUTPUT_PATH = BASE_DIR / "results" / "mitigation_results.json"


def train_and_evaluate(train_df, test_df, name):
    print(f"\nTraining: {name}")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(
        train_df["text"].astype(str)
    )

    X_test = vectorizer.transform(
        test_df["text"].astype(str)
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, train_df["label"])

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        test_df["label"],
        predictions
    )

    f1 = f1_score(
        test_df["label"],
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1)
    }


def main():

    print("=" * 60)
    print("LLM DATAGUARD — MITIGATION EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    clean_df = pd.read_csv(CLEAN_TRAIN)
    poisoned_df = pd.read_csv(POISONED_TRAIN)
    mitigated_df = pd.read_csv(MITIGATED_TRAIN)
    test_df = pd.read_csv(TEST_DATA)

    print(f"Clean training samples:     {len(clean_df):,}")
    print(f"Poisoned training samples:  {len(poisoned_df):,}")
    print(f"Mitigated training samples: {len(mitigated_df):,}")
    print(f"Test samples:               {len(test_df):,}")

    # --------------------------------------------------------
    # Train three conditions
    # --------------------------------------------------------

    clean = train_and_evaluate(
        clean_df,
        test_df,
        "CLEAN DATA"
    )

    poisoned = train_and_evaluate(
        poisoned_df,
        test_df,
        "POISONED DATA"
    )

    mitigated = train_and_evaluate(
        mitigated_df,
        test_df,
        "MITIGATED DATA"
    )

    # --------------------------------------------------------
    # Calculate recovery
    # --------------------------------------------------------

    accuracy_drop = clean["accuracy"] - poisoned["accuracy"]
    f1_drop = clean["f1"] - poisoned["f1"]

    accuracy_recovery = mitigated["accuracy"] - poisoned["accuracy"]
    f1_recovery = mitigated["f1"] - poisoned["f1"]

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "experiment": "mitigation",
        "clean": clean,
        "poisoned": poisoned,
        "mitigated": mitigated,
        "impact": {
            "accuracy_drop": float(accuracy_drop),
            "f1_drop": float(f1_drop),
            "accuracy_recovery": float(accuracy_recovery),
            "f1_recovery": float(f1_recovery)
        },
        "dataset": {
            "clean_samples": len(clean_df),
            "poisoned_samples": len(poisoned_df),
            "mitigated_samples": len(mitigated_df),
            "test_samples": len(test_df)
        }
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MITIGATION RESULTS")
    print("=" * 60)

    print(
        f"\nClean Accuracy:      {clean['accuracy'] * 100:.2f}%"
    )

    print(
        f"Poisoned Accuracy:   {poisoned['accuracy'] * 100:.2f}%"
    )

    print(
        f"Mitigated Accuracy:  {mitigated['accuracy'] * 100:.2f}%"
    )

    print(
        f"\nClean F1:             {clean['f1'] * 100:.2f}%"
    )

    print(
        f"Poisoned F1:          {poisoned['f1'] * 100:.2f}%"
    )

    print(
        f"Mitigated F1:         {mitigated['f1'] * 100:.2f}%"
    )

    print(
        f"\nAccuracy Recovery:   {accuracy_recovery * 100:.2f} percentage points"
    )

    print(
        f"F1 Recovery:         {f1_recovery * 100:.2f} percentage points"
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("MITIGATION EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()