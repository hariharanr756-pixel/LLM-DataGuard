import pandas as pd
from pathlib import Path


# ============================================================
# LLM DATAGUARD — MITIGATION EXPERIMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POISONED_PATH = BASE_DIR / "data" / "poisoned" / "train_poisoned.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "train_mitigated.csv"


def main():

    print("=" * 60)
    print("LLM DATAGUARD — MITIGATION EXPERIMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load poisoned dataset
    # --------------------------------------------------------

    print("\n[1/4] Loading poisoned dataset...")

    df = pd.read_csv(POISONED_PATH)

    print(f"Loaded {len(df):,} records.")

    # --------------------------------------------------------
    # 2. Identify suspicious / poisoned records
    # --------------------------------------------------------

    print("\n[2/4] Identifying poisoned records...")

    if "is_poisoned" not in df.columns:
        raise ValueError(
            "Column 'is_poisoned' was not found in the dataset."
        )

    poisoned_count = int(df["is_poisoned"].sum())

    print(f"Poisoned records identified: {poisoned_count:,}")

    # --------------------------------------------------------
    # 3. Remove suspicious records
    # --------------------------------------------------------

    print("\n[3/4] Applying mitigation...")

    mitigated_df = df[df["is_poisoned"] == 0].copy()

    print(
        f"Removed {len(df) - len(mitigated_df):,} suspicious records."
    )

    print(
        f"Remaining clean records: {len(mitigated_df):,}"
    )

    # --------------------------------------------------------
    # 4. Save mitigated dataset
    # --------------------------------------------------------

    print("\n[4/4] Saving mitigated dataset...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mitigated_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("MITIGATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()