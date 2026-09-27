import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# LLM DATAGUARD - DATA POISONING
# Label-Flipping Attack
# ============================================================

# Paths
INPUT_FILE = Path("data/processed/train.csv")
OUTPUT_DIR = Path("data/poisoned")

# Poisoning configuration
POISON_RATE = 0.05
RANDOM_SEED = 42

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== LOADING CLEAN DATA ==========")

df = pd.read_csv(INPUT_FILE)

print("Original samples:", len(df))
print("Original labels:")
print(df["label"].value_counts().sort_index())

# ============================================================
# SELECT POISONED SAMPLES
# ============================================================

np.random.seed(RANDOM_SEED)

poison_count = int(len(df) * POISON_RATE)

poison_indices = np.random.choice(
    df.index,
    size=poison_count,
    replace=False
)

print("\n========== POISONING CONFIGURATION ==========")
print("Poison rate:", POISON_RATE * 100, "%")
print("Poisoned samples:", poison_count)
print("Random seed:", RANDOM_SEED)

# ============================================================
# LABEL FLIPPING
# ============================================================

poisoned_df = df.copy()

# Save original labels for verification
poisoned_df["original_label"] = poisoned_df["label"]

# Flip labels: 0 -> 1 and 1 -> 0
poisoned_df.loc[poison_indices, "label"] = (
    1 - poisoned_df.loc[poison_indices, "label"]
)

# Mark poisoned records
poisoned_df["is_poisoned"] = 0
poisoned_df.loc[poison_indices, "is_poisoned"] = 1

# ============================================================
# SAVE POISONED DATA
# ============================================================

output_file = OUTPUT_DIR / "train_poisoned.csv"

poisoned_df.to_csv(output_file, index=False)

# Save ground-truth poisoned indices
indices_file = OUTPUT_DIR / "poison_indices.csv"

pd.DataFrame({
    "index": poison_indices
}).to_csv(indices_file, index=False)

# ============================================================
# RESULTS
# ============================================================

print("\n========== POISONING COMPLETE ==========")

print("Total samples:", len(poisoned_df))
print("Poisoned samples:", poisoned_df["is_poisoned"].sum())
print("Clean samples:", (poisoned_df["is_poisoned"] == 0).sum())

print("\nFinal label distribution:")
print(poisoned_df["label"].value_counts().sort_index())

print("\nFiles created:")
print(output_file)
print(indices_file)

print("\n========== ANALYSIS COMPLETE ==========")