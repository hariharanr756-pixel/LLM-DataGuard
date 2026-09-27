from datasets import load_dataset
from pathlib import Path

print("Downloading IMDB dataset...")

# Load dataset
dataset = load_dataset("stanfordnlp/imdb")

# Create required folders
raw_dir = Path("data/raw")
processed_dir = Path("data/processed")

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

# Convert Hugging Face datasets to pandas
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# Keep only the columns needed by our project
train_df = train_df[["text", "label"]]
test_df = test_df[["text", "label"]]

# Save CSV files
train_file = processed_dir / "train.csv"
test_file = processed_dir / "test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print("\nDATASET DOWNLOADED SUCCESSFULLY")
print("Training samples:", len(train_df))
print("Test samples:", len(test_df))

print("\nFILES CREATED:")
print(train_file)
print(test_file)

print("\nDownload and preprocessing complete!")