from datasets import load_dataset
import pandas as pd

print("Loading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

# Convert the training split into a Pandas DataFrame
df = pd.DataFrame(dataset["train"])

print("\n========== DATASET OVERVIEW ==========")

print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LABEL DISTRIBUTION ==========")
print(df["label"].value_counts())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print("Duplicate reviews:", df["text"].duplicated().sum())

# Calculate review length
df["review_length"] = df["text"].str.len()

print("\n========== REVIEW LENGTH ==========")
print("Average characters:", round(df["review_length"].mean(), 2))
print("Minimum characters:", df["review_length"].min())
print("Maximum characters:", df["review_length"].max())

print("\n========== SAMPLE REVIEWS ==========")

for i in range(3):
    print(f"\nReview {i + 1}:")
    print(df.iloc[i]["text"][:300])
    print("Label:", df.iloc[i]["label"])

print("\n========== ANALYSIS COMPLETE ==========")