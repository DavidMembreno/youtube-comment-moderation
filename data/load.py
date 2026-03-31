import pandas as pd

# ── Jigsaw ────────────────────────────────────────────────────────────────────
jigsaw = pd.read_csv('data/raw/jigsaw/train.csv')
print("=== JIGSAW ===")
print(f"Shape: {jigsaw.shape}")
print(f"Columns: {jigsaw.columns.tolist()}")
print(f"Nulls:\n{jigsaw.isnull().sum()}")
print(f"Sample:\n{jigsaw.head(3)}\n")

# ── Spam ──────────────────────────────────────────────────────────────────────
spam = pd.read_csv('data/raw/spam/Youtube-Spam-Dataset.csv')
print("=== SPAM ===")
print(f"Shape: {spam.shape}")
print(f"Columns: {spam.columns.tolist()}")
print(f"Nulls:\n{spam.isnull().sum()}")
print(f"Class distribution:\n{spam['CLASS'].value_counts()}\n")
print(f"Sample:\n{spam.head(3)}\n")

# ── Toxicity ──────────────────────────────────────────────────────────────────
toxicity = pd.read_csv('data/raw/toxicity/youtoxic_english_1000.csv')
print("=== TOXICITY ===")
print(f"Shape: {toxicity.shape}")
print(f"Columns: {toxicity.columns.tolist()}")
print(f"Nulls:\n{toxicity.isnull().sum()}")
print(f"Class distribution:\n{toxicity['IsToxic'].value_counts()}\n")
print(f"Sample:\n{toxicity.head(3)}\n")