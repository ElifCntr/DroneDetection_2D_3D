import pandas as pd

# Load both files
df1 = pd.read_csv("train_index_neg.csv")
df2 = pd.read_csv("../../../data/tubelets/train_index.csv")

# Concatenate vertically
merged = pd.concat([df1, df2], ignore_index=True)

# Save to CSV
merged.to_csv("all_train_index.csv", index=False)
