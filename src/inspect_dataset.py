"""Inspect the FakeAVCeleb parquet metadata."""
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import pandas as pd

parquet_files = {
    "train": [f"data/train-{i:05d}-of-00010.parquet" for i in range(10)],
    "validation": [f"data/validation-{i:05d}-of-00002.parquet" for i in range(2)],
    "test": [f"data/test-{i:05d}-of-00002.parquet" for i in range(2)],
}

all_data = []
for split, files in parquet_files.items():
    for f in files:
        local = hf_hub_download(
            repo_id="thenewsupercell/fixed-fakeavceleb",
            filename=f, repo_type="dataset"
        )
        df = pq.read_table(local).to_pandas()
        df["split"] = split
        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
print(f"Total rows: {len(df_all)}")
print(f"\nLabel distribution:")
print(df_all["label"].value_counts().sort_index())
print(f"\nType distribution:")
print(df_all["type"].value_counts())
print(f"\nCategory distribution:")
print(df_all["category"].value_counts())
print(f"\nMethod distribution:")
print(df_all["method"].value_counts())
print(f"\nPer-split counts:")
print(df_all.groupby("split")["label"].value_counts())
src_col = "source"
print(f"\nUnique sources: {df_all[src_col].nunique()}")
print(f"\nSample types per label:")
for label in sorted(df_all["label"].unique()):
    types = df_all[df_all["label"]==label]["type"].unique()
    print(f"  label={label}: {list(types)}")
print(f"\nVideo path samples:")
for i in range(min(5, len(df_all))):
    v = df_all["video"].iloc[i]
    if isinstance(v, dict):
        print(f"  {v.get('path', 'no path')}")
print(f"\nHas text: {df_all['text'].notna().sum()}/{len(df_all)}")
print(f"\nRace distribution:")
print(df_all["race"].value_counts())
print(f"\nGender distribution:")
print(df_all["gender"].value_counts())

# Save full metadata
df_all.drop(columns=["video"], errors="ignore").to_csv("d:/deepfake/data/fakeavceleb_metadata.csv", index=False)
print("\nSaved metadata to d:/deepfake/data/fakeavceleb_metadata.csv")
