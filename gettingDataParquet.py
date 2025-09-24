from datasets import load_dataset, Dataset
import os

# Folder to save
os.makedirs("webcode_large_subset", exist_ok=True)

dataset_iter = load_dataset("xcodemind/webcode2m", split="train", streaming=True)

max_samples = 1000
buffer_size = 100
buffer = []
count = 0
file_index = 0

for sample in dataset_iter:
    buffer.append(sample)
    count += 1

    if len(buffer) >= buffer_size:
        ds = Dataset.from_list(buffer)
        file_index += 1
        ds.to_parquet(f"webcode_large_subset/part_{file_index}.parquet")
        buffer = []

    if count >= max_samples:
        break

# Flush any remaining samples
if buffer:
    ds = Dataset.from_list(buffer)
    file_index += 1
    ds.to_parquet(f"webcode_large_subset/part_{file_index}.parquet")

print(f"✅ Saved {count} samples across {file_index} Parquet files in ./webcode_large_subset")
