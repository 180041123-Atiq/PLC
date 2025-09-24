from datasets import load_dataset, Dataset
import os

# Folder to save
os.makedirs("webcode_subset", exist_ok=True)

# Stream dataset
dataset_iter = load_dataset("xcodemind/webcode2m", split="train", streaming=True)

# Collect first 317 samples
samples = []
for i, sample in enumerate(dataset_iter):
    samples.append(sample)
    if i >= 1000:  # stop after 317
        break

small_dataset = Dataset.from_list(samples)
small_dataset.save_to_disk("webcode_subset")
print("✅ Saved as Hugging Face dataset at ./webcode_subset")