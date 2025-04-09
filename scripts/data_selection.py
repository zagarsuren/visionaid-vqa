import json
import os
import shutil
import random
from collections import defaultdict
from math import ceil

# Paths
base_dir = "/Users/zagaraa/Documents/GitHub/visionaid-vqa/data/full_vizwiz"
annotations_dir = os.path.join(base_dir, "Annotations")
image_dirs = {"train": "train", "val": "val", "test": "test"}
output_dir = "/Users/zagaraa/Documents/GitHub/visionaid-vqa/data"

# Desired output directory
output_base = os.path.join(output_dir, "balanced_subset")
os.makedirs(output_base, exist_ok=True)

# Final split sizes
splits = {"train": 2000, "val": 286, "test": 571}
answer_type_distribution = {
    "yes/no": 0.048,
    "number": 0.0169,
    "other": 0.5891,
    "unanswerable": 0.346
}

# Load all annotations
def load_annotations():
    all_data = []
    for split in ["train", "val", "test"]:
        with open(os.path.join(annotations_dir, f"{split}.json"), "r") as f:
            data = json.load(f)
            for item in data:
                item["split"] = split
                all_data.append(item)
    return all_data

# Group by answer_type
def group_by_answer_type(data):
    grouped = defaultdict(list)
    for item in data:
        if "answer_type" in item:
            grouped[item["answer_type"]].append(item)
    return grouped

# Sample items
def sample_balanced(grouped, total, seed=42):
    random.seed(seed)
    desired_counts = {k: int(v * total) for k, v in answer_type_distribution.items()}
    selected = []

    for answer_type, count in desired_counts.items():
        pool = grouped.get(answer_type, [])
        if len(pool) < count:
            raise ValueError(f"Not enough samples for '{answer_type}' — found {len(pool)}, needed {count}")
        selected.extend(random.sample(pool, count))

    return selected

# Save new splits
def save_split(split_name, items):
    # Save JSON
    save_ann_path = os.path.join(output_base, f"{split_name}.json")
    with open(save_ann_path, "w") as f:
        json.dump(items, f, indent=2)

    # Copy images
    images_dir = os.path.join(output_base, split_name)
    os.makedirs(images_dir, exist_ok=True)

    for item in items:
        image_filename = item["image"]
        src_path = os.path.join(base_dir, image_dirs[item["split"]], image_filename)
        dst_path = os.path.join(images_dir, image_filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

# Main
def main():
    all_data = load_annotations()
    grouped = group_by_answer_type(all_data)

    used_ids = set()
    for split_name, count in splits.items():
        sampled = sample_balanced(grouped, count)
        save_split(split_name, sampled)

        # Remove used items to prevent duplicates across splits
        for item in sampled:
            grouped[item["answer_type"]].remove(item)

if __name__ == "__main__":
    main()