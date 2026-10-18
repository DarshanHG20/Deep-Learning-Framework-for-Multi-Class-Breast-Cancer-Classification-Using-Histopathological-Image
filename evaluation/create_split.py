import os
import json
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# COLLECT ALL IMAGES
# ============================================================

class_names = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

image_paths = []
labels = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

for label, class_name in enumerate(class_names):

    class_dir = os.path.join(
        DATASET_DIR,
        class_name
    )

    class_count = 0

    for root, dirs, files in os.walk(class_dir):

        for file in files:

            if file.lower().endswith(valid_extensions):

                full_path = os.path.join(
                    root,
                    file
                )

                image_paths.append(full_path)
                labels.append(label)

                class_count += 1

    print(
        f"{class_name}: {class_count} images"
    )

# ============================================================
# CHECK DATA
# ============================================================

print("\nTotal images:", len(image_paths))

if len(image_paths) == 0:
    raise ValueError("No images found in dataset.")

# ============================================================
# STRATIFIED SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# ============================================================
# SHOW VALIDATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION DISTRIBUTION")
print("=" * 60)

for i, class_name in enumerate(class_names):

    count = val_labels.count(i)

    print(
        f"{class_name:25s}: {count}"
    )

# ============================================================
# SAVE SPLIT
# ============================================================

split_data = {
    "class_names": class_names,

    "train_paths": train_paths,
    "train_labels": train_labels,

    "val_paths": val_paths,
    "val_labels": val_labels
}

split_path = os.path.join(
    RESULTS_DIR,
    "stratified_split.json"
)

with open(
    split_path,
    "w"
) as file:

    json.dump(
        split_data,
        file,
        indent=4
    )

print("\n" + "=" * 60)
print("STRATIFIED SPLIT CREATED")
print("=" * 60)

print("\nTraining images  :", len(train_paths))
print("Validation images:", len(val_paths))

print("\nSaved to:")
print(split_path)

print("\nDONE")