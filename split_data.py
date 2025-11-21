import os, json, random

DATA_ROOT = "/dtu/datasets1/02516/potholes"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
END_PATH = "/zhome/4d/5/147570/IDLCV/Project_4/results"

files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".png"))]

random.seed(42)
random.shuffle(files)

train_size = int(0.7 * len(files))
val_size = int(0.1 * len(files))
train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

splits = {"train": train_files, "val": val_files, "test": test_files}

with open(os.path.join(END_PATH, "splits.json"), "w") as f:
    json.dump(splits, f, indent=2)

print("Created splits.json")
