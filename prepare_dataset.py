import os, shutil, random
from pathlib import Path

SRC = Path("data/OACE")
OPEN = SRC / "open"
CLOSE = SRC / "close"

OUT = Path("data/eyes_split")
TRAIN = OUT / "train"
VAL = OUT / "val"
TEST = OUT / "test"

VAL_PCT = 0.15
TEST_PCT = 0.15
SEED = 42

def split_and_copy(cls_name, src_dir):
    imgs = [p for p in src_dir.glob("**/*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]]
    random.shuffle(imgs)
    n = len(imgs)
    if n == 0:
        print(f"Warning: no images found in {src_dir}")
        return
    n_test = int(n*TEST_PCT)
    n_val  = int(n*VAL_PCT)
    test_set = imgs[:n_test]
    val_set  = imgs[n_test:n_test+n_val]
    train_set= imgs[n_test+n_val:]

    for subset, subset_dir in [(train_set, TRAIN/cls_name), (val_set, VAL/cls_name), (test_set, TEST/cls_name)]:
        subset_dir.mkdir(parents=True, exist_ok=True)
        for i,p in enumerate(subset):
            shutil.copy2(p, subset_dir / f"{cls_name}_{i}{p.suffix.lower()}")

def main():
    assert OPEN.exists() and CLOSE.exists(), "Expected data/OACE/open and data/OACE/close"
    random.seed(SEED)
    for d in [TRAIN, VAL, TEST]:
        d.mkdir(parents=True, exist_ok=True)
    split_and_copy("open", OPEN)
    split_and_copy("close", CLOSE)
    (Path("models")).mkdir(exist_ok=True)
    with open("models/label_map.txt","w") as f:
        f.write("0 open\n1 close\n")
    print("Dataset prepared at data/eyes_split with train/val/test.")

if __name__ == "__main__":
    main()