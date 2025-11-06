# %%

from pathlib import Path
import shutil
import random

SRC_X = Path("/home/dils/astroai/task/data/split/x")
SRC_Y = Path("/home/dils/astroai/task/data/split/y")
OUT_ROOT = Path("/home/dils/astroai/task/data/split")  # where the train/test folders will be made
TEST_SIZE = 0.2            # 20% test split
SEED = 42                  # reproducibility
COPY_MODE = "copy"         # "copy" | "symlink" | "move"


IMG_EXTS = {".png"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def find_y_for_x(x_path: Path, y_dir: Path) -> Path | None:
    # match by stem; accept any known image extension in y folder
    stem = x_path.stem
    for ext in IMG_EXTS:
        candidate = y_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # as a fallback, exact name match (including extension)
    candidate = y_dir / x_path.name
    return candidate if candidate.exists() else None

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def transfer(src: Path, dst: Path) -> None:
    if COPY_MODE == "copy":
        shutil.copy2(src, dst)
    elif COPY_MODE == "symlink":
        # relative symlink keeps tree portable
        dst.symlink_to(src.resolve())
    elif COPY_MODE == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown COPY_MODE: {COPY_MODE}")

def main():
    assert SRC_X.exists(), f"Missing {SRC_X}"
    assert SRC_Y.exists(), f"Missing {SRC_Y}"

    x_files = sorted([p for p in SRC_X.iterdir() if is_img(p)])
    if not x_files:
        raise SystemExit(f"No images found in {SRC_X}")

    pairs: list[tuple[Path, Path]] = []
    missing = []

    for x in x_files:
        y = find_y_for_x(x, SRC_Y)
        if y is None:
            missing.append(x.name)
        else:
            pairs.append((x, y))

    if missing:
        print(f"[WARN] {len(missing)} X files have no matching Y. First few:", missing[:10])

    if not pairs:
        raise SystemExit("No matched X/Y pairs found. Check filenames and extensions.")

    random.Random(SEED).shuffle(pairs)

    n_total = len(pairs)
    n_test = max(1, int(round(TEST_SIZE * n_total)))
    n_train = n_total - n_test
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    x_train_dir = OUT_ROOT / "x_train"
    x_test_dir  = OUT_ROOT / "x_test"
    y_train_dir = OUT_ROOT / "y_train"
    y_test_dir  = OUT_ROOT / "y_test"
    ensure_dirs(x_train_dir, x_test_dir, y_train_dir, y_test_dir)

    def dump(pairs, x_out, y_out):
        for x, y in pairs:
            transfer(x, x_out / x.name)
            transfer(y, y_out / y.name)

    dump(train_pairs, x_train_dir, y_train_dir)
    dump(test_pairs,  x_test_dir,  y_test_dir)

if __name__ == "__main__":
    main()
# %%
