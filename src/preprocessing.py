"""Clean, resize and split the raw images into data/processed/.

Implements WBS 3.4-3.8: data cleaning, labelling and verification, resizing and
normalisation, augmentation, and dataset validation.

    python src/preprocessing.py --dry-run    # report every step, write nothing
    python src/preprocessing.py

Pipeline, with counts logged after each step so the effect of every stage is
visible and quotable in the methodology chapter:

  1. MAP     Assign each file under data/raw/ to one of the four project
             classes using config.FOLDER_TO_CLASS. Unmapped files are reported
             and skipped rather than guessed at.
  2. CLEAN   Drop unreadable, truncated, undersized and duplicate images.
  3. CROP    Remove the zero-padding from the ulcer images (see below).
  4. CONVERT Force 3-channel RGB.
  5. RESIZE  Aspect-preserving resize to config.IMAGE_SIZE, then centre crop.
  6. SPLIT   Stratified, grouped 70/15/15 with config.RANDOM_SEED.
  7. WRITE   data/processed/<split>/<class>/, plus CSV manifests.
  8. VERIFY  Re-read what was written and assert the split is sound.

Why the ulcer crop (step 3). The FUSeg and Medetec publishers cropped each
image to the wound and zero-padded it to square, so 25-32% of every ulcer image
is pure black — a marking no other class carries. Without this step a model
learns "black border => ulcer", scores near-perfectly, and has learned nothing
about ulcers. This is data cleaning in the WBS 3.4 sense: removing an artefact
of how the data was prepared, not altering its clinical content.

Why pixel scaling is NOT applied here. MobileNetV2 expects inputs in [-1, 1];
ResNet50 expects caffe-style mean subtraction. Each applies its own
`preprocess_input` inside its own graph at training time. Baking either into the
files on disk would handicap the other architecture and invalidate the
comparison that this dissertation is built around.

Why augmentation is NOT applied here. Augmentation must vary between epochs to
be useful, and must never touch validation or test. It is therefore defined as a
`tf.data` stage (see `build_augmentation`) and applied to the training split at
training time, not written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 300_000_000

SPLITS = ("train", "val", "test")


@dataclass
class StepLog:
    """Running record of what each stage kept and dropped."""

    steps: list[dict] = field(default_factory=list)

    def record(self, step: str, kept: int, dropped: int = 0, detail: str = "") -> None:
        self.steps.append({"step": step, "kept": kept, "dropped": dropped, "detail": detail})
        line = f"[{step:<22}] kept {kept:>6}"
        if dropped:
            line += f"   dropped {dropped:>5}"
        if detail:
            line += f"   {detail}"
        print(line)


# ---------------------------------------------------------------------------
# 1. MAP
# ---------------------------------------------------------------------------
def map_to_class(path: Path, raw_root: Path) -> str | None:
    """Resolve one file to a project class, or None if excluded/unmapped.

    Leaf rules are checked first so that a specific sub-folder name
    (``onychomycosis``) wins over a broader parent pattern.
    """
    relative = path.relative_to(raw_root).as_posix()
    for leaf, target in config.LEAF_FOLDER_TO_CLASS.items():
        if f"/{leaf}/" in f"/{relative}":
            return target
    for fragment, target in config.FOLDER_TO_CLASS.items():
        if fragment in relative:
            return target
    return None


def build_inventory(raw_root: Path, log: StepLog) -> pd.DataFrame:
    paths = [
        p for p in sorted(raw_root.rglob("*"))
        if p.is_file() and p.suffix.lower() in config.VALID_EXTENSIONS
        and not p.name.startswith(".")
    ]
    if not paths:
        raise SystemExit(
            f"No images under '{raw_root}'. Run src/download_data.py first."
        )
    log.record("scan data/raw", len(paths))

    rows, unmapped = [], Counter()
    for path in paths:
        target = map_to_class(path, raw_root)
        if target is None:
            unmapped[path.relative_to(raw_root).parent.as_posix()] += 1
            continue
        rows.append({"path": path.as_posix(), "label": target,
                     "source": _source_of(path, raw_root)})

    frame = pd.DataFrame(rows)
    log.record("map to classes", len(frame), sum(unmapped.values()),
               f"{len(unmapped)} folder(s) excluded or unmapped")
    for folder, count in unmapped.most_common(12):
        print(f"      excluded: {folder:<55} {count:>6}")
    if frame.empty:
        raise SystemExit(
            "Nothing mapped to a class. Check config.FOLDER_TO_CLASS against the "
            "folder names reported by src/inspect_data.py."
        )
    return frame


def _source_of(path: Path, raw_root: Path) -> str:
    """Top-level dataset directory, kept so per-source effects stay visible."""
    return path.relative_to(raw_root).parts[0]


# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------
def _file_hash(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def clean(frame: pd.DataFrame, log: StepLog) -> pd.DataFrame:
    """Drop unreadable, undersized and duplicate images."""
    keep, corrupt, tiny = [], 0, 0
    for row in tqdm(frame.itertuples(), total=len(frame), desc="  verifying", unit="img"):
        path = Path(row.path)
        if path.stat().st_size < config.MIN_FILE_SIZE_BYTES:
            tiny += 1
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
            with Image.open(path) as img:
                img.verify()
        except (OSError, ValueError):
            corrupt += 1
            continue
        if min(width, height) < config.MIN_IMAGE_DIMENSION:
            tiny += 1
            continue
        keep.append(row.Index)

    started = len(frame)
    frame = frame.loc[keep].copy()
    # Both filters run in one pass, so the intermediate "kept" is derived rather
    # than measured — otherwise each line would report the same final count and
    # the log would misrepresent where images were lost.
    log.record("clean: readable", started - corrupt, corrupt, "unreadable or truncated")
    log.record("clean: min size", len(frame), tiny,
               f"< {config.MIN_IMAGE_DIMENSION}px or < {config.MIN_FILE_SIZE_BYTES}B")

    digests = [_file_hash(Path(p)) for p in
               tqdm(frame["path"], desc="  hashing", unit="img")]
    frame["md5"] = digests
    before = len(frame)

    cross = frame.groupby("md5")["label"].nunique()
    conflicting = set(cross[cross > 1].index)
    if conflicting:
        # The same picture filed under two classes: the ground truth is
        # genuinely ambiguous, so neither copy is trustworthy.
        frame = frame[~frame["md5"].isin(conflicting)]
        log.record("clean: label conflicts", len(frame), before - len(frame),
                   f"{len(conflicting)} image(s) appeared under >1 class")

    before = len(frame)
    frame = frame.drop_duplicates(subset="md5", keep="first")
    log.record("clean: duplicates", len(frame), before - len(frame),
               "identical copies removed")
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2b. CAP
# ---------------------------------------------------------------------------
def cap_sources(frame: pd.DataFrame, log: StepLog) -> pd.DataFrame:
    """Limit how many images any one source folder contributes.

    Sampling is spread evenly across groups rather than taken in filename order.
    Taking the first N tiles would draw them all from the first one or two
    montage sheets, collapsing the already-limited session diversity to almost
    nothing while appearing to keep a healthy sample size.
    """
    if not config.MAX_IMAGES_PER_SOURCE:
        return frame

    rng = np.random.default_rng(config.RANDOM_SEED)
    frame = frame.copy()
    if "group" not in frame.columns:
        frame["group"] = [_group_key(r) for r in frame.itertuples()]

    keep_index: list = []
    for source, subset in frame.groupby("source"):
        cap = next(
            (c for marker, c in config.MAX_IMAGES_PER_SOURCE.items() if marker in source),
            None,
        )
        if cap is None or len(subset) <= cap:
            keep_index.extend(subset.index)
            continue

        # Round-robin over groups: take one image from each in turn, so every
        # sheet contributes before any sheet contributes twice.
        by_group = {g: list(rng.permutation(idx.to_numpy()))
                    for g, idx in subset.groupby("group").groups.items()}
        chosen: list = []
        while len(chosen) < cap and any(by_group.values()):
            for bucket in by_group.values():
                if bucket and len(chosen) < cap:
                    chosen.append(bucket.pop())
        keep_index.extend(chosen)
        log.record(f"cap: {source[:28]}", len(chosen), len(subset) - len(chosen),
                   f"capped at {cap}, spread over {len(by_group)} group(s)")

    return frame.loc[sorted(keep_index)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3-5. CROP / CONVERT / RESIZE
# ---------------------------------------------------------------------------
def crop_black_padding(image: Image.Image, threshold: int) -> Image.Image:
    """Crop to the bounding box of non-black content.

    Returns the image unchanged when no padding is found, so a photograph that
    merely happens to be dark is never cropped into.
    """
    array = np.asarray(image.convert("RGB"))
    content = array.max(axis=2) > threshold
    if not content.any():
        return image
    rows = np.flatnonzero(content.any(axis=1))
    cols = np.flatnonzero(content.any(axis=0))
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    if box == (0, 0, image.width, image.height):
        return image
    return image.crop(box)


def resize_with_aspect(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale the shorter side to target, then centre crop.

    Chosen over a plain stretch, which distorts the 153 images with extreme
    aspect ratios, and over pad-to-square, which would reintroduce exactly the
    black border this pipeline removes from the ulcer images.
    """
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale))),
        Image.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def process_image(path: Path, source: str) -> Image.Image:
    """Crop padding (where applicable), force RGB, resize. Steps 3-5."""
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if any(marker in source for marker in config.CROP_BLACK_BORDERS):
        image = crop_black_padding(image, config.BLACK_THRESHOLD)
    return resize_with_aspect(image, config.IMAGE_SIZE)


# ---------------------------------------------------------------------------
# 6. SPLIT
# ---------------------------------------------------------------------------
def _group_key(row) -> str:
    """Images that must not be separated across splits.

    Tiles cut from one montage sheet come from the same photographic session and
    frequently the same person, so they are grouped by sheet. Everything else is
    its own group, leaving an ordinary stratified split.
    """
    path = Path(row.path)
    if "figshare_nail_tiles" in path.as_posix():
        # "..._r012c004.png" -> the sheet it was cut from
        return path.stem.rsplit("_r", 1)[0]
    return path.as_posix()


def stratified_group_split(frame: pd.DataFrame, log: StepLog) -> pd.DataFrame:
    """Assign each row to train/val/test, grouped by source image.

    Stratified by class AND by source dataset, not class alone. Stratifying on
    class only allowed an entire sub-population to be confined to one split: the
    montage tiles arrive as ~20 groups of ~50 images each, far larger than the
    single-image groups around them, so a largest-first assignment sent every
    one of them to training and left the test set with no healthy nail images at
    all. Performance on precisely the category the tiles were extracted to
    provide then went unmeasured.

    Splitting within each (class, source) pair guarantees every source appears
    in every split in the intended proportions, which also makes the per-source
    breakdown in evaluation meaningful.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    frame = frame.copy()
    frame["group"] = [_group_key(r) for r in frame.itertuples()]

    targets = {"train": config.TRAIN_SPLIT, "val": config.VAL_SPLIT, "test": config.TEST_SPLIT}
    assignment: dict[str, str] = {}

    for (label, source), subset in frame.groupby(["label", "source"], sort=True):
        sizes = subset.groupby("group").size().to_dict()
        groups = sorted(sizes)
        rng.shuffle(groups)
        # Largest groups first keeps the realised ratios closest to target.
        groups.sort(key=lambda g: -sizes[g])

        filled = dict.fromkeys(SPLITS, 0)
        total = float(sum(sizes.values()))
        for group in groups:
            deficit = {s: targets[s] - filled[s] / total for s in SPLITS if targets[s] > 0}
            chosen = max(deficit, key=lambda s: (deficit[s], s == "train"))
            assignment[group] = chosen
            filled[chosen] += sizes[group]

    frame["split"] = frame["group"].map(assignment)
    counts = frame["split"].value_counts()
    log.record("split (70/15/15)", len(frame), 0,
               " ".join(f"{s}={int(counts.get(s, 0))}" for s in SPLITS))
    return frame


# ---------------------------------------------------------------------------
# 7-8. WRITE / VERIFY
# ---------------------------------------------------------------------------
def _repo_relative(path: Path) -> str:
    """Path relative to the project root where possible, absolute otherwise.

    Manifests stay portable for the normal case, without breaking when --out
    points somewhere outside the repository.
    """
    try:
        return path.relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _clear_output_dir(out_root: Path) -> None:
    """Empty the output directory without destroying it.

    On Colab this path is typically a symlink into Google Drive, and rmtree
    refuses to follow one. Clearing the contents keeps the link (and therefore
    the persistence across runtime restarts) intact.
    """
    if not out_root.exists():
        out_root.mkdir(parents=True, exist_ok=True)
        return
    for child in out_root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def write_processed(frame: pd.DataFrame, out_root: Path, log: StepLog) -> pd.DataFrame:
    _clear_output_dir(out_root)
    records, failures = [], 0

    for row in tqdm(frame.itertuples(), total=len(frame), desc="  writing", unit="img"):
        out_dir = out_root / row.split / row.label
        out_dir.mkdir(parents=True, exist_ok=True)
        # md5 prefix keeps filenames unique across sources without collisions.
        out_path = out_dir / f"{row.md5[:10]}_{Path(row.path).stem[:60]}.jpg"
        try:
            image = process_image(Path(row.path), row.source)
            image.save(out_path, "JPEG", quality=95)
        except (OSError, ValueError):
            failures += 1
            continue
        records.append({
            "path": _repo_relative(out_path),
            "source_path": row.path, "label": row.label, "split": row.split,
            "source": row.source, "group": row.group,
        })

    processed = pd.DataFrame(records)
    log.record("write processed", len(processed), failures, f"-> {out_root}")
    return processed


def verify(processed: pd.DataFrame, log: StepLog) -> dict:
    """Assert the split is sound before anything downstream trusts it."""
    problems = []

    overlap = processed.groupby("group")["split"].nunique()
    leaked = overlap[overlap > 1]
    if len(leaked):
        problems.append(f"{len(leaked)} group(s) span more than one split — data leakage")

    for split in SPLITS:
        present = set(processed[processed["split"] == split]["label"])
        missing = set(config.CLASS_NAMES) - present
        if missing:
            problems.append(f"split '{split}' has no examples of {sorted(missing)}")

    # A source confined to one split leaves its sub-population unmeasured, which
    # is how the test set ended up containing no healthy nail images.
    all_sources = set(processed["source"])
    for split in SPLITS:
        present = set(processed[processed["split"] == split]["source"])
        missing = all_sources - present
        if missing:
            problems.append(
                f"split '{split}' contains nothing from source(s) {sorted(missing)} — "
                f"performance on those images would go unmeasured"
            )

    sizes = Counter()
    for path in processed["path"].head(200):
        with Image.open(config.PROJECT_ROOT / path) as img:  # absolute paths pass through
            sizes[img.size] += 1
    if set(sizes) != {config.IMAGE_SIZE}:
        problems.append(f"unexpected image sizes on disk: {dict(sizes)}")

    if problems:
        for problem in problems:
            print(f"      FAIL: {problem}")
        raise SystemExit("dataset validation failed — see above")

    log.record("verify", len(processed), 0, "no leakage, all classes present, sizes correct")
    return {"groups": int(processed['group'].nunique()), "images": int(len(processed))}


def distribution_table(processed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in config.CLASS_NAMES:
        subset = processed[processed["label"] == label]
        row = {"class": config.CLASS_DISPLAY_NAMES[label]}
        for split in SPLITS:
            row[split] = int((subset["split"] == split).sum())
        row["total"] = len(subset)
        rows.append(row)
    table = pd.DataFrame(rows)
    totals = {"class": "TOTAL"}
    for column in (*SPLITS, "total"):
        totals[column] = int(table[column].sum())
    return pd.concat([table, pd.DataFrame([totals])], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="run every step and report, but write no images")
    args = parser.parse_args()

    config.ensure_dirs()
    log = StepLog()
    print(f"seed={config.RANDOM_SEED}  target size={config.IMAGE_SIZE}  "
          f"classes={config.CLASS_NAMES}\n")

    frame = build_inventory(args.raw, log)
    print("\n   per-class after mapping:")
    for label, count in frame["label"].value_counts().items():
        print(f"      {label:<14} {count:>6}")
    print()

    missing = [c for c in config.CLASS_NAMES if (frame["label"] == c).sum() == 0]
    if missing:
        raise SystemExit(
            f"\nNo images mapped to: {missing}\n"
            f"Only {sorted(frame['label'].unique())} were found, from "
            f"{sorted(frame['source'].unique())}.\n\n"
            f"On Colab this usually means Google Drive has not finished mounting —\n"
            f"large directories are listed lazily, so data/raw can look partly\n"
            f"empty for several minutes after drive.mount(). Check with:\n"
            f"    !ls /content/drive/MyDrive/dissertation_foot_nail/data/raw\n"
            f"    !find /content/drive/MyDrive/dissertation_foot_nail/data/raw "
            f"-type f | wc -l\n"
            f"and re-run once the counts look right.\n\n"
            f"Otherwise check config.FOLDER_TO_CLASS against the folder names "
            f"reported by src/inspect_data.py."
        )

    frame = clean(frame, log)
    frame = cap_sources(frame, log)
    frame = stratified_group_split(frame, log)

    if args.dry_run:
        print("\n--dry-run: stopping before any image is written.\n")
        print(distribution_table(frame.rename(columns={"split": "split"})).to_string(index=False))
        return 0

    processed = write_processed(frame, args.out, log)
    stats = verify(processed, log)

    # Manifests live beside the images they describe, so a run written to a
    # non-default --out stays self-contained.
    args.out.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.out / "manifest.csv", index=False)
    for split in SPLITS:
        processed[processed["split"] == split].to_csv(args.out / f"{split}.csv", index=False)

    table = distribution_table(processed)
    table.to_csv(config.METRICS_DIR / "class_distribution.csv", index=False)

    source_table = (
        processed.groupby(["label", "source", "split"]).size()
        .unstack(fill_value=0).reindex(columns=list(SPLITS), fill_value=0).reset_index()
    )
    source_table.to_csv(config.METRICS_DIR / "split_by_source.csv", index=False)
    (config.METRICS_DIR / "preprocessing_log.json").write_text(
        json.dumps({"steps": log.steps, "stats": stats,
                    "seed": config.RANDOM_SEED,
                    "image_size": list(config.IMAGE_SIZE)}, indent=2)
    )

    print("\n" + "=" * 64)
    print("FINAL CLASS DISTRIBUTION")
    print("=" * 64)
    print(table.to_string(index=False))

    print("\n" + "=" * 64)
    print("SPLIT BY CLASS AND SOURCE")
    print("=" * 64)
    print(source_table.to_string(index=False))

    counts = table[table["class"] != "TOTAL"]["total"]
    print(f"\nimbalance ratio (largest : smallest) = {counts.max() / max(counts.min(), 1):.1f} : 1")
    print(f"\nwritten to {args.out}")
    print(f"manifests  {args.out}/train.csv, val.csv, test.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
