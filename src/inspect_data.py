"""Audit whatever is currently in data/raw/ and report what preprocessing needs.

Run this before writing or tuning any preprocessing, because the answers decide
the design:

* How many images per class?      -> whether class weighting or oversampling is
                                     needed, and whether the split ratios are
                                     even viable for the rarest class.
* What resolutions are present?    -> whether 224x224 is upscaling (blurring) or
                                     downscaling (safe).
* What formats and colour modes?   -> whether greyscale or RGBA files need
                                     converting, and whether any file is a
                                     non-image that would crash training.
* Are there corrupt/tiny files?    -> what cleaning has to remove.
* Are there duplicate images?      -> exact duplicates across classes are a
                                     labelling conflict; duplicates within a
                                     class inflate the dataset and, if they land
                                     on both sides of the split, leak test data
                                     into training.

    python src/inspect_data.py
    python src/inspect_data.py --dir data/raw/mendeley_foot
    python src/inspect_data.py --sample 500      # faster pass on a huge set

A machine-readable copy of everything printed is written to
results/metrics/raw_data_inventory.json, and the per-image table to
results/metrics/raw_data_files.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

# Pillow refuses very large images by default as a decompression-bomb guard.
# Clinical photos can legitimately be large, so raise the ceiling rather than
# silently skipping them — but keep a limit so a malicious file cannot exhaust
# memory.
Image.MAX_IMAGE_PIXELS = 300_000_000


def find_images(root: Path) -> list[Path]:
    """Every file under `root` with a recognised image extension."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in config.VALID_EXTENSIONS
        and not path.name.startswith(".")
    )


def group_label(path: Path, root: Path) -> str:
    """Best-guess class label: the immediate parent folder name.

    This is only a *guess* at how the publishers organised their data. The
    mapping from these folder names to the project's four classes is decided
    after reading this report, not here.
    """
    relative = path.relative_to(root)
    return relative.parent.as_posix() if relative.parent != Path(".") else "(root)"


def inspect_image(path: Path) -> dict:
    """Read one image's properties without decoding the full pixel data.

    `Image.open` is lazy, so size/mode/format come from the header — fast even
    on tens of thousands of files. `verify()` then confirms the file is not
    truncated, which is the failure that would otherwise crash training halfway
    through an epoch.
    """
    record: dict = {
        "path": path.as_posix(),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "file_size_bytes": path.stat().st_size,
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "corrupt": False,
        "error": "",
    }
    try:
        with Image.open(path) as img:
            record["width"], record["height"] = img.size
            record["mode"] = img.mode
            record["format"] = img.format
        # verify() consumes the file object, so it needs a fresh open.
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        record["corrupt"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """MD5 of the file bytes, for exact-duplicate detection.

    Byte-level only: it will not catch an image re-saved at a different quality.
    That is fine here — exact duplicates are the common case in datasets that
    were assembled by merging folders.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(root: Path, sample: int | None, hash_files: bool) -> pd.DataFrame:
    paths = find_images(root)
    if not paths:
        raise SystemExit(
            f"No images found under '{root}'.\n"
            f"Expected image files ({', '.join(config.VALID_EXTENSIONS)}) somewhere "
            f"beneath that directory.\n"
            f"Put the datasets there first:  python src/download_data.py"
        )

    if sample and sample < len(paths):
        # Evenly spaced rather than random, so every sub-folder is represented.
        step = len(paths) / sample
        paths = [paths[int(i * step)] for i in range(sample)]
        print(f"Sampling {len(paths)} of the images found (use --sample 0 for all).")

    records = []
    for path in tqdm(paths, desc="Reading image headers", unit="img"):
        record = inspect_image(path)
        record["label"] = group_label(path, root)
        if hash_files and not record["corrupt"]:
            record["md5"] = file_hash(path)
        records.append(record)

    frame = pd.DataFrame(records)
    frame["megapixels"] = (frame["width"] * frame["height"]) / 1e6
    frame["aspect_ratio"] = frame["width"] / frame["height"]
    return frame


def report_class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Images per source folder — the table that drives the class-balance plan."""
    counts = frame["label"].value_counts().sort_values(ascending=False)
    table = pd.DataFrame({"folder": counts.index, "images": counts.to_numpy()})
    table["percent"] = (table["images"] / table["images"].sum() * 100).round(2)

    print("\n" + "=" * 72)
    print("CLASS DISTRIBUTION (by source folder)")
    print("=" * 72)
    print(table.to_string(index=False))

    if len(table) > 1:
        largest, smallest = int(table["images"].max()), int(table["images"].min())
        ratio = largest / smallest
        print(f"\nImbalance ratio (largest : smallest) = {ratio:.1f} : 1")
        if ratio > 3:
            print(
                "  -> Notable imbalance. Plan for class weighting "
                "(config.USE_CLASS_WEIGHTS) and report macro-averaged F1, not just\n"
                "     accuracy: a model ignoring the rarest class could still look good."
            )
        else:
            print("  -> Reasonably balanced; standard training should be fine.")

    too_small = table[table["images"] < 20]
    if not too_small.empty:
        print(
            f"\n  WARNING: {len(too_small)} folder(s) have under 20 images. A 70/15/15 "
            f"split leaves\n  barely any test images there, so per-class metrics will be "
            f"very noisy:\n  {', '.join(too_small['folder'].tolist())}"
        )
    return table


def report_formats(frame: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("FILE FORMATS AND COLOUR MODES")
    print("=" * 72)

    extensions = Counter(frame["extension"])
    formats = Counter(frame["format"].dropna())
    modes = Counter(frame["mode"].dropna())

    print("\nExtensions:")
    for ext, count in extensions.most_common():
        print(f"  {ext:<8} {count:>7}")
    print("\nDecoded formats:")
    for fmt, count in formats.most_common():
        print(f"  {str(fmt):<8} {count:>7}")
    print("\nColour modes:")
    for mode, count in modes.most_common():
        print(f"  {str(mode):<8} {count:>7}")

    non_rgb = {m: c for m, c in modes.items() if m != "RGB"}
    if non_rgb:
        print(
            f"\n  -> {sum(non_rgb.values())} image(s) are not RGB ({', '.join(non_rgb)}).\n"
            f"     Preprocessing must convert everything to 3-channel RGB: both\n"
            f"     MobileNetV2 and ResNet50 expect {config.INPUT_SHAPE}."
        )
    return {"extensions": dict(extensions), "formats": {str(k): v for k, v in formats.items()}, "modes": dict(modes)}


def report_dimensions(frame: pd.DataFrame) -> dict:
    valid = frame[~frame["corrupt"]].dropna(subset=["width", "height"])
    if valid.empty:
        print("\nNo readable images to measure.")
        return {}

    print("\n" + "=" * 72)
    print("IMAGE DIMENSIONS")
    print("=" * 72)
    stats = valid[["width", "height", "megapixels", "aspect_ratio"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    print(stats.round(2).to_string())

    file_mb = valid["file_size_bytes"] / 1e6
    print(
        f"\nFile size: min {file_mb.min():.3f} MB | median {file_mb.median():.3f} MB | "
        f"max {file_mb.max():.3f} MB | total {file_mb.sum():.1f} MB"
    )

    target_w, target_h = config.IMAGE_SIZE
    upscaled = valid[(valid["width"] < target_w) | (valid["height"] < target_h)]
    print(f"\nResizing to {target_w}x{target_h}:")
    print(
        f"  {len(valid) - len(upscaled):>7} image(s) will be DOWNSCALED (no detail invented)"
    )
    print(f"  {len(upscaled):>7} image(s) will be UPSCALED (blurring, no new detail)")
    if len(upscaled) > 0.2 * len(valid):
        print(
            "  -> Over 20% would be upscaled. Consider a smaller input size, or\n"
            "     report this as a dataset limitation in the dissertation."
        )

    tiny = valid[
        (valid["width"] < config.MIN_IMAGE_DIMENSION)
        | (valid["height"] < config.MIN_IMAGE_DIMENSION)
    ]
    if not tiny.empty:
        print(
            f"\n  WARNING: {len(tiny)} image(s) are under {config.MIN_IMAGE_DIMENSION}px on a "
            f"side and will be dropped\n  by cleaning (config.MIN_IMAGE_DIMENSION)."
        )

    extreme = valid[(valid["aspect_ratio"] < 0.4) | (valid["aspect_ratio"] > 2.5)]
    if not extreme.empty:
        print(
            f"\n  NOTE: {len(extreme)} image(s) have an extreme aspect ratio. A square "
            f"resize will\n  distort these; consider pad-to-square instead of a plain stretch."
        )

    return {
        "width": {k: float(v) for k, v in valid["width"].describe().items()},
        "height": {k: float(v) for k, v in valid["height"].describe().items()},
        "total_megabytes": float(file_mb.sum()),
        "would_be_upscaled": int(len(upscaled)),
        "below_min_dimension": int(len(tiny)),
        "extreme_aspect_ratio": int(len(extreme)),
    }


def report_problems(frame: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("DATA QUALITY PROBLEMS")
    print("=" * 72)

    corrupt = frame[frame["corrupt"]]
    print(f"\nUnreadable / corrupt files: {len(corrupt)}")
    for _, row in corrupt.head(10).iterrows():
        print(f"  {row['path']}\n    {row['error']}")
    if len(corrupt) > 10:
        print(f"  ... and {len(corrupt) - 10} more")

    tiny_files = frame[frame["file_size_bytes"] < config.MIN_FILE_SIZE_BYTES]
    print(f"\nSuspiciously small files (<{config.MIN_FILE_SIZE_BYTES} bytes): {len(tiny_files)}")

    duplicates: dict[str, list[str]] = {}
    cross_class: list[str] = []
    if "md5" in frame.columns:
        by_hash = defaultdict(list)
        for _, row in frame.dropna(subset=["md5"]).iterrows():
            by_hash[row["md5"]].append(row)
        for digest, rows in by_hash.items():
            if len(rows) > 1:
                duplicates[digest] = [r["path"] for r in rows]
                if len({r["label"] for r in rows}) > 1:
                    cross_class.append(digest)

        duplicate_files = sum(len(v) - 1 for v in duplicates.values())
        print(f"\nExact duplicate images: {duplicate_files} redundant copies "
              f"across {len(duplicates)} group(s)")
        if duplicates:
            print(
                "  -> Remove duplicates before splitting. If the same image lands in\n"
                "     both train and test, the reported test accuracy is inflated."
            )
        if cross_class:
            print(
                f"\n  WARNING: {len(cross_class)} identical image(s) appear under MORE THAN ONE\n"
                f"  folder — the same picture carries two different labels. These must be\n"
                f"  resolved by hand; the label is genuinely ambiguous:"
            )
            for digest in cross_class[:5]:
                print(f"    {' <-> '.join(duplicates[digest])}")

    return {
        "corrupt_count": int(len(corrupt)),
        "corrupt_files": corrupt["path"].tolist()[:100],
        "tiny_file_count": int(len(tiny_files)),
        "duplicate_groups": len(duplicates),
        "duplicate_redundant_files": sum(len(v) - 1 for v in duplicates.values()),
        "cross_class_duplicate_groups": len(cross_class),
    }


def report_folder_tree(root: Path, max_depth: int = 3) -> None:
    """Show the directory layout, since the class mapping is derived from it."""
    print("\n" + "=" * 72)
    print(f"FOLDER STRUCTURE OF {root}")
    print("=" * 72)

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        subdirs = sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith("."))
        for index, sub in enumerate(subdirs):
            last = index == len(subdirs) - 1
            images = sum(
                1
                for p in sub.rglob("*")
                if p.is_file() and p.suffix.lower() in config.VALID_EXTENSIONS
            )
            print(f"{prefix}{'└── ' if last else '├── '}{sub.name}/  ({images} images)")
            walk(sub, prefix + ("    " if last else "│   "), depth + 1)

    walk(root, "", 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=config.RAW_DIR,
        help="directory to inspect (default: data/raw)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="inspect only N evenly-spaced images (0 = all, the default)",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="skip duplicate detection (faster; hashing reads every byte)",
    )
    args = parser.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        raise SystemExit(
            f"'{root}' does not exist.\nRun `python src/download_data.py` first, or "
            f"copy your dataset into data/raw/."
        )

    config.ensure_dirs()
    print(f"Inspecting: {root}")

    frame = build_inventory(root, args.sample or None, hash_files=not args.no_hash)
    print(f"\nTotal image files found: {len(frame)}")

    report_folder_tree(root)
    distribution = report_class_distribution(frame)
    formats = report_formats(frame)
    dimensions = report_dimensions(frame)
    problems = report_problems(frame)

    inventory = {
        "inspected_directory": str(root),
        "total_images": int(len(frame)),
        "class_distribution": distribution.to_dict(orient="records"),
        "formats": formats,
        "dimensions": dimensions,
        "problems": problems,
        "project_classes": config.CLASS_NAMES,
    }
    inventory_path = config.METRICS_DIR / "raw_data_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, default=str))

    files_path = config.METRICS_DIR / "raw_data_files.csv"
    frame.to_csv(files_path, index=False)

    print("\n" + "=" * 72)
    print("NEXT STEP")
    print("=" * 72)
    print(
        f"Saved: {inventory_path.relative_to(config.PROJECT_ROOT)}\n"
        f"       {files_path.relative_to(config.PROJECT_ROOT)}\n\n"
        f"Read the folder listing above and decide how each source folder maps onto\n"
        f"the four project classes: {', '.join(config.CLASS_NAMES)}.\n"
        f"That mapping is the input to src/preprocessing.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
