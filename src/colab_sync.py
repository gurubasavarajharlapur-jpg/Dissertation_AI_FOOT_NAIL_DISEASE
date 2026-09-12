"""Persist work across Colab runtime restarts, without trusting Drive with many files.

Colab wipes /content whenever the VM is recycled, so anything expensive to
rebuild has to live somewhere else. The obvious answer — mounting Google Drive
and working directly inside it — does not survive contact with this dataset.
Drive's FUSE layer buffers writes and syncs them in the background; when the
runtime is recycled before that finishes, recent writes are simply gone. Writing
18,000 small tile images is precisely the workload it handles worst, and doing
so lost an entire session's preparation.

This script takes the opposite approach. All work happens on /content, which is
fast local disk. Finished artefacts are packed into a single archive each and
copied to Drive, because one large file syncs reliably where thousands of small
ones do not. A fresh session restores from those archives.

    python src/colab_sync.py status            # what exists locally and on Drive
    python src/colab_sync.py save              # local -> Drive archives
    python src/colab_sync.py restore           # Drive archives -> local
    python src/colab_sync.py save --what models
    python src/colab_sync.py unlink            # undo an older Drive-symlink setup

Raw data is deliberately not archived: Figshare and the ulcer repository are
re-downloadable, and the Mendeley zips you uploaded already sit on Drive as
single files, which is the shape Drive handles well.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

DEFAULT_DRIVE = Path("/content/drive/MyDrive/dissertation_foot_nail")

# Directory -> archive name. Raw data is excluded by design (see module docstring).
ARCHIVES: dict[str, str] = {
    "processed": "processed_dataset.tar",
    "models": "models.tar",
    "results": "results.tar",
}


def local_dir(what: str) -> Path:
    return {
        "processed": config.PROCESSED_DIR,
        "models": config.MODELS_DIR,
        "results": config.RESULTS_DIR,
    }[what]


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def _tree_stats(path: Path) -> tuple[int, int]:
    """(file count, total bytes) for a directory, ignoring .gitkeep."""
    if not path.exists():
        return 0, 0
    files = [p for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    return len(files), sum(p.stat().st_size for p in files)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# unlink: migrate away from the older "symlink data/ into Drive" setup
# ---------------------------------------------------------------------------
def unlink_drive_symlinks() -> None:
    """Replace Drive symlinks with real local directories, keeping any content.

    Anything already on Drive is copied back to local disk first, so switching
    over never discards work.
    """
    for rel in ("data/raw", "data/processed", "models", "results"):
        path = config.PROJECT_ROOT / rel
        if not path.is_symlink():
            print(f"  {rel:<16} not a symlink, left alone")
            continue
        target = path.resolve()
        count, size = _tree_stats(target)
        path.unlink()
        path.mkdir(parents=True, exist_ok=True)
        if count:
            print(f"  {rel:<16} copying {count} file(s), {_human(size)} back from Drive...")
            for item in target.iterdir():
                dest = path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        print(f"  {rel:<16} now a real local directory ({count} file(s) recovered)")


# ---------------------------------------------------------------------------
# save / restore
# ---------------------------------------------------------------------------
def save(what: str, drive_root: Path) -> bool:
    source = local_dir(what)
    count, size = _tree_stats(source)
    if count == 0:
        print(f"  {what:<10} nothing to save (empty)")
        return True

    archive_dir = drive_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    staged = Path("/tmp") / ARCHIVES[what]
    final = archive_dir / ARCHIVES[what]

    print(f"  {what:<10} packing {count} file(s), {_human(size)}...")
    # Built on local disk first: writing a tar directly onto the Drive mount is
    # slow and is exactly the interrupted-write case this script exists to avoid.
    with tarfile.open(staged, "w") as tar:
        tar.add(source, arcname=what, filter=lambda ti: None if ti.name.endswith(".gitkeep") else ti)

    digest = _md5(staged)
    print(f"  {what:<10} copying {_human(staged.stat().st_size)} to Drive...")
    shutil.copy2(staged, final)

    # Re-read from the mount and compare. This cannot prove the bytes reached
    # Google's servers — the mount may serve its own cache — but it does catch a
    # truncated or failed copy, which is the common failure.
    ok = final.exists() and final.stat().st_size == staged.stat().st_size
    if ok:
        verify = _md5(final)
        ok = verify == digest
    staged.unlink(missing_ok=True)

    print(f"  {what:<10} {'OK' if ok else 'MISMATCH — copy failed'}  -> {final}")
    return ok


def restore(what: str, drive_root: Path) -> bool:
    archive = drive_root / "archives" / ARCHIVES[what]
    if not archive.exists():
        print(f"  {what:<10} no archive on Drive ({archive.name})")
        return False

    destination = local_dir(what)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"  {what:<10} extracting {_human(archive.stat().st_size)}...")
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SystemExit(f"refusing to extract unsafe entry: {member.name}")
        tar.extractall(destination.parent, filter="data")

    count, size = _tree_stats(destination)
    print(f"  {what:<10} {count} file(s), {_human(size)} -> {destination}")
    return count > 0


def status(drive_root: Path) -> None:
    print(f"project : {config.PROJECT_ROOT}")
    print(f"drive   : {drive_root}  ({'mounted' if drive_root.parent.exists() else 'NOT MOUNTED'})")

    print("\nlocal (/content — lost on runtime restart):")
    for what in ARCHIVES:
        path = local_dir(what)
        count, size = _tree_stats(path)
        link = "  [SYMLINK -> Drive]" if path.is_symlink() else ""
        print(f"  {what:<10} {count:>7} file(s)  {_human(size):>10}{link}")

    raw_count, raw_size = _tree_stats(config.RAW_DIR)
    link = "  [SYMLINK -> Drive]" if config.RAW_DIR.is_symlink() else ""
    print(f"  {'raw':<10} {raw_count:>7} file(s)  {_human(raw_size):>10}{link}")

    print("\ndrive archives (survive restarts):")
    archive_dir = drive_root / "archives"
    if not archive_dir.exists():
        print("  none yet — run `save` after preprocessing or training")
        return
    for what, name in ARCHIVES.items():
        path = archive_dir / name
        if path.exists():
            stat = path.stat()
            age = (time.time() - stat.st_mtime) / 3600
            print(f"  {what:<10} {_human(stat.st_size):>10}  saved {age:.1f}h ago")
        else:
            print(f"  {what:<10} {'-':>10}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=["status", "save", "restore", "unlink"])
    parser.add_argument(
        "--what", nargs="+", choices=list(ARCHIVES), default=list(ARCHIVES),
        help="which artefacts to act on (default: all)",
    )
    parser.add_argument("--drive", type=Path, default=DEFAULT_DRIVE)
    args = parser.parse_args()

    config.ensure_dirs()

    if args.action == "status":
        status(args.drive)
        return 0

    if args.action == "unlink":
        print("replacing Drive symlinks with real local directories:")
        unlink_drive_symlinks()
        return 0

    if not args.drive.parent.exists():
        raise SystemExit(
            f"Drive is not mounted at {args.drive.parent}.\n"
            f"Run this first:\n"
            f"    from google.colab import drive\n"
            f"    drive.mount('/content/drive')"
        )

    print(f"{args.action} : {', '.join(args.what)}")
    results = [
        (save if args.action == "save" else restore)(what, args.drive) for what in args.what
    ]

    if args.action == "save":
        if all(results):
            print("\nSaved. Give Drive a minute to finish syncing before closing the "
                  "session,\nand check the files appear at drive.google.com.")
        else:
            print("\nSome archives failed to copy — do not rely on them.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
