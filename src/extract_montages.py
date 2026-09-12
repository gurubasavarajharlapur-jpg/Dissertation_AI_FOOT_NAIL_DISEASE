"""Tile the Figshare montage sheets into individual nail images.

The Figshare A1/A2 archives do not contain individual photographs. Each file is
a *contact sheet*: several hundred small nail thumbnails tiled into one very
large image (2260px wide, up to 13,252px tall, ~50MB). Fed to a classifier
as-is, one sheet would be a single training example containing hundreds of
nails — so these must be cut apart before they can be used at all.

The sheet's class is encoded in its filename, after the last '#':

    L#NAIL#5nail#5nail2005_2#normalnail.png     -> normalnail
    L#NAIL#5nail#5nail2013_1#naildystrophy.png  -> naildystrophy
    L#NAIL#5nail#5nail2007_2#-focus.png         -> -focus

Every tile on a sheet shares that sheet's label, which is what makes the
extraction worthwhile: it yields labelled healthy-nail images, the one category
missing from the per-image folders.

    python src/extract_montages.py --dry-run     # report geometry, write nothing
    python src/extract_montages.py --preview     # + contact sheets to eyeball
    python src/extract_montages.py               # extract

Always run --dry-run first and check the reported grid against the previews. A
misdetected grid silently produces thousands of half-nail crops, and no metric
downstream would reveal it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

# These sheets are far larger than Pillow's default bomb threshold.
Image.MAX_IMAGE_PIXELS = 300_000_000

# Sheet label (after the final '#') -> project class. Sheets whose label maps to
# None are recognised but deliberately not extracted.
LABEL_TO_CLASS: dict[str, str | None] = {
    "normalnail": "healthy",
    "naildystrophy": None,  # not one of the four classes fixed by the proposal
    "-focus": None,         # focus-variant sheets; class not stated by the source
}


@dataclass
class Grid:
    """Detected tile layout of one montage sheet."""

    rows: list[tuple[int, int]]
    cols: list[tuple[int, int]]
    background: int
    method: str

    @property
    def n_tiles(self) -> int:
        return len(self.rows) * len(self.cols)

    def describe(self) -> str:
        rh = _modal_span(self.rows)
        cw = _modal_span(self.cols)
        return (
            f"{len(self.cols)} cols x {len(self.rows)} rows = {self.n_tiles} tiles "
            f"| cell ~{cw}x{rh}px | bg={self.background} | {self.method}"
        )


def _modal_span(bands: list[tuple[int, int]]) -> int:
    if not bands:
        return 0
    return Counter(b - a for a, b in bands).most_common(1)[0][0]


def sheet_label(path: Path) -> str:
    """The text after the final '#' in the filename, lowercased."""
    stem = path.stem
    return (stem.rsplit("#", 1)[-1] if "#" in stem else stem).strip().lower()


def _background_value(gray: np.ndarray) -> int:
    """Modal intensity of the sheet border, taken as the separator colour."""
    border = np.concatenate(
        [gray[0], gray[-1], gray[:, 0], gray[:, -1]]
    )
    return int(np.bincount(border, minlength=256).argmax())


def _bands(is_separator: np.ndarray, min_span: int) -> list[tuple[int, int]]:
    """Contiguous runs of non-separator lines, as (start, end) pairs."""
    bands: list[tuple[int, int]] = []
    start = None
    for i, sep in enumerate(is_separator):
        if not sep and start is None:
            start = i
        elif sep and start is not None:
            if i - start >= min_span:
                bands.append((start, i))
            start = None
    if start is not None and len(is_separator) - start >= min_span:
        bands.append((start, len(is_separator)))
    return bands


def detect_grid(gray: np.ndarray, min_cell: int, tolerance: float) -> Grid:
    """Locate tile boundaries from background-coloured separator lines.

    A row or column consisting almost entirely of the background colour is a
    separator; everything between separators is a tile band. This adapts to
    sheets with different tile counts, which a hard-coded grid could not.
    """
    background = _background_value(gray)
    near_bg = np.abs(gray.astype(np.int16) - background) <= 12

    row_sep = near_bg.mean(axis=1) >= tolerance
    col_sep = near_bg.mean(axis=0) >= tolerance

    rows = _bands(row_sep, min_cell)
    cols = _bands(col_sep, min_cell)

    if _plausible(rows, cols, gray.shape):
        return Grid(rows, cols, background, "separator-detection")

    # Tiles butted together with no separator line. Separator detection returns
    # one enormous band in that case, which is why the result is validated above
    # rather than trusted: silently emitting a single 2260x8192 "tile" would be
    # worse than failing. Recover the cell size from the periodicity of the
    # sheet's edge profile instead — tile boundaries are intensity
    # discontinuities even when nothing separates them.
    period_c = _dominant_period(_edge_profile(gray, axis=0), min_cell)
    period_r = _dominant_period(_edge_profile(gray, axis=1), min_cell)
    if not period_c or not period_r:
        return Grid([], [], background, "failed")

    cols = [(x, x + period_c) for x in range(0, gray.shape[1] - period_c + 1, period_c)]
    rows = [(y, y + period_r) for y in range(0, gray.shape[0] - period_r + 1, period_r)]
    if not _plausible(rows, cols, gray.shape):
        return Grid([], [], background, "failed")
    return Grid(rows, cols, background, f"uniform-grid({period_c}x{period_r})")


def _plausible(rows: list, cols: list, shape: tuple[int, int]) -> bool:
    """Reject a 'grid' that is really one undivided block.

    A montage has many tiles, each small relative to the sheet. One or two bands
    spanning most of the image means detection failed, however well-formed the
    result looks.
    """
    if len(rows) < 2 or len(cols) < 2:
        return False
    height, width = shape
    return _modal_span(rows) <= height / 3 and _modal_span(cols) <= width / 3


def _edge_profile(gray: np.ndarray, axis: int) -> np.ndarray:
    """Mean absolute intensity change along one axis.

    Peaks mark tile boundaries: adjacent tiles rarely meet at matching
    intensities, so the seam shows as a discontinuity even without a gap.
    """
    diff = np.abs(np.diff(gray.astype(np.int16), axis=1 - axis))
    return diff.mean(axis=axis)


def _dominant_period(profile: np.ndarray, min_period: int) -> int | None:
    """Strongest repeating spacing in a 1-D profile, via autocorrelation."""
    signal = profile.astype(np.float64) - profile.mean()
    if not np.any(signal):
        return None
    corr = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
    search = corr[min_period : min(len(corr), 400)]
    if search.size == 0 or search.max() <= 0:
        return None
    return int(search.argmax() + min_period)


def _is_blank(tile: np.ndarray, background: int) -> bool:
    """Reject padding cells: near-uniform, or almost entirely background."""
    gray = tile.mean(axis=2) if tile.ndim == 3 else tile
    if np.abs(gray - background).mean() < 6:
        return True
    return float(gray.std()) < 8.0


def _fixed_grid(shape: tuple[int, int], cell: tuple[int, int], background: int) -> Grid:
    """Uniform grid of a caller-specified cell size (the --cell override)."""
    height, width = shape
    cw, ch = cell
    cols = [(x, x + cw) for x in range(0, width - cw + 1, cw)]
    rows = [(y, y + ch) for y in range(0, height - ch + 1, ch)]
    return Grid(rows, cols, background, f"fixed-cell({cw}x{ch})")


def extract_sheet(
    path: Path, out_dir: Path, min_cell: int, tolerance: float, write: bool,
    cell: tuple[int, int] | None = None,
) -> tuple[Grid, int]:
    """Cut one sheet into tiles. Returns the detected grid and tiles kept."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    gray = rgb.mean(axis=2).astype(np.uint8)

    grid = (
        _fixed_grid(gray.shape, cell, background=_background_value(gray))
        if cell
        else detect_grid(gray, min_cell, tolerance)
    )
    if grid.method == "failed":
        return grid, 0

    kept = 0
    stem = path.stem.replace("#", "_")
    for r, (y0, y1) in enumerate(grid.rows):
        for c, (x0, x1) in enumerate(grid.cols):
            tile = rgb[y0:y1, x0:x1]
            if tile.shape[0] < min_cell or tile.shape[1] < min_cell:
                continue
            if _is_blank(tile, grid.background):
                continue
            kept += 1
            if write:
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile).save(out_dir / f"{stem}_r{r:03d}c{c:03d}.png")
    return grid, kept


def write_preview(path: Path, grid: Grid, out_path: Path, n: int = 24) -> None:
    """Contact sheet of the first N tiles the detected grid produces."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))

    tiles = []
    for y0, y1 in grid.rows:
        for x0, x1 in grid.cols:
            tile = rgb[y0:y1, x0:x1]
            if not _is_blank(tile, grid.background):
                tiles.append(tile)
            if len(tiles) >= n:
                break
        if len(tiles) >= n:
            break
    if not tiles:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 8
    rows = (len(tiles) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    for ax, tile in zip(np.atleast_1d(axes).ravel(), tiles):
        ax.imshow(tile)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(tiles):]:
        ax.axis("off")
    fig.suptitle(f"{path.name}\n{grid.describe()}", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=config.RAW_DIR / "figshare_nail")
    parser.add_argument(
        "--out", type=Path, default=config.RAW_DIR / "figshare_nail_tiles",
        help="where extracted tiles are written, as <out>/<class>/",
    )
    parser.add_argument(
        "--label", default="normalnail",
        help="only process sheets with this filename label (default: normalnail); "
             "'all' processes every sheet with a mapped class",
    )
    parser.add_argument("--min-cell", type=int, default=40, help="smallest acceptable tile side, px")
    parser.add_argument(
        "--tolerance", type=float, default=0.97,
        help="fraction of background pixels for a line to count as a separator",
    )
    parser.add_argument("--dry-run", action="store_true", help="report geometry, write nothing")
    parser.add_argument("--preview", action="store_true", help="also write contact sheets")
    parser.add_argument("--limit", type=int, default=0, help="process at most N sheets")
    parser.add_argument(
        "--cell", default="",
        help="force a uniform grid of WxH pixel cells (e.g. 113x128), bypassing "
             "automatic detection. Use when --dry-run reports a failure or a "
             "grid the previews show to be wrong.",
    )
    args = parser.parse_args()

    cell = None
    if args.cell:
        try:
            cw, ch = (int(v) for v in args.cell.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--cell must look like 113x128, got {args.cell!r}")
        cell = (cw, ch)

    if not args.source.is_dir():
        raise SystemExit(f"'{args.source}' does not exist. Download the Figshare data first.")

    sheets = [
        p for p in sorted(args.source.rglob("*"))
        if p.is_file() and p.suffix.lower() in config.VALID_EXTENSIONS
    ]
    by_label = Counter(sheet_label(p) for p in sheets)
    print(f"{len(sheets)} file(s) under {args.source}")
    print("labels found:")
    for label, count in by_label.most_common():
        target = LABEL_TO_CLASS.get(label, "UNMAPPED")
        print(f"  {label:<20} {count:>5}  -> {target}")

    selected = [
        p for p in sheets
        if (args.label == "all" or sheet_label(p) == args.label)
        and LABEL_TO_CLASS.get(sheet_label(p)) is not None
    ]
    if not selected:
        raise SystemExit(
            f"\nNo sheets matched --label {args.label!r} with a mapped class. "
            f"Pick one of the labels listed above."
        )
    if args.limit:
        selected = selected[: args.limit]

    print(f"\nprocessing {len(selected)} sheet(s)"
          f"{' (dry run — nothing written)' if args.dry_run else ''}")

    total, failures = 0, []
    for path in tqdm(selected, desc="sheets", unit="sheet"):
        target_class = LABEL_TO_CLASS[sheet_label(path)]
        grid, kept = extract_sheet(
            path, args.out / target_class, args.min_cell, args.tolerance,
            write=not args.dry_run, cell=cell,
        )
        if grid.method == "failed":
            failures.append(path.name)
            continue
        total += kept
        if args.preview:
            write_preview(path, grid, config.FIGURES_DIR / "montage_previews" / f"{path.stem.replace('#','_')}.png")
        if len(selected) <= 12 or args.dry_run:
            tqdm.write(f"  {path.name[:58]:<58} {grid.describe()} -> kept {kept}")

    print(f"\n{total} tile(s) {'would be' if args.dry_run else ''} extracted -> {args.out}")
    if failures:
        print(f"grid detection FAILED on {len(failures)} sheet(s): {failures[:5]}")
        print("  Try a lower --tolerance (e.g. 0.90), a smaller --min-cell, or")
        print("  force the geometry with --cell WxH once you know it.")
    if args.preview:
        print(f"previews -> {config.FIGURES_DIR / 'montage_previews'}")
    if args.dry_run:
        print("\nCheck the grid and previews, then re-run without --dry-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
