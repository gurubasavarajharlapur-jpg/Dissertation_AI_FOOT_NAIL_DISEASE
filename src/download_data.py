"""Download the public source datasets into data/raw/.

Two public datasets back this project:

1. Mendeley Data, DOI 10.17632/hsj38fwnvr.3
   https://data.mendeley.com/datasets/hsj38fwnvr/3
   Foot imagery — the source for the wound/ulcer categories.

2. Figshare, article 5398573
   https://figshare.com/articles/dataset/5398573
   "Model Onychomycosis Training Datasets (JPG thumbnails) and Validation
   Datasets (JPG images)" — the source for the nail fungal category.

Both are fetched through their public REST APIs, which expose per-file download
URLs without requiring an account.

    python src/download_data.py                 # both datasets
    python src/download_data.py --source figshare
    python src/download_data.py --list          # show files, download nothing

IMPORTANT — this script downloads into `data/raw/_downloads/` and unpacks into
`data/raw/<source>/`, preserving whatever folder names the publishers used. It
does NOT sort images into this project's four classes: that mapping depends on
the real folder structure and is decided after `src/inspect_data.py` reports
what is actually there. Inventing a mapping before looking at the data is how
mislabelled training sets happen.

LICENSING — check and record each dataset's licence and citation before using
it in the dissertation. Both must be cited in the methodology chapter. The
`--list` output includes the licence field where the API provides it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

MENDELEY_DOI = "hsj38fwnvr"
MENDELEY_VERSION = 3
MENDELEY_API = (
    f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DOI}"
    f"/files?folder_id=root&version={MENDELEY_VERSION}"
)
MENDELEY_LANDING = f"https://data.mendeley.com/datasets/{MENDELEY_DOI}/{MENDELEY_VERSION}"
# The "Download all" button on a Mendeley Data page serves a pre-built archive
# from this S3 cache. Used as a fallback when the JSON API refuses the request,
# which it does from some networks and automated clients.
MENDELEY_BULK_ZIP = (
    f"https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/"
    f"{MENDELEY_DOI}-{MENDELEY_VERSION}.zip"
)

# Both hosts reject requests that do not look like a browser, returning 403 with
# no explanation. Sending a normal User-Agent is enough to be served.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# Metadata endpoints return JSON.
API_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"}

# File downloads must NOT advertise a JSON preference: asking a download
# endpoint for application/json can get an empty body back instead of the
# archive, which is what silently produced three 0-byte "downloads".
DOWNLOAD_HEADERS = {"User-Agent": USER_AGENT, "Accept": "*/*"}

FIGSHARE_ARTICLE_ID = 5398573
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
FIGSHARE_LANDING = f"https://figshare.com/articles/dataset/{FIGSHARE_ARTICLE_ID}"

DOWNLOAD_DIR = config.RAW_DIR / "_downloads"
REQUEST_TIMEOUT = 60


class DatasetUnavailable(RuntimeError):
    """The dataset host could not be reached or refused the request."""


def _get_json(url: str, what: str) -> dict | list:
    """GET a JSON endpoint, turning network failures into an actionable error."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=API_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        hint = ""
        if status in (401, 403):
            hint = (
                "\n  A 403 here is the host refusing the request, not a network "
                "problem — the\n  endpoint may now require a browser session or an "
                "account."
            )
        raise DatasetUnavailable(
            f"the {what} API returned HTTP {status} for {url}{hint}\n"
            f"  Fall back to a manual download (see --help) and unzip into data/raw/."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise DatasetUnavailable(
            f"could not reach the {what} API at {url}\n"
            f"  reason: {exc}\n"
            f"  If you are behind a proxy, firewall or restricted network, download\n"
            f"  the dataset manually (see --help) and unzip it into data/raw/."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DatasetUnavailable(
            f"the {what} API returned a non-JSON response — the endpoint may have "
            f"changed, or a captive portal intercepted the request: {exc}"
        ) from exc


def list_mendeley_files() -> list[dict]:
    """File listing for the Mendeley dataset: name, size and download URL.

    Tries the JSON API first, since it gives per-file names and sizes. That
    endpoint refuses some clients outright with a 403, so the "Download all"
    bulk archive is used as a fallback — it fetches the same data as one zip,
    just without a per-file breakdown beforehand.
    """
    try:
        return _list_mendeley_via_api()
    except DatasetUnavailable as api_error:
        print(f"  Mendeley JSON API unavailable ({api_error.args[0].splitlines()[0]})")
        print(f"  Falling back to the bulk archive: {MENDELEY_BULK_ZIP}")
        return _list_mendeley_bulk()


def _list_mendeley_bulk() -> list[dict]:
    """Single-entry listing for the whole-dataset zip.

    A HEAD request confirms the archive is actually there and reports its size,
    so a broken fallback fails here rather than part-way through a download.
    """
    try:
        response = requests.head(
            MENDELEY_BULK_ZIP, timeout=REQUEST_TIMEOUT, headers=DOWNLOAD_HEADERS,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise DatasetUnavailable(
            f"the Mendeley bulk archive is not reachable either: {exc}\n"
            f"  Download it manually from {MENDELEY_LANDING} (the 'Download all'\n"
            f"  button) and unzip into data/raw/mendeley_foot/."
        ) from exc

    return [
        {
            "name": f"{MENDELEY_DOI}-{MENDELEY_VERSION}.zip",
            "size": int(response.headers.get("content-length") or 0),
            "url": MENDELEY_BULK_ZIP,
        }
    ]


def _list_mendeley_via_api() -> list[dict]:
    payload = _get_json(MENDELEY_API, "Mendeley Data")
    # The endpoint returns a bare list of file objects for a folder listing.
    entries = payload if isinstance(payload, list) else payload.get("results", [])
    files = []
    for entry in entries:
        content = entry.get("content_details") or {}
        url = content.get("download_url") or entry.get("download_url")
        if not url:
            continue
        files.append(
            {
                "name": entry.get("filename") or content.get("filename") or "unknown",
                "size": int(content.get("size") or entry.get("size") or 0),
                "url": url,
            }
        )
    if not files:
        raise DatasetUnavailable(
            f"the Mendeley API returned no downloadable files for {MENDELEY_DOI} "
            f"version {MENDELEY_VERSION}. Check {MENDELEY_LANDING} in a browser — "
            f"the version number may have moved on."
        )
    return files


def list_figshare_files() -> list[dict]:
    """File listing for the Figshare article."""
    payload = _get_json(FIGSHARE_API, "Figshare")
    if not isinstance(payload, dict):
        raise DatasetUnavailable("unexpected Figshare response shape")
    files = [
        {
            "name": entry.get("name", "unknown"),
            "size": int(entry.get("size") or 0),
            "url": entry.get("download_url"),
        }
        for entry in payload.get("files", [])
        if entry.get("download_url")
    ]
    if not files:
        raise DatasetUnavailable(
            f"the Figshare API returned no files for article {FIGSHARE_ARTICLE_ID}. "
            f"Check {FIGSHARE_LANDING} in a browser."
        )
    return files


def _human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download_file(url: str, destination: Path, expected_size: int = 0) -> Path:
    """Stream a file to disk with a progress bar, skipping a complete download.

    Writes to a `.part` file and renames on success, so an interrupted download
    is never mistaken for a finished one on the next run.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected_size or destination.stat().st_size == expected_size):
        print(f"  already downloaded: {destination.name}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(
            url, stream=True, timeout=REQUEST_TIMEOUT, headers=DOWNLOAD_HEADERS
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or expected_size or 0)
            with open(partial, "wb") as handle, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=f"  {destination.name}",
                leave=False,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    bar.update(len(chunk))
    except requests.exceptions.RequestException as exc:
        partial.unlink(missing_ok=True)
        raise DatasetUnavailable(f"download failed for {url}: {exc}") from exc

    # Check the bytes before accepting them. A request that returns an empty or
    # truncated body otherwise renames cleanly, fails to unzip, gets copied
    # verbatim, and reports "Download complete" over an empty dataset.
    got = partial.stat().st_size
    if got == 0:
        partial.unlink(missing_ok=True)
        raise DatasetUnavailable(
            f"'{destination.name}' downloaded as 0 bytes from {url}\n"
            f"  The server accepted the request but sent no content. Download the\n"
            f"  dataset manually (see --help) and unzip it into data/raw/."
        )
    if expected_size and abs(got - expected_size) > max(1024, expected_size * 0.01):
        partial.unlink(missing_ok=True)
        raise DatasetUnavailable(
            f"'{destination.name}' is {_human(got)} but the API reported "
            f"{_human(expected_size)}.\n  The transfer was truncated; re-run to retry."
        )

    partial.replace(destination)
    print(f"  downloaded {destination.name} ({_human(destination.stat().st_size)})")
    return destination


def extract_archive(archive: Path, target_dir: Path) -> bool:
    """Unpack a zip/tar archive. Returns False if the file is not an archive."""
    target_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _assert_safe_members(zf.namelist(), archive)
            zf.extractall(target_dir)
        print(f"  extracted {archive.name} -> {target_dir}")
        return True

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            _assert_safe_members(tf.getnames(), archive)
            # `filter="data"` blocks absolute paths, device files and symlink
            # escapes; it is the default from Python 3.14 and explicit here.
            tf.extractall(target_dir, filter="data")
        print(f"  extracted {archive.name} -> {target_dir}")
        return True

    return False


def _assert_safe_members(names: list[str], archive: Path) -> None:
    """Reject archives whose entries would write outside the target directory."""
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise DatasetUnavailable(
                f"refusing to extract '{archive.name}': entry '{name}' would write "
                f"outside the target directory."
            )


def fetch_source(name: str, lister, target_dir: Path, list_only: bool) -> None:
    """List, download and unpack one dataset source."""
    print(f"\n=== {name} ===")
    files = lister()
    total = sum(f["size"] for f in files)
    print(f"{len(files)} file(s), {_human(total)} total")
    for entry in files:
        print(f"  - {entry['name']:<55} {_human(entry['size']):>10}")

    if list_only:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for entry in files:
        archive = download_file(entry["url"], DOWNLOAD_DIR / entry["name"], entry["size"])
        if not extract_archive(archive, target_dir):
            if archive.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
                raise DatasetUnavailable(
                    f"'{archive.name}' has an archive extension but is not a readable "
                    f"zip or tar.\n  The download is corrupt; delete "
                    f"{DOWNLOAD_DIR / archive.name} and re-run."
                )
            # A loose image rather than an archive: copy it across as-is.
            shutil.copy2(archive, target_dir / archive.name)
            print(f"  copied {archive.name} -> {target_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Manual download (if this machine cannot reach the hosts):\n"
            f"  1. Open {MENDELEY_LANDING}\n"
            f"     and {FIGSHARE_LANDING}\n"
            "  2. Download the archives from each page.\n"
            "  3. Unzip them into data/raw/mendeley_foot/ and data/raw/figshare_nail/.\n"
            "  4. Run: python src/inspect_data.py\n"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["mendeley", "figshare", "both"],
        default="both",
        help="which dataset to fetch (default: both)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="show the available files and exit without downloading",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    sources = {
        "mendeley": ("Mendeley Data hsj38fwnvr (foot)", list_mendeley_files, config.RAW_DIR / "mendeley_foot"),
        "figshare": (f"Figshare {FIGSHARE_ARTICLE_ID} (onychomycosis / nail)", list_figshare_files, config.RAW_DIR / "figshare_nail"),
    }
    selected = list(sources) if args.source == "both" else [args.source]

    failures = []
    for key in selected:
        label, lister, target = sources[key]
        try:
            fetch_source(label, lister, target, args.list_only)
        except DatasetUnavailable as exc:
            print(f"\n[FAILED] {label}\n{exc}", file=sys.stderr)
            failures.append(key)

    if failures:
        print(
            f"\n{len(failures)} of {len(selected)} source(s) failed. "
            f"Use the manual steps in `--help` for those, then run "
            f"`python src/inspect_data.py`.",
            file=sys.stderr,
        )
        return 1

    if not args.list_only:
        images = sum(
            1
            for p in config.RAW_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in config.VALID_EXTENSIONS
        )
        print(f"\nDownload complete: {images} image(s) now under {config.RAW_DIR}")
        if images == 0:
            print("  ...but no images were produced. Check the output above.", file=sys.stderr)
            return 1
        print("Next: python src/inspect_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
