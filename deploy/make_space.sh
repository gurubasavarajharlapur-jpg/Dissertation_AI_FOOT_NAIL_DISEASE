#!/usr/bin/env bash
#
# Assemble a Hugging Face Space from this repository.
#
#   ./deploy/make_space.sh [target-directory]
#
# A Space is its own git repository, so this builds one: the entry point and
# requirements from deploy/, the source tree, the MobileNetV2 weights and the
# calibration that goes with them. The weights are tracked with Git LFS, which
# is how a 100MB file reaches a Space at all.
#
# It stops short of pushing. Create the Space in the browser first, then follow
# the three lines this prints.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$ROOT/build/space}"
MODEL="mobilenetv2"

WEIGHTS="$ROOT/models/${MODEL}_best.keras"
CALIBRATION="$ROOT/results/metrics/${MODEL}_calibration.json"

for required in "$WEIGHTS" "$CALIBRATION"; do
    if [[ ! -f "$required" ]]; then
        echo "missing: $required" >&2
        echo "Train the model and run src/calibrate.py before deploying." >&2
        exit 1
    fi
done

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Git LFS is not installed, and the weights cannot be pushed without it." >&2
    echo "  macOS:  brew install git-lfs     Debian/Ubuntu:  apt install git-lfs" >&2
    exit 1
fi

rm -rf "$TARGET"
mkdir -p "$TARGET/src" "$TARGET/models" "$TARGET/results/metrics"

cp "$ROOT/deploy/app.py" "$TARGET/app.py"
cp "$ROOT/deploy/requirements.txt" "$TARGET/requirements.txt"
cp "$ROOT/deploy/README.md" "$TARGET/README.md"
cp -R "$ROOT/src/." "$TARGET/src/"
find "$TARGET/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cp "$WEIGHTS" "$TARGET/models/"
cp "$CALIBRATION" "$TARGET/results/metrics/"

cat > "$TARGET/.gitignore" <<'IGNORE'
__pycache__/
*.py[cod]
.DS_Store
IGNORE

cd "$TARGET"
git init -q
git lfs install --local >/dev/null
git lfs track "*.keras" >/dev/null
git add -A
git -c user.email=noreply@example.com -c user.name="Space build" \
    commit -qm "Screening prototype for demonstration"

echo
echo "Space assembled in $TARGET"
du -sh "$TARGET" | sed 's/^/  /'
echo
echo "Create the Space at https://huggingface.co/new-space (SDK: Streamlit), then:"
echo
echo "  cd $TARGET"
echo "  git remote add origin https://huggingface.co/spaces/<your-username>/<space-name>"
echo "  git push -u origin main"
echo
echo "First build takes about ten minutes. Open it once, take one photograph,"
echo "and leave it: the model then stays loaded."
