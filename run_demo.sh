#!/usr/bin/env bash
#
# Start the screening prototype for a live demonstration and keep it up.
#
#   ./run_demo.sh
#
# Written for the poster session, where the app has to stay available for
# hours while nobody is watching the terminal. It differs from
# `streamlit run src/prototype/app.py` in three ways:
#
#   it refuses to start quietly broken -- missing weights or missing
#   calibration are reported before the browser opens, not discovered when
#   an examiner uploads a photograph;
#
#   it restarts the server if it dies, so a crash at half past four costs
#   five seconds rather than the rest of the session;
#
#   it prints the one thing the script cannot do for you. Streamlit caches
#   the loaded model inside its own process (@st.cache_resource in
#   src/prototype/app.py), so no outside process can warm that cache. The
#   first prediction after launch pays the ten second model load whoever
#   makes it. Make it yourself, before the session starts.
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8501}"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

python - <<'PY' || exit 1
"""Report anything that would make the demonstration fail, before it starts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src import config

ready, problems = [], []
for name in config.MODEL_NAMES:
    weights = config.model_path(name)
    if not weights.exists():
        problems.append(f"no trained weights for {name}: expected {weights}")
        continue
    size = weights.stat().st_size / 1e6
    calibration = config.calibration_path(name)
    if calibration.exists():
        ready.append(f"{name:12} {size:6.1f} MB   calibrated")
    else:
        ready.append(f"{name:12} {size:6.1f} MB   UNCALIBRATED")
        problems.append(
            f"no calibration for {name}: {calibration} is missing, so the app "
            f"will show raw softmax confidence and will not abstain. Run "
            f"`python -m src.calibrate` first.")

for line in ready:
    print("  " + line)
for line in problems:
    print("  ! " + line)
if not ready:
    print("\nNothing to demonstrate: train or copy the weights into models/ first.")
    sys.exit(1)
if problems:
    print("\nStarting anyway, but read the warnings above before anyone sees this.")
PY

mkdir -p logs
echo
echo "  http://localhost:${PORT}"
echo "  Upload one photograph yourself now, so the model is loaded before 3pm."
echo "  Ctrl-C to stop. Restarts are logged to logs/demo.log"
echo

trap 'echo; echo "stopped"; exit 0' INT TERM

attempt=0
while true; do
    streamlit run src/prototype/app.py \
        --server.port "$PORT" \
        --server.headless false \
        --server.fileWatcherType none \
        --browser.gatherUsageStats false
    attempt=$((attempt + 1))
    printf '%s  server exited, restart %d\n' "$(date '+%F %T')" "$attempt" \
        | tee -a logs/demo.log
    sleep 3
done
