#!/usr/bin/env bash
# Cross-platform software-stack check on the SAME hardware: Linux (WSL2 Ubuntu) with a fresh virtual environment
# built from the dependency lock (versions as close as the Linux wheels allow). Steps:
#   1. environment record; 2. frozen-input check (FrozenFamily: hex coefficients -> float assembly -> SHA-256);
#   3. paired ST S3 cells on frozen inputs for seeds 200-204, repeat 0, both families (all seven arms);
#   4. replay of the Linux records on Linux.  (Replay of the Linux records on Windows is run separately.)
# This is NOT an independent-hardware replication; timings are descriptive only.
set -u
CARS="$1"            # path of the CARS directory as seen from WSL
OUT="$CARS/FINAL_DEV/crossplatform/linux"
VENV="$HOME/cars_xplat_venv"
mkdir -p "$OUT/cells"
export PYTHONDONTWRITEBYTECODE=1
{
  echo "== uname"; uname -a; echo "== os-release"; cat /etc/os-release | head -4; echo "== python"; python3 --version
  echo "== cpu"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core" ; echo "== mem"; free -g | head -2
} > "$OUT/environment.txt" 2>&1
if [ ! -x "$VENV/bin/python" ]; then
  # Ubuntu's python3 lacks ensurepip and no admin rights are used: create the venv without pip and bootstrap pip with
  # the official PyPA script.
  python3 -m venv --without-pip "$VENV" >> "$OUT/setup.log" 2>&1 || { echo "VENV_FAILED" >> "$OUT/setup.log"; exit 2; }
  curl -sS https://bootstrap.pypa.io/get-pip.py -o "$HOME/get-pip.py" >> "$OUT/setup.log" 2>&1
  "$VENV/bin/python" "$HOME/get-pip.py" >> "$OUT/setup.log" 2>&1 || { echo "PIP_BOOTSTRAP_FAILED" >> "$OUT/setup.log"; exit 2; }
fi
"$VENV/bin/python" -m pip install numpy==2.2.6 scipy==1.15.3 pypardiso==0.4.7 mkl==2026.2.1 python-flint==0.9.0 \
    cholespy==2.2.0 psutil==7.1.3 >> "$OUT/setup.log" 2>&1 || echo "PIP_EXACT_FAILED" >> "$OUT/setup.log"
"$VENV/bin/python" -m pip freeze > "$OUT/pip_freeze.txt" 2>&1
[ "${SETUP_ONLY:-0}" = "1" ] && { echo SETUP_DONE >> "$OUT/setup.log"; exit 0; }
cd "$CARS/FINAL_DEV/release_tools"
"$VENV/bin/python" -B check_inputs.py > "$OUT/check_inputs.log" 2>&1; cp check_inputs_result.json "$OUT/check_inputs_result.json" 2>/dev/null
for SEED in 200 201 202 203 204; do
  for FAM in E3-TS D3-TS; do
    D="$OUT/cells/${FAM}_S3_s${SEED}_r0_ST"
    timeout 1800 "$VENV/bin/python" -B run_cell_frozen.py --family $FAM --size S3 --seed $SEED --repeat 0 --mode ST --out "$D" > "$D.log" 2>&1
    echo "$FAM $SEED cell exit $?" >> "$OUT/progress.log"
    timeout 1800 "$VENV/bin/python" -B replay_frozen.py "$D" replay_linux.json > "$D.replay.log" 2>&1
    echo "$FAM $SEED replay exit $?" >> "$OUT/progress.log"
  done
done
echo DONE >> "$OUT/progress.log"
