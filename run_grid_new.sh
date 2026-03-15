#!/usr/bin/env bash
# Grid search (3 activations × 5 batch sizes) with:
# 1) Interactive input for MPI process count (1..8)
# 2) Auto-detecting local slots (CPU/hardware threads)
# 3) Friendly fallback when NP > slots
# 4) Auto-create and use a hostfile (more robust on some setups)
# 5) Pin internal BLAS/OMP threads to 1 to avoid oversubscription
# 6) Output summary CSV with only test_RMSE_USD

set -euo pipefail

# ---- Interactive process count (1..8) ----
DEFAULT_NP=4
read -rp "Please enter MPI process count [1-8] (default ${DEFAULT_NP}): " USER_NP || true
if [[ -z "${USER_NP:-}" ]]; then
  NP="$DEFAULT_NP"
else
  NP="$USER_NP"
fi
if ! [[ "$NP" =~ ^[0-9]+$ ]]; then
  echo "Error: process count must be a non-negative integer." >&2; exit 2
fi
if (( NP < 1 || NP > 8 )); then
  echo "Error: process count must be within 1 to 8 (inclusive). Got: $NP" >&2; exit 2
fi

# ---- Detect local slots (prefer nproc, fallback to lscpu, then 1) ----
if command -v nproc >/dev/null 2>&1; then
  SLOTS="$(nproc --all)"
elif command -v lscpu >/dev/null 2>&1; then
  SLOTS="$(lscpu -p=CPU | grep -v '^#' | wc -l | awk '{print $1}')"
else
  SLOTS=1
fi
if ! [[ "$SLOTS" =~ ^[0-9]+$ ]] || (( SLOTS < 1 )); then
  SLOTS=1
fi
echo "Detected local slots ≈ ${SLOTS} (CPU/hardware threads)"

# ---- Limit internal threads per process to avoid hidden oversubscription ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Prepare a hostfile (robust default) ----
HOSTFILE="${HOME}/hostfile.txt"
if [[ ! -f "${HOSTFILE}" ]]; then
  echo "localhost slots=${SLOTS}" > "${HOSTFILE}"
else
  if ! grep -q "localhost" "${HOSTFILE}"; then
    echo "localhost slots=${SLOTS}" >> "${HOSTFILE}"
  fi
fi

# ---- Fallbacks when NP exceeds slots ----
MPIRUN_EXTRAS=""
if (( NP > SLOTS )); then
  echo "⚠️  Requested NP=${NP} exceeds available slots=${SLOTS}."
  echo "Choose a fallback:"
  echo "  1) Reduce NP to ${SLOTS} (recommended)"
  echo "  2) Use --use-hwthread-cpus (treat hardware threads as slots)"
  echo "  3) Use --oversubscribe (ignore slot limits; may be slower)"
  echo "  4) Cancel and exit"
  read -rp "Enter 1/2/3/4 [default 1]: " CHOICE
  CHOICE="${CHOICE:-1}"
  case "$CHOICE" in
    1) NP="$SLOTS"; echo "NP set to ${NP}.";;
    2) MPIRUN_EXTRAS="--use-hwthread-cpus"; echo "Using --use-hwthread-cpus.";;
    3) MPIRUN_EXTRAS="--oversubscribe"; echo "Using --oversubscribe (note: may be slower).";;
    4) echo "Canceled."; exit 4;;
    *) NP="$SLOTS"; echo "Unrecognized input. NP set to ${NP}.";;
  esac
fi

echo "== Final NP=${NP}, extra mpirun options: '${MPIRUN_EXTRAS}' =="

# ---- Grid configuration ----
ACTS=("relu" "tanh" "sigmoid")
BSS=(256 512 1024 2048 4096)

HIDDEN=128
EPOCHS=4
LR=1e-3

PY="${PY:-$HOME/venv/bin/python}"
TRAIN_DIR="${TRAIN_DIR:-$HOME/taxi_data}"
TEST_DIR="${TEST_DIR:-$HOME/taxi_test}"
LOGDIR="${LOGDIR:-$HOME/logs}"
CKPTDIR="${CKPTDIR:-$HOME/ckpts}"
OUTCSV="${OUTCSV:-$HOME/logs/results_grid_new.csv}"

mkdir -p "$LOGDIR" "$CKPTDIR"

# Dependencies
command -v mpirun >/dev/null 2>&1 || { echo "Error: mpirun not found in PATH." >&2; exit 3; }
[ -x "$PY" ] || { echo "Error: Python interpreter not executable: $PY" >&2; exit 3; }

# Write CSV header if missing (only test_RMSE_USD)
if [[ ! -f "$OUTCSV" ]]; then
  echo "timestamp,activation,batch,hidden,epochs,lr,train_time_s,test_RMSE_USD,ckpt,train_log,eval_log" > "$OUTCSV"
fi

# Unified mpirun prefix (using hostfile by default)
MPI_PREFIX=(mpirun --hostfile "${HOSTFILE}" ${MPIRUN_EXTRAS})
# Alternatively, if you prefer not to use a hostfile:
# MPI_PREFIX=(mpirun ${MPIRUN_EXTRAS})

for a in "${ACTS[@]}"; do
  for b in "${BSS[@]}"; do
    ck="${CKPTDIR}/taxi_${a}_h${HIDDEN}_b${b}_ep${EPOCHS}.npz"
    trlog="${LOGDIR}/train_${a}_b${b}.log"
    evlog="${LOGDIR}/eval_${a}_b${b}.log"

    echo "== Train a=${a} b=${b} =="
    "${MPI_PREFIX[@]}" -np "${NP}" "${PY}" -u mpi_train_save.py \
      --data_dir "${TRAIN_DIR}" \
      --target amount \
      --hidden ${HIDDEN} --activation "${a}" \
      --batch ${b} --epochs ${EPOCHS} --lr ${LR} --clip 5 \
      --max_steps_per_epoch 200 \
      --save_path "${ck}" |& tee "${trlog}"

    # Parse training time (line like: 'Train time ~111.1s')
    train_time=$(awk '/Train time/{ if (match($0, /~([0-9.]+)s/, m)) print m[1] }' "${trlog}")
    if [[ -z "${train_time:-}" ]]; then train_time="NaN"; fi

    echo "== Eval a=${a} b=${b} =="
    "${MPI_PREFIX[@]}" -np "${NP}" "${PY}" -u mpi_eval_stream.py \
      --test_dir "${TEST_DIR}" \
      --ckpt "${ck}" \
      --batch 16384 --progress_every_files 5 |& tee "${evlog}"

    # Parse test RMSE (line like: '[Eval] RMSE_USD=19.28')
    test_rmse=$(awk -F= '/\[Eval\] RMSE_USD=/{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}' "${evlog}")
    if [[ -z "${test_rmse:-}" ]]; then test_rmse="NaN"; fi

    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "${ts},${a},${b},${HIDDEN},${EPOCHS},${LR},${train_time},${test_rmse},${ck},${trlog},${evlog}" >> "${OUTCSV}"
    echo "--> Logged to ${OUTCSV}"
  done
done

echo "✅ Grid done. Summary at ${OUTCSV}"
