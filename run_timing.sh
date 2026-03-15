#!/usr/bin/env bash
set -u
PY=~/venv/bin/python
TRAIN_DIR=~/taxi_data
LOGDIR=~/logs
CKPTDIR=~/ckpts
OUTCSV=~/logs/results_timing.csv

# 固定参数
ACT="relu"; BATCH=1024; HIDDEN=128; EPOCHS=2; LR=1e-3
NP_LIST=(1 2 4 8)

mkdir -p "$LOGDIR" "$CKPTDIR"
[[ -f "$OUTCSV" ]] || echo "timestamp,np,activation,batch,hidden,epochs,lr,train_time_s,ckpt,train_log,flags" > "$OUTCSV"

# 逻辑 CPU（含超线程）
LOGICAL=$(nproc)
# 物理核数（不含超线程）
CORES=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)

echo "Detected cores=${CORES}, logical_cpus=${LOGICAL}"

export OMP_NUM_THREADS=1  # 避免 BLAS 二次超订阅

for P in "${NP_LIST[@]}"; do
  ck="${CKPTDIR}/timing_np${P}_${ACT}_h${HIDDEN}_b${BATCH}_ep${EPOCHS}.npz"
  trlog="${LOGDIR}/timing_np${P}.log"
  EXTRA=""

  if (( P > CORES && P <= LOGICAL )); then
    EXTRA="--use-hwthread-cpus"
    echo "np=${P} uses hyperthreads -> ${EXTRA}"
  elif (( P > LOGICAL )); then
    EXTRA="--oversubscribe --bind-to none"
    echo "np=${P} > logical=${LOGICAL} -> ${EXTRA}"
  fi

  echo "== Timing np=${P} =="
  mpirun -np ${P} ${EXTRA} "${PY}" -u mpi_train_save.py \
    --data_dir "${TRAIN_DIR}" \
    --target amount \
    --hidden ${HIDDEN} --activation "${ACT}" \
    --batch ${BATCH} --epochs ${EPOCHS} --lr ${LR} --clip 5 \
    --max_steps_per_epoch 200 \
    --save_path "${ck}" |& tee "${trlog}"

  t=$(awk '/Train time/{ if (match($0, /~([0-9.]+)s/, m)) print m[1] }' "${trlog}")
  [[ -z "${t:-}" ]] && t="NaN"
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "${ts},${P},${ACT},${BATCH},${HIDDEN},${EPOCHS},${LR},${t},${ck},${trlog},\"${EXTRA}\"" >> "${OUTCSV}"
done

echo "✅ Timing done. Summary at ${OUTCSV}"
