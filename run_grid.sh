#!/usr/bin/env bash
# 参数网格：激活函数 × 批量大小，固定 hidden=128, epochs=4, lr=1e-3
# 跑完自动把每组结果写入 results_grid.csv

set -u
ACTS=("relu" "tanh" "sigmoid")
BSS=(256 512 1024 2048 4096)

# 可按需修改
NP=4
HIDDEN=128
EPOCHS=4
LR=1e-3

PY=~/venv/bin/python
TRAIN_DIR=~/taxi_data
TEST_DIR=~/taxi_test
LOGDIR=~/logs
CKPTDIR=~/ckpts
OUTCSV=~/logs/results_grid.csv

mkdir -p "$LOGDIR" "$CKPTDIR"

# 写表头（若文件不存在）
if [[ ! -f "$OUTCSV" ]]; then
  echo "timestamp,activation,batch,hidden,epochs,lr,train_time_s,test_RMSE_USD,ckpt,train_log,eval_log" > "$OUTCSV"
fi

for a in "${ACTS[@]}"; do
  for b in "${BSS[@]}"; do
    ck="${CKPTDIR}/taxi_${a}_h${HIDDEN}_b${b}_ep${EPOCHS}.npz"
    trlog="${LOGDIR}/train_${a}_b${b}.log"
    evlog="${LOGDIR}/eval_${a}_b${b}.log"

    echo "== Train a=${a} b=${b} =="
    mpirun -np ${NP} "${PY}" -u mpi_train_save.py \
      --data_dir "${TRAIN_DIR}" \
      --target amount \
      --hidden ${HIDDEN} --activation "${a}" \
      --batch ${b} --epochs ${EPOCHS} --lr ${LR} --clip 5 \
      --max_steps_per_epoch 200 \
      --save_path "${ck}" |& tee "${trlog}"

    # 解析训练时间（行形如: 'Train time ~111.1s'）
    train_time=$(awk '/Train time/{ if (match($0, /~([0-9.]+)s/, m)) print m[1] }' "${trlog}")
    if [[ -z "${train_time:-}" ]]; then train_time="NaN"; fi

    echo "== Eval a=${a} b=${b} =="
    mpirun -np ${NP} "${PY}" -u mpi_eval_stream.py \
      --test_dir "${TEST_DIR}" \
      --ckpt "${ck}" \
      --batch 16384 --progress_every_files 5 |& tee "${evlog}"

    # 解析RMSE（行形如: '[Eval] RMSE_USD=19.28'）
    rmse=$(awk -F= '/\[Eval\] RMSE_USD=/{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}' "${evlog}")
    if [[ -z "${rmse:-}" ]]; then rmse="NaN"; fi

    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "${ts},${a},${b},${HIDDEN},${EPOCHS},${LR},${train_time},${rmse},${ck},${trlog},${evlog}" >> "${OUTCSV}"
    echo "--> Logged to ${OUTCSV}"
  done
done

echo "✅ Grid done. Summary at ${OUTCSV}"
