#!/usr/bin/env python3
import argparse, os, glob, time, math, json
import numpy as np
import pandas as pd
from mpi4py import MPI

# -------- activations --------
def relu(z): return np.maximum(0, z)
def d_relu(z): return (z > 0).astype(z.dtype)
def tanh(z): return np.tanh(z)
def d_tanh(z): return 1 - np.tanh(z)**2
def sigmoid(z): return 1/(1+np.exp(-z))
def d_sigmoid(z): s = sigmoid(z); return s*(1-s)
ACTS = {"relu": (relu, d_relu), "tanh": (tanh, d_tanh), "sigmoid": (sigmoid, d_sigmoid)}

# -------- model --------
def init_params(m, n_hidden, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((n_hidden, m)).astype(np.float64) / np.sqrt(m)
    b1 = np.zeros((n_hidden, 1), dtype=np.float64)
    W2 = rng.standard_normal((1, n_hidden)).astype(np.float64) / np.sqrt(n_hidden)
    b2 = np.zeros((1, 1), dtype=np.float64)
    return [W1, b1, W2, b2]

def forward(X, params, act):
    W1, b1, W2, b2 = params
    Z1 = W1 @ X + b1
    A1 = act(Z1)
    Z2 = W2 @ A1 + b2
    return Z1, A1, Z2

def compute_grads(X, y, params, act, d_act):
    W1, b1, W2, b2 = params
    B = X.shape[1]
    Z1, A1, Z2 = forward(X, params, act)
    E = Z2 - y
    dW2 = (E @ A1.T) / B
    db2 = np.mean(E, axis=1, keepdims=True)
    dA1 = W2.T @ E
    dZ1 = dA1 * d_act(Z1)
    dW1 = (dZ1 @ X.T) / B
    db1 = np.mean(dZ1, axis=1, keepdims=True)
    loss = float(np.mean(0.5*(E**2)))
    for g in (dW1, db1, dW2, db2):
        if not np.all(np.isfinite(g)):
            raise FloatingPointError("NaN/Inf in gradients")
    return (dW1, db1, dW2, db2), loss

def apply_update(params, grads, lr, clip=None):
    if clip is not None and clip > 0:
        total = 0.0
        for g in grads: total += float(np.sum(g*g))
        norm = math.sqrt(total) + 1e-12
        if norm > clip:
            scale = clip / norm
            grads = [g*scale for g in grads]
    for p, g in zip(params, grads):
        p -= lr * g
        if not np.all(np.isfinite(p)):
            raise FloatingPointError("NaN/Inf in params after update")

def allreduce_same_shape(comm, arr):
    shape = np.array(arr.shape, dtype=np.int64)
    all_shapes = comm.allgather(shape)
    for s in all_shapes:
        if not np.array_equal(s, shape):
            raise RuntimeError(f"Allreduce shape mismatch across ranks: got {all_shapes}")
    out = np.empty_like(arr)
    comm.Allreduce(arr, out, op=MPI.SUM)
    return out

def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--target", choices=["amount","log"], default="log",
                    help="amount=total_amount; log=log(1+amount)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--activation", choices=list(ACTS.keys()), default="relu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=1)  # 占位，不在本脚本做测试
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--max_steps_per_epoch", type=int, default=20,
                    help="限制每个 epoch 的 batch 数")
    ap.add_argument("--save_path", type=str, required=True,
                    help="保存 npz 的路径，例如 ~/ckpts/taxi_sgd_log_relu_h128_b1024.npz")
    args = ap.parse_args()
    act_f, d_act_f = ACTS[args.activation]

    # 1) 列文件 & 特征列
    if rank == 0:
        all_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.data_dir), "*.parquet")))
        if not all_files:
            raise FileNotFoundError(f"No parquet files in {args.data_dir}")
        probe = pd.read_parquet(all_files[0])
        drop_cols = ["tpep_pickup_datetime","tpep_dropoff_datetime","total_amount","total_amount_log"]
        feat_cols = [c for c in probe.columns if c not in drop_cols]
        file_splits = np.array_split(all_files, size)
        meta = (feat_cols, args.target, args.activation, args.hidden)
    else:
        file_splits = None
        meta = None

    my_files = comm.scatter(file_splits, root=0)
    feat_cols, target_mode, act_name, hidden_size = comm.bcast(meta, root=0)

    # 2) 读本 rank 的数据
    dfs = [pd.read_parquet(f) for f in my_files]
    df_local = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # 3) 特征/目标
    X_local = df_local[feat_cols].to_numpy(dtype=np.float64, copy=False) if len(df_local)>0 else np.empty((0, len(feat_cols)), dtype=np.float64)
    y_amt_local = df_local["total_amount"].to_numpy(dtype=np.float64, copy=False) if len(df_local)>0 else np.empty((0,), dtype=np.float64)
    y_local = y_amt_local if target_mode=="amount" else (df_local["total_amount_log"].to_numpy(dtype=np.float64, copy=False) if len(df_local)>0 else np.empty((0,), dtype=np.float64))
    m = X_local.shape[1]

    # 4) 全局特征标准化
    n_local = X_local.shape[0]
    sum_local   = np.sum(X_local, axis=0) if n_local>0 else np.zeros(m, dtype=np.float64)
    sumsq_local = np.sum(X_local*X_local, axis=0) if n_local>0 else np.zeros(m, dtype=np.float64)
    n_total = comm.allreduce(n_local, op=MPI.SUM)
    sum_global   = allreduce_same_shape(comm, sum_local)
    sumsq_global = allreduce_same_shape(comm, sumsq_local)
    mu = sum_global / max(1, n_total)
    var = np.maximum(1e-12, sumsq_global / max(1, n_total) - mu*mu)
    sigma = np.sqrt(var)
    if n_local>0:
        X_local = (X_local - mu) / sigma

    # 5) 全量训练（无本地验证）
    Xtr_local, ytr_local = X_local, y_local

    # 6) 目标标准化统计（训练集）
    if target_mode == "log":
        sumy_local   = float(np.sum(ytr_local)) if len(ytr_local)>0 else 0.0
        sumy2_local  = float(np.sum(ytr_local**2)) if len(ytr_local)>0 else 0.0
        ny_local     = int(len(ytr_local))
        sumy  = comm.allreduce(sumy_local,  op=MPI.SUM)
        sumy2 = comm.allreduce(sumy2_local, op=MPI.SUM)
        ny    = comm.allreduce(ny_local,    op=MPI.SUM)
        mu_y = sumy / max(1, ny)
        var_y = max(1e-12, sumy2 / max(1, ny) - mu_y*mu_y)
        sigma_y = math.sqrt(var_y)
        ytr_std_local = (ytr_local - mu_y) / sigma_y if len(ytr_local)>0 else ytr_local
        if rank == 0:
            print(f"[Info] Target standardized: mu_y={mu_y:.4f}, sigma_y={sigma_y:.4f}", flush=True)
    else:
        mu_y, sigma_y = 0.0, 1.0
        ytr_std_local = ytr_local
        if rank == 0:
            print(f"[Info] Target mean/std skipped (amount mode).", flush=True)

    # 7) 参数
    params = init_params(m, args.hidden, seed=42)
    for p in params:
        comm.Bcast(p, root=0)

    # 8) 训练
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        steps_done = 0
        if len(ytr_std_local) > 0:
            order = np.random.permutation(len(ytr_std_local))
            total_steps = (len(order) + args.batch - 1) // args.batch
        else:
            order = np.array([], dtype=int)
            total_steps = 1

        step = 0
        while step < total_steps and steps_done < args.max_steps_per_epoch:
            if len(order) > 0:
                start = step * args.batch
                bid = order[start:start+args.batch]
                Xb = Xtr_local[bid].T
                yb = ytr_std_local[bid][None, :]
                grads, _ = compute_grads(Xb, yb, params, ACTS[act_name][0], ACTS[act_name][1])
            else:
                grads = [np.zeros_like(p) for p in params]

            # Allreduce 平均梯度 + 更新
            grads_avg = []
            for g in grads:
                g = g.astype(np.float64, copy=False)
                g_sum = np.empty_like(g)
                comm.Allreduce(g, g_sum, op=MPI.SUM)
                grads_avg.append(g_sum / size)
            apply_update(params, grads_avg, args.lr, clip=args.clip)

            step += 1
            steps_done += 1
            if rank == 0 and (steps_done % 5 == 0 or steps_done == min(total_steps, args.max_steps_per_epoch)):
                print(f"[epoch {epoch}] progress: {steps_done}/{min(total_steps,args.max_steps_per_epoch)} batches", flush=True)

        if rank == 0:
            print(f"[Epoch {epoch}] (training only)", flush=True)

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.expanduser(args.save_path)), exist_ok=True)
        np.savez_compressed(
            os.path.expanduser(args.save_path),
            W1=params[0], b1=params[1], W2=params[2], b2=params[3],
            mu=mu, sigma=sigma, mu_y=np.array([mu_y]), sigma_y=np.array([sigma_y]),
            feat_cols=np.array(feat_cols, dtype=object),
            activation=np.array([act_name], dtype=object),
            hidden=np.array([hidden_size]),
            target=np.array([target_mode], dtype=object)
        )
        print(f"[Saved] {os.path.expanduser(args.save_path)}", flush=True)
        print(f"Train time ~{time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
