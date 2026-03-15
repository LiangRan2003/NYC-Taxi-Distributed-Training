#!/usr/bin/env python3
import argparse, os, glob, math, time
import numpy as np
import pandas as pd
from mpi4py import MPI

# --- activations ---
def relu(z): return np.maximum(0, z)
def tanh(z): return np.tanh(z)
def sigmoid(z): return 1/(1+np.exp(-z))
ACT = {"relu": relu, "tanh": tanh, "sigmoid": sigmoid}

def forward(X, params, act):
    W1, b1, W2, b2 = params
    Z1 = W1 @ X + b1
    A1 = act(Z1)
    Z2 = W2 @ A1 + b2
    return Z2  # 1 x B

def rmse_parallel(comm, sse_local, n_local):
    sse = comm.allreduce(float(sse_local), op=MPI.SUM)
    n   = comm.allreduce(int(n_local),   op=MPI.SUM)
    n = max(1, n)
    return float(np.sqrt(sse / n))

def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clip_log", type=float, default=8.0)
    ap.add_argument("--batch", type=int, default=8192, help="评测前向批大小")
    ap.add_argument("--progress_every_files", type=int, default=10, help="每处理多少个文件打印一次进度")
    args = ap.parse_args()

    # 1) 读 ckpt（rank0）并广播
    if rank == 0:
        ck = np.load(os.path.expanduser(args.ckpt), allow_pickle=True)
        W1 = ck["W1"]; b1 = ck["b1"]; W2 = ck["W2"]; b2 = ck["b2"]
        mu = ck["mu"]; sigma = ck["sigma"]
        mu_y = float(ck["mu_y"][0]); sigma_y = float(ck["sigma_y"][0])
        feat_cols = ck["feat_cols"].tolist()
        activation = str(ck["activation"][0])
        target_mode = str(ck["target"][0])
        meta = (feat_cols, activation, target_mode, mu, sigma, mu_y, sigma_y, W1, b1, W2, b2)
    else:
        meta = None
    (feat_cols, activation, target_mode, mu, sigma, mu_y, sigma_y, W1, b1, W2, b2) = comm.bcast(meta, root=0)
    params = [W1, b1, W2, b2]
    act = ACT[activation]

    # 2) 列出测试文件并分发
    if rank == 0:
        all_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.test_dir), "*.parquet")))
        if not all_files:
            raise FileNotFoundError(f"No parquet files in {args.test_dir}")
        splits = np.array_split(all_files, size)
    else:
        splits = None
    my_files = comm.scatter(splits, root=0)
    n_files_local = len(my_files)

    # 3) 先做一次“轻量统计”跑一遍文件头，估计测试集 log 的均值方差（不 concat）
    n_local = 0
    sum_log_local = 0.0
    sumsq_log_local = 0.0
    for f in my_files:
        df = pd.read_parquet(f, engine="pyarrow")
        # 容错补列
        if "total_amount_log" not in df.columns:
            if "total_amount" not in df.columns:
                raise KeyError(f"Missing column 'total_amount' in {f}")
            df["total_amount_log"] = np.log1p(df["total_amount"].astype(np.float64))
        # 累计统计
        ylog = df["total_amount_log"].to_numpy(np.float64, copy=False)
        n_local += ylog.shape[0]
        sum_log_local += float(np.sum(ylog))
        sumsq_log_local += float(np.sum(ylog * ylog))

    n_total = comm.allreduce(n_local, op=MPI.SUM)
    s1 = comm.allreduce(sum_log_local, op=MPI.SUM)
    s2 = comm.allreduce(sumsq_log_local, op=MPI.SUM)
    mu_log_test = s1 / max(1, n_total)
    var_log_test = max(0.0, s2 / max(1, n_total) - mu_log_test * mu_log_test)
    std_log_test = math.sqrt(var_log_test)
    if rank == 0:
        print(f"[Diag] test_size={n_total}, test_log_mean={mu_log_test:.4f}, test_log_std={std_log_test:.4f}", flush=True)

    # 4) 真正评测：逐文件→分批前向→即时累计 SSE（log 与 USD）
    sse_log_local, n_log = 0.0, 0
    sse_usd_local, n_usd = 0.0, 0
    B = args.batch

    t0 = time.time()
    for idx_f, f in enumerate(my_files, 1):
        df = pd.read_parquet(f, engine="pyarrow")

        # 特征列检查（只在需要时检查一次）
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            # 汇报到 rank0 抛错
            miss_all = comm.gather(missing, root=0)
            if rank == 0:
                miss_any = sorted(set(sum(miss_all, [])))
                raise KeyError(f"[Eval] Missing feature columns in test data: {miss_any}")
            else:
                # 其它 rank 提交占位
                comm.gather([], root=0)
            return

        # 自动补 total_amount_log
        if "total_amount_log" not in df.columns:
            df["total_amount_log"] = np.log1p(df["total_amount"].astype(np.float64))

        # 提取需要列
        X = df[feat_cols].to_numpy(np.float64, copy=False)
        y_amt = df["total_amount"].to_numpy(np.float64, copy=False)
        y_log = df["total_amount_log"].to_numpy(np.float64, copy=False)

        # 特征标准化
        if X.shape[0] > 0:
            X = (X - mu) / sigma

        # 分批前向
        for i in range(0, X.shape[0], B):
            Xb = X[i:i+B].T
            if Xb.shape[1] == 0: break
            Z2 = forward(Xb, params, act)  # 1 x bsz

            if target_mode == "log":
                y_pred_log = (Z2 * sigma_y + mu_y).ravel()
                y_pred_log = np.clip(y_pred_log, -5, args.clip_log)

                diff_log = y_pred_log - y_log[i:i+B]
                sse_log_local += float(np.sum(diff_log**2))
                n_log += len(diff_log)

                y_pred_usd = np.expm1(y_pred_log)
                diff_usd = y_pred_usd - y_amt[i:i+B]
                sse_usd_local += float(np.sum(diff_usd**2))
                n_usd += len(diff_usd)
            else:
                y_pred_usd = Z2.ravel()
                diff_usd = y_pred_usd - y_amt[i:i+B]
                sse_usd_local += float(np.sum(diff_usd**2))
                n_usd += len(diff_usd)

        # 进度日志（各 rank 各自报；数量不大时很有用）
        if (idx_f % max(1, args.progress_every_files) == 0) or (idx_f == n_files_local):
            print(f"[rank {rank}] eval progress: {idx_f}/{n_files_local} files "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)

    # 5) 汇总并打印
    rmse_log = rmse_parallel(comm, sse_log_local, n_log) if target_mode == "log" else float('nan')
    rmse_usd = rmse_parallel(comm, sse_usd_local, n_usd)

    if rank == 0:
        if target_mode == "log":
            print(f"[Eval] RMSE_log={rmse_log:.4f}  RMSE_USD={rmse_usd:.2f}", flush=True)
        else:
            print(f"[Eval] RMSE_USD={rmse_usd:.2f}", flush=True)

if __name__ == "__main__":
    main()
