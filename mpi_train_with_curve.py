#!/usr/bin/env python3
"""
Single-hidden-layer MPI-SGD trainer with:
- Parallel stochastic gradients (per-rank minibatch, MPI Allreduce -> average -> sync update)
- Optional gradient clipping (global L2 on concatenated grads)
- Training risk curve logging R(θ_k) vs iteration k (parallel SSE Allreduce)
- Debug prints to demonstrate per-rank local grad norms vs global-avg grad norm
- Optional theta consistency checks across ranks
- Parquet streaming: each rank randomly samples row-groups from its shard of files

Dependencies: numpy, pandas, pyarrow, mpi4py

Example:
mpirun -np 4 ~/venv/bin/python -u mpi_train_with_curve.py \
  --data_dir ~/taxi_data \
  --target amount \
  --hidden 128 --activation relu \
  --batch 1024 --epochs 4 --lr 1e-3 --clip 5 \
  --max_steps_per_epoch 200 \
  --save_path ~/ckpts/taxi_relu_h128_b1024_ep4.npz \
  --curve_every 20 --curve_log ~/logs/train_curve.csv \
  --debug_parallel --check_theta_every 100
"""

from __future__ import annotations
import os, sys, math, time, glob, argparse
import numpy as np
import pandas as pd
from mpi4py import MPI
import pyarrow.parquet as pq

# -------------------- MPI setup --------------------
comm = MPI.COMM_WORLD
WORLD = comm.Get_size()

rank = comm.Get_rank()

# -------------------- Utils --------------------
ACTS = {
    "relu": (lambda x: np.maximum(x, 0.0),
              lambda x: (x > 0.0).astype(x.dtype)),
    "tanh": (np.tanh,
              lambda x: 1.0 - np.tanh(x)**2),
    "sigmoid": (lambda x: 1.0 / (1.0 + np.exp(-x)),
                 lambda x: (1.0 / (1.0 + np.exp(-x))) * (1.0 - 1.0 / (1.0 + np.exp(-x))))
}

def set_seed(seed: int):
    if seed is None:
        return
    try:
        import random
        random.seed(seed + rank)
    except Exception:
        pass
    np.random.seed(seed + rank)

# Pack/unpack helpers for gradients

def pack_params(dW1, db1, dW2, db2):
    return np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), np.array([db2])])

def unpack_params(vec, shapes):
    # shapes: ((D,H),(H,),(H,),(1,)) for W1,b1,W2,b2
    D,H = shapes[0]
    i=0
    dW1 = vec[i:i+D*H].reshape(D,H); i+=D*H
    db1 = vec[i:i+H]; i+=H
    dW2 = vec[i:i+H]; i+=H
    db2 = float(vec[i])
    return dW1, db1, dW2, db2

# -------------------- Data --------------------

FEATURE_COLUMNS_CANDIDATES = [
    "passenger_count", "trip_distance", "extra",
    "RatecodeID", "PULocationID", "DOLocationID", "payment_type",
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "fare_amount", "mta_tax", "improvement_surcharge", "tip_amount", "tolls_amount"
]

TARGET_COLUMN = "total_amount"


def _cyc(x, period):
    # return sin, cos
    return np.sin(2*np.pi*x/period), np.cos(2*np.pi*x/period)


def make_features(df: pd.DataFrame):
    """Build numerical features robustly from common NYC taxi columns.
    No one-hot for location IDs to keep dimension small.
    Returns X (float32, n×D) and y_amount (float32, n, raw USD total_amount).
    """
    n = len(df)
    # Parse datetimes if present
    if "tpep_pickup_datetime" in df.columns:
        p = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        d = pd.to_datetime(df.get("tpep_dropoff_datetime"), errors="coerce")
        hour = p.dt.hour.fillna(0).astype(np.float32)
        wday = p.dt.weekday.fillna(0).astype(np.float32)
        dur_min = (d - p).dt.total_seconds().fillna(0) / 60.0
        dur_min = dur_min.clip(lower=0, upper=240).astype(np.float32)
        sh, ch = _cyc(hour, 24)
        sw, cw = _cyc(wday, 7)
    else:
        dur_min = pd.Series(np.zeros(n, dtype=np.float32))
        sh = ch = sw = cw = pd.Series(np.zeros(n, dtype=np.float32))

    def fget(name, default=0.0):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(np.float32)
        return pd.Series(np.full(n, default, dtype=np.float32))

    passenger = fget("passenger_count")
    trip_distance = fget("trip_distance")
    extra = fget("extra")
    ratecode = fget("RatecodeID")
    pu = fget("PULocationID") / 300.0  # scale down IDs
    do = fget("DOLocationID") / 300.0
    paytype = fget("payment_type")

    # Light transforms
    lg_dist = np.log1p(trip_distance)
    lg_dur  = np.log1p(dur_min)

    X = np.stack([
        passenger, trip_distance, lg_dist, extra,
        ratecode, pu, do, paytype,
        dur_min, lg_dur,
        sh.astype(np.float32), ch.astype(np.float32), sw.astype(np.float32), cw.astype(np.float32)
    ], axis=1).astype(np.float32)

    # Target
    if TARGET_COLUMN not in df.columns:
        raise RuntimeError(f"Missing target column '{TARGET_COLUMN}' in parquet")
    y_amount = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0.0).astype(np.float32).to_numpy()

    # Filter bad rows
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y_amount) & (y_amount >= 0.0)
    X = X[mask]
    y_amount = y_amount[mask]
    return X, y_amount


class ParquetMinibatcher:
    def __init__(self, files, batch, rng: np.random.Generator):
        self.files = list(files)
        if not self.files:
            raise FileNotFoundError("No parquet files for this rank")
        self.batch = int(batch)
        self.rng = rng
        self._cache = None

    def sample(self):
        # Pick a random file and random row-group, then sample up to batch rows
        pf_path = self.rng.choice(self.files)
        pf = pq.ParquetFile(pf_path)
        rg_idx = int(self.rng.integers(0, max(1, pf.num_row_groups)))
        tbl = pf.read_row_group(rg_idx)  # read needed columns later in pandas
        df = tbl.to_pandas()  # might be ~100k rows; ok for sampling
        X, y_amount = make_features(df)
        if len(X) == 0:
            # fallback: try another group
            return self.sample()
        if len(X) > self.batch:
            idx = self.rng.choice(len(X), size=self.batch, replace=False)
            X = X[idx]
            y_amount = y_amount[idx]
        return X, y_amount

# -------------------- Model --------------------

class OneHiddenMLP:
    def __init__(self, D, H, act_name: str, rng: np.random.Generator):
        if act_name not in ACTS:
            raise ValueError(f"activation must be one of {list(ACTS)}")
        self.act_name = act_name
        self.act, self.actp = ACTS[act_name]
        # Xavier/He-like init depending on act
        scale = {
            "relu": math.sqrt(2.0 / D),
            "tanh": math.sqrt(1.0 / D),
            "sigmoid": math.sqrt(1.0 / D),
        }[act_name]
        self.W1 = rng.normal(0, scale, size=(D, H)).astype(np.float32)
        self.b1 = np.zeros(H, dtype=np.float32)
        self.W2 = rng.normal(0, 1.0/math.sqrt(H), size=(H,)).astype(np.float32)
        self.b2 = np.float32(0.0)

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1  # (N,H)
        H1 = self.act(Z1)
        y = H1 @ self.W2 + self.b2  # (N,)
        return y, (Z1, H1)

    def grads(self, X, y_true, y_pred, cache):
        # Loss: (1/(2M)) * sum (y_pred - y)^2
        M = max(1, y_true.shape[0])
        diff = (y_pred - y_true).astype(np.float32)
        dyp = diff / M  # dL/dy_pred
        Z1, H1 = cache
        # dW2, db2
        dW2 = (H1.T @ dyp).astype(np.float32)        # (H,)
        db2 = np.sum(dyp, dtype=np.float32)
        # dL/dH1 -> dL/dZ1
        g = (dyp[:, None] * self.W2[None, :]).astype(np.float32)
        if self.act_name == "relu":
            mask = (Z1 > 0.0).astype(np.float32)
            dZ1 = g * mask
        elif self.act_name == "tanh":
            dZ1 = g * (1.0 - np.tanh(Z1)**2)
        else:  # sigmoid
            s = 1.0 / (1.0 + np.exp(-Z1))
            dZ1 = g * s * (1.0 - s)
        # dW1, db1
        dW1 = (X.T @ dZ1).astype(np.float32)        # (D,H)
        db1 = np.sum(dZ1, axis=0, dtype=np.float32) # (H,)
        return dW1, db1, dW2, db2

    def pack_grads(self, dW1, db1, dW2, db2):
        return pack_params(dW1, db1, dW2, db2)

    def apply_grad(self, gvec, lr, shapes, clip=None):
        # Optional global gradient clipping on the averaged gradient vector
        if clip is not None and clip > 0:
            gnorm = float(np.linalg.norm(gvec))
            if gnorm > clip:
                gvec = gvec * (clip / (gnorm + 1e-12))
        dW1, db1, dW2, db2 = unpack_params(gvec, shapes)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def shapes(self):
        return (self.W1.shape, self.b1.shape, self.W2.shape, (1,))

    def save(self, path, meta: dict):
        if rank == 0:
            os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)
            np.savez(os.path.expanduser(path), W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, **meta)

# -------------------- Training --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory of *.parquet (train shard)")
    ap.add_argument("--target", choices=["amount", "log"], default="amount")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--activation", choices=list(ACTS.keys()), default="relu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_steps_per_epoch", type=int, default=200)
    ap.add_argument("--clip", type=float, default=0.0, help="Global grad clip (L2) AFTER averaging")
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--seed", type=int, default=2025)
    # Curve/Debug
    ap.add_argument("--curve_every", type=int, default=20,
                    help="Log training risk estimate every K updates")
    ap.add_argument("--curve_log", type=str, default=None,
                    help="CSV path to save training curve (rank0)")
    ap.add_argument("--debug_parallel", action="store_true",
                    help="Print per-rank local/global grad norms periodically")
    ap.add_argument("--check_theta_every", type=int, default=0,
                    help="If >0, check theta norm equality every K steps; broadcast if diverged")
    args = ap.parse_args()

    set_seed(args.seed)

    # Discover files and shard across ranks
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.data_dir), "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {args.data_dir}")
    my_files = files[rank::WORLD]
    print(f"[rank {rank}] using {len(my_files)} files (of {len(files)})", flush=True)

    rng = np.random.default_rng(args.seed + 1000*rank)
    # Build one minibatch to get input dimensionality
    X0, y0_amount = ParquetMinibatcher(my_files, args.batch, rng).sample()
    if args.target == "log":
        y0 = np.log(np.clip(y0_amount, 1e-3, None)).astype(np.float32)
    else:
        y0 = y0_amount
    D = X0.shape[1]
    model = OneHiddenMLP(D, args.hidden, args.activation, rng)

    # Curve logging setup
    t0 = time.time()
    curve_fp = None
    if args.curve_log and rank == 0:
        os.makedirs(os.path.dirname(os.path.expanduser(args.curve_log)), exist_ok=True)
        curve_fp = open(os.path.expanduser(args.curve_log), "w", buffering=1)
        print("step,epoch,elapsed_s,R_hat,lr", file=curve_fp)
    global_step = 0

    batcher = ParquetMinibatcher(my_files, args.batch, rng)

    # Training
    for epoch in range(1, args.epochs+1):
        for step in range(1, args.max_steps_per_epoch+1):
            Xb, yb_amount = batcher.sample()
            yb = np.log(np.clip(yb_amount, 1e-3, None)).astype(np.float32) if args.target=="log" else yb_amount

            # Forward
            y_pred, cache = model.forward(Xb)
            # Parallel risk estimate using current mini-batch: R_hat = 0.5 * SSE / M_total
            residual = (y_pred - yb).astype(np.float64)  # promote for summation stability
            sse_local = float(residual.dot(residual))
            m_local = float(residual.size)
            buf = np.array([sse_local, m_local], dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
            sse_tot, m_tot = buf
            R_hat = 0.5 * (sse_tot / max(m_tot, 1.0))

            # Local grads, then Allreduce-average
            dW1, db1, dW2, db2 = model.grads(Xb, yb, y_pred, cache)
            gvec_local = model.pack_grads(dW1, db1, dW2, db2).astype(np.float64)
            gvec_global = np.empty_like(gvec_local)
            comm.Allreduce(gvec_local, gvec_global, op=MPI.SUM)
            gvec_global /= float(WORLD)

            # Debug: show local vs global grad norms
            if args.debug_parallel and (global_step % 20 == 0):
                l2_local = float(np.linalg.norm(gvec_local))
                l2_global = float(np.linalg.norm(gvec_global))
                print(f"[rank {rank}] step {global_step} ||g_local||={l2_local:.4e} ||g_avg||={l2_global:.4e}", flush=True)

            # Apply update (optionally clip the averaged gradient)
            model.apply_grad(gvec_global.astype(np.float32), args.lr, model.shapes(), clip=args.clip)

            global_step += 1
            if args.curve_every > 0 and (global_step % args.curve_every == 0) and rank == 0:
                elapsed = time.time() - t0
                print(f"[curve] k={global_step} R_hat={R_hat:.6f} elapsed={elapsed:.1f}s")
                if curve_fp:
                    print(f"{global_step},{epoch},{elapsed:.3f},{R_hat:.8f},{args.lr}", file=curve_fp)

            # Optional consistency check (more stable relative check across all params) 
            if args.check_theta_every and (global_step % args.check_theta_every == 0):
                v = np.concatenate([
                    model.W1.ravel().astype(np.float64),
                    model.b1.ravel().astype(np.float64),
                    model.W2.ravel().astype(np.float64),
                    np.array([model.b2], dtype=np.float64),
                ])
                # two simple signatures: L2 norm and partial sum (first 2048 elems)
                sig = np.array([np.linalg.norm(v), v[:2048].sum()], dtype=np.float64)
                sig_all = comm.allgather(sig)
                ref = sig_all[0]
                ok = all(np.allclose(s, ref, rtol=1e-5, atol=1e-7) for s in sig_all)
                if not ok and rank == 0:
                    print("[Warn] tiny param mismatch; broadcasting rank0", flush=True)
                # force resync (rarely triggers)
                comm.Bcast(model.W1, root=0)
                comm.Bcast(model.b1, root=0)
                comm.Bcast(model.W2, root=0)
                tmp = np.array([model.b2], dtype=np.float32)
                comm.Bcast(tmp, root=0)
                model.b2 = float(tmp[0])

            # Pretty progress line

            if step % 5 == 0 and rank == 0:
                print(f"[epoch {epoch}] progress: {step}/{args.max_steps_per_epoch} batches")

        # Epoch boundary marker (optional)
        if rank == 0:
            print(f"[Epoch {epoch}] (training only)")

    if curve_fp is not None:
        curve_fp.close()

    # Save checkpoint (rank 0)
    meta = {"hidden": args.hidden, "activation": args.activation, "input_dim": D,
            "target": args.target, "world": WORLD}
    model.save(args.save_path, meta)
    if rank == 0:
        print(f"[Saved] {os.path.expanduser(args.save_path)}")
        print(f"Train time ~{time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
