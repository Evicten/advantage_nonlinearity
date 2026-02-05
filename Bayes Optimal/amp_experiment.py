import os
import json
import time
import numpy as np
from pathlib import Path
from mpi4py import MPI

# Your AMP implementation file should expose AMP_algo (you pasted it above)
from amp import AMP_algo, generate_data

# ====== GLOBAL CONFIG (edit by hand) ======
# Single (gamma, delta) pair
GAMMA = 0.0
DELTA = 0.0

# Single planting value (0.90 for planted, 0.01 for "unplanted")
PLANT = 0.90

# Alpha grid
ALPHA_MIN = 0.1
ALPHA_MAX = 80.0
N_ALPHA   = 300
ALPHA_VALUES = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)

# Model / algorithm params
D        = 2000
BETA_U   = 1.0
BETA_V   = 2.0
MAX_ITER = 1000
TOL      = 1e-4
DAMP     = 0.6
EPS      = 1e-6
ASSUME_CONCENTRATION = True

# Runs per alpha
RUNS = 72

# Base output directory
BASE_OUT = Path("data/amp_runs")

# ----- OPTIONAL env overrides (keep globals as defaults) -----
def _env_float(name, default):
    v = os.environ.get(name, None)
    return float(v) if v is not None else default

def _env_int(name, default):
    v = os.environ.get(name, None)
    return int(v) if v is not None else default

def _env_bool(name, default):
    v = os.environ.get(name, None)
    if v is None:
        return default
    if v.lower() in ["1", "true", "yes", "on"]:
        return True
    if v.lower() in ["0", "false", "no", "off"]:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {v}")


GAMMA = _env_float("AMP_GAMMA", GAMMA)
DELTA = _env_float("AMP_DELTA", DELTA)
PLANT = _env_float("AMP_PLANT", PLANT)

D        = _env_int("AMP_D", D)
BETA_U   = _env_float("AMP_BETA_U", BETA_U)
BETA_V   = _env_float("AMP_BETA_V", BETA_V)
MAX_ITER = _env_int("AMP_MAX_ITER", MAX_ITER)
TOL      = _env_float("AMP_TOL", TOL)
DAMP     = _env_float("AMP_DAMP", DAMP)
EPS      = _env_float("AMP_EPS", EPS)
RUNS     = _env_int("AMP_RUNS", RUNS)
ASSUME_CONCENTRATION = _env_bool("AMP_CONCENTRATION", ASSUME_CONCENTRATION)

ALPHA_MIN = _env_float("AMP_ALPHA_MIN", ALPHA_MIN)
ALPHA_MAX = _env_float("AMP_ALPHA_MAX", ALPHA_MAX)
N_ALPHA   = _env_int("AMP_N_ALPHA", N_ALPHA)
ALPHA_VALUES = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)
# -------------------------------------------------------------


# ------------------------------------------

def alabel(a: float) -> str:
    """Safe tag for filenames: 0.500 -> '0p500'; handles small/negatives."""
    s = f"{a:.6g}"
    return s.replace(".", "p").replace("-", "m")

def build_top_dir() -> Path:
    """
    Top-level folder encodes global params (date + fixed settings).
    e.g., data/amp_runs/20251021/betaU=1p0_betaV=2p0_d=1000_tol=1p0e-04_damp=0p4_maxit=500_eps=1p0e-06_runs=72_alphaMin=0p1_alphaMax=80_nAlpha=300
    """
    today = time.strftime("%Y%m%d")
    parts = [
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        f"d={D}",
        f"tol={alabel(TOL)}",
        f"damp={alabel(DAMP)}",
        f"maxit={MAX_ITER}",
        f"eps={alabel(EPS)}",
        f"runs={RUNS}",
        f"alphaMin={alabel(ALPHA_MIN)}",
        f"alphaMax={alabel(ALPHA_MAX)}",
        f"nAlpha={N_ALPHA}",
    ]
    return BASE_OUT / today / "_".join(parts)

def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split RUNS across ranks (not alphas)
    all_runs = np.arange(RUNS, dtype=int)
    run_chunks = np.array_split(all_runs, size)
    my_runs = run_chunks[rank]

    # --- directories
    top_dir = build_top_dir()
    if rank == 0:
        os.makedirs(top_dir, exist_ok=True)
        meta = dict(
            beta_u=BETA_U, beta_v=BETA_V, d=D,
            tol=TOL, damping=DAMP, max_iter=MAX_ITER, eps=EPS,
            runs=RUNS,
            alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX, n_alpha=N_ALPHA,
            gamma=GAMMA, delta=DELTA, plant=PLANT,
            note="Single (gamma,delta) & single planting run."
        )
        write_json(top_dir / "meta.json", meta)
    comm.Barrier()

    plant_dir = top_dir / f"plant={alabel(PLANT)}"
    gd_dir    = plant_dir / f"gamma={alabel(GAMMA)}_delta={alabel(DELTA)}"
    if rank == 0:
        os.makedirs(gd_dir, exist_ok=True)
        np.save(gd_dir / "alpha_values.npy", ALPHA_VALUES)
        print(f"\n=== plant={PLANT:.2f} | (gamma={GAMMA:.3f}, delta={DELTA:.3f}) ===")
    comm.Barrier()

    # Sweep alphas; each alpha: ranks compute their subset of RUNS, gather to root, save
    for alpha in ALPHA_VALUES:
        n = int(alpha * D)
        local_overlaps, local_flags, local_indices = [], [], []
        local_histories = [] 

        for r in my_runs:
            # seed per (alpha, run)
            seed = ((int(alpha*1_000_000) % (2**31-1)) ^ (r*7919) ^ (rank*2654435761)) & 0x7FFFFFFF
            np.random.seed(seed)

            X, y, u_star, v_star = generate_data(n=n, d=D, beta_u=BETA_U, beta_v=BETA_V, gamma=GAMMA, delta=DELTA)
            w_star = np.stack([u_star, v_star], axis=1)  # (D,2)

            try:
                w, overlap_hist, converged = AMP_algo(
                    X, y, K=2,
                    beta_u=BETA_U, beta_v=BETA_V, gamma=GAMMA, delta=DELTA,
                    max_iter=MAX_ITER, tol=TOL, plant=PLANT,
                    damp=DAMP, w_star=w_star, eps=EPS, rank=rank, assume_concentration=ASSUME_CONCENTRATION
                )
                if overlap_hist.shape[0] > 0:
                    local_overlaps.append(overlap_hist[-1])
                else:
                    local_overlaps.append(np.full((2,2), np.nan))

                local_histories.append(overlap_hist.copy())
                local_flags.append(bool(converged))
                local_indices.append(int(r))
            except Exception as e:
                print(f"[Rank {rank}] α={alpha:.3f}, run={r} failed: {repr(e)}")
                local_overlaps.append(np.full((2,2), np.nan))
                local_histories.append(np.zeros((0,2,2)))
                local_flags.append(False)
                local_indices.append(int(r))

        gathered = comm.gather((local_indices, local_overlaps, local_flags, local_histories), root=0)

        if rank == 0:
            final_overlaps = np.full((RUNS, 2, 2), np.nan, dtype=float)
            converged      = np.zeros((RUNS,), dtype=bool)
            histories = [np.zeros((0,2,2)) for _ in range(RUNS)]
            iters = np.zeros((RUNS,), dtype=int)

            for idxs, ovs, flg, hists in gathered:
                for i, O, f, H in zip(idxs, ovs, flg, hists):
                    final_overlaps[i] = O
                    converged[i] = f
                    histories[i] = H
                    iters[i] = H.shape[0] if isinstance(H, np.ndarray) else 0

            tag = alabel(alpha)
            np.save(gd_dir / f"final_overlaps_alpha_{tag}.npy", final_overlaps)
            np.save(gd_dir / f"converged_alpha_{tag}.npy",      converged)
            np.save(gd_dir / f"overlap_histories_alpha_{tag}.npy", np.array(histories, dtype=object))
            np.save(gd_dir / f"iters_alpha_{tag}.npy", iters)

            ok = np.isfinite(final_overlaps.reshape(RUNS, -1)).all(axis=1).sum()
            print(f"[γ={GAMMA:.3f}, δ={DELTA:.3f}, α={alpha:.3f}, plant={PLANT:.2f}] "
                  f"saved. ok_runs={ok}/{RUNS}")

    comm.Barrier()

if __name__ == "__main__":
    main()

