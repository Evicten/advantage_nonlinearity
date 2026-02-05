import os
import numpy as np
from pathlib import Path
from mpi4py import MPI
import json
import time

from state_evolution import run_one_alpha  # must accept mode="replica" or "rBP"

# ========= ENV / USER SETTINGS =========
# Defaults are here; env vars can override them.

def get_float_env(name, default):
    return float(os.getenv(name, str(default)))

def get_int_env(name, default):
    return int(os.getenv(name, str(default)))

# Model / algorithm params (can be overridden by env)
BETA_U = get_float_env("SE_BETA_U", 1.0)
BETA_V = get_float_env("SE_BETA_V", 2.0)
BETA   = (BETA_U, BETA_V)

SAMPLES = get_int_env("SE_SAMPLES", 1000)
ITERS   = get_int_env("SE_ITERS",   100)
DAMPING = get_float_env("SE_DAMPING", 0.3)

# Single (gamma, delta) and mode for this run
GAMMA = get_float_env("SE_GAMMA", 1.0)
DELTA = get_float_env("SE_DELTA", 0.0)
MODE  = os.getenv("SE_MODE", "replica")  # "replica" or "rBP"

# Alpha grid (one grid per job)
ALPHA_MIN = get_float_env("SE_ALPHA_MIN", 0.1)
ALPHA_MAX = get_float_env("SE_ALPHA_MAX", 10.0)
N_ALPHA   = get_int_env("SE_N_ALPHA", 2)
ALPHA_VALUES = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)

# q_init entries (upper triangle). Default: [[0.10, 0.01],
#                                            [0.01, 0.10]]
Q11 = get_float_env("SE_Q11", 0.001)
Q22 = get_float_env("SE_Q22", 0.001)
Q12 = get_float_env("SE_Q12", 0.0001)

Q_INIT = np.array([[Q11, Q12],
                   [Q12, Q22]], dtype=float)

# Base output directory
BASE_OUT = Path("data/runs")


# ==========================================

def alabel(a: float) -> str:
    """Safe tag for filenames: 0.500 -> '0p500'."""
    return f"{a:.3f}".replace(".", "p")

def tag_qinit(q):
    """Encode q_init upper triangle (q11, q22, q12) into a short tag."""
    return f"qinit={alabel(q[0,0])}_{alabel(q[1,1])}_{alabel(q[0,1])}"

def build_top_dir() -> Path:
    """Top-level folder encodes the global params (same as your previous script)."""
    today = time.strftime("%Y%m%d")
    parts = [
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        tag_qinit(Q_INIT),
        f"samples={SAMPLES}",
        f"iters={ITERS}",
        f"damp={alabel(DAMPING)}",
    ]
    return BASE_OUT / today / "_".join(parts)

def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ---- create top-level directory and meta (root only)
    top_dir = build_top_dir()
    if rank == 0:
        top_dir.mkdir(parents=True, exist_ok=True)
        top_meta = dict(
            beta_u=BETA_U, beta_v=BETA_V,
            q_init=Q_INIT.tolist(),
            samples=SAMPLES, iters=ITERS,
            damping=DAMPING,
            mode=MODE,
            gamma=GAMMA,
            delta=DELTA,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            n_alpha=N_ALPHA,
            note="Single (gamma, delta, mode) per job. Params may be overridden via env SE_*."
        )
        write_json(top_dir / "meta.json", top_meta)
    comm.Barrier()

    # ---- Mode-level folder
    mode_dir = top_dir / f"mode={MODE}"
    if rank == 0:
        mode_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # ---- Subfolder for this (gamma, delta)
    sub_dir = mode_dir / f"gamma={alabel(GAMMA)}_delta={alabel(DELTA)}"
    if rank == 0:
        sub_dir.mkdir(parents=True, exist_ok=True)
        # Save alpha grid once for this job
        np.save(sub_dir / "alpha_values.npy", ALPHA_VALUES)
        print(f"\n=== mode={MODE} (gamma={GAMMA:.3f}, delta={DELTA:.3f}) ===")
    comm.Barrier()


    # Sweep over alpha
    for a in ALPHA_VALUES:
        tag = alabel(a)

        # Define per-alpha filenames
        f_q_list    = sub_dir / f"q_list_alpha_{tag}.npy"
        f_qhat_list = sub_dir / f"q_hat_list_alpha_{tag}.npy"

        # Run one alpha (uses the chosen mode)
        q_list, q_hat_list = run_one_alpha(
            alpha=a, beta=BETA, gamma=GAMMA, delta=DELTA,
            q_init=Q_INIT, samples=SAMPLES, iters=ITERS,
            damping=DAMPING, mode=MODE
        )

        # Save (root only)
        if rank == 0:
            np.save(f_q_list,    q_list)
            np.save(f_qhat_list, q_hat_list)
            print(f"[mode={MODE} | γ={GAMMA:.3f}, δ={DELTA:.3f}, α={a:.3f}] saved.")

    comm.Barrier()  # sync between (g, d) pairs inside this mode

if __name__ == "__main__":
    main()
