import numpy as np
from mpi4py import MPI
import os
from pathlib import Path
import json

from ERM_SE import run_one_alpha

# ========= ENV / USER SETTINGS =========
# Defaults are here; env vars can override them.

def get_float_env(name, default):
    return float(os.getenv(name, str(default)))

def get_int_env(name, default):
    return int(os.getenv(name, str(default)))

def get_str_env(name, default):
    return os.getenv(name, str(default))

def shp(x):
    if isinstance(x, np.ndarray): return x.shape
    if np.isscalar(x): return "scalar"
    try: return f"len={len(x)}"
    except: return type(x).__name_



# Model / algorithm params (can be overridden by env)
ACTIVATION = get_str_env("SE_ACTIVATION", 'tanh_half') # linear, relu, he1he2, sigmoid, tanh, elu, softsign, tanh_half
BETA_U = get_float_env("SE_BETA_U", 1.0)
BETA_V = get_float_env("SE_BETA_V", 2.0)
BETA   = (BETA_U, BETA_V)

SAMPLES = get_int_env("SE_SAMPLES", 100_000) # samples per rank
ITERS   = get_int_env("SE_ITERS",   500)  # Number of fixed-point iterations
DAMPING = get_float_env("SE_DAMPING", 0.95) 
SAMPLES_LOSS = get_int_env("SE_SAMPLES_LOSS", 1_000_000) # samples to compute the losses. We use more samples than for the iterations, as this is done only once.


# Single (gamma, delta) and mode for this run.
GAMMA = get_float_env("SE_GAMMA", 0.0)
DELTA = get_float_env("SE_DELTA", 1.0)

# Alpha grid
ALPHA_MIN = get_float_env("SE_ALPHA_MIN", 2) # For small alpha, might need to increase damping
ALPHA_MAX = get_float_env("SE_ALPHA_MAX", 6.2)
N_ALPHA   = get_int_env("SE_N_ALPHA", 1) # 123 
ALPHA_VALUES = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)

SOLVER='grid_search_refined' # Solver for the proximal. Options include: Newton_Raphson, Newton_Raphson_robust, golden_section, gradient_descent or grid_search_refined
NUM_STEP_SOLVER=25  # Num steps for the solver, exact meaning depends on the solver
CLIPPING=20         # Proximal values are clipped at -CLIPPING and +CLIPPING. This is also the border when doing a grid search or golden_section


# Optional annealing
ANNEALING_LAMBDA=False
START_LAMBDA=1
RATE_LAMBDA=100
STEPS_WITHOUT_ANNEALING_LAMBDA=500
ANNEALING_ALPHA=False
START_ALPHA=10
RATE_ALPHA=10
STEPS_WITHOUT_ANNEALING_ALPHA=500

# Regularisation
LAM = get_float_env("SE_LAM", 0.0)  # Regularization parameter

# initial order parameters (m, q, V)
M0_U = get_float_env("SE_M0_U", 0.001) 
M0_V = get_float_env("SE_M0_V", 0.001)
Q0   = get_float_env("SE_Q0", 4.00)  # 
V0   = get_float_env("SE_V0", 2.00)  # Note: for difficult activations function such as tanh, it is very much possible that it does not converge for small V initialization at small alpha

M0 = np.array([[M0_U, M0_V]])
STATE_INIT = (M0, Q0, V0)

# Base output directory
BASE_OUT = Path("results_SE_ERM/")


# ==========================================

def alabel(a: float) -> str:
    return f"{a:.3f}" # convert floats to 3 decimal place strings.

def tag_init(state):
    m0, q0, V0 = state
    m0 = np.asarray(m0).ravel()
    return (
        f"init_m=({alabel(m0[0])},{alabel(m0[1])})_"
        f"q={alabel(q0)}_V={alabel(V0)}"
    )

def build_top_dir(rank) -> Path:
    """Top-level folder encodes the global params."""
    parts = [
        f"act={ACTIVATION}",        
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        f"lambda={alabel(LAM)}",
        f"iters={ITERS}",
        f'gamma={alabel(GAMMA)}',
        f'delta={alabel(DELTA)}',
        f'' # Add optional comments here
    ]
    return BASE_OUT / "_".join(parts)

def hyperparams_txt(num_ranks):
    parts = [
        f"act={ACTIVATION}",        
        f"betaU={alabel(BETA_U)}",
        f"betaV={alabel(BETA_V)}",
        f"lambda={alabel(LAM)}",
        tag_init(STATE_INIT),
        f"samples={SAMPLES}",
        f"samples_loss={SAMPLES_LOSS}",
        f"rank={num_ranks}",
        f"iters={ITERS}",
        f"damp={alabel(DAMPING)}",
        f'gamma={alabel(GAMMA)}',
        f'delta={alabel(DELTA)}',
        f'solver={SOLVER}',
        f'steps_solver={alabel(NUM_STEP_SOLVER)}',
        f'clip={alabel(CLIPPING)}',
        f'annealing lambda={ANNEALING_LAMBDA}',     # Annealing from higher regularization lambda
        f'start_lambda={START_LAMBDA}',
        f'rate lambda={RATE_LAMBDA}',
        f'steps without annealing lambda={STEPS_WITHOUT_ANNEALING_LAMBDA}',
        f'annealing alpha={ANNEALING_ALPHA}',
        f'start alpha={START_ALPHA}',
        f'rate alpha={RATE_ALPHA}',
        f'steps without annealing alpha={STEPS_WITHOUT_ANNEALING_ALPHA}'
    ]
    return parts

def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(f"[Rank {rank}] Starting ERM SE run")

    # ---- create top-level directory and meta (root only)
    top_dir = build_top_dir(comm.Get_size())
    if rank == 0:
        top_dir.mkdir(parents=True, exist_ok=True)

    comm.Barrier()
    sub_dir = top_dir

    if rank == 0:
        num_ranks= comm.Get_size()
        sub_dir.mkdir(parents=True, exist_ok=True)
        # Save alpha grid once for this job
        np.save(sub_dir / "alpha_values.npy", ALPHA_VALUES)
        with open(sub_dir / '1params.txt', 'w') as f:
            f.write("\n".join(hyperparams_txt(num_ranks)))
        print(f"\n=====================================================\nStarting the run with\nACTIVATION={ACTIVATION}\nITERS={ITERS}\nSOLVER={SOLVER}=============================")
    comm.Barrier()


    # Sweep over alpha
    for a in ALPHA_VALUES:
        tag = alabel(a)

        # Define per-alpha filenames
        f_state_list    = sub_dir / f"state_list_alpha_{tag}.npy"
        f_hat_list = sub_dir / f"hat_list_alpha_{tag}.npy"
        losses_dir= sub_dir / f"losses_{tag}.npy"

        state_traj, hat_traj, losses = run_one_alpha(
            alpha=a,
            beta=BETA,
            gamma=GAMMA,
            delta=DELTA,
            init=STATE_INIT,
            samples=SAMPLES,
            samples_loss=SAMPLES_LOSS,
            iters=ITERS,
            activation=ACTIVATION,
            lam=LAM,
            damping=DAMPING,
            print_every=50, # Print every print_every fixed-point step
            annealing_lambda=ANNEALING_LAMBDA,
            lambda_start=START_LAMBDA,
            annealing_rate_lambda=RATE_LAMBDA,
            steps_without_annealing_lambda=STEPS_WITHOUT_ANNEALING_LAMBDA,
            annealing_alpha=ANNEALING_ALPHA,
            alpha_start=START_ALPHA,
            annealing_rate_alpha=RATE_ALPHA,
            steps_without_annealing_alpha=STEPS_WITHOUT_ANNEALING_ALPHA,
            solver=SOLVER,
            num_steps_solver=NUM_STEP_SOLVER,
            clip_param=CLIPPING,
            seed=0 # Random seed for reproducibility. The actual used seed is seed+rank for multiple processes.
        )

        # Save (root only)
        if rank == 0:
            np.save(f_state_list, np.array(state_traj, dtype=object))
            np.save(f_hat_list,   np.array(hat_traj,   dtype=object))
            np.save(losses_dir, losses)
            print(f"Run with α={a:.3f}, β_u={BETA_U:.3f}, β_v={BETA_V:.3f}, activation={ACTIVATION} saved. \nTrain MSE : {losses[0]:.4f} +- {losses[1]:.4f}, Test MSE: {losses[2]:.4f} +- {losses[3]:.4f}")

    comm.Barrier()

if __name__ == "__main__":
    main()