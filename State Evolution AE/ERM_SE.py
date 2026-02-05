import numpy as np
from scipy.stats import norm
from mpi4py import MPI
import os

#  Prevent JAX from pre-allocating 90% of VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# (Optional) Force JAX to be aggressive about freeing memory
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


# --- JAX Setup ---
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)
try:
    jax.devices('gpu')
    JAX_PLATFORM = 'gpu'
except RuntimeError:
    JAX_PLATFORM = 'cpu'
print(f"JAX is running on: {JAX_PLATFORM.upper()}")

# Matrix square root in jax. Not needed as we only consider the scalar p=1 case.
'''
@jax.jit
def sqrtm_eigh(A):
    """GPU-compatible matrix square root for symmetric matrices."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    return (eigvecs * jnp.sqrt(jnp.maximum(0, eigvals))) @ eigvecs.T
# --- END JAX Setup ---
'''

def beta_tilde(beta_v):
    return beta_v/(1+beta_v+np.sqrt(1+beta_v))

# Compute non-hatted order parameters from hatted order parameters
def state_func(state_hat, lam):
    m_hat, q_hat, V_hat = state_hat
    m = m_hat/(V_hat + lam)
    q = (q_hat+float(m_hat@m_hat.T))/(V_hat + lam)**2
    V = 1/(V_hat + lam)
    return (m, q, V)

# Generate latent in the generic gamma and delta case. Suboptimal if gamma=0, delta=1
def generate_latents(rng, n, gamma=0, delta=1): 
    lambda_vals = rng.standard_normal(n) 
    unif = rng.random((n,)) 
    nu_vals = np.zeros(n, dtype=int) 

    mask1 = unif < gamma 
    nu_vals[mask1] = np.sign(lambda_vals[mask1]) 
    nu_vals[mask1][nu_vals[mask1]==0] = 1 #if lambda is exactly zero, set nu to 1 

    mask2 = (unif >= gamma) & (unif < gamma + delta)
    t = norm.ppf(0.75)
    nu_vals[mask2] = np.where(np.abs(lambda_vals[mask2]) > t, 1, -1) 

    mask3 = ~(mask1 | mask2) 
    nu_vals[mask3] = rng.choice([-1, 1], size=np.sum(mask3), p=[1/2, 1/2]) 

    return lambda_vals, nu_vals

def F_1(beta_tilde_v):
    F1 = np.zeros((2,2))
    F1[1,1] = -beta_tilde_v
    return F1

def F_2(beta, lam_star, nu_star, Q, beta_tilde_v):
    beta_u, beta_v = beta
    F2 = np.zeros((2,))
    F2[0] = np.sqrt(beta_u)*lam_star
    F2[1] = np.sqrt(beta_v)*nu_star*(1-beta_tilde_v*Q[1,1])
    return F2

def F_2_batch(beta, lambda_vals, nu_vals, Q, beta_tilde_v):
    beta_u, beta_v = beta

    c0 = np.sqrt(beta_u)                             # scalar
    c1 = np.sqrt(beta_v) * (1 - beta_tilde_v * Q[1, 1])  # scalar

    # Each term is (n,), then we stack into (n, 2)
    F2_array = np.column_stack((
        c0 * lambda_vals,   # first coordinate for all samples
        c1 * nu_vals        # second coordinate for all samples
    ))

    return F2_array


# =============================================================================== ANALYTICAL PROXIMAL SOLVERS FOR SIMPLE ACTIVATIONS ========================================================
def find_proximal_linear_analytic(Y, q, V):
    """
    Analytic solution for Linear activation sigma(x) = x.
    Loss: -h^2 + 0.5*q*h^2 + 0.5/V*(h-Y)^2
    Derivative: -2h + qh + (h-Y)/V = 0
    h(q - 2 + 1/V) = Y/V
    h(V(q-2) + 1) = Y
    """
    denom = 1.0 + V * (q - 2.0)
    return Y / denom

def find_proximal_relu_analytic(Y, q, V):
    """
    Analytic solution for ReLU.
    """
    denom = 1.0 + V * (q - 2.0)
    g_pos = Y / denom
    # If Y <= 0, the penalty pulls h negative (where grad is 0), so h=Y.
    # If Y > 0, we check the positive solution.
    return np.where(Y/denom > 0, g_pos, Y)

# NOTE: for He1+He2 an analytical solution exists, but doing a few numerical steps using solver below is faster and reliable.

# =============================================================================== NUMERICAL PROXIMAL SOLVERS FOR GENERIC ACTIVATIONS ========================================================

# --- OBJECTIVE ---
def objective_elementwise(h, Y, q, inv_V, sigma):
    """
    Scalar objective function for a single sample.
    """
    sig_g = sigma(h)
    # Loss Term
    # (Only for scalar)
    loss_linear = -h * sig_g
    loss_quad   = 0.5 * q * (sig_g ** 2)
    
    # Proximal Quadratic Penalty
    diff = h - Y
    prox_penalty = 0.5 * inv_V * (diff ** 2)
    
    return loss_linear + loss_quad + prox_penalty

def get_newton_solver(num_iterations, sigma): # This version is less robust than the one below
    """
    Creates a JIT-compiled Newton-Raphson solver.
    """
    # Gradient
    grad_fn = jax.grad(partial(objective_elementwise, sigma=sigma), argnums=0)
    # Hessian
    hess_fn = jax.grad(grad_fn, argnums=0)
    # Newton Step
    def newton_step(h, Y, q, inv_V):
        g = grad_fn(h, Y, q, inv_V)
        H = hess_fn(h, Y, q, inv_V)
        return h - g / (H + 1e-10)
    # Vectorize over the batch (axis 0)
    vmapped_step = jax.vmap(newton_step, in_axes=(0, 0, None, None))
    # Loop
    def run_newton(h_init, Y_batch, q, inv_V):
        def body_fun(i, h):
            return vmapped_step(h, Y_batch, q, inv_V)
        final_h = jax.lax.fori_loop(0, num_iterations, body_fun, h_init)
        return final_h
    return jax.jit(run_newton)

def get_newton_robust_solver(num_iterations, sigma): # More robust to non-convexity
    # Automatic differentiation
    obj = partial(objective_elementwise, sigma=sigma)
    grad_fn = jax.grad(obj, argnums=0)
    hess_fn = jax.grad(grad_fn, argnums=0)
    
    # Robust Newton Step
    def newton_step(h, Y, q, inv_V):
        g = grad_fn(h, Y, q, inv_V)
        H = hess_fn(h, Y, q, inv_V)
        # If H is negative (concave region), Newton diverges.
        # We clamp H to be at least a small positive value.
        # This effectively switches to Gradient Descent in concave regions.
        H_safe = jnp.maximum(H, 1e-2)
        # Damping factor (0.5 to 1.0) helps with oscillations in non-convex regions
        lr = 1.0 
        
        return h - lr * (g / H_safe)
    vmapped_step = jax.vmap(newton_step, in_axes=(0, 0, None, None))

    def run_newton(h_init, Y_batch, q, inv_V):
        def body_fun(i, h):
            return vmapped_step(h, Y_batch, q, inv_V)
        return jax.lax.fori_loop(0, num_iterations, body_fun, h_init)

    return jax.jit(run_newton)

def get_golden_section_solver(sigma, min_val=-50.0, max_val=50.0, num_iters=35):
    """
    Creates a JIT-compiled Golden Section Search solver.
    Guaranteed to stay within [min_val, max_val].
    """
    obj_fn = partial(objective_elementwise, sigma=sigma)
    
    # Golden Ratio constants
    GR = (jnp.sqrt(5) - 1) / 2  # approx 0.618
    
    def solve_single(Y, q, inv_V):
        # Initialize Brackets
        a = min_val
        b = max_val
        
        # Initialize Internal Points
        c = b - GR * (b - a)
        d = a + GR * (b - a)
        
        # Evaluate initial function values
        fc = obj_fn(c, Y, q, inv_V)
        fd = obj_fn(d, Y, q, inv_V)
        
        # The Loop body
        def body_fun(i, state):
            a, b, c, d, fc, fd = state
            
            # Compare values
            # If fc < fd, the min is in [a, d]. New bracket is [a, d].
            # If fc > fd, the min is in [c, b]. New bracket is [c, b].
            
            # Logic for update if fc < fd:
            # new_b = d
            # new_d = c, new_fd = fc
            # new_c = a + (1-GR)*(new_b - a)
            
            # Logic for update if fc > fd:
            # new_a = c
            # new_c = d, new_fc = fd
            # new_d = new_a + GR*(b - new_a)
            
            cond = fc < fd
            
            # Update bounds
            b_new = jnp.where(cond, d, b)
            a_new = jnp.where(cond, a, c)
            
            c_new = b_new - GR * (b_new - a_new)
            d_new = a_new + GR * (b_new - a_new)
            
            fc_new = obj_fn(c_new, Y, q, inv_V)
            fd_new = obj_fn(d_new, Y, q, inv_V)
            
            return a_new, b_new, c_new, d_new, fc_new, fd_new

        # Run loop
        final_state = jax.lax.fori_loop(0, num_iters, body_fun, (a, b, c, d, fc, fd))
        a_f, b_f, _, _, _, _ = final_state
        
        return (a_f + b_f) / 2.0
    # Vectorize
    vmapped_solve = jax.vmap(solve_single, in_axes=(0, None, None))

    def run_solver(h_init_ignored, Y_batch, q, inv_V):
        # We ignore h_init because this is a global search within bounds
        return vmapped_solve(Y_batch, q, inv_V)
    return jax.jit(run_solver)
    
def get_gradient_descent_solver(num_iterations=100, learning_rate=0.1, sigma=None):
    """
    Finds a LOCAL minimum using Gradient Descent with momentum
    """
    # Define Gradient
    obj_fn = partial(objective_elementwise, sigma=sigma)
    grad_fn = jax.grad(obj_fn, argnums=0)
    
    # Vectorize
    vmapped_grad = jax.vmap(grad_fn, in_axes=(0, 0, None, None))

    def run_gd(h_init, Y_batch, q, inv_V):
        h = h_init
        # Simple GD with momentum
        def body_fun(i, state):
            h, v = state
            g = vmapped_grad(h, Y_batch, q, inv_V)
            
            # Momentum update
            beta = 0.9
            v = beta * v + (1 - beta) * g
            h = h - learning_rate * v
            return h, v

        final_h, _ = jax.lax.fori_loop(0, num_iterations, body_fun, (h, jnp.zeros_like(h)))
        return final_h

    return jax.jit(run_gd)


def get_global_local_solver(sigma, grid_min=-20.0, grid_max=20.0, grid_size=1000, refined_iters=20):
    """
    Creates a solver that:
    1. Evaluates the objective on a fixed grid of size `grid_size`.
    2. Identifies the best index for each sample.
    3. Runs Golden Section Search in the interval around that index.
    """
    
    # Pre-compute the fixed grid
    fixed_grid = jnp.linspace(grid_min, grid_max, grid_size)
    grid_step = (grid_max - grid_min) / (grid_size - 1)
    
    # Partial objective for internal use
    obj_fn = partial(objective_elementwise, sigma=sigma)
    
    # --- STEP 1: GRID SEARCH ---
    def grid_search_step(Y, q, inv_V):
        # Y: (Batch,) -> (Batch, 1)
        Y_col = Y[:, None]
        # h: (1, Grid)
        h_row = fixed_grid[None, :]
        
        # Evaluate Cost: Result shape (Batch, Grid)
        # q and inv_V are scalars, broadcast automatically
        costs = obj_fn(h_row, Y_col, q, inv_V)
        
        # Find best index for each sample
        # best_idx: (Batch,)
        best_idx = jnp.argmin(costs, axis=1)
        
        # Extract the center points
        # centers: (Batch,)
        centers = fixed_grid[best_idx]
        
        # Define dynamic brackets [center - step, center + step]
        # We widen slightly (* 1.1) to ensure we don't miss the minimum if it's on the neighboring points
        radius = grid_step * 1.1
        low_bounds = centers - radius
        high_bounds = centers + radius
        
        # Clip to global bounds to not get out of the grid
        low_bounds = jnp.maximum(low_bounds, grid_min)
        high_bounds = jnp.minimum(high_bounds, grid_max)
        
        return low_bounds, high_bounds

    # --- STEP 2: GOLDEN SECTION REFINEMENT ---
    # Accepts arrays 'low' and 'high' of shape (Batch,)
    def refinement_step(low, high, Y, q, inv_V):
        GR = (jnp.sqrt(5) - 1) / 2 
        
        # Initial internal points (Batch,)
        c = high - GR * (high - low)
        d = low + GR * (high - low)
        
        fc = obj_fn(c, Y, q, inv_V)
        fd = obj_fn(d, Y, q, inv_V)
        
        def body_fun(i, state):
            a, b, c, d, fc, fd = state
            
            # Vectorized condition
            cond = fc < fd
            
            # Update bounds
            b_new = jnp.where(cond, d, b)
            a_new = jnp.where(cond, a, c)
            
            # Standard Golden Section update
            c_new = b_new - GR * (b_new - a_new)
            d_new = a_new + GR * (b_new - a_new)
            
            fc_new = obj_fn(c_new, Y, q, inv_V)
            fd_new = obj_fn(d_new, Y, q, inv_V)
            
            return a_new, b_new, c_new, d_new, fc_new, fd_new

        # Run refinement loop
        final_state = jax.lax.fori_loop(0, refined_iters, body_fun, (low, high, c, d, fc, fd))
        a_f, b_f = final_state[0], final_state[1]
        
        return (a_f + b_f) / 2.0

    # --- MAIN SOLVER FUNCTION ---
    def solve(h_init_ignored, Y_batch, q, inv_V):
        # 1. Coarse Search
        low, high = grid_search_step(Y_batch, q, inv_V)

        # 2. Fine Search
        # Note: Y_batch must be (Batch,) here, not (Batch, 1)
        best_h = refinement_step(low, high, Y_batch, q, inv_V)

        return best_h

    return jax.jit(solve)



# ==========================================================================================================================================================================================

# --- JAX ACTIVATIONS ---
def relu_jax(x): return jnp.maximum(x, 0.0)
def linear_jax(x): return x
def he1he2_jax(x): return x**2 + x - 1.0
def sigmoid_jax(x): return jax.nn.sigmoid(x)
def tanh_jax(x): return jnp.tanh(x)
def tanh_half_jax(x): return jnp.tanh(x/2)
def elu_jax(x, alpha=1.0): return jax.nn.elu(x, alpha=alpha)
def softsign_jax(x): return jax.nn.soft_sign(x)

# --- GLOBAL CACHE ---
_SOLVER_CACHE = {}

def get_cached_solver(activation, solver='Newton_Raphson_robust', num_iterations=50, clip_param=5):

    cache_key = (activation, solver, num_iterations, float(clip_param))

    if cache_key not in _SOLVER_CACHE:
        if activation == 'relu': sigma_fn = relu_jax
        elif activation == 'linear': sigma_fn = linear_jax
        elif activation == 'he1he2': sigma_fn = he1he2_jax
        elif activation == 'sigmoid': sigma_fn = sigmoid_jax
        elif activation == 'tanh': sigma_fn = tanh_jax
        elif activation == 'tanh_half': sigma_fn= tanh_half_jax
        elif activation == 'elu': sigma_fn = elu_jax
        elif activation == 'softsign': sigma_fn = softsign_jax
        else: raise NotImplementedError(f"Activation {activation} not supported")
        
        if solver=='Newton_Raphson':
            _SOLVER_CACHE[cache_key] = get_newton_solver(num_iterations=num_iterations, sigma=sigma_fn)            
        elif solver=='Newton_Raphson_robust':        
            _SOLVER_CACHE[cache_key] = get_newton_robust_solver(num_iterations=num_iterations, sigma=sigma_fn)
        elif solver=='golden_section':
            _SOLVER_CACHE[cache_key] = get_golden_section_solver(sigma=sigma_fn, min_val=-clip_param, max_val=clip_param, num_iters=num_iterations) # Could use less iters if needed.
        elif solver=='gradient_descent':
            _SOLVER_CACHE[cache_key] = get_gradient_descent_solver(num_iterations=num_iterations, learning_rate=1.0, sigma=sigma_fn)
        elif solver=='grid_search_refined':
            _SOLVER_CACHE[cache_key]=get_global_local_solver(sigma=sigma_fn, grid_min=-float(clip_param), grid_max=float(clip_param), grid_size=1000, refined_iters=num_iterations)
        else:
            print('Solver should be Newton_Raphson, Newton_Raphson_robust, golden_section or gradient_descent')

    return _SOLVER_CACHE[cache_key]


#==============================================================================================================================================================
# =============================
# 2. LANDSCAPE PLOTTING HELPER
# =============================

import matplotlib.pyplot as plt
def plot_single_landscape(rng, Y_val, h_sol, q, V, activation, alpha, init, h_val_2=None, save_dir="plots"):
    """
    Plots the exact objective function landscape for a single realization.
    Marks the initialization (h=Y) and the solution (h=h_sol).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if activation == 'relu': sigma_fn = relu_jax
    elif activation == 'linear': sigma_fn = linear_jax
    elif activation == 'he1he2': sigma_fn = he1he2_jax
    elif activation == 'sigmoid': sigma_fn = sigmoid_jax
    elif activation == 'tanh': sigma_fn = tanh_jax
    elif activation == 'tanh_half': sigma_fn= tanh_half_jax
    elif activation == 'elu': sigma_fn = elu_jax
    elif activation == 'softsign': sigma_fn = softsign_jax
    else: raise NotImplementedError(f"Activation {activation} not supported")

    # Define grid centered around Y and 0
    center = 0.0
    span = max(15.0, abs(Y_val) * 2.0, abs(h_sol) * 2.0)
    h_grid = np.linspace(center - span, center + span, 1000)
    
    # Prepare JAX function for the grid
    # We fix Y, q, V, sigma and vmap over h
    obj_fn_fixed = partial(objective_elementwise, Y=Y_val, q=q, inv_V=1.0/V, sigma=sigma_fn)
    vmapped_obj = jax.vmap(obj_fn_fixed)
    
    # Compute curve
    F_grid = vmapped_obj(jnp.array(h_grid))
    
    # Compute specific points
    F_start = obj_fn_fixed(init) # objective at inital h
    F_sol   = obj_fn_fixed(h_sol) # objective at final h
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(h_grid, F_grid, label='Objective F(h)', color='blue')
    
    # Initialization (Black Dot)
    plt.scatter([init], [F_start], color='black', s=50, zorder=5, label='Initial h')
    
    # Solution (Red Star)
    plt.scatter([h_sol], [F_sol], color='red', marker='*', s=150, zorder=6, label='Solution')
    if h_val_2 is not None:
        F_sol_2=obj_fn_fixed(h_val_2)
        plt.scatter([h_val_2], [F_sol_2], color='green', marker='*', s=150, zorder=6, label='Solution 2')
       
    
    plt.title(f"Landscape ({activation}): alpha={alpha:.2f}\nq={q:.2f}, V={V:.2f}\nY={Y_val:.2f}, h_prox={h_sol:.2f}")
    plt.xlabel("h")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save with unique ID (random int to avoid overwrite issues in parallel)
    rnd_id = rng.randint(0, 100000)
    plt.savefig(f"{save_dir}/landscape_{activation}_alpha{alpha:.2f}_{rnd_id}.png")
    plt.close()

#=========================================================================================================================================================================

# --- MAIN MCMC FUNCTION ---
from scipy.stats import norm
ppf_075 = norm.ppf(0.75)


def hat_vars_MCMC(rng, alpha, beta, q_vars, samples, gamma, delta, activation='relu', solver='Newton_Raphson_robust', num_iterations=50, clip_param=10):
    """
    Computes m_hat, q_hat, V_hat using the Clean derivation (Eq 10.8).
    """
    beta_u, beta_v = beta
    m, q, V = q_vars
    
    # Precomputations
    Sigma_Y = q - np.sum(m**2)
    if Sigma_Y < 1e-10: Sigma_Y = 1e-10

    Sigma_Y_sqrt = np.sqrt(Sigma_Y)
    Sigma_Y_inv_sqrt = 1.0 / Sigma_Y_sqrt

    # Random samples
    h_star = rng.standard_normal((samples, 2))
    xi = rng.standard_normal((samples, 1))
    
    # Latents (case gamma=0, delta=1)
    lam_vals, nu_vals=None, None
    if gamma==0 and delta==1:
        lam_vals = rng.standard_normal(samples)
        nu_vals = np.where(np.abs(lam_vals) > ppf_075, 1, -1)
    else:
        lam_vals, nu_vals = generate_latents(rng, n=samples, gamma=gamma, delta=delta)

    # F functions
    beta_tilde_v = beta_tilde(beta_v)
    F1 = F_1(beta_tilde_v) 
    F2 = F_2_batch(beta=beta, lambda_vals=lam_vals, nu_vals=nu_vals, Q=np.eye(2), beta_tilde_v=beta_tilde_v)

    # Effective Y
    F_term = h_star @ F1.T + F2 
    A = F_term + h_star 
    mu_Y = A @ m.T
    
    # Y shape is (samples, 1). 
    Y = mu_Y + Sigma_Y_sqrt * xi

    # Finding the proximal
    h_init=None
    if activation == 'relu':
        h_prox= find_proximal_relu_analytic(Y, q, V)  # (samples, 1)
    elif activation == 'linear':
        h_prox= find_proximal_linear_analytic(Y, q, V)  # (samples, 1)
    else:
        # Compiled Solver
        solve_prox = get_cached_solver(activation, solver=solver, num_iterations=num_iterations, clip_param=clip_param)
        # Proximal Step (JAX)
        # Convert (N, 1) -> (N,) so vmap sees scalars
        Y_flat = Y.flatten()
        # Convert to JAX arrays
        Y_jax = jnp.array(Y_flat)
        # Initialization:
        #h_init = jnp.array(Y_flat)                                 # init at Y
        #h_init = jnp.zeros_like(Y_flat)                            # init at 0
        h_init = jnp.array(rng.standard_normal(len(Y_flat))*0.01)       # init at 0 + noise
        
        # Run Solver
        # q and V^-1 passed as scalars
        h_prox_jax = solve_prox(h_init, Y_jax, q, 1.0/V)

        # Convert back to NumPy and reshape to (N, 1)
        h_prox = np.array(h_prox_jax).reshape(samples, 1)

        # Second solver if need to compare:
        '''
        solve_prox_2 = get_cached_solver(activation, solver='grid_search_refined', num_iterations=25, clip_param=25)
        h_prox_jax_2=solve_prox_2(h_init, Y_jax, q, 1.0/V)
        h_prox_2 = np.array(h_prox_jax_2).reshape(samples, 1)
        max_diff_prox=np.max(np.abs(h_prox-h_prox_2))
        argmax=np.argmax(np.abs(h_prox-h_prox_2))
        num_diff=np.count_nonzero(np.abs(h_prox-h_prox_2)>1e-3)
        print('Num diff prox : ', num_diff, ' out of ', len(h_prox))
        if max_diff_prox>1e-3 and rng.random()<1/100:
            print('Max diff prox : ', max_diff_prox)
            # Extract values for this sample
            Y_val = float(Y[argmax, 0])
            h_val = float(h_prox[argmax, 0])
            h_val_2 = float(h_prox_2[argmax, 0])
            h_init_single=float(np.array(h_init)[argmax])
            try:
                plot_single_landscape(
                    rng, Y_val, h_val, q, V, 
                    activation, alpha, init=h_init_single, h_val_2=h_val_2)
            except Exception as e:
                print(f"Plotting failed: {e}")
        '''

    # Visualization of the proximal landscape
    '''
    # ---------------- VISUALIZATION BLOCK ----------------
    # Plot with given probability
    if rng.random() < 1/1000:
        # Pick one random sample index
        idx = rng.randint(0, samples)
        # Extract values for this sample
        Y_val = float(Y[idx, 0])
        h_val = float(h_prox[idx, 0])
        h_init_single=float(np.array(h_init)[idx])
        try:
            plot_single_landscape(
                rng, Y_val, h_val, q, V, 
                activation, alpha, init=h_init_single)
        except Exception as e:
            print(f"Plotting failed: {e}")
    # -----------------------------------------------------
    '''

    num_clipped = np.sum((h_prox < -clip_param+1e-4) | (h_prox > clip_param-1e-4))
    h_prox = np.clip(h_prox, -clip_param, clip_param)
    if num_clipped > 0 and rng.random()<1e-3:
        print(f"Warning: {num_clipped} proximal values were clipped for stability out of {h_prox.shape[0]} samples.")

    # Derivatives (in NumPy)
    if activation == 'relu':
        sigma_g = np.maximum(h_prox, 0)
    elif activation == 'linear':
        sigma_g = h_prox
    elif activation == 'he1he2':
        sigma_g = h_prox**2 + h_prox - 1.0
    elif activation == 'sigmoid':
        sigma_g = 1.0 / (1.0 + np.exp(-h_prox))
    elif activation == 'tanh':
        sigma_g = np.tanh(h_prox)
    elif activation == 'tanh_half':
        sigma_g = np.tanh(h_prox/2)
    elif activation == 'elu':
        sigma_g = np.where(h_prox > 0, h_prox, np.exp(h_prox) - 1.0)
    elif activation == 'softsign':
        sigma_g=h_prox/(1+np.abs(h_prox))
        
    d_loss_dq = 0.5 * sigma_g**2

    # Compute Order Parameters
    delta_ = (h_prox - Y) / V
    
    q_hat = alpha * np.mean(delta_**2)
    
    loss_part_V = np.mean(d_loss_dq)
    second_part_V = (np.mean(h_prox * xi) * Sigma_Y_inv_sqrt) - 1.0
    V_hat = (2 * alpha * loss_part_V) - (alpha / V) * second_part_V
    
    first_term_m = np.mean(delta_ * A, axis=0).reshape(1, 2)
    second_term_m = np.mean(delta_ * xi) * Sigma_Y_inv_sqrt * m
    m_hat = alpha * (first_term_m - second_term_m)

    # Optionally plot the MSE interaction term:
    '''
    if rng.random()<1e-2:
        print('MSE interaction term : ', np.mean( -h_prox * sigma_g + 0.5 * q * (sigma_g**2) ))
    '''
    return m_hat, q_hat, V_hat


# ============================================================= MSE ============================================================

def energetic_potential_MCMC(rng, alpha, beta, q_vars, samples, gamma, delta, activation='relu', solver='grid_search_refined', num_iterations=50, clip_param=50):
    """
    Computes Psi_y = - alpha * E[ Moreau_Envelope ] using the Clean derivation.
    """
    beta_u, beta_v = beta
    m, q, V = q_vars
    
    # Dimensions and type
    m = np.array(m).reshape(1, 2)
    q = float(q)
    V = float(V)

    # Sigma_Y (Clean)
    # Sigma_Y = q - m @ m.T (assuming q* = I) if vector order parameters
    Sigma_Y = q - np.sum(m**2)
    if Sigma_Y < 1e-10: Sigma_Y = 1e-10
    Sigma_Y_sqrt = np.sqrt(Sigma_Y)

    # Generate Independent Random Variables
    # h* ~ N(0, I)
    h_star = rng.standard_normal((samples, 2))
    # xi_c ~ N(0, 1)
    xi_c = rng.standard_normal((samples, 1))

    # Latents (Simplified case gamma=0, delta=1 logic or general)
    lam_vals, nu_vals=None, None
    if gamma==0 and delta==1:
        lam_vals = rng.standard_normal(samples)
        nu_vals = np.where(np.abs(lam_vals) > ppf_075, 1, -1)
    else:
        lam_vals, nu_vals = generate_latents(rng, n=samples, gamma=gamma, delta=delta)

    beta_tilde_v = beta_tilde(beta_v)
    F1 = F_1(beta_tilde_v) 
    F2 = F_2_batch(beta=beta, lambda_vals=lam_vals, nu_vals=nu_vals, Q=np.eye(2), beta_tilde_v=beta_tilde_v)

    # Effective Variables
    F_term = h_star @ F1.T + F2 
    
    # Effective Input A (Signal)
    A = F_term + h_star 
    
    # Effective Observation Y (Clean)
    # Y = m A^T + sqrt(Sigma) xi_c
    mu_Y = A @ m.T
    Y = mu_Y + Sigma_Y_sqrt * xi_c
    
    # Proximal Step & Loss Evaluation
    
    if activation == 'relu':
        h_prox = find_proximal_relu_analytic(Y, q, V)
        sigma_g = np.maximum(h_prox, 0)
        loss_val = -h_prox * sigma_g + 0.5 * q * (sigma_g**2)
        
    elif activation == 'linear':
        h_prox = find_proximal_linear_analytic(Y, q, V)
        sigma_g = h_prox
        loss_val = -h_prox * sigma_g + 0.5 * q * (sigma_g**2)
        
    else:
        # JAX Solver
        solve_prox = get_cached_solver(activation, solver=solver, num_iterations=num_iterations, clip_param=clip_param)
        Y_flat = Y.flatten()
        Y_jax = jnp.array(Y_flat)
        # Initialization: change here if needed
        #h_init = jnp.array(Y_flat)                             # init on Y        
        #h_init = jnp.zeros_like(Y_flat)                        # init on 0
        h_init = jnp.array(rng.standard_normal(len(Y_flat))*0.1)    # init on 0 + noise
        
        # Solve 
        h_prox_jax = solve_prox(h_init, Y_jax, q, 1.0/V)
        h_prox = np.array(h_prox_jax).reshape(samples, 1)
        
        # Calculate Loss in NumPy
        if activation == 'he1he2': sigma_g = h_prox**2 + h_prox - 1.0
        elif activation == 'sigmoid': sigma_g = 1.0 / (1.0 + np.exp(-h_prox))
        elif activation == 'tanh': sigma_g = np.tanh(h_prox)
        elif activation == 'tanh_half': sigma_g = np.tanh(h_prox/2)        
        elif activation == 'elu': sigma_g = np.where(h_prox > 0, h_prox, np.exp(h_prox) - 1.0)
        elif activation == 'softsign': sigma_g = h_prox/(1+np.abs(h_prox))
            
        loss_val = -h_prox * sigma_g + 0.5 * q * (sigma_g**2)

    # Moreau_envelope = Loss(h) + 1/2V * (h - Y)^2
    penalty = (0.5 / V) * ((h_prox - Y)**2)
    Moreau_envelope = loss_val + penalty
    
    # Sign of envelope and alpha:
    psi_y = - alpha * np.mean(Moreau_envelope)
    
    return psi_y

def training_loss_from_order_params(rng, alpha, 
    q_vars, q_hat, m_hat, V_hat,
    beta: tuple[float, float],
    lam: float,
    gamma: float,
    delta: float,
    samples: int = 200_000,
    activation='relu',
    solver='grid_search_refined',
    num_iterations=50,
    clip_param=50
) -> float:
    """
    Returns training loss = - free_entropy.
    """
    m, q, V = q_vars
    m = np.array(m).reshape(1, 2)
    m_hat = np.array(m_hat).reshape(1, 2)

    # --- Trace Term (Psi_t) ---
    # Psi_t = - m.m_hat - 1/2 V*q_hat + 1/2 q*V_hat
    trace = - np.sum(m * m_hat) - 0.5 * V * q_hat + 0.5 * V_hat * q

    # --- Entropic Term (Psi_w) ---
    # Ridge: 1/2 * Tr( (lam I + V_hat)^-1 * (q_hat + m_hat m_hat^T) )
    # Only case p=1, so matrices are scalars or (1,2) vectors.
    denom = lam + V_hat
    num = q_hat + np.sum(m_hat**2)
    entropic = 0.5 * (num / denom)

    # --- Energetic Term (Psi_y) ---
    energetic = energetic_potential_MCMC(rng, alpha, beta, q_vars, samples, gamma, delta, activation, solver, num_iterations, clip_param)

    # --- Total Free Entropy ---
    free_entropy = trace + entropic + energetic

    # --- Training Loss ---
    # - sign to have free energy, and rescaling by alpha
    return -free_entropy/alpha

def test_loss_from_order_params(
    rng,
    q_vars,
    beta: tuple[float, float],
    gamma: float,
    delta: float,
    activation: str = 'relu',
    samples: int = 100_000
) -> float:
    """
    Computes the Test Loss (Generalization Error) 
    Test Loss ~ E[ -g * sigma(g) + 0.5 * q * sigma(g)^2 ]
    """
    beta_u, beta_v = beta
    m, q, V = q_vars
    
    # 1. Dimensions
    m = np.array(m).reshape(1, 2)
    q = float(q)
    
    if q <= 0:
        raise ValueError(f"q must be > 0, got q={q}")

    # Compute "Clean" Variance
    # We construct h from h* using the correlation m.
    # h = (m . h*) + noise
    # Var(h) = q. Var(m.h*) = ||m||^2 (since Var(h*) = I).
    # Var(noise) = q - ||m||^2
    m_sq_sum = np.sum(m**2)
    Sigma_h = q - m_sq_sum
    
    # Numerical stability
    if Sigma_h < 0: Sigma_h = 0.0
    Sigma_h_sqrt = np.sqrt(Sigma_h)

    # Generate Independent Random Variables
    # h* ~ N(0, I_2) (Latents)
    h_star = rng.standard_normal((samples, 2))
    # xi ~ N(0, 1) (Independent noise part of student pre-activation)
    xi = rng.standard_normal((samples, 1))

    # Latent Generation for F functions
    lam_vals, nu_vals=None, None
    if gamma==0 and delta==1:
        lam_vals = rng.standard_normal(samples)
        nu_vals = np.where(np.abs(lam_vals) > ppf_075, 1, -1)
    else:
        lam_vals, nu_vals = generate_latents(rng, n=samples, gamma=gamma, delta=delta)    
    
    # Construct F Terms
    beta_tilde_v = beta_tilde(beta_v)
    F1 = F_1(beta_tilde_v) 
    F2 = F_2_batch(beta=beta, lambda_vals=lam_vals, nu_vals=nu_vals, Q=np.eye(2), beta_tilde_v=beta_tilde_v)

    # Construct Effective Field g
    # g = h + S
    # where h is the student pre-activation on the noise and S is the shift due to the structure S = m @ (F1 h* + F2)^T
    
    # Construct h (Student field correlated with h*)
    # h = m @ h*^T + sqrt(Sigma) * xi
    # Shapes: m(1,2), h_star(N,2) -> (N,1)
    h_aligned = h_star @ m.T 
    h = h_aligned + Sigma_h_sqrt * xi
    
    # Construct S (Shift)
    # F_term = F1 h* + F2. (N, 2)
    F_term = h_star @ F1.T + F2 
    S = F_term @ m.T
    
    # Total Field
    g = h + S

    # Compute Activation
    # We apply the activation function element-wise. 
    
    if activation == 'relu':
        act = np.maximum(0, g)
    elif activation == 'linear':
        act = g
    elif activation == 'he1he2':
        act = g**2 + g - 1.0
    elif activation == 'sigmoid':
        act = 1.0 / (1.0 + np.exp(-g))
    elif activation == 'tanh':
        act = np.tanh(g)
    elif activation == 'tanh_half':
        act= np.tanh(g/2)
    elif activation == 'elu':
        act = np.where(g > 0, g, np.exp(g) - 1.0)
    elif activation == 'softsign':
        act = g/(1+np.abs(g))
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Compute Loss
    test_loss = -np.mean(g * act) + (q / 2.0) * np.mean(act**2)

    return test_loss

#============================================================= RUN WITH MPI ============================================================

def run_one_alpha(
    alpha,
    beta,
    gamma,
    delta,
    init,
    samples,
    samples_loss,
    iters,
    activation='relu',
    lam=0.0,
    damping=0.7,
    print_every=50,
    annealing_lambda=False,
    lambda_start=1,
    annealing_rate_lambda=100,
    steps_without_annealing_lambda=500,
    annealing_alpha=False,
    alpha_start=5,
    annealing_rate_alpha=100,
    steps_without_annealing_alpha=500,
    solver='grid_search_refined',
    num_steps_solver=50,
    clip_param=100,
    seed=0
):
    """
    Run state evolution with MPI across ranks.
    """
    assert 0.0 <= gamma <= 1.0 and 0.0 <= delta <= 1.0 and gamma + delta <= 1.0, \
        "Require 0 ≤ γ, δ and γ+δ ≤ 1"


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # To have different seeds for each rank, add the rank to the given seed.
    seed=seed+rank
    rng=np.random.default_rng(seed)
    m0, q0, V0 = init
    state = (m0.copy(), float(q0), float(V0))

    state_list = []
    state_hat_list = []

    # ANNEALING PARAMETERS for regularization lambda and sample complexity alpha
    lam_start = lambda_start
    lam_target = lam
    tau_anneal_lambda = annealing_rate_lambda
    current_lam=lam_start if annealing_lambda else lam_target

    alpha_start=alpha_start
    alpha_target=alpha
    tau_anneal_alpha=annealing_rate_alpha
    current_alpha=alpha_start if annealing_alpha else alpha_target

    for i in range(iters):
        if annealing_alpha:
            if i < iters-steps_without_annealing_alpha: # No annealing for the last steps, goal alpha reached
                current_alpha= alpha_target + (alpha_start - alpha_target) * np.exp(-i / tau_anneal_alpha)
            else:
                current_alpha = alpha_target

        m_hat_local, q_hat_local, V_hat_local = hat_vars_MCMC(rng,
            alpha=current_alpha, beta=beta, q_vars=state, samples=samples, 
            gamma=gamma, delta=delta, activation=activation, solver=solver, num_iterations=num_steps_solver, clip_param=clip_param
        )

        # GATHERING HATTED VARIABLES
        # comm.allreduce aggregated all the hatted variables
        
        m_hat_sum = comm.allreduce(m_hat_local, op=MPI.SUM)
        q_hat_sum = comm.allreduce(q_hat_local, op=MPI.SUM)
        V_hat_sum = comm.allreduce(V_hat_local, op=MPI.SUM)

        # Above the variables are summed, we take the average.
        m_hat = m_hat_sum / size
        q_hat = q_hat_sum / size
        V_hat = V_hat_sum / size
        state_hat = (m_hat, q_hat, V_hat)

        # UPDATE NON HATTED ORDER PARAMETERS
        # Every rank updates the state locally. This is not costly, so it is not problematic to do this same operations for all ranks.
        (m, q, V) = state
        
        if annealing_lambda:
            if i < iters-steps_without_annealing_lambda: # No annealing for the last steps, goal lambda reached
                current_lam = lam_target + (lam_start - lam_target) * np.exp(-i / tau_anneal_lambda)
            else:
                current_lam = lam_target
        
        (m_new, q_new, V_new) = state_func(state_hat=state_hat, lam=current_lam)

        # UPDATE WITH DAMPING
        m = damping * m + (1 - damping) * m_new
        q = damping * q + (1 - damping) * q_new
        V = damping * V + (1 - damping) * V_new
        state = (m, q, V)

        # Store evolution
        state_list.append(state)
        state_hat_list.append(state_hat)

        # LOGGING (Root only)
        if rank == 0:
            if (i % print_every) == 0:
                print(f"===  ITER {i}, alpha={current_alpha}, lambda={current_lam} ===")
                print(f"m, q, V =\n{state}")
                print(f"m_hat, q_hat, V_hat =\n{state_hat}")


    # Determine window for averaging (last 200 or length of history if shorter)
    window = 200
    history_len = len(state_list)
    start_idx = max(0, history_len - window)

    # Extract and Average Order Parameters (Temporal Mean/Std)
    
    # Unpack history into arrays for vectorized operations
    # state_list is a list of tuples (m, q, V)
    m_hist = np.array([s[0] for s in state_list[start_idx:]])
    q_hist = np.array([s[1] for s in state_list[start_idx:]])
    V_hist = np.array([s[2] for s in state_list[start_idx:]])
    
    # Same for hat variables
    m_hat_hist = np.array([s[0] for s in state_hat_list[start_idx:]])
    q_hat_hist = np.array([s[1] for s in state_hat_list[start_idx:]])
    V_hat_hist = np.array([s[2] for s in state_hat_list[start_idx:]])

    # Compute Means over the consider window
    m_mean = np.mean(m_hist, axis=0)
    q_mean = np.mean(q_hist, axis=0)
    V_mean = np.mean(V_hist, axis=0)
    
    m_hat_mean = np.mean(m_hat_hist, axis=0)
    q_hat_mean = np.mean(q_hat_hist, axis=0)
    V_hat_mean = np.mean(V_hat_hist, axis=0)

    # Compute standard deviation (Stability Metric - how much it fluctuated in the window)
    m_std_temporal = np.std(m_hist, axis=0)
    q_std_temporal = np.std(q_hist, axis=0)
    V_std_temporal = np.std(V_hist, axis=0)
    m_hat_std_temporal = np.std(m_hat_hist, axis=0)
    q_hat_std_temporal = np.std(q_hat_hist, axis=0)
    V_hat_std_temporal = np.std(V_hat_hist, axis=0)

    # Pack averaged states for the loss function
    state_avg = (m_mean, q_mean, V_mean)
    
    # Compute Local Loss using Averaged Parameters
    local_train_loss = training_loss_from_order_params(
        rng=rng,
        alpha=alpha,
        q_vars=state_avg,       # Use the averaged q, V, m
        q_hat=q_hat_mean,       # Use the averaged hat variables
        m_hat=m_hat_mean,
        V_hat=V_hat_mean,
        beta=beta,
        lam=lam,
        gamma=gamma,
        delta=delta,
        samples=samples_loss,     
        activation=activation,
        solver=solver,
        num_iterations=num_steps_solver,
        clip_param=clip_param
    )

    local_test_loss= test_loss_from_order_params(
        rng=rng,
        q_vars=state_avg,
        beta=beta,
        gamma=gamma,
        delta=delta,
        activation=activation,
        samples=samples_loss
    )

    # MPI Aggregation: Mean and Std of Loss across Ranks
    # We need E[L] and E[L^2] to compute Std[L] = sqrt(E[L^2] - E[L]^2)
    local_metrics = np.array([local_train_loss, local_train_loss**2, local_test_loss, local_test_loss**2], dtype=np.float64)
    global_metrics_sum = comm.allreduce(local_metrics, op=MPI.SUM)

    # Calculate global statistics
    loss_mean = global_metrics_sum[0] / size
    loss_sq_mean = global_metrics_sum[1] / size
    test_loss_mean = global_metrics_sum[2] / size
    test_loss_sq_mean = global_metrics_sum[3] / size
    
    # Variance = E[X^2] - (E[X])^2. Max(0, ...) protects against tiny negative floats due to precision
    loss_var = max(0.0, loss_sq_mean - loss_mean**2)
    loss_std_spatial = np.sqrt(loss_var)
    test_loss_var = max(0.0, test_loss_sq_mean - test_loss_mean**2)
    test_loss_std_spatial = np.sqrt(test_loss_var)
    
    comm.Barrier()

    if rank == 0:
        print('Order Parameters Mean:\nm =', m_mean, '+-', m_std_temporal, 'q =', q_mean, '+-', q_std_temporal, 'V =', V_mean, '+-', V_std_temporal, '\nhatm =', m_hat_mean, '+-', m_hat_std_temporal, 'hatq =', q_hat_mean, '+-', q_hat_std_temporal, 'hatV =', V_hat_mean, '+-', V_hat_std_temporal)
        return state_list, state_hat_list, np.array([loss_mean, loss_std_spatial, test_loss_mean, test_loss_std_spatial]) # Last array contains mean train MSE over ranks using the mean order parameters, then the std over the ranks, then the test mse, etc.

    # non-zero rank return nothing
    return None, None, None