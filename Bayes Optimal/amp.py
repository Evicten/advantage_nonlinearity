import numpy as np
from amp_denoising_functions_vectorized import integral_parameters, f_out, df_out, f_Q, bmv, bmm, bouter, compute_M_N_from_V_c1, beta_tilde, c_1, c_2
from mpi4py import MPI
from state_evolution import generate_latents

def update_mean(X, V, w, fout):
    n , d = X.shape
    omega = X @ w / np.sqrt(d) - bmv(V, fout)

    return omega

def update_covariance(X2, C_hat, assume_concentration=False):
    n, d = X2.shape
    if assume_concentration:
        V = 1/d * np.sum(C_hat, axis=0)
        V = np.tile(V, (n, 1, 1))
    else:
        V = 1/d * np.einsum('nd, dkj -> nkj', X2, C_hat)
    
    return V

def update_Q(w, C_hat):
    d, s = w.shape
    return 1/d*(np.sum(C_hat + bouter(w), axis = 0))

def update_b(X, X2, fout, dfout, w, assume_concentration=False): 
    n, d = X.shape
    if assume_concentration:
        interm = 1/d * np.sum(dfout, axis=0)
        interm = np.tile(interm, (d, 1, 1))
    else:
        interm = 1/d * np.einsum('nd,njk->djk', X2, dfout)

    return 1/np.sqrt(d) * X.T @ fout - bmv(interm, w)

def update_A(X2, dfout, fQ, assume_concentration=False): 
    n, d = X2.shape
    n, s, _ = dfout.shape
    if assume_concentration:
        interm = 1/d * np.sum(dfout, axis=0)
        interm = np.tile(interm, (d, 1, 1))
    else:
        interm = 1/d * np.einsum('nd,nkj->dkj', X2, dfout)

    Q_term = 2/d*np.sum(fQ, axis=0) #(s,s) shape

    return - interm - Q_term # broadcasts to (d,s,s)

def update_w(A, b): 
    d, k, k = A.shape
    mat = np.linalg.inv(np.eye(k) + A)
    return bmv(mat, b)

def update_C_hat(A): 
    d, k, k = A.shape
    C_hat = np.linalg.inv(np.eye(k) + A)
    
    return C_hat

def channel(beta_u, beta_v, gamma, delta, y, omega, V, Q, eps=1e-6): #vectorized version

    bt = beta_tilde(beta_v)
    c2 = c_2(bt, Q[1,1])
    c1 = c_1(c2, bt)
    M, N, V_inv, gamma_mat = compute_M_N_from_V_c1(V, c1, eps)
    a, b, c, d = integral_parameters(beta_u, beta_v, omega, Q, N, M, c1, c2)
    
    fout = f_out(beta_u, beta_v, gamma, delta, gamma_mat, c1, c2, y, omega, Q, N, a, b, c, d)
    dfout = df_out(fout, beta_u, beta_v, gamma, delta, gamma_mat, c1, c2, y, omega, Q, N, a, b, c, d)
    fQ = f_Q(beta_u, beta_v, gamma, delta, bt, c1, c2, y, omega, Q, M, V, a, b, c, d, fout, dfout)

    return fout, dfout, fQ

## Helper functions 

def check_convergence(old, new, d, tol=1e-6):
    diff = np.linalg.norm(new - old) / np.sqrt(d)
    return diff < tol

def damping(x_new, x_old, coeff=0.2):
    if coeff > 1:
        print('Coefficient must be between 0 and 1. Returning new value.')
        return x_new
    else:
        return (1-coeff) * x_new + coeff * x_old

def pad_histories(histories, pad_to=None, pad_value=np.nan):
    """
    histories: list of arrays shaped (T_r, 2, 2), variable T_r
    pad_to: if None, uses max(T_r); otherwise pads/clips to this length
    pad_value: usually np.nan

    returns:
      padded: (R, T, 2, 2) float array
      lengths: (R,) int array with each original T_r (clipped to T if pad_to given)
    """
    if len(histories) == 0:
        return np.empty((0,0,2,2)), np.zeros((0,), dtype=int)

    lengths = np.array([h.shape[0] for h in histories], dtype=int)
    T_max = int(pad_to) if pad_to is not None else int(lengths.max())
    R = len(histories)

    padded = np.full((R, T_max, 2, 2), pad_value, dtype=float)
    for i, h in enumerate(histories):
        T = min(h.shape[0], T_max)         # clip if longer than pad_to
        padded[i, :T] = h[:T]

    lengths = np.minimum(lengths, T_max)   # keep consistency if clipped
    return padded, lengths

## AMP!


def AMP_algo(X, y, K, beta_u=1, beta_v=2, gamma=1, delta=0, max_iter=50, tol=1e-4, plant=0.9, damp=0.2, w_star = None, eps=1e-6, rank = None, assume_concentration=False):
    n = X.shape[0]
    d = X.shape[1]

    X2 = X**2

    fout = np.zeros((n, K))

    w = np.sqrt(plant)*w_star + np.sqrt(1-plant)*np.random.randn(d, K)
    overlap_hist = []

    overlap = w.T @ w_star / d
    #print("initial overlap")
    #print(overlap)

    overlap_hist.append(overlap)

    C_hat = [np.eye(K) for _ in range(d)]
    C_hat = np.array(C_hat)

    converged = False
    for _ in range(max_iter):
        #update mean, covariance and overlap Qxs
        V = update_covariance(X2, C_hat, assume_concentration=assume_concentration)
        omega = update_mean(X, V, w, fout)
        Q = update_Q(w, C_hat)

        #update f_out, df_out and f_Q
        fout, dfout, fQ = channel(beta_u, beta_v, gamma, delta, y, omega, V, Q, eps)

        #update b and A
        b = update_b(X, X2, fout, dfout, w, assume_concentration=assume_concentration)
        A = update_A(X2, dfout, fQ, assume_concentration=assume_concentration)
        #update w and C_hat
        w_old = w.copy()
        w = update_w(A, b)
        w = damping(w, w_old, coeff=damp)
        C_hat_old = C_hat.copy()
        C_hat = update_C_hat(A)
        C_hat = damping(C_hat, C_hat_old, coeff=damp)

        overlap = w.T @ w_star / d
        overlap_hist.append(overlap)

        def _p(msg):
            if rank is None:
                print(msg)
            else:
                print(f"[Rank {rank}] {msg}")

        if np.isnan(overlap).any():
            _p(f"NaN encountered at iter {_}")
            break

        if _ % 20 == 0:
            _p(f"Iteration {_}")

        if check_convergence(w_old, w, d, tol):
            _p(f"Converged at iter {_}")
            converged = True
            break

    overlap_hist = np.array(overlap_hist)

    return w, overlap_hist, converged


def generate_data(n, d, beta_u, beta_v, gamma=0, delta=1):
    """
    Returns:
      X: (n,d)
      y: (n,)
      u_star, v_star: (d,)
    Notes:
      For y == -1 we overwrite X with standard Gaussian (null class).
      For y == +1 we use the model with lambda, nu latents and the S whitening matrix.
    """
    # Ground-truth directions
    u_star = np.random.randn(d)
    u_star_norm = u_star / np.sqrt(d)
    v_star = np.random.randn(d)
    v_star_norm = v_star / np.sqrt(d)

    # Labels (balanced)
    # y = np.random.choice([-1, 1], size=n, p=[0.5, 0.5])
    y = np.ones(n)

    # Latents for the channel (only used for y==+1 rows)
    lambda_vals, nu_vals = generate_latents(n, gamma=gamma, delta=delta)

    # S from your definition
    beta_tilde_v = beta_v / (1 + beta_v + np.sqrt(1 + beta_v))
    S = np.eye(d) - beta_tilde_v * np.outer(v_star_norm, v_star_norm)

    # Build X
    X = np.sqrt(beta_u) * np.outer(lambda_vals, u_star_norm) \
        + (np.sqrt(beta_v) * np.outer(nu_vals, v_star_norm) + np.random.randn(n, d)) @ S.T

    # Overwrite y == -1 rows with N(0, I)
    mask_neg = (y == -1)
    if np.any(mask_neg):
        X[mask_neg] = np.random.randn(np.sum(mask_neg), d)

    return X, y, u_star, v_star