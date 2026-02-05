import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from lambda_nu_integrals import int_1, int_2, int_3, int_4, int_5, gamma_matrix
from scipy.stats import norm
from mpi4py import MPI
from scipy.linalg import sqrtm


def beta_tilde(beta_v):
    return beta_v/(1+beta_v+np.sqrt(1+beta_v))

def beta_vec(nu, lambda_, beta):
    beta_u, beta_v = beta
    return np.array([np.sqrt(beta_u)*lambda_, np.sqrt(beta_v*(1+beta_v))*nu])

def q_func(q_hat):
    return q_hat @ np.linalg.inv(q_hat + np.eye(2))

def f_out(beta_u, beta_v, gamma, delta, beta_tilde_v, gamma_matrix, y, omega, N, a, d):
    if y == -1:
        return np.zeros((2,))
    elif y == 1:
        omega1, omega2 = omega
        b = np.sqrt(beta_u)*(omega1*N[0,0]+omega2*N[1,0])
        c = np.sqrt(beta_v)/(1-beta_tilde_v)*(omega1*N[0,1]+omega2*N[1,1])
        denominator = int_1(a, b, c, d, gamma, delta)
        numerator_1 = np.sqrt(beta_u)*int_2(a, b, c, d, gamma, delta) 
        numerator_2 = np.sqrt(beta_v*(1+beta_v))*int_3(a, b, c, d, gamma, delta)
        numerator_vec = np.array([numerator_1, numerator_2])
        return N @ numerator_vec/denominator - N @ gamma_matrix @ omega
    
def Z_out(beta_u, beta_v, gamma, delta, beta_tilde_v, y, omega, M, N, a, d):
    if y == -1:
        return 1/2
    elif y == 1:
        omega1, omega2 = omega
        b = np.sqrt(beta_u)*(omega1*N[0,0]+omega2*N[1,0])
        c = np.sqrt(beta_v)/(1-beta_tilde_v)*(omega1*N[0,1]+omega2*N[1,1])
        integral = int_1(a, b, c, d, gamma, delta)
        constant_piece = 1/(1-beta_tilde_v)*np.exp(-1/2*beta_v)*np.sqrt(np.linalg.det(N))*np.exp(-1/2*omega2**2*beta_v*(1-beta_v*M[1,1]))*np.exp(1/2*beta_v*(1+beta_v)*M[1,1])
        return constant_piece/2 * integral



def q_hat_func_MCMC_replica(alpha, beta, q, samples, gamma, delta, eps=1e-6): 
    expectation_xi = np.zeros((2,2))
    beta_u, beta_v = beta
    sqrt_q = sqrtm(q)
    V = np.eye(2) - q
    gamma_mat = gamma_matrix(beta_v)
    V_inv = np.linalg.inv(V+np.eye(2)*eps)
    M = np.linalg.inv(V_inv+gamma_mat+np.eye(2)*eps)
    N = V_inv @ M
    beta_tilde_v = beta_tilde(beta_v)
    a = beta_u*(1-M[0,0])
    d = np.sqrt(beta_u*beta_v)/(1-beta_tilde_v)*M[0,1]

    for _ in range(samples):
        Xi = np.random.normal(0,1, 2)
        omega = sqrt_q @ Xi
        Zout = Z_out(beta_u = beta_u, beta_v= beta_v, gamma= gamma, delta = delta, beta_tilde_v = beta_tilde_v, y = 1, omega = omega, M = M, N = N, a = a, d = d) 
        fout = f_out(beta_u = beta_u, beta_v= beta_v, gamma= gamma, delta = delta, beta_tilde_v = beta_tilde_v, gamma_matrix = gamma_mat, y = 1, omega = omega, N = N, a = a, d = d)
        expectation_xi += Zout*np.outer(fout, fout)

    q_hat = 2*alpha * (expectation_xi / samples) #added the 2 to remove the half noise for discrimination task

    return q_hat

def generate_latents(n, gamma=0, delta=1): 
    lambda_vals = np.random.randn(n) 
    unif = np.random.rand(n) 
    nu_vals = np.zeros(n, dtype=int) 

    mask1 = unif < gamma 
    nu_vals[mask1] = np.sign(lambda_vals[mask1]) 
    nu_vals[mask1][nu_vals[mask1]==0] = 1 #if lambda is exactly zero, set nu to 1 

    mask2 = (unif >= gamma) & (unif < gamma + delta) 
    t = norm.ppf(0.75) 
    nu_vals[mask2] = np.where(np.abs(lambda_vals[mask2]) > t, 1, -1) 

    mask3 = ~(mask1 | mask2) 
    nu_vals[mask3] = np.random.choice([-1, 1], size=np.sum(mask3), p=[1/2, 1/2]) 

    return lambda_vals, nu_vals

def modelF(beta, h_star, lambda_star, nu_star, y, beta_tilde_v):
    if y == 1:
        beta_u, beta_v = beta
        F1 = np.zeros((2,2))
        F1[1,1] = - beta_tilde_v
        F2 = np.zeros((2,))
        F2[0] = np.sqrt(beta_u)*lambda_star
        F2[1] = np.sqrt(beta_v)/np.sqrt(1+beta_v)*nu_star
        return F1 @ h_star + F2
    else:
        return np.zeros((2,))

def q_hat_func_MCMC_rBP(alpha, beta, q, samples, gamma, delta, eps=1e-6): 
    expectation_xi = np.zeros((2,2))
    beta_u, beta_v = beta
    sqrt_q = sqrtm(q)
    V = np.eye(2) - q
    gamma_mat = gamma_matrix(beta_v)
    V_inv = np.linalg.inv(V+np.eye(2)*eps)
    M = np.linalg.inv(V_inv+gamma_mat+np.eye(2)*eps)
    N = V_inv @ M
    beta_tilde_v = beta_tilde(beta_v)
    a = beta_u*(1-M[0,0])
    d = np.sqrt(beta_u*beta_v)/(1-beta_tilde_v)*M[0,1]

    for _ in range(samples):
        Xi = np.random.normal(0,1, 2)
        Xi_star = np.random.normal(0, 1, 2)
        (lambda_star,), (nu_star,) = generate_latents(n=1, gamma=gamma, delta=delta)
        h_star = sqrt_q @ Xi_star

        omega = sqrt_q @ Xi + q @ modelF(beta=beta, h_star = h_star, lambda_star=lambda_star, nu_star=nu_star, y=1, beta_tilde_v=beta_tilde_v)

        fout = f_out(beta_u = beta_u, beta_v= beta_v, gamma= gamma, delta = delta, beta_tilde_v = beta_tilde_v, gamma_matrix = gamma_mat, y = 1, omega = omega, N = N, a = a, d = d)

        expectation_xi += np.outer(fout, fout)

    q_hat = alpha/2 * (expectation_xi / samples) 

    return q_hat


def run_one_alpha(
    alpha,
    beta,
    gamma,
    delta,
    q_init,
    samples,
    iters,
    damping=0.7,
    mode="replica",           # "replica" or "rBP"
    print_every=50
):
    """
    Run state evolution with MPI across ranks.

    mode:
      - "replica" -> uses q_hat_func_MCMC_replica 
      - "rBP"     -> uses q_hat_func_MCMC_rBP     
    """
    assert 0.0 <= gamma <= 1.0 and 0.0 <= delta <= 1.0 and gamma + delta <= 1.0, \
        "Require 0 ≤ γ, δ and γ+δ ≤ 1"

    if mode == "replica":
        qhat_fn = q_hat_func_MCMC_replica
    elif mode == "rBP":
        qhat_fn = q_hat_func_MCMC_rBP
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'replica' or 'rBP'.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    q = q_init.copy()

    if rank == 0:
        q_list = []
        q_hat_list = []

    for i in range(iters):
        # Each rank computes its local q_hat
        q_hat_local = qhat_fn(alpha, beta, q, samples, gamma, delta)

        if rank != 0:
            # send local estimate to root, receive updated q
            comm.send(q_hat_local, dest=0)
            q = comm.recv(source=0)
            if q is None:
                break
            continue

        # --- root aggregates ---
        if (i % print_every) == 0:
            print(f"[mode {mode} | alpha {alpha:.3f} | iter {i}] q =\n{q}")
            print(f"[mode {mode} | alpha {alpha:.3f} | iter {i}] q_hat (local) =\n{q_hat_local}")

        q_hat_all = np.zeros((size, 2, 2), dtype=np.float64)
        q_hat_all[0] = q_hat_local
        for j in range(1, size):
            q_hat_all[j] = comm.recv(source=j)

        q_hat = np.mean(q_hat_all, axis=0)

        # Update q with damping
        q = damping * q_func(q_hat) + (1 - damping) * q

        # store trajectories on root
        q_list.append(q)
        q_hat_list.append(q_hat)

        # broadcast updated q to workers
        for j in range(1, size):
            comm.send(q, dest=j)

    comm.Barrier()

    if rank == 0:
        return np.array(q_list), np.array(q_hat_list)

    return None, None

