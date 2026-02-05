import numpy as np
from lambda_nu_integrals import int_1, int_2, int_3, int_4, int_5, corr_int

def beta_tilde(beta_v):
    return beta_v/(1+beta_v+np.sqrt(1+beta_v))

def c_2(beta_tilde_v, Q_22):
    return 1 - beta_tilde_v*Q_22

def c_1(c2, beta_tilde_v):
    return beta_tilde_v*(1+c2)/c2**2

def gamma_matrix(c1):
    mat = np.zeros((2,2))
    mat[1,1] = c1
    return mat

def compute_M_N_from_V_c1(V, c1, eps=1e-10, symmetrize=True):
    """
    Inputs
      V  : (n,2,2)  symmetric positive-definite matrices
      c1 : (1,)     c1 used in Gamma

    Returns
      M      : (n,2,2)  = (V^{-1} + Gamma)^{-1}
      N      : (n,2,2)  = V^{-1} @ M
      V_inv  : (n,2,2)  (returned in case you need it elsewhere)
      G      : (2,2)  Gamma matrix
    """
    # ensure symmetry + tiny jitter for stability
    if symmetrize:
        V = 0.5 * (V + np.swapaxes(V, -1, -2))
    V = V + eps * np.eye(2)[None, :, :]

    # build Gamma
    G = gamma_matrix(c1)  # (2,2)

    # invert V in batch (NumPy supports stacked inv)
    V_inv = np.linalg.inv(V)      # (n,2,2)
    M = np.linalg.inv(V_inv + G)

    # N = V^{-1} M
    N = bmm(V_inv, M)             # (n,2,2)

    return M, N, V_inv, G

# -------- batch helpers --------
def bmv(M, v):
    """Batch matrix–vector: (n,2,2) @ (n,2) -> (n,2)"""
    return np.einsum('nij,nj->ni', M, v)

def bmm(A, B):
    """Batch matrix–matrix: (n,2,2) @ (n,2,2) -> (n,2,2)"""
    return np.einsum('nij,njk->nik', A, B)

def bouter(v):
    """Batch outer: (n,2) -> (n,2,2)"""
    return np.einsum('ni,nj->nij', v, v)

# New helpers for constant right/left multiplies (Gamma global)
def bmm_const_right(A, B):
    """(n,2,2) @ (2,2) -> (n,2,2)"""
    return np.einsum('nij,jk->nik', A, B)

def bmm_const_left(B, A):
    """(2,2) @ (n,2,2) -> (n,2,2)"""
    return np.einsum('ij,njk->nik', B, A)

    
def f_out(beta_u, beta_v, gamma, delta, gamma_mat, c1, c2, y, omega, Q, N, a, b, c, d):
    """
    Shapes:
      y: (n,) with entries in {-1, 1}
      omega: (n,2)
      Q, N: (n,2,2)
      gamma_mat: (2,2)
      beta_u, beta_v, c1, c2: (1,)
      a,b,c,d: (n,)
    Returns:
      f_out: (n,2)
    """

    n = y.shape[0]
    out = np.zeros((n, 2))

    # compute only for y == 1
    mask = (y == 1)
    if not np.any(mask):
        return out

    # select active samples
    a_, b_, c_, d_ = a[mask], b[mask], c[mask], d[mask]
    N_, omega_= N[mask], omega[mask]

    int1 = int_1(a_, b_, c_, d_, gamma, delta)
    int2 = int_2(a_, b_, c_, d_, gamma, delta)
    int3 = int_3(a_, b_, c_, d_, gamma, delta)

    num1 = np.sqrt(beta_u) * int2
    num2 = (np.sqrt(beta_v)/c2) * int3 + np.sqrt(beta_u)*c1*Q[0,1]*int2
    numer = np.stack([num1, num2], axis=-1)

    fout_pos = bmv(N_, numer) / int1[:, None] - bmv(N_, omega_ @ gamma_mat.T)
    out[mask] = fout_pos
    return out
    
def df_out(fout, beta_u, beta_v, gamma, delta, gamma_mat, c1, c2, y, omega, Q, N, a, b, c, d):
    n = y.shape[0]
    dfout = np.zeros((n, 2, 2))
    mask = (y == 1)
    if not np.any(mask):
        return dfout

    # select active samples
    a_, b_, c_, d_ = a[mask], b[mask], c[mask], d[mask]
    N_, omega_, fout_ =  N[mask], omega[mask], fout[mask]

    # integrals
    int1 = int_1(a_, b_, c_, d_, gamma, delta)
    int4 = int_4(a_, b_, c_, d_, gamma, delta)
    int5 = int_5(a_, b_, c_, d_, gamma, delta)

    # construct B
    B = np.zeros((np.sum(mask), 2, 2))
    B[:,0,0] = beta_u * int5
    B[:,0,1] = np.sqrt(beta_u*beta_v)/c2 * int4 + beta_u*c1*Q[0,1]*int5
    B[:,1,0] = B[:,0,1]
    B[:,1,1] = (beta_v/c2**2)*int1 + beta_u*(c1**2)*(Q[0,1]**2)*int5 + 2*np.sqrt(beta_u*beta_v)*c1/c2*Q[0,1]*int4

    vec = fout_ + bmv(N_, omega_ @ gamma_mat.T)
    term1 = bmm(bmm(N_, B), np.swapaxes(N_, -1, -2)) / int1[:, None, None]
    term2 = bmm_const_right(N_, gamma_mat)
    term3 = bouter(vec)

    dfout[mask] = term1 - term2 - term3
    return dfout
    
def f_Q(beta_u, beta_v, gamma, delta, beta_tilde_v, c1, c2, y,
                omega, Q, M, V, a, b, c, d, fout, dfout):
    n = y.shape[0]
    fQ = np.zeros((n, 2, 2))
    mask = (y == 1)
    if not np.any(mask):
        return fQ

    # select active samples
    a_, b_, c_, d_ = a[mask], b[mask], c[mask], d[mask]
    M_, V_, omega_, fout_, dfout_ = M[mask], V[mask], omega[mask], fout[mask], dfout[mask]

    int1 = int_1(a_, b_, c_, d_, gamma, delta)
    int2 = int_2(a_, b_, c_, d_, gamma, delta)
    int3 = int_3(a_, b_, c_, d_, gamma, delta)
    int4 = int_4(a_, b_, c_, d_, gamma, delta)
    int5 = int_5(a_, b_, c_, d_, gamma, delta)

    Q01 = Q[0,1]
    M01, M11 =  M_[:,0,1], M_[:,1,1]
    M01p = M01 + c1*Q01*M11
    w2 = omega_[:,1]

    fQ_pos = np.zeros((np.sum(mask), 2, 2))

    # fQ[0,0]
    fQ_pos[:,0,0] = -0.5 * beta_u * (int5 / int1)

    # fQ[0,1] = fQ[1,0]
    t_left  = np.sqrt(beta_u*beta_v)/c2 * int4 + beta_u*c1*Q01*int5
    t_right = np.sqrt(beta_u)*c1 * (
        np.sqrt(beta_u)*M01p*int5 + np.sqrt(beta_v)/c2*M11*int4 + (1 - M11*c1)*w2*int2
    )
    f01 = -(t_left - t_right)/int1
    fQ_pos[:,0,1] = f01/2 #changed here the 1/2 factor
    fQ_pos[:,1,0] = f01/2 #changed here the 1/2 factor

    # fQ[1,1]
    A = -0.5*beta_v + beta_tilde_v/c2
    C = beta_tilde_v*np.sqrt(beta_v)/c2**2 * (
        np.sqrt(beta_u)*M01p*int4 + np.sqrt(beta_v)/c2*M11*int1 + (1 - M11*c1)*w2*int3
    )
    D = np.sqrt(beta_u)*beta_tilde_v**2/c2**2*(1+2/c2)*Q01 * (
        np.sqrt(beta_u)*M01p*int5 + np.sqrt(beta_v)/c2*M11*int4 + w2*(1 - M11*c1)*int2
    )
    E = -beta_tilde_v/c2**2 * np.sqrt(beta_u*beta_v)*Q01*int4
    F = -0.5*beta_u*beta_tilde_v**2/c2**2*(1+2/c2)*Q01**2*int5

    Vf = bmv(V_, fout_)
    z = omega_ + Vf
    covZ = bouter(z) + bmm(bmm(V_, dfout_), np.swapaxes(V_, -1, -2)) + V_
    B = -0.5*beta_tilde_v**2/c2**2*(1+2/c2)*covZ[:,1,1]

    fQ_pos[:,1,1] = A + B + (C + D + E + F)/int1

    fQ[mask] = fQ_pos
    return fQ

def G_out(beta_u, beta_v, gamma, beta_tilde_v):
    """
    linearized, returns Gout of shape (2, 2)
    """
    correlation = corr_int(gamma)
    off_diagonal_term = np.sqrt(beta_u*beta_v/(1+beta_v))*correlation
    return np.array([[0, 0], [0, beta_tilde_v*(beta_tilde_v - 2)]]) + np.array([[beta_u, off_diagonal_term],[off_diagonal_term, beta_v/(1+beta_v)]])


def integral_parameters(beta_u, beta_v, omega, Q, N, M, c1, c2):
    """
    Inputs:
      beta_u, beta_v, c1, c2: (1,)
      omega: (n,2)
      Q: (2,2)
      N, M: (n,2,2)
    Returns:
      a,b,c,d: each (n,)
    """
    n = omega.shape[0]

    # unpack useful entries per-sample
    Q01 = Q[0, 1]
    Q00 = Q[0, 0]
    M00 = M[:, 0, 0]
    M01 = M[:, 0, 1]
    M11 = M[:, 1, 1]
    N00 = N[:, 0, 0]
    N01 = N[:, 0, 1]
    N10 = N[:, 1, 0]
    N11 = N[:, 1, 1]
    w1 = omega[:, 0]
    w2 = omega[:, 1]

    a = beta_u * (Q00 + c1*Q01**2 - M00 - 2*c1*Q01*M01 - c1**2 * Q01**2 * M11)
    b = np.sqrt(beta_u) * (w1*N00 + w2*N10 + c1*Q01*(w1*N01 + w2*N11))
    c = (np.sqrt(beta_v)/c2) * (w1*N01 + w2*N11)
    d = (np.sqrt(beta_u*beta_v)/c2) * (-Q01 + M01 + c1*Q01*M11)
    return a, b, c, d
