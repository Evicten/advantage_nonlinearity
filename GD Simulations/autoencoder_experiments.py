"""
Autoencoder Training on Synthetic Data with Multiple SNR Levels

This script trains single-hidden-layer autoencoders on synthetic data generated
from a two-spike model with higher-order correlations. It compares the learned
representations to PCA/SVD baselines across different sample complexity regimes.

"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# Configuration and Hyperparameters
# =============================================================================

# Random seeds
RNG_SEED_NUMPY = 2
RNG_SEED_TORCH = 1
rng = np.random.default_rng(RNG_SEED_NUMPY)
torch.manual_seed(RNG_SEED_TORCH)

# Data parameters
D = 2000  # Input dimension
BETA_U = 1  # Signal strength for first spike
BETA_V = 2  # Signal strength for second spike
LATENT_DEPENDENCE = 'dependent_ord3'  # Options: 'independent', 'correlated', 'dependent', 'dependent_ord3'

# Test set parameters
N_TEST_TOTAL = 2_000_000
N_TEST_BATCH = 20_000
N_BATCHES = N_TEST_TOTAL // N_TEST_BATCH

# Model hyperparameters
N_HIDDEN = 1  # Number of hidden neurons
NONLINEARITY = 'tanh'  # Options: 'relu', 'elu', 'tanh', etc.
TIED_WEIGHTS = True  # Whether encoder and decoder share weights
USE_BIAS = False

# Training hyperparameters
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0  # L2 regularization strength
EPOCHS = 1200

# Experimental design
ALPHA_ARRAY = np.logspace(-0.4, 1.5, 20)[::-1]  # Sample complexity ratios (n/d)
N_SEEDS = 20  # Number of random seeds per alpha

# Flags
WRITE_DATA = True
PLOT_RESULTS = False

# Auxiliary constants for dependent latent variables
AUX_K2 = norm.ppf(0.75)  # For k=2 dependent case
AUX_K3 = math.sqrt(2 * math.log(2))  # For k=3 dependent case

# Device configuration
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else ('mps' if torch.backends.mps.is_available() else 'cpu')
)
print(f"Device: {device}")


# =============================================================================
# Model Definition
# =============================================================================

class AutoEncoder(nn.Module):
    """
    Single-hidden-layer autoencoder with optional weight tying.
    
    The forward pass computes:
        h = activation(W @ x / sqrt(D) + bias)
        x_hat = W^T @ h / sqrt(D)
    
    Parameters
    ----------
    D : int
        Input dimension
    K : int
        Number of hidden units
    activation : str
        Activation function name
    tied : bool
        If True, decoder uses transposed encoder weights
    init_scale : float
        Scale for weight initialization
    use_bias : bool
        If True, add bias to hidden layer
    seed : int
        Random seed for initialization
    """
    
    def __init__(self, D, K, activation='relu', tied=False, init_scale=1.0, 
                 use_bias=False, seed=0):
        super(AutoEncoder, self).__init__()
        
        torch.manual_seed(seed)
        
        self.D = D
        self.K = K
        self.tied = tied
        self.activation_name = activation
        
        # Define activation function
        self.activation = self._get_activation(activation)
        
        # Initialize weights
        self.W = nn.Parameter(torch.randn(K, D) * init_scale)
        self.V = nn.Parameter(torch.randn(D, K) * init_scale) if not tied else None
        self.bias = nn.Parameter(torch.zeros(K)) if use_bias else None
    
    def _get_activation(self, name):
        """Map activation name to function."""
        activations = {
            'relu': F.relu,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'gelu': F.gelu,
            'elu': F.elu,
            'softsign': F.softsign,
            'linear': lambda x: x,
            'he_2': lambda z: z**2 - 1.0,
            'he_3': lambda z: z**3 - 3.0 * z,
            'he_4': lambda z: z**4 - 6.0 * z**2 + 3.0,
            'swish': lambda z: z * torch.sigmoid(z),
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        
        return activations[name]
    
    def forward(self, x):
        """
        Forward pass through autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, D)
        
        Returns
        -------
        torch.Tensor
            Reconstructed data of shape (batch_size, D)
        """
        batch_size = x.shape[0]
        
        # Encoder
        if self.tied:
            h = self.W @ x.reshape(-1, self.D, 1) / np.sqrt(self.D)
        else:
            h = self.V.T @ x.reshape(-1, self.D, 1) / np.sqrt(self.D)
        
        if self.bias is not None:
            h = h + self.bias.view(1, self.K, 1)
        
        # Activation
        h = self.activation(h)
        
        # Decoder
        x_recon = (self.W.T / np.sqrt(self.D)) @ h
        x_recon = x_recon.squeeze(-1)
        
        return x_recon


# =============================================================================
# Loss Functions
# =============================================================================

def reconstruction_loss(x, x_hat):
    """
    Compute squared reconstruction error.
    
    Parameters
    ----------
    x : torch.Tensor
        Original data
    x_hat : torch.Tensor
        Reconstructed data
    
    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    return torch.sum((x - x_hat) ** 2) / 2


def l2_regularized_loss(model, x, x_hat, weight_decay=0.0):
    """
    Reconstruction loss with L2 weight regularization.
    
    Parameters
    ----------
    model : AutoEncoder
        The autoencoder model
    x : torch.Tensor
        Original data
    x_hat : torch.Tensor
        Reconstructed data
    weight_decay : float
        L2 regularization coefficient
    
    Returns
    -------
    torch.Tensor
        Total loss (reconstruction + regularization)
    """
    recon_loss = reconstruction_loss(x, x_hat)
    
    if weight_decay == 0.0:
        return recon_loss
    
    reg_loss = torch.sum(model.W ** 2) / 2
    if not model.tied and model.V is not None:
        reg_loss = reg_loss + torch.sum(model.V ** 2) / 2
    
    return recon_loss + weight_decay * reg_loss


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(u, v, beta_u, beta_v, n_samples, d, seed, 
                  latent_dependence='independent', device=None):
    """
    Generate synthetic data from two-spike model with configurable latent dependence.
    
    The data model is:
        X = sqrt(beta_u)/sqrt(d) * g_u * u^T + 
            sqrt(beta_v)/sqrt(d) * g_v * v^T * S^T + Z * S^T
    
    where g_u, g_v are latent variables with specified dependence structure,
    Z is standard Gaussian noise, and S is a normalization matrix.
    
    Parameters
    ----------
    u, v : torch.Tensor
        Signal directions (d-dimensional)
    beta_u, beta_v : float
        Signal strengths
    n_samples : int
        Number of samples to generate
    d : int
        Dimension
    seed : int
        Random seed
    latent_dependence : str
        Type of dependence between g_u and g_v
    device : torch.device
        Device for computation
    
    Returns
    -------
    torch.Tensor
        Generated data of shape (n_samples, d)
    """
    if device is None:
        device = torch.device('cpu')
    
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    # Generate first latent variable
    g_u = torch.randn((n_samples, 1), generator=gen, device=device)
    
    # Generate second latent variable with specified dependence
    if latent_dependence == 'independent':
        g_v = torch.randn((n_samples, 1), generator=gen, device=device)
        g_v = torch.sign(g_v)
    elif latent_dependence == 'correlated':
        g_v = torch.sign(g_u)
    elif latent_dependence == 'dependent':  # k=2 dependent case
        g_v = torch.where(
            torch.abs(g_u) >= AUX_K2,
            torch.ones_like(g_u),
            -torch.ones_like(g_u)
        )
    elif latent_dependence == 'dependent_ord3':  # k=3 dependent case
        g_v = torch.sign(g_u) * torch.sign(torch.abs(g_u) - AUX_K3)
    else:
        raise ValueError(f"Unknown latent dependence: {latent_dependence}")
    
    # Normalization matrix
    S = torch.eye(d, device=device) - \
        (beta_v / (1 + beta_v + math.sqrt(1 + beta_v))) * (v @ v.T) / d
    
    # Generate data
    z_noise = torch.randn((n_samples, d), generator=gen, device=device)
    rhs_base = (math.sqrt(beta_v) / math.sqrt(d)) * (g_v @ v.T) + z_noise
    rhs = rhs_base @ S.T
    lhs = (math.sqrt(beta_u) / math.sqrt(d)) * (g_u @ u.T)
    
    X = lhs + rhs
    
    return X


# =============================================================================
# Training and Evaluation
# =============================================================================

def compute_overlaps(model, u, v, d):
    """
    Compute overlaps between learned weights and ground truth directions.
    
    Parameters
    ----------
    model : AutoEncoder
        Trained model
    u, v : torch.Tensor
        Ground truth signal directions
    d : int
        Dimension
    
    Returns
    -------
    tuple
        (q_w, q_u_w, q_v_w) where:
        - q_w: self-overlap of weights
        - q_u_w: overlap with u
        - q_v_w: overlap with v
    """
    with torch.no_grad():
        w = model.W[0]  # Shape: (d,)
        
        q_w = float((w @ w / d).item())
        q_u_w = float((u.reshape(-1) @ w / (torch.norm(u) * torch.norm(w))).item())
        q_v_w = float((v.reshape(-1) @ w / (torch.norm(v) * torch.norm(w))).item())
    
    return q_w, q_u_w, q_v_w


def train_autoencoder(X_train, n_hidden, activation, epochs, learning_rate,
                      weight_decay, use_bias, tied_weights, u, v, d, seed=0):
    """
    Train autoencoder on training data.
    
    Parameters
    ----------
    X_train : torch.Tensor
        Training data
    n_hidden : int
        Number of hidden units
    activation : str
        Activation function name
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        L2 regularization strength
    use_bias : bool
        Whether to use bias in hidden layer
    tied_weights : bool
        Whether to tie encoder/decoder weights
    u, v : torch.Tensor
        Ground truth directions for tracking
    d : int
        Dimension
    seed : int
        Random seed for initialization
    
    Returns
    -------
    tuple
        (model, train_loss_history, q_u_history, q_v_history, q_w_history)
    """
    n_train = X_train.shape[0]
    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=n_train,
        shuffle=False
    )
    
    # Initialize model
    model = AutoEncoder(
        D=d,
        K=n_hidden,
        activation=activation,
        tied=tied_weights,
        init_scale=1.0,
        use_bias=use_bias,
        seed=seed
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Tracking
    train_loss_history = []
    q_u_history = []
    q_v_history = []
    q_w_history = []
    
    # Initial overlaps
    q_w, q_u_w, q_v_w = compute_overlaps(model, u, v, d)
    q_u_history.append(q_u_w)
    q_v_history.append(q_v_w)
    q_w_history.append(q_w)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for (xb,) in train_loader:
            xb = xb.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            x_hat = model(xb)
            loss = l2_regularized_loss(model, xb, x_hat, weight_decay=weight_decay)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Record metrics
        model.eval()
        train_loss_history.append(epoch_loss / n_train)
        
        q_w, q_u_w, q_v_w = compute_overlaps(model, u, v, d)
        q_u_history.append(q_u_w)
        q_v_history.append(q_v_w)
        q_w_history.append(q_w)
    
    return (model, 
            np.array(train_loss_history),
            np.array(q_u_history),
            np.array(q_v_history),
            np.array(q_w_history))


@torch.no_grad()
def compute_svd_baseline(X_train, u, v, random_baseline, k=1):
    """
    Compute SVD baseline and overlaps with ground truth directions.
    
    Parameters
    ----------
    X_train : torch.Tensor or np.ndarray
        Training data
    u, v, random_baseline : torch.Tensor or np.ndarray
        Ground truth directions and random baseline
    k : int
        Number of principal components to use
    
    Returns
    -------
    tuple
        (mse_train, V_k_transpose, overlap_u, overlap_v, overlap_random)
    """
    # Convert to tensors
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device)
        return torch.as_tensor(x, device=device)
    
    X_train = to_tensor(X_train)
    u = to_tensor(u).flatten()
    v = to_tensor(v).flatten()
    random_baseline = to_tensor(random_baseline).flatten()
    
    n_train, d = X_train.shape
    k = min(int(k), d, n_train)
    
    # Compute SVD
    _, S, Vh = torch.linalg.svd(X_train, full_matrices=False)
    
    # Reconstruction error
    tail_energy = torch.sum(S[k:] ** 2)
    mse_train = float((tail_energy / n_train / 2.0).item())
    
    V_k_transpose = Vh[:k, :]
    
    # Compute overlaps
    def overlap(vec, target):
        return float(torch.abs(torch.dot(vec, target) / torch.norm(target)).item())
    
    overlap_u = overlap(Vh[0, :], u)
    overlap_v = overlap(Vh[1, :] if Vh.shape[0] > 1 else Vh[0, :], v)
    overlap_random = overlap(Vh[1, :] if Vh.shape[0] > 1 else Vh[0, :], random_baseline)
    
    return mse_train, V_k_transpose, overlap_u, overlap_v, overlap_random


# =============================================================================
# Main Experimental Loop
# =============================================================================

def main():
    """Run complete experimental pipeline."""
    
    # Generate ground truth signal directions
    gen = torch.Generator(device=device)
    
    gen.manual_seed(8082125)
    u = torch.randn((D, 1), generator=gen, device=device)
    u = u / torch.norm(u) * math.sqrt(D)
    
    gen.manual_seed(1313)
    v = torch.randn((D, 1), generator=gen, device=device)
    v = v / torch.norm(v) * math.sqrt(D)
    
    gen.manual_seed(20242)
    random_baseline = torch.randn((D, 1), generator=gen, device=device)
    random_baseline = random_baseline / torch.norm(random_baseline) * math.sqrt(D)
    
    # Convert to numpy for SVD baseline
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    random_np = random_baseline.detach().cpu().numpy()
    
    # Print experimental configuration
    print('=' * 70)
    print('Autoencoder Experiment Configuration')
    print('=' * 70)
    print(f'Signal strengths: beta_u = {BETA_U}, beta_v = {BETA_V}')
    print(f'Hidden neurons: {N_HIDDEN}')
    print(f'Dimension: d = {D}')
    print(f'Latent dependence: {LATENT_DEPENDENCE}')
    print(f'Activation: {NONLINEARITY}')
    print(f'Tied weights: {TIED_WEIGHTS}')
    print(f'Bias: {USE_BIAS}')
    print(f'Weight decay: {WEIGHT_DECAY}')
    print(f'Seeds per alpha: {N_SEEDS}')
    print(f'Test samples: {N_TEST_TOTAL:,}')
    print(f'Device: {device}')
    print('=' * 70)
    
    # Initialize results storage
    results = {
        'alpha': [],
        'seed': [],
        'lambda': [],
        'svd_overlap_u': [],
        'svd_overlap_v': [],
        'svd_overlap_rpy': [],
        'svd_mse_train': [],
        'svd_mse_test': [],
        'ae_mse_train': [],
        'ae_mse_test': [],
        'ae_overlap_u': [],
        'ae_overlap_v': [],
        'ae_q_final': []
    }
    
    results_dynamics = {
        'alpha': [],
        'seed': [],
        'lambda': [],
        'tr_hist': [],
        'q_u_w': [],
        'q_v_w': [],
        'q_track': []
    }
    
    results_weights = {
        'alpha': [],
        'seed': [],
        'lambda': [],
        'weights': []
    }
    
    total_time_start = time.time()
    
    # Main experimental loop
    for alpha_idx, alpha in enumerate(ALPHA_ARRAY):
        print(f"\n{'=' * 70}")
        print(f"Alpha = {alpha:.3f} ({alpha_idx + 1}/{len(ALPHA_ARRAY)})")
        print('=' * 70)
        
        n_train = int(alpha * D)
        
        for seed_idx in range(N_SEEDS):
            print(f"\nSeed {seed_idx + 1}/{N_SEEDS}")
            print('-' * 70)
            
            # Store metadata
            results['alpha'].append(alpha)
            results['lambda'].append(WEIGHT_DECAY)
            results['seed'].append(seed_idx)
            
            results_dynamics['alpha'].append(alpha)
            results_dynamics['lambda'].append(WEIGHT_DECAY)
            results_dynamics['seed'].append(seed_idx)
            
            results_weights['alpha'].append(alpha)
            results_weights['lambda'].append(WEIGHT_DECAY)
            results_weights['seed'].append(seed_idx)
            
            # Generate training data
            data_seed = int(rng.integers(1 << 30))
            X_train = generate_data(
                u, v, BETA_U, BETA_V, n_train, D,
                seed=data_seed,
                latent_dependence=LATENT_DEPENDENCE,
                device=device
            )
            
            # Train autoencoder
            model, train_hist, q_u_hist, q_v_hist, q_w_hist = train_autoencoder(
                X_train=X_train,
                n_hidden=N_HIDDEN,
                activation=NONLINEARITY,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                use_bias=USE_BIAS,
                tied_weights=TIED_WEIGHTS,
                u=u,
                v=v,
                d=D,
                seed=data_seed + 33_000
            )
            
            print(f"AE overlap u: {np.abs(q_u_hist[-1]):.4f}")
            print(f"AE overlap v: {np.abs(q_v_hist[-1]):.4f}")
            print(f"AE q_final: {q_w_hist[-1]:.4f}")
            
            # Compute SVD baseline
            _, V_k_T, svd_overlap_u, svd_overlap_v, svd_overlap_random = \
                compute_svd_baseline(
                    X_train.detach().cpu().numpy(),
                    u_np, v_np, random_np,
                    k=1
                )
            
            # SVD training reconstruction
            V_k_T_torch = torch.tensor(V_k_T, dtype=torch.float32, device=device)
            X_recon_svd_train = (X_train @ V_k_T_torch.T) @ V_k_T_torch
            svd_mse_train = (torch.sum((X_train - X_recon_svd_train) ** 2) / 2.0).item()
            svd_mse_train /= n_train
            
            print(f"SVD overlap u: {svd_overlap_u:.4f}")
            print(f"SVD overlap v: {svd_overlap_v:.4f}")
            print(f"SVD overlap random: {svd_overlap_random:.4f}")
            
            # Extract weights
            w_learned = model.W.detach().cpu().numpy().tolist()
            
            # Free training data memory
            del X_train, X_recon_svd_train
            
            # Evaluate on test set
            print("Evaluating on test set...")
            test_mse_ae = 0.0
            test_mse_svd = 0.0
            total_test_samples = 0
            
            for batch_idx in range(1, N_BATCHES + 1):
                test_seed = data_seed + batch_idx * 1000
                X_test_batch = generate_data(
                    u, v, BETA_U, BETA_V, N_TEST_BATCH, D,
                    seed=test_seed,
                    latent_dependence=LATENT_DEPENDENCE,
                    device=device
                )
                total_test_samples += N_TEST_BATCH
                
                # AE reconstruction
                model.eval()
                with torch.no_grad():
                    X_hat_ae = model(X_test_batch)
                    batch_mse_ae = (torch.sum((X_test_batch - X_hat_ae) ** 2) / 2.0).item()
                    test_mse_ae += batch_mse_ae
                
                # SVD reconstruction
                X_recon_svd = (X_test_batch @ V_k_T_torch.T) @ V_k_T_torch
                batch_mse_svd = (torch.sum((X_test_batch - X_recon_svd) ** 2) / 2.0).item()
                test_mse_svd += batch_mse_svd
                
                del X_test_batch, X_hat_ae, X_recon_svd
            
            test_mse_ae /= total_test_samples
            test_mse_svd /= total_test_samples
            
            print(f"AE test MSE: {test_mse_ae:.6f}")
            print(f"SVD test MSE: {test_mse_svd:.6f}")
            
            # Store results
            results['svd_overlap_u'].append(svd_overlap_u)
            results['svd_overlap_v'].append(svd_overlap_v)
            results['svd_overlap_rpy'].append(svd_overlap_random)
            results['svd_mse_train'].append(svd_mse_train)
            results['svd_mse_test'].append(test_mse_svd)
            results['ae_mse_train'].append(train_hist[-1])
            results['ae_mse_test'].append(test_mse_ae)
            results['ae_overlap_u'].append(np.abs(q_u_hist[-1]))
            results['ae_overlap_v'].append(np.abs(q_v_hist[-1]))
            results['ae_q_final'].append(q_w_hist[-1])
            
            results_dynamics['tr_hist'].append(train_hist)
            results_dynamics['q_u_w'].append(q_u_hist)
            results_dynamics['q_v_w'].append(q_v_hist)
            results_dynamics['q_track'].append(q_w_hist)
            
            results_weights['weights'].append(w_learned)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
    
    # Compute and print total time
    total_time = time.time() - total_time_start
    print("\n" + "=" * 70)
    print(f"Total simulation time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print("=" * 70)
    
    # Save results
    if WRITE_DATA:
        output_dir = './data_ERM_AE/'
        
        base_filename = (
            f'results_HOC_{LATENT_DEPENDENCE}_{NONLINEARITY}_'
            f'beta_u{BETA_U}_beta_v{BETA_V}_'
            f'nk{N_HIDDEN}_d{D}_seeds{N_SEEDS}_'
            f'tied_{TIED_WEIGHTS}_bias_{USE_BIAS}'
        )
        
        # Save main results
        filepath = output_dir + base_filename + '.npz'
        np.savez_compressed(filepath, **results)
        print(f"\nResults saved to: {filepath}")
        
        # Save dynamics
        filepath_dynamics = output_dir + 'results_dynamics_' + base_filename + '.npz'
        safe_dynamics = {k: np.array(v, dtype=object) for k, v in results_dynamics.items()}
        np.savez_compressed(filepath_dynamics, **safe_dynamics)
        print(f"Dynamics saved to: {filepath_dynamics}")
        
        # Save weights
        filepath_weights = output_dir + 'results_weights_' + base_filename + '.npz'
        safe_weights = {k: np.array(v, dtype=object) for k, v in results_weights.items()}
        np.savez_compressed(filepath_weights, **safe_weights)
        print(f"Weights saved to: {filepath_weights}")


if __name__ == '__main__':
    main()
