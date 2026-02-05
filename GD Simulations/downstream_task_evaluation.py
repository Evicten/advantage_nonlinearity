"""
Downstream Task Evaluation for Learned Autoencoder Representations

This script evaluates the performance of learned autoencoder representations on a
downstream binary classification task. It loads pre-trained autoencoder weights and
tests their ability to predict sign labels based on conditional means.

"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch


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
BETA_U = 1  # Signal strength for first spike (training)
BETA_V = 2  # Signal strength for second spike (training)
LATENT_DEPENDENCE = 'dependent'  # Latent variable dependence structure

# Downstream task parameters
BETA_U_DOWNSTREAM = 1  # Signal strength for downstream task
BETA_V_DOWNSTREAM = 2  # Signal strength for downstream task
N_DOWNSTREAM = 100  # Batch size for downstream evaluation
TOTAL_SAMPLES = 5000  # Total number of evaluation samples

# Model configuration (must match pre-trained models)
N_HIDDEN = 1  # Number of hidden neurons
TIED_WEIGHTS = True  # Whether weights were tied during training
USE_BIAS = False  # Whether bias was used during training
N_SEEDS = 30  # Number of random seeds in pre-trained models

# Experimental design
ALPHA_ARRAY = np.linspace(0.35, 10, 30)  # Sample complexity ratios
LAMBDA_REG = 0.0  # Regularization strength (must match training)

# Nonlinearities to evaluate
NONLINEARITIES = ('linear', 'relu', 'elu', 'tanh')

# Flags
WRITE_DATA = True

# Auxiliary constant for dependent latent variables
AUX_K2 = norm.ppf(0.75)  # For k=2 dependent case

# Device configuration
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else ('mps' if torch.backends.mps.is_available() else 'cpu')
)
print(f"Device: {device}")


# =============================================================================
# Ground Truth Signal Directions
# =============================================================================

def generate_ground_truth_signals(d, device):
    """
    Generate ground truth signal directions u and v.
    
    Parameters
    ----------
    d : int
        Dimension
    device : torch.device
        Device for computation
    
    Returns
    -------
    tuple
        (u, v) normalized to norm sqrt(d)
    """
    gen = torch.Generator(device=device)
    
    # Generate u
    gen.manual_seed(8082125)
    u = torch.randn((d, 1), generator=gen, device=device)
    u = u / torch.norm(u) * math.sqrt(d)
    
    # Generate v
    gen.manual_seed(1313)
    v = torch.randn((d, 1), generator=gen, device=device)
    v = v / torch.norm(v) * math.sqrt(d)
    
    return u, v


# =============================================================================
# Data Generation for Downstream Task
# =============================================================================

def index_model(X, w):
    """
    Compute sign predictions based on linear projections.
    
    Parameters
    ----------
    X : torch.Tensor
        Input data of shape (n, d)
    w : torch.Tensor
        Weight vector of shape (d, 1)
    
    Returns
    -------
    torch.Tensor
        Sign predictions of shape (n, 1)
    """
    y = X @ w
    y_output = torch.sign(y)
    return y_output


def generate_downstream_data(u, v, beta_u, beta_v, n_samples, d, seed,
                             latent_dependence='independent', device=None):
    """
    Generate data for downstream binary classification task.
    
    Generates synthetic data and computes conditional means for each class
    defined by sign(X @ v).
    
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
        Type of dependence between latent variables
    device : torch.device
        Device for computation
    
    Returns
    -------
    tuple
        (X_pos, label_pos, X_neg, label_neg) where:
        - X_pos: conditional mean for positive class
        - label_pos: positive class label (+1)
        - X_neg: conditional mean for negative class
        - label_neg: negative class label (-1)
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
    
    # Compute labels based on index model
    f_z = index_model(X, v)
    
    # Compute conditional means for each class
    X_pos = torch.mean(X[f_z.flatten() == 1], dim=0, keepdim=True)
    X_neg = torch.mean(X[f_z.flatten() == -1], dim=0, keepdim=True)
    
    pos_label = 1
    neg_label = -1
    
    return X_pos, pos_label, X_neg, neg_label


# =============================================================================
# Downstream Task Evaluation
# =============================================================================

def evaluate_downstream_task(w_vector, u, v, beta_u, beta_v, n_batch,
                             total_samples, d, latent_dependence, device):
    """
    Evaluate a learned weight vector on the downstream classification task.
    
    Parameters
    ----------
    w_vector : torch.Tensor
        Learned weight vector of shape (d, 1)
    u, v : torch.Tensor
        Ground truth signal directions
    beta_u, beta_v : float
        Signal strengths for downstream task
    n_batch : int
        Batch size for evaluation
    total_samples : int
        Total number of evaluation samples
    d : int
        Dimension
    latent_dependence : str
        Latent variable dependence structure
    device : torch.device
        Device for computation
    
    Returns
    -------
    float
        Mean squared error on downstream task
    """
    mse_total = 0.0
    
    for _ in range(total_samples):
        # Generate new random seed for each sample
        sample_seed = int(rng.integers(1 << 30))
        
        # Generate downstream data
        X_pos, y_pos, X_neg, y_neg = generate_downstream_data(
            u, v, beta_u, beta_v, n_batch, d,
            seed=sample_seed,
            latent_dependence=latent_dependence,
            device=device
        )
        
        # Make predictions
        y_pred_pos = torch.sign((X_pos @ w_vector)).item()
        y_pred_neg = torch.sign((X_neg @ w_vector)).item()
        
        # Compute MSE for both classes
        mse_loss = 0.25 * (y_pos - y_pred_pos) ** 2 + \
                   0.25 * (y_neg - y_pred_neg) ** 2
        
        mse_total += mse_loss
    
    # Average over all samples and classes
    mse_total = mse_total / (2 * total_samples)
    
    return mse_total


def load_pretrained_weights(alpha, nonlinearity, beta_u, beta_v, n_hidden,
                           d, n_seeds, tied, bias, lambda_reg, data_dir='./data_ERM_AE/'):
    """
    Load pre-trained autoencoder weights for a specific configuration.
    
    Parameters
    ----------
    alpha : float
        Sample complexity ratio
    nonlinearity : str
        Activation function name
    beta_u, beta_v : float
        Signal strengths used during training
    n_hidden : int
        Number of hidden neurons
    d : int
        Dimension
    n_seeds : int
        Number of random seeds
    tied : bool
        Whether weights were tied
    bias : bool
        Whether bias was used
    lambda_reg : float
        Regularization strength
    data_dir : str
        Directory containing pre-trained weights
    
    Returns
    -------
    np.ndarray
        Array of learned weight vectors, shape (n_seeds, d)
    """
    # Construct filename matching training script convention
    filename = (
        f'{data_dir}results_weights_HOC_{LATENT_DEPENDENCE}_{nonlinearity}_'
        f'beta_u{beta_u}_beta_v{beta_v}_'
        f'nk{n_hidden}_d{d}_seeds{n_seeds}_'
        f'tied_{tied}_bias_{bias}.npz'
    )
    
    # Load data
    data = np.load(filename, allow_pickle=True)
    alphas = np.array(data['alpha'], dtype=float)
    lambdas = np.array(data['lambda'], dtype=float)
    weights_obj = data['weights']
    
    # Filter for specified alpha and lambda
    mask = np.isclose(alphas, alpha) & np.isclose(lambdas, lambda_reg)
    indices = np.where(mask)[0]
    
    # Extract and reshape weights
    weights_selected = [np.array(weights_obj[i], dtype=float) for i in indices]
    weights_array = np.array(weights_selected).reshape(n_seeds, d)
    
    return weights_array


# =============================================================================
# Main Experimental Loop
# =============================================================================

def main():
    """Run complete downstream task evaluation pipeline."""
    
    # Generate ground truth signal directions
    u, v = generate_ground_truth_signals(D, device)
    
    # Save ground truth signals
    u_np = u.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    weights_dict = {'u': u_np, 'v': v_np}
    np.savez(f'./data_ERM_AE/weights_spikes_d{D}.npz', **weights_dict)
    print(f"Ground truth signals saved to: ./data_ERM_AE/weights_spikes_d{D}.npz")
    
    # Print experimental configuration
    print('=' * 70)
    print('Downstream Task Evaluation Configuration')
    print('=' * 70)
    print(f'Dimension: d = {D}')
    print(f'Training signal strengths: beta_u = {BETA_U}, beta_v = {BETA_V}')
    print(f'Downstream signal strengths: beta_u = {BETA_U_DOWNSTREAM}, beta_v = {BETA_V_DOWNSTREAM}')
    print(f'Latent dependence: {LATENT_DEPENDENCE}')
    print(f'Batch size: {N_DOWNSTREAM}')
    print(f'Total evaluation samples: {TOTAL_SAMPLES}')
    print(f'Number of seeds: {N_SEEDS}')
    print(f'Alpha values: {len(ALPHA_ARRAY)} points from {ALPHA_ARRAY[0]:.2f} to {ALPHA_ARRAY[-1]:.2f}')
    print(f'Nonlinearities: {NONLINEARITIES}')
    print(f'Device: {device}')
    print('=' * 70)
    
    # Initialize results storage
    results = {
        'alpha': [],
        'non_linearity': [],
        'mean_error': [],
        'std_error': []
    }
    
    # Evaluate each nonlinearity
    for nonlinearity in NONLINEARITIES:
        print(f"\n{'=' * 70}")
        print(f"Evaluating nonlinearity: {nonlinearity}")
        print('=' * 70)
        
        mse_downstream_mean = []
        mse_downstream_std = []
        
        # Evaluate each alpha value
        for alpha_idx, alpha in enumerate(ALPHA_ARRAY):
            print(f"\nAlpha = {alpha:.3f} ({alpha_idx + 1}/{len(ALPHA_ARRAY)})")
            print('-' * 70)
            
            # Load pre-trained weights
            try:
                weights_array = load_pretrained_weights(
                    alpha=alpha,
                    nonlinearity=nonlinearity,
                    beta_u=BETA_U,
                    beta_v=BETA_V,
                    n_hidden=N_HIDDEN,
                    d=D,
                    n_seeds=N_SEEDS,
                    tied=TIED_WEIGHTS,
                    bias=USE_BIAS,
                    lambda_reg=LAMBDA_REG
                )
            except FileNotFoundError:
                print(f"WARNING: Weights file not found for alpha={alpha}, nonlinearity={nonlinearity}")
                print("Skipping this configuration...")
                continue
            
            # Evaluate each seed
            mse_per_seed = []
            for seed_idx in range(N_SEEDS):
                # Convert weight to torch tensor
                w_vec = weights_array[seed_idx]
                w_vec_torch = torch.as_tensor(w_vec, device=device, dtype=torch.float32)
                w_vec_torch = w_vec_torch.reshape(-1, 1)
                
                # Evaluate on downstream task
                mse_seed = evaluate_downstream_task(
                    w_vector=w_vec_torch,
                    u=u,
                    v=v,
                    beta_u=BETA_U_DOWNSTREAM,
                    beta_v=BETA_V_DOWNSTREAM,
                    n_batch=N_DOWNSTREAM,
                    total_samples=TOTAL_SAMPLES,
                    d=D,
                    latent_dependence=LATENT_DEPENDENCE,
                    device=device
                )
                
                mse_per_seed.append(mse_seed)
            
            # Compute statistics across seeds
            mean_mse = np.mean(mse_per_seed)
            std_mse = np.std(mse_per_seed)
            
            mse_downstream_mean.append(mean_mse)
            mse_downstream_std.append(std_mse)
            
            print(f"Mean MSE: {mean_mse:.6f} ± {std_mse:.6f}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Store results for this nonlinearity
        results['alpha'].append(ALPHA_ARRAY.tolist())
        results['non_linearity'].append(nonlinearity)
        results['mean_error'].append(mse_downstream_mean)
        results['std_error'].append(mse_downstream_std)
        
        print(f"\nCompleted evaluation for {nonlinearity}")
        print(f"Mean MSE range: [{np.min(mse_downstream_mean):.6f}, {np.max(mse_downstream_mean):.6f}]")
    
    # Save results
    if WRITE_DATA:
        output_filename = (
            f'./data_ERM_AE/downstream_task_results_'
            f'beta_u{BETA_U}_beta_v{BETA_V}_'
            f'nk{N_HIDDEN}_d{D}_seeds{N_SEEDS}_'
            f'tied_{TIED_WEIGHTS}_bias_{USE_BIAS}_'
            f'n_downstream_{N_DOWNSTREAM}_total_samples_{TOTAL_SAMPLES}.npz'
        )
        
        np.savez(output_filename, **results)
        print(f"\n{'=' * 70}")
        print(f"Results saved to: {output_filename}")
        print('=' * 70)


if __name__ == '__main__':
    main()
