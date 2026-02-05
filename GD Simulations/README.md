# Autoencoder Experiments on Synthetic Two-Spike Data

This repository contains code for training and evaluating autoencoders on synthetic data with higher-order correlations, comparing learned representations to PCA/SVD baselines and evaluating them on downstream tasks.

## Overview

The code implements experiments that:
- Generate synthetic data from a two-spike model with configurable latent dependencies
- Train single-hidden-layer autoencoders across multiple sample complexity regimes
- Compare autoencoder performance to PCA/SVD baselines
- Track learning dynamics and overlaps with ground truth signal directions
- Evaluate learned representations on downstream binary classification tasks

## Requirements

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
torch>=2.0.0
```

## File Structure

- `autoencoder_experiments.py`: Main training script for autoencoders
- `downstream_task_evaluation.py`: Evaluation script for downstream tasks
- `data_ERM_AE/`: Output directory for results (created automatically)

## Usage

### Training Autoencoders

First, train the autoencoders on synthetic data:

```bash
python autoencoder_experiments.py
```

### Evaluating on Downstream Tasks

After training, evaluate the learned representations on downstream tasks:

```bash
python downstream_task_evaluation.py
```

**Note:** The downstream evaluation script requires the output files from the training script to be present in `./data_ERM_AE/`.

### Configuration

All hyperparameters are defined at the top of the script in the "Configuration and Hyperparameters" section:

**Data Parameters:**
- `D`: Input dimension (default: 2000)
- `BETA_U`, `BETA_V`: Signal strengths for the two spikes
- `LATENT_DEPENDENCE`: Type of correlation between latent variables
  - Options: `'independent'`, `'correlated'`, `'dependent'`, `'dependent_ord3'`

**Model Parameters:**
- `N_HIDDEN`: Number of hidden neurons (default: 1)
- `NONLINEARITY`: Activation function (default: `'tanh'`)
  - Options: `'relu'`, `'elu'`, `'tanh'`, `'sigmoid'`, etc.
- `TIED_WEIGHTS`: Whether encoder and decoder share weights (default: `True`)
- `USE_BIAS`: Whether to include bias terms (default: `False`)

**Training Parameters:**
- `LEARNING_RATE`: Learning rate (default: 0.1)
- `WEIGHT_DECAY`: L2 regularization strength (default: 0.0)
- `EPOCHS`: Number of training epochs (default: 1200)

**Experimental Design:**
- `ALPHA_ARRAY`: Sample complexity ratios (n/d)
- `N_SEEDS`: Number of random seeds per alpha value

### Downstream Task Configuration

The `downstream_task_evaluation.py` script has similar configuration options:

**Downstream Task Parameters:**
- `BETA_U_DOWNSTREAM`, `BETA_V_DOWNSTREAM`: Signal strengths for downstream evaluation
- `N_DOWNSTREAM`: Batch size for downstream evaluation (default: 100)
- `TOTAL_SAMPLES`: Total number of evaluation samples (default: 5000)

**Evaluation Settings:**
- `NONLINEARITIES`: Tuple of activation functions to evaluate (default: `('linear', 'relu', 'elu', 'tanh')`)
- `ALPHA_ARRAY`: Sample complexity ratios to evaluate (default: 30 values from 0.35 to 10)

## Output

### Training Script Output

The `autoencoder_experiments.py` script generates three compressed numpy archives:

1. **Main Results** (`results_HOC_*.npz`):
   - Final metrics for each experiment
   - Fields: alpha, seed, lambda, overlaps, MSE values

2. **Training Dynamics** (`results_dynamics_HOC_*.npz`):
   - Per-epoch training history
   - Fields: loss trajectories, overlap evolution

3. **Learned Weights** (`results_weights_HOC_*.npz`):
   - Final learned weight matrices
   - Fields: encoder weights for each experiment

### Downstream Evaluation Output

The `downstream_task_evaluation.py` script generates:

1. **Ground Truth Signals** (`weights_spikes_d*.npz`):
   - Saved u and v vectors used for data generation

2. **Downstream Task Results** (`downstream_task_results_*.npz`):
   - Mean and standard deviation of classification error across seeds
   - Fields: alpha, non_linearity, mean_error, std_error

### Loading Results

**Training results:**
```python
import numpy as np

# Load main results
data = np.load('data_ERM_AE/results_HOC_*.npz', allow_pickle=True)
alphas = data['alpha']
ae_overlap_u = data['ae_overlap_u']
svd_overlap_u = data['svd_overlap_u']

# Load dynamics
dynamics = np.load('data_ERM_AE/results_dynamics_HOC_*.npz', allow_pickle=True)
train_histories = dynamics['tr_hist']
```

**Downstream task results:**
```python
# Load downstream task results
downstream = np.load('data_ERM_AE/downstream_task_results_*.npz', allow_pickle=True)
nonlinearities = downstream['non_linearity']
mean_errors = downstream['mean_error']
std_errors = downstream['std_error']

# Access results for specific nonlinearity
idx = np.where(nonlinearities == 'tanh')[0][0]
alpha_values = downstream['alpha'][idx]
tanh_mean_error = mean_errors[idx]
tanh_std_error = std_errors[idx]
```

## Key Functions

### Training Script Functions

#### Data Generation

`generate_data(u, v, beta_u, beta_v, n_samples, d, seed, latent_dependence, device)`

Generates synthetic data according to:
```
X = sqrt(beta_u)/sqrt(d) * g_u * u^T + 
    sqrt(beta_v)/sqrt(d) * g_v * v^T * S^T + Z * S^T
```

#### Model Training

`train_autoencoder(X_train, n_hidden, activation, epochs, learning_rate, weight_decay, use_bias, tied_weights, u, v, d, seed)`

Trains an autoencoder and tracks:
- Training loss per epoch
- Overlaps with ground truth directions u and v
- Weight self-overlap

#### Baseline Computation

`compute_svd_baseline(X_train, u, v, random_baseline, k)`

Computes PCA/SVD baseline:
- Reconstruction error
- Overlaps with ground truth directions
- Top principal components

### Downstream Evaluation Functions

#### Downstream Data Generation

`generate_downstream_data(u, v, beta_u, beta_v, n_samples, d, seed, latent_dependence, device)`

Generates data for binary classification task:
- Computes conditional means E[X | sign(X @ v) = +1] and E[X | sign(X @ v) = -1]
- Returns class-conditional means and labels

#### Task Evaluation

`evaluate_downstream_task(w_vector, u, v, beta_u, beta_v, n_batch, total_samples, d, latent_dependence, device)`

Evaluates learned representation:
- Tests prediction accuracy on conditional means
- Computes mean squared error across multiple samples
- Returns average classification performance

## Hardware

- Automatically detects and uses GPU (CUDA), MPS (Apple Silicon), or CPU
- Supports multi-batch test set evaluation for memory efficiency