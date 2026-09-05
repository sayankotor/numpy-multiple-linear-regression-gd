"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    np.random.seed(seed)
    
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    return (X_shuffled, y_shuffled)

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    X_shuffled, y_shuffled = X, y
    common_len = X.shape[0]
    X_train, X_val,X_test, = X_shuffled[:int(common_len*train_frac)],X_shuffled[int(common_len*train_frac):int(common_len*(train_frac)) + int(common_len*val_frac)], X_shuffled[int(common_len*(train_frac)) + int(common_len*val_frac):]
    y_train, y_val,y_test, = y_shuffled[:int(common_len*train_frac)],y_shuffled[int(common_len*train_frac):int(common_len*(train_frac)) + int(common_len*val_frac)], y_shuffled[int(common_len*(train_frac)) + int(common_len*val_frac):]
    return (X_train, y_train, X_val, y_val, X_test, y_test)

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    stds = np.std(X, axis = 0)
    stds = np.where(stds ==0.0, 1.0, stds)
    return np.mean(X, axis = 0), stds

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    return (X - mean)/std

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    column = np.ones(X.shape[0])
    return np.concat([column[:,None], X], axis = 1)

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    X_new = standardize_features(X, mean, std)
    X_new_plus_bias = add_bias_column(X_new)
    return X_new_plus_bias

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return X@weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    return np.sum((y_true - y_pred)**2/(len(y_true)))

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    return (2/len(y_true))*X.T@(y_pred - y_true)

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    #return np.linalg.solve(X.T@X, X.T@y)
    return np.linalg.pinv (X.T@X)@X.T@y

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # TODO: Return (n_features,) weights sampled from N(0, 0.01)
    np.random.seed(seed)
    return np.random.normal(0.0, 0.01, n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step
    y_pred = predict_linear(X, weights)
    dw = mse_gradient(X, y, y_pred)
    weights = weights - lr*dw
    return weights

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # TODO: return the pair (train_loss, val_loss) as MSE floats
    y_pred = predict_linear(X_val, weights)
    val_loss = mse_loss(y_val, y_pred)
    y_pred = predict_linear(X_train, weights)
    train_loss = mse_loss(y_train, y_pred)
    return (train_loss, val_loss)

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # TODO: Update best weights and patience counter; signal stop when val loss stalls...
    if (val_loss < best_val_loss):
        best_val_loss = val_loss
        best_weights = weights.copy()
        wait = 0
        is_stop = False
    elif (wait+1 >= patience):
        is_stop = True
        wait = wait+1
    else:
        is_stop = False
        wait = wait+1

    
    return best_val_loss, wait, best_weights,is_stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # TODO: Build the initial training-state dictionary for the GD epoch loop.
    init_weights = initialize_weights(n_features, seed)
    return {"weights":init_weights, "best_weights":init_weights.copy(), "best_val_loss":np.inf, "wait":0, "train_losses": [], "val_losses":[], "stopped":False}

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # TODO: Take one GD step, log train/val losses, refresh early-stopping fields...
    new_weights = gd_step(X_train, y_train, state['weights'], lr)
    state['weights'] = new_weights
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, new_weights)
    best_val_loss, wait, best_weights,is_stop = update_early_stop_state(val_loss, state['best_val_loss'], state['wait'], new_weights, state['best_weights'], patience)
    state['best_val_loss'] = best_val_loss
    state['wait'] = wait
    state['best_weights'] = best_weights
    state['stopped'] = is_stop
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)

    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # TODO: Train weights with full-batch GD for up to epochs, with early stopping.
    
    state = init_training_state(X_train.shape[1], seed)
    for epoch in range(epochs):
        if not state['stopped']:
            state = run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience)

    return state['best_weights'], state['train_losses'], state['val_losses']

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: Compute the mean absolute error between true targets and predictions
    return np.mean(np.abs(y_true - y_pred))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # TODO: Return the root mean squared error between y_true and y_pred.
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Step 20 - r_squared
import numpy as np

def r_squared(y_true, y_pred):
    # TODO: Compute the coefficient of determination R^2.
    RSS = np.sum((y_true - y_pred)**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    
    # Если изменчивости нет (TSS == 0), тесты требуют вернуть NaN
    if TSS == 0:
        return np.nan
        
    return 1 - RSS / TSS

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, and R^2 into one metrics dictionary for test-set reporting.
    return {'mae':mean_absolute_error(y_true, y_pred), 'rmse':root_mean_squared_error(y_true, y_pred), 'r2':r_squared(y_true, y_pred)}

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):
    # TODO: Return epoch indices and loss series for external plotting...
    epoch_indices = np.arange(len(train_losses)) + 1
    if isinstance(train_losses, np.ndarray):
        train_losses = train_losses.tolist()
    if isinstance(val_losses, np.ndarray):
        val_losses = val_losses.tolist()
    return epoch_indices.tolist(),train_losses, val_losses

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    # TODO: Compute the L2 distance between two weight vectors
    return np.sqrt(np.sum((w_gd - w_closed)**2))

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    # Настраиваем начальный словарь модели в стиле LinearRegressionGD
    model_state = {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience,
        'seed': seed,
        'weights': None,
        'normal_weights': None,
        'mean': None,
        'std': None,
        'train_losses': [],
        'val_losses': []
    }
    return model_state

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    # TODO: Fit model with train stats, design matrices, GD, and normal eq
    mean, std = compute_feature_stats(X_train)
    model['mean'] = mean
    model['std'] = std
    X_new = standardize_features(X_train, model['mean'], model['std'])
    X_new = add_bias_column(X_new)
    X_new_val = standardize_features(X_val, model['mean'], model['std'])
    X_new_val = add_bias_column(X_new_val)
    best_weights, train_losses, val_losses = train_batch_gd(X_new, y_train, X_new_val, y_val, model['learning_rate'], model['epochs'], model['patience'], model['seed'])
    model['weights'] = best_weights
    model['train_losses'] = train_losses
    model['val_losses'] = val_losses
    model['normal_weights'] =normal_equation(X_new, y_train)
    return model

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    # TODO: Return predicted targets for raw X using the fitted model.
    mean, std = compute_feature_stats(X)
    X_new = standardize_features(X, model['mean'], model['std'])
    X_new = add_bias_column(X_new)
    return X_new @ model['weights']

# Step 27 - score_lr_model
import numpy as np
def score_lr_model(model, X, y):
    # TODO: Predict on raw features and return MAE, RMSE, and R^2 metrics.
    y_pred = predict_lr_model(model, X)
    return evaluate_regression(y_pred, y)

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

