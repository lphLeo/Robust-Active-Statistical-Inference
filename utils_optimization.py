import numpy as np
import torch
from typing import Tuple
import xgboost as xgb
from training import SimpleNN, train_nn, train_tree, train_tree_2
from scipy.stats import bernoulli
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import DataLoader, TensorDataset
from utils import ols, active_logistic_pointestimate, logistic_cov

class MinMaxOptimizer_l2:
    def __init__(self, e_hat: np.ndarray, pi: np.ndarray, nb: float, n: int, constraint_sum: float):
        """
        Initialize the optimizer with known parameters
        
        Args:
            e_hat: Array of error terms
            pi: Array of pi values
            nb: nb parameter
            n: Number of samples
            constraint_sum: Sum constraint for delta
        """
        self.e_hat = e_hat
        self.pi = pi
        self.nb = nb
        self.n = n
        self.constraint_sum = constraint_sum
        self.n_nonzero = np.sum(self.pi != 0)
        self.nonzero_mask = self.pi != 0
        
    def objective_function(self, r: float) -> float:
        ratio = self.nb / self.n
        denominator = self.pi ** (1 - r) * ratio ** r / np.sum(self.pi ** (1 - r) * ratio ** r) * self.nb
        denominator = np.clip(denominator, 0, 1)
        terms = (self.e_hat[self.nonzero_mask]**2) / denominator[self.nonzero_mask]
        return (np.sum(terms)  + self.constraint_sum * np.sqrt(np.sum(1/denominator[self.nonzero_mask]**2))) / self.n_nonzero if self.n_nonzero > 0 else 0
    
    def optimize(self, r_bounds: Tuple[float, float] = (0, 1), r_steps: int = 100) -> Tuple[float, float, np.ndarray]:
        """
        Solve the optimization problem
        
        Args:
            r_bounds: Tuple of (min_r, max_r) to search
            r_steps: Number of steps for r grid search
            
        Returns:
            Tuple of (optimal_value, optimal_r, optimal_delta)
        """
        r_values = np.linspace(r_bounds[0], r_bounds[1], r_steps)

        best_value = float('inf')
        best_r = None
        obj_values = []
        for r in r_values:
            value = self.objective_function(r)
            obj_values.append(value)
            if value < best_value:
                best_value = value
                best_r = r
                
        return best_value, best_r
    

def process_fold(args):
    """Process a single fold of cross-validation"""
    X_train, Y_train, X_test, Y_test, model, bg, c, device = args
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Get predictions
    Yhat = model.predict(xgb.DMatrix(X_test))
    error_train = np.abs(Y_train - model.predict(xgb.DMatrix(X_train)))
    error_train_tensor = torch.tensor(error_train, dtype=torch.float32).view(-1,1).to(device)
    
    dataset = TensorDataset(X_train_tensor, error_train_tensor)
    loader = DataLoader(dataset, batch_size=min(256, len(X_train)), shuffle=True)
    
    # Train neural network
    nn_ = SimpleNN(X_train_tensor.shape[1]).to(device)
    train_nn(nn_, loader, epochs=1000, lr=0.01)
    
    # Get uncertainty predictions
    with torch.no_grad():
        uncertainty = np.abs(nn_(X_test_tensor).cpu().numpy().flatten())
    
    uncertainty = np.clip(uncertainty, 0, 1)
    n = len(Y_test)
    nb = bg * n
    error = uncertainty.copy()
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimal_value, optimal_r = optimizer.optimize()
    
    # Vectorized Monte Carlo trials
    probs = np.clip((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r / 
                    np.sum((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r) * bg * n, 
                    0.0, 1.0)
    
    num_trials = 200
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    probs_ = np.clip(probs, 0.0001, 1.0)
    
    # Vectorized computation of active robust labels
    active_robust_labels = Yhat + (Y_test - Yhat) * (xi / probs_)
    std = np.mean(np.std(active_robust_labels, axis=1))
    
    return std

def constraint_cross_validation(X_bi: np.ndarray, Y_bi: np.ndarray, model, cv_list: np.ndarray, k: int, budgets: np.ndarray, device: torch.device) -> np.ndarray:
    
    optimal_c_list = []
    n_samples = len(X_bi)
    fold_size = n_samples // k
    
    # Create all fold indices once
    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            # Prepare arguments for parallel processing
            process_args = []
            for train_idx, test_idx in fold_indices:
                X_train = X_bi[train_idx]
                Y_train = Y_bi[train_idx]
                X_test = X_bi[test_idx]
                Y_test = Y_bi[test_idx]
                process_args.append((X_train, Y_train, X_test, Y_test, model, bg, c, device))
            
            # Process folds in parallel
            with Pool() as pool:
                fold_results = pool.map(process_fold, process_args)
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list

def process_fold_regression(args):
    """Process a single fold of cross-validation"""
    X_train, income_features_unlabeled_train, Y_train, X_test, income_features_unlabeled_test, Y_test, model, bg, c, Hessian_inv, enc, device = args

    Yhat = model.predict(xgb.DMatrix(income_features_unlabeled_test))
    # Get predictions

    error_train = np.abs(Y_train - model.predict(xgb.DMatrix(income_features_unlabeled_train)))
    xgb_err = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=20)
    xgb_err.fit(X_train, error_train)
    predicted_errs = np.clip(np.abs(xgb_err.predict(X_test)), 0, np.inf)
    h = Hessian_inv[:,0]
    uncertainty = np.abs(h.dot(X_test.T)) * predicted_errs
    
    uncertainty = np.clip(uncertainty, 0, 1)
    n = len(Y_test)
    nb = bg * n
    error = uncertainty.copy()
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimal_value, optimal_r = optimizer.optimize()
    
    # Vectorized Monte Carlo trials
    probs = np.clip((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r / 
                    np.sum((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r) * bg * n, 
                    0.0, 1.0)
    
    num_trials = 200
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    
    # Vectorized computation of active robust labels
    active_robust_labels = Yhat + (Y_test - Yhat) * (xi / probs)
    
    # Compute OLS for each trial
    pointest_active_robust = np.array([ols(X_test, active_robust_labels[i]) for i in range(num_trials)])
    grads = np.array([(np.dot(X_test, pointest_active_robust[i]) - active_robust_labels[i])[:, np.newaxis] * X_test for i in range(num_trials)])

    # Compute covariance for each trial
    V = np.array([np.cov(grads[i].T) for i in range(num_trials)])
    Sigma_active_robust = np.array([Hessian_inv @ V[i] @ Hessian_inv for i in range(num_trials)])
    std = np.mean(np.sqrt(Sigma_active_robust[:, 0, 0]))
    
    return std



def constraint_cross_validation_regression(X_bi, income_features_unlabeled_bi: np.ndarray, Y_bi: np.ndarray, model, cv_list: np.ndarray, k: int, budgets: np.ndarray, Hessian_inv: np.ndarray, enc, device: torch.device) -> np.ndarray:

    optimal_c_list = []
    n_samples = len(Y_bi)
    fold_size = n_samples // k
    
    # Create all fold indices once
    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            # Prepare arguments for parallel processing
            process_args = []
            for train_idx, test_idx in fold_indices:
                X_train = X_bi[train_idx]
                income_features_unlabeled_train = income_features_unlabeled_bi[train_idx]
                Y_train = Y_bi[train_idx]
                X_test = X_bi[test_idx]
                income_features_unlabeled_test = income_features_unlabeled_bi[test_idx]
                Y_test = Y_bi[test_idx]
                process_args.append((X_train, income_features_unlabeled_train, Y_train, X_test, income_features_unlabeled_test, Y_test, model, bg, c, Hessian_inv, enc, device))
            
            # Process folds in parallel
            with Pool() as pool:
                fold_results = pool.map(process_fold_regression, process_args)
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list


def process_fold_bias(args):
    """Process a single fold of cross-validation"""
    X_train, Y_train, Yhat_train, X_test, Y_test, Yhat_test, bg, c, device = args
    
    # Get predictions
    error_train = np.abs(Y_train - Yhat_train)
    tree = train_tree_2(X_train, error_train)
    
    error = tree.predict(X_test)
    uncertainty = (1 - 2 * np.abs(X_test - 0.5)).reshape(-1)

    n = len(Y_test)
    nb = bg * n
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimal_value, optimal_r = optimizer.optimize()
    
    # Vectorized Monte Carlo trials
    probs = np.clip((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r / 
                    np.sum((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r) * bg * n, 
                    0.0, 1.0)
    
    num_trials = 200
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    probs_ = np.clip(probs, 0.0001, 1.0)
    
    # Vectorized computation of active robust labels
    active_robust_labels = Yhat_test + (Y_test - Yhat_test) * (xi / probs_)
    std = np.mean(np.std(active_robust_labels, axis=1))
    
    return std

def constraint_cross_validation_bias(X_bi: np.ndarray, Y_bi: np.ndarray, Yhat_bi: np.ndarray, cv_list: np.ndarray, k: int, budgets: np.ndarray, device: torch.device) -> np.ndarray:
    
    optimal_c_list = []
    n_samples = len(X_bi)
    fold_size = n_samples // k
    
    # Create all fold indices once
    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            process_args = []
            for train_idx, test_idx in fold_indices:
                X_train = X_bi[train_idx]
                Y_train = Y_bi[train_idx]
                Yhat_train = Yhat_bi[train_idx]
                X_test = X_bi[test_idx]
                Y_test = Y_bi[test_idx]
                Yhat_test = Yhat_bi[test_idx]
                process_args.append((X_train, Y_train, Yhat_train, X_test, Y_test, Yhat_test, bg, c, device))
            
            # Process folds in parallel
            with Pool() as pool:
                fold_results = pool.map(process_fold_bias, process_args)
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list


def process_fold_politeness(args):
    """Process a single fold of cross-validation"""
    confidence_train, X_train, Y_train, Yhat_train, confidence_test, X_test, Y_test, Yhat_test, bg, c, h, device = args
    
    # Get predictions
    error_train = np.abs(Y_train - Yhat_train)
    tree = train_tree_2(confidence_train, error_train**2)
    
    error = np.sqrt(tree.predict(confidence_test)) * np.abs(X_test.dot(h))
    uncertainty = (1 - 2 * np.abs(confidence_test - 0.5)).reshape(-1)

    n = len(Y_test)
    nb = bg * n
    eta = bg / np.mean(uncertainty)
    pi = eta * uncertainty
    pi = np.clip(pi, 0, 1)
    constraint_sum = c * np.sqrt(n)
    
    # Optimize
    optimizer = MinMaxOptimizer_l2(error, pi, nb, n, constraint_sum)
    optimal_value, optimal_r = optimizer.optimize()
    
    # Vectorized Monte Carlo trials
    probs = np.clip((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r / 
                    np.sum((eta * uncertainty) ** (1-optimal_r) * bg ** optimal_r) * bg * n, 
                    0.0, 1.0)
    
    num_trials = 200
    # Generate all random samples at once
    xi = bernoulli.rvs(probs, size=(num_trials, len(probs)))
    probs_ = np.clip(probs, 0.0001, 1.0)
    
    # Compute std across trials
    stds = []
    for i in range(num_trials):
        # Use single trial weights
        weights = xi[i] / probs_
        pointest = active_logistic_pointestimate(X_test, Y_test, Yhat_test, weights, 1)
        Sigmahat = logistic_cov(pointest, X_test, Y_test, Yhat_test, weights, 1)
        stds.append(np.sqrt(Sigmahat[0, 0]))
    
    std = np.mean(stds)
    
    return std

def constraint_cross_validation_politeness(confidence_bi: np.ndarray, X_bi: np.ndarray, Y_bi: np.ndarray, Yhat_bi: np.ndarray, cv_list: np.ndarray, k: int, budgets: np.ndarray, h: np.ndarray, device: torch.device) -> np.ndarray:
    
    optimal_c_list = []
    n_samples = len(confidence_bi)
    fold_size = n_samples // k

    fold_indices = []
    for j in range(k):
        test_indices = np.arange(j * fold_size, (j + 1) * fold_size)
        train_indices = np.concatenate([
            np.arange(0, j * fold_size),
            np.arange((j + 1) * fold_size, n_samples)
        ])
        fold_indices.append((train_indices, test_indices))
    
    for bg in budgets:
        cv_error_list = []
        
        for c in tqdm(cv_list, desc=f"Testing constraints (bg={bg:.2f})", leave=False):
            process_args = []
            for train_idx, test_idx in fold_indices:
                confidence_train = confidence_bi[train_idx]
                X_train = X_bi[train_idx]
                Y_train = Y_bi[train_idx]
                Yhat_train = Yhat_bi[train_idx]
                confidence_test = confidence_bi[test_idx]
                X_test = X_bi[test_idx]
                Y_test = Y_bi[test_idx]
                Yhat_test = Yhat_bi[test_idx]
                process_args.append((confidence_train, X_train, Y_train, Yhat_train, confidence_test, X_test, Y_test, Yhat_test, bg, c, h, device))
            
            # Process folds in parallel
            with Pool() as pool:
                fold_results = pool.map(process_fold_politeness, process_args)
            
            cv_error = sum(fold_results)
            cv_error_list.append(cv_error)
            
        optimal_c = cv_list[np.argmin(cv_error_list)]
        optimal_c_list.append(optimal_c)
        
    return optimal_c_list