import numpy as np

def measure_mses(betas, mus, X):
    """Measures MSEs for each sample given regression coefficients and data.

    Args:
        betas (np.ndarray): Tensor of shape (n_samples, n_features, n_features
        mus (np.ndarray): Tensor of shape (n_samples, n_features, n_features)
        X (np.ndarray): Data matrix of shape (n_samples, n_features)

    Returns:
        np.ndarray: Array of shape (n_samples,) containing MSE for each sample.
    """
    mses = np.zeros(len(X))
    for i in range(len(X)):
        sample_mse = 0
        for j in range(X.shape[-1]):
            for k in range(X.shape[-1]):
                residual = X[i, j] - betas[i, j, k] * X[i, k] - mus[i, j, k]
                sample_mse += residual**2 / (X.shape[-1] ** 2)
        mses += sample_mse / len(X)
    return mses

def regression_to_correlation(betas):
        """Converts univariate regression coefficients to Pearson's correlation coefficients.

        Args:
            betas (np.ndarray): Tensor of shape (n_samples, n_features, n_features)

        Returns:
            np.ndarray: Tensor of shape (n_samples, n_features, n_features) containing Pearson's correlation coefficients.
        """
        beta_T = np.transpose(betas, (0, 2, 1))
        signs = np.sign(betas)
        signs[signs != np.sign(beta_T)] = 0  # should not have any mismatched signs, no negative squared correlations
        correlations = signs * np.sqrt(np.abs(betas * beta_T))
        return correlations