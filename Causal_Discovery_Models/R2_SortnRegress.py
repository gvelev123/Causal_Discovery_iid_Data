import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC
from scipy import linalg


def r2coeff(X):
    r"""
    Compute the :math:`R^2` of each variable using partial correlations obtained through matrix inversion.

    Args:
        X: Data (:math:`d \times n` np.array - note that the dimensions here are different from other methods, following np.corrcoef).

    Returns: 
        Array of :math:`R^2` values for all variables.
    """
    try:
        return 1 - 1/np.diag(linalg.inv(np.corrcoef(X)))
    except linalg.LinAlgError:
        # fallback if correlation matrix is singular
        d = X.shape[0]
        r2s = np.zeros(d)
        LR = LinearRegression()
        X = X.T
        for k in range(d):
            parents = np.arange(d) != k
            LR.fit(X[:, parents], X[:, k])
            r2s[k] = LR.score(X[:, parents], X[:, k])
        return r2s

def sort_regress(X, scores):
    """
    Regress each variable onto all predecessors in
    the ordering implied by the scores.

    Args:
        X: Data (:math:`n \times d` np.array).
        scores: Vector of scores (np.array with :math:`d` entries).

    Returns:
        Candidate causal structure matrix with coefficients
    """
    LR = LinearRegression()
    LL = LassoLarsIC(criterion='bic')
    d = X.shape[1]
    W = np.zeros((d, d))
    ordering = np.argsort(scores)

    # backward regression
    for k in range(1, d):
        cov = ordering[:k]
        target = ordering[k]
        LR.fit(X[:, cov], X[:, target].ravel())
        weight = np.abs(LR.coef_)
        LL.fit(X[:, cov] * weight, X[:, target].ravel())
        W[cov, target] = LL.coef_ * weight
    return W


def r2_sort_regress(X):
    r"""
    Perform sort_regress using :math:`R^2` as ordering criterion.

    Args:
        X: Data (:math:`n \times d` np.array).
    
    Returns:
        Candidate causal structure matrix with coefficients.
    """
    return sort_regress(X, r2coeff(X.T))