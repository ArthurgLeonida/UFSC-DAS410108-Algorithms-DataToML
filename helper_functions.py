from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mutual_info_score
import numpy as np
import pandas as pd

############################ DEPENDENCE MEASURES ############################

def mutual_information(x, y, bins=10):
    """Calculates mutual information using the standard sklearn method."""
    # Discretize continuous variable x before calculating MI
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        x_binned = pd.cut(x, bins=bins, labels=False, duplicates='drop')
    else:
        x_binned = x
        
    # Use the direct function from scikit-learn
    return mutual_info_score(y, x_binned)

def hsic(X, Y, sigma_X=1.0, sigma_Y=1.0):
    """Calculates the Hilbert-Schmidt Independence Criterion."""
    n = len(X)
    if n < 2: return 0.0
    X, Y = np.asarray(X).reshape(-1, 1), np.asarray(Y).reshape(-1, 1)
    K, L = rbf_kernel(X, gamma=1/(2*sigma_X**2)), rbf_kernel(Y, gamma=1/(2*sigma_Y**2))
    H = np.eye(n) - (1/n) * np.ones((n,n))
    Kc, Lc = H @ K @ H, H @ L @ H
    return np.trace(Kc @ Lc) / ((n - 1)**2)

############################ BINNING STRATEGIES ############################

def knuth_rule(x):
    """Knuth's rule for optimal binning."""
    n = len(x)
    
    def cost_function(k):
        if k < 2:
            return np.inf
        
        hist, _ = np.histogram(x, bins=int(k))
        hist = hist[hist > 0]  # Remove empty bins
        
        if len(hist) < 2:
            return np.inf
        
        # Log-likelihood
        log_likelihood = np.sum(hist * np.log(hist / n))
        
        # BIC penalty
        penalty = 0.5 * (k - 1) * np.log(n)
        
        return -log_likelihood + penalty
    
    # Search for optimal k
    k_range = range(2, min(50, n//5))
    costs = [cost_function(k) for k in k_range]
    return k_range[np.argmin(costs)]

def freedman_diaconis_rule(x):
    n = len(x)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    bin_width = 2 * iqr / (n ** (1/3))
    bins = int((np.max(x) - np.min(x)) / bin_width)
    return max(2, min(bins, n//5))

def adaptive_binning(x):
    """Chooses a binning strategy based on data characteristics."""
    x = x.dropna()
    n = len(x)
    x_unique = x.nunique()
    
    # Logic is the same, but only for x
    if x_unique <= 10:
        bins_x = x_unique
    elif n < 50:
        bins_x = int(np.sqrt(n))
    elif n < 200:
        bins_x = freedman_diaconis_rule(x)
    else:
        bins_x = knuth_rule(x)
    
    # Return only the bins for x
    return max(2, min(bins_x, 50))

def robust_outlier_detection(x):
    """Detects outliers using multiple methods and returns a consensus."""
    # Method 1: Z-score (assumes normality)
    z_outliers = abs((x - x.mean()) / x.std()) > 3
    
    # Method 2: IQR method (non-parametric)
    Q1, Q3 = x.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    iqr_outliers = (x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))
    
    # Method 3: Percentile method (robust to distribution shape)
    p5, p95 = x.quantile([0.05, 0.95])
    percentile_outliers = (x < p5) | (x > p95)
    
    # Consensus approach: flagged by multiple methods
    consensus_outliers = z_outliers & iqr_outliers
    return consensus_outliers