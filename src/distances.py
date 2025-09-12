from scipy.stats import wasserstein_distance,entropy
from scipy.spatial.distance import cityblock 
import pandas as pd
import numpy as np
from scipy.special import rel_entr

@staticmethod
def hellinger(X: pd.DataFrame, x_ref: pd.Series, weight: pd.Series = None,normalize=False):
    epsilon = 1e-10 
    X_index = X.index
    X = X + epsilon
    x_ref = x_ref + epsilon
    if normalize:
        x_ref = x_ref.to_numpy()
        x_ref  = x_ref / x_ref.sum()

        X = X.to_numpy()
        X = X / X.sum(axis=1,keepdims=True)
    else:
        X=X.to_numpy()
        x_ref = x_ref.to_numpy()
        
    # Convert to numpy arrays for faster operations
    X_sqrt = np.sqrt(x_ref)
    x_ref_sqrt = np.sqrt(X)
    
    # Compute the difference only once
    diff = X_sqrt - x_ref_sqrt

    # Hellinger distance calculation based on whether weight is provided
    if weight is not None:
        weight = weight.to_numpy()  # Convert weight to numpy array
        result = np.sqrt(0.5 * np.sum(diff**2 * weight, axis=1))
    else:
        result = np.sqrt(0.5 * np.sum(diff**2, axis=1))
        
    return pd.Series(result, name="Hellinger distance",index=X_index)

def jensenshannon(p,q,base=None,axis=0,keepdims=False):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)


    
@staticmethod
def jsd(X: pd.DataFrame, x_ref: pd.Series,log_base=2,normalize=False):

    epsilon =  1e-9 # the extremely small value for adding for avoiding the zero division
    X = X + epsilon
    x_ref = x_ref + epsilon
    P = X.to_numpy()
    q = x_ref.to_numpy()
    
    if normalize:
        q = q / q.sum()
        js_divergences = np.array([
            jensenshannon(row / row.sum(), q , base=log_base) for row in P
        ])
    else:
        js_divergences = np.array([
            jensenshannon(row , q , base=log_base) for row in P
        ])
        
    return pd.Series(js_divergences,name = 'JS divergence',index=X.index)

@staticmethod
def emd(X: pd.DataFrame, x_ref: pd.Series, normalize=False):
    epsilon = 1e-10
    P = X.to_numpy() + epsilon
    x_ref = x_ref + epsilon
    if normalize:
        q = x_ref.to_numpy() / x_ref.sum()  # Normalize q

        # Calculate Earth Mover's Distance (Wasserstein distance) for each row in P
        emd_distances = np.array([
            wasserstein_distance(row / row.sum(), q) for row in P
        ])
    else:
        q = x_ref.to_numpy() 

        # Calculate Earth Mover's Distance (Wasserstein distance) for each row in P
        emd_distances = np.array([
            wasserstein_distance(row, q) for row in P
        ])
    return pd.Series(emd_distances, name='Earth Mover\'s Distance', index=X.index)

@staticmethod
def tvd(X: pd.DataFrame, x_ref: pd.Series,normalize=False):
    epsilon = 1e-10
    P = X.to_numpy() + epsilon
    x_ref = x_ref+epsilon
    if normalize:
        q = x_ref.to_numpy() / x_ref.sum()  # Normalize q

        # Calculate Total Variation Distance for each row in P
        tvd_distances = np.array([
            0.5 * cityblock(row / row.sum(), q) for row in P
        ])
    else:
        q = x_ref.to_numpy() 

        tvd_distances = np.array([
            0.5 * cityblock(row, q) for row in P
        ])
        
    return pd.Series(tvd_distances, name='Total Variation Distance', index=X.index) 