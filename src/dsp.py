
import numpy as np
from scipy import signal
from scipy import linalg
import numpy as np


def ssd(Cband, Cnoise):
    """
    This fuction performs broadband spatio-spectral decomposition (SSD) on input signal
    Args:
        Cband: covariance matrix of signal at frequency band of interest
        Cnoise: covariance matrix of signal at noise frequency band

    Returns:
    """
    D, V = linalg.eigh(Cband)
    indx_descend = np.argsort(D)[::-1]
    V = V[:, indx_descend]
    D = D[indx_descend]

    # Estimate rank of covariance matrix
    tol = D[0] * 1e-6
    k = np.sum(D > tol)
    if(k < Cnoise.shape[0]):
        M = V[:, 0:k] @ np.diag(D[0:k]**(-0.5))
    else:
        M = np.eye(Cnoise.shape[0])

    Cband_r = M.T @ Cband @ M
    Cnoise_r = M.T @ Cnoise @ M

    # Generalized eigenvalue decomposition
    # Here I assume that the covariance matrix is positive definite.
    #D, W = linalg.eig(Cband_r, Cnoise_r)
    D, W = linalg.eigh(Cband_r, Cnoise_r)
    indx_descend = np.argsort(D)[::-1]
    W = W[:, indx_descend]
    W = M @ W

    return W


