
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def k1_butter_bandpass(flc, fhc, fs, order, plot=False):
    nyq = 0.5 * fs
    low = flc / nyq
    high = fhc / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    if plot:
        k1_freqz(b, a)
    return b, a

def k1_butter_lowpass(fc, fs, order, plot=False):
    nyq = 0.5 * fs
    fcutoff = fc / nyq
    b, a = signal.butter(order, [fcutoff], btype='low')
    if plot:
        k1_freqz(b, a)
    return b, a


def k1_butter_highpass(fc, fs, order, plot=False):
    nyq = 0.5 * fs
    fcutoff = fc / nyq
    b, a = signal.butter(order, [fcutoff], btype='high')
    if plot:
        k1_freqz(b, a)
    return b, a


def k1_freqz(b, a, fs, worN=2000):
    w, h = signal.freqz(b, a, worN=2000)
    f = (fs/(2*np.pi)) * w # rad/sec to Hz conversion
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(f, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [Hz]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(f, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')


def k1_freqz_old(b, a, worN=2000):
    w, h = signalfreqz(b, a, worN=2000)
    f = (fs/(2*np.pi)) * w # rad/sec to Hz conversion
    plt.plot(f, abs(h))


def k1_ged(Cband, Cbroad):

    from scipy import linalg
    import numpy as np

    D, V = linalg.eigh(Cband)
    indx_descend = np.argsort(D)[::-1]
    V = V[:, indx_descend]
    D = D[indx_descend]


    # Estimate rank of covariance matrix
    tol = D[0] * 1e-6
    k = np.sum(D > tol)
    if(k < Cbroad.shape[0]):
        M = V[:, 0:k] @ np.diag(D[0:k]**(-0.5))
    else:
        M = np.eye(Cbroad.shape[0])

    Cband_r = M.T @ Cband @ M
    Cbroad_r = M.T @ Cbroad @ M

    #breakpoint()
    # Generalized eigenvalue decomposition
    # Here I assume that the covariance matrix is positive definite.
    #D, W = linalg.eig(Cband_r, Cbroad_r)
    D, W = linalg.eigh(Cband_r, Cbroad_r)
    indx_descend = np.argsort(D)[::-1]
    W = W[:, indx_descend]
    W = M @ W

    return W


def k1_robust_pca(C):
    """
    This function performs PCA on covariance matrix C, which is not full rank.
    :param C:  covraince matrix obtained from multiple time series. Dimension: N x N
    :return: V_r: Robust principal components. Dimension: N x N

    Example: Given times series TS (Dimension: N x T), find the 1st principal component:
    TS -= TS.mean(axis=1).reshape((-1, 1))
    C = np.cov(TS)
    V = k1_pca(C)
    TS_pca_1stcomp = V[:, 1] * TS
    """

    from scipy import linalg
    import numpy as np

    D, V = linalg.eigh(C)
    indx_descend = np.argsort(D)[::-1]
    V = V[:, indx_descend]
    D = D[indx_descend]

    # Estimate rank of covariance matrix
    tol = D[0] * 1e-6
    k = np.sum(D > tol)
    if k < C.shape[0]:
        M = V[:, 0:k] @ np.diag(D[0:k] ** -0.5)
    else:
        M = np.eye(C.shape[0])

    C_r = M.T @ C @ M
    D_r, V_r = linalg.eigh(C_r)
    indx_descend = np.argsort(D_r)[::-1]
    V_r = V_r[:, indx_descend]
    V_r = M @ V_r

    return V_r



def k1_pca(C):
    """
    This function performs PCA on covariance matrix C, which is not full rank.
    :param C:  covraince matrix obtained from multiple time series. Dimension: N x N
    :return: V_r: Robust principal components. Dimension: N x N

    Example: Given times series TS (Dimension: N x T), find the 1st principal component:
    TS -= TS.mean(axis=1).reshape((-1, 1))
    C = np.cov(TS)
    V = k1_pca(C)
    TS_pca_1stcomp = V[:, 1] * TS
    """

    from scipy import linalg
    import numpy as np

    D, V = linalg.eigh(C)
    indx_descend = np.argsort(D)[::-1]
    V = V[:, indx_descend]
    D = D[indx_descend]

    return D, V