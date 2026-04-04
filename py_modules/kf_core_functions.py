from numba import njit
import numpy as np

@njit
def kf_predict(A, xk_lag, Pk_lag, Q):
    xk_pred = A @ xk_lag
    Pk_pred = A @ Pk_lag @ A.T + Q
    return xk_pred, Pk_pred

@njit
def kf_update(xk_pred, Pk_pred, z_k, H, R):
    v_k = z_k - (H @ xk_pred)[0, 0]
    S_k = (H @ Pk_pred @ H.T + R)[0, 0]
    K_k = Pk_pred @ H.T * (1.0 / S_k)
    xk = xk_pred + K_k * v_k
    I = np.eye(xk.shape[0])
    Pk = (I - K_k @ H) @ Pk_pred
    return xk, Pk, v_k, S_k