import numpy as np
from py_modules.kf_core_functions import kf_update, kf_predict

def build_llt_matrices(Q_level, Q_trend, R_val):

    A = np.array([[1.0, 1.0],
                [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.diag([Q_level, Q_trend])
    R = np.array([[R_val]])

    return A, H, Q, R

def neg_log_likelihood_llt(params, measurements):
    Q_level, Q_trend, R_val = np.exp(params)
    
    A, H, Q, R = build_llt_matrices(Q_level, Q_trend, R_val)
    
    xk = np.array([[measurements[0]], [0.0]])  # [initial level, initial trend = 0]
    Pk = np.eye(2) * 10.0
    nll = 0.0
    
    for k in range(1, len(measurements)):
        xk_pred, Pk_pred = kf_predict(A, xk, Pk, Q)
        xk, Pk, v_k, S_k = kf_update(xk_pred, Pk_pred, measurements[k], H, R)
        nll += 0.5 * (np.log(S_k) + (v_k ** 2) / S_k)
    
    return nll

def run_llt_filter(params, measurements, xk_init=None, Pk_init=None):
    Q_level, Q_trend, R_val = params
    
    A, H, Q, R = build_llt_matrices(Q_level, Q_trend, R_val)

    if xk_init is not None:
        xk = xk_init
        Pk = Pk_init
    else:
        xk = np.array([[measurements[0]], [0.0]])
        Pk = np.eye(2) * 10.0
    
    states, trends, upper, lower, xk_rts, Pk_rts, Pk_bars, innovations, nll_contribs = [], [], [], [], [], [], [], [], []

    for k in range(len(measurements)):
        xk_pred, Pk_pred = kf_predict(A, xk, Pk, Q)
        xk, Pk, v_k, S_k = kf_update(xk_pred, Pk_pred, measurements[k], H, R)

        states.append(xk[0, 0])
        trends.append(xk[1, 0])
        sigma = np.sqrt(Pk[0, 0])
        upper.append(xk[0, 0] + 2 * sigma)
        lower.append(xk[0, 0] - 2 * sigma)
        xk_rts.append(xk)
        Pk_rts.append(Pk)
        Pk_bars.append(Pk_pred)
        innovations.append(v_k / np.sqrt(S_k))
        nll_contribs.append(0.5 * (np.log(S_k) + (v_k ** 2) / S_k).item())

    return np.array(states), np.array(trends), np.array(upper), np.array(lower), xk, Pk, xk_rts, Pk_rts, Pk_bars, np.array(innovations), np.array(nll_contribs)