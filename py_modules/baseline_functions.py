import numpy as np
from py_modules.kf_core_functions import kf_update, kf_predict

def build_baseline_matrices(A_val, Q_val, R_val):
    A = np.array([[A_val]])
    H = np.array([[1.0]])
    Q = np.array([[Q_val]])
    R = np.array([[R_val]])
    return A, H, Q, R

def neg_log_likelihood_baseline(params, measurements, A_val):
    Q_val, R_val = np.exp(params)
    A, H, Q, R = build_baseline_matrices(A_val, Q_val, R_val)
    
    xk = np.array([[measurements[0]]])
    Pk = np.array([[1.0]])
    nll = 0.0
    
    for k in range(1, len(measurements)):
        xk_pred, Pk_pred = kf_predict(A, xk, Pk, Q)
        xk, Pk, v_k, S_k = kf_update(xk_pred, Pk_pred, measurements[k], H, R)
        nll += 0.5 * (np.log(S_k) + (v_k ** 2) / S_k)
    
    return nll

def run_baseline_filter(params, measurements, xk_init=None, Pk_init=None):
    A_val, Q_val, R_val = params
    A, H, Q, R = build_baseline_matrices(A_val, Q_val, R_val)
    
    if xk_init is not None:
        xk = xk_init
        Pk = Pk_init
    else:
        xk = np.array([[measurements[0]]])
        Pk = np.array([[1.0]])

    states, upper, lower, xk_rts, Pk_rts, Pk_bars, innovations, nll_contribs = [], [], [], [], [], [], [], []

    for k in range(len(measurements)):
        xk_pred, Pk_pred = kf_predict(A, xk, Pk, Q)
        xk, Pk, v_k, S_k = kf_update(xk_pred, Pk_pred, measurements[k], H, R)

        states.append(xk.item())
        sigma = np.sqrt(Pk.item())
        upper.append(xk.item() + 2 * sigma)
        lower.append(xk.item() - 2 * sigma)
        xk_rts.append(xk)
        Pk_rts.append(Pk)
        Pk_bars.append(Pk_pred)
        innovations.append(v_k / np.sqrt(S_k))
        nll_contribs.append(0.5 * (np.log(S_k) + (v_k ** 2) / S_k).item())

    return np.array(states), np.array(upper), np.array(lower), xk, Pk, xk_rts, Pk_rts, Pk_bars, np.array(innovations), np.array(nll_contribs)