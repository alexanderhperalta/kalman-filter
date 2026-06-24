import numpy as np
from numba import njit
from py_modules.kf_core_functions import kf_update, kf_predict
import pandas as pd

@njit
def build_fourier_features(week_array, period=52, n_harmonics=3):
    n = len(week_array)
    fourier = np.zeros((n, 2 * n_harmonics))
    
    for h in range(1, n_harmonics + 1):
        for i in range(n):
            angle = 2.0 * np.pi * h * week_array[i] / period
            fourier[i, 2*(h-1)] = np.sin(angle)
            fourier[i, 2*(h-1)+1] = np.cos(angle)
    
    return fourier

@njit
def build_seasonal_matrices(Q_level, Q_trend, Q_seasonal, R_val):
    A = np.eye(8)
    A[0, 1] = 1.0
    Q = np.zeros((8, 8))
    Q[0, 0] = Q_level
    Q[1, 1] = Q_trend
    for i in range(2, 8):
        Q[i, i] = Q_seasonal
    R = np.zeros((1, 1))
    R[0, 0] = R_val
    return A, Q, R

@njit
def neg_log_likelihood_seasonal(params, measurements, fourier, covid_mask):
    Q_level = np.exp(params[0])
    Q_trend = np.exp(params[1])
    Q_seasonal_pre = np.exp(params[2])
    Q_seasonal_post = np.exp(params[3])
    R_val = np.exp(params[4])
    alpha = 1.0 / (1.0 + np.exp(-params[5]))  # sigmoid to keep in (0, 1)

    n = len(measurements)
    xk = np.zeros((8, 1))
    xk[0, 0] = measurements[0]
    Pk = np.eye(8) * 10.0
    R_k = R_val  # initialize adaptive R
    nll = 0.0

    for k in range(1, n):
        Q_seasonal_k = Q_seasonal_post if covid_mask[k] else Q_seasonal_pre
        A, Q, R = build_seasonal_matrices(Q_level, Q_trend, Q_seasonal_k, R_k)

        H_k = np.zeros((1, 8))
        H_k[0, 0] = 1.0
        H_k[0, 2:] = fourier[k]

        xk_pred, Pk_pred = kf_predict(A, xk, Pk, Q)
        xk, Pk, v_k, S_k = kf_update(xk_pred, Pk_pred, measurements[k], H_k, R)
        nll += 0.5 * (np.log(S_k) + (v_k ** 2) / S_k)

        # Anchored decay — spikes up, always returns toward R_val
        R_k = alpha * (v_k ** 2) + (1.0 - alpha) * R_val

    return nll

def run_seasonal_filter(params, measurements, weeks, dates, xk_init=None, Pk_init=None):
    Q_level, Q_trend, Q_seasonal_pre, Q_seasonal_post, R_val, alpha = params
    covid_start = pd.Timestamp('2020-03-11')
    
    fourier = build_fourier_features(weeks)
    
    if xk_init is not None:
        xk = xk_init
        Pk = Pk_init
    else:
        xk = np.zeros((8, 1))
        xk[0, 0] = measurements[0]
        Pk = np.eye(8) * 10.0

    R_k = R_val
    states, trends, seasonal, upper, lower, innovations, nll_contribs = [], [], [], [], [], [], []
    xk_list, Pk_list, Pk_pred_list = [], [], []

    for k in range(len(measurements)):
        Q_seasonal_k = Q_seasonal_post if dates[k] >= covid_start else Q_seasonal_pre
        A, Q, R = build_seasonal_matrices(Q_level, Q_trend, Q_seasonal_k, R_k)

        H_k = np.zeros((1, 8))
        H_k[0, 0] = 1
        H_k[0, 2:] = fourier[k]

        xk_pred, Pk_pred = kf_predict(A, xk, Pk, Q)
        xk, Pk, v_k, S_k = kf_update(xk_pred, Pk_pred, measurements[k], H_k, R)

        states.append(xk[0, 0])
        trends.append(xk[1, 0])
        seasonal_k = sum(xk[2 + j, 0] * fourier[k, j] for j in range(6))
        seasonal.append(seasonal_k)
        sigma = np.sqrt((H_k @ Pk @ H_k.T + R)[0, 0])
        upper.append((H_k @ xk)[0, 0] + 1.5 * sigma)
        lower.append((H_k @ xk)[0, 0] - 1.5 * sigma)
        innovations.append(v_k / np.sqrt(S_k))
        nll_contribs.append(0.5 * (np.log(S_k) + (v_k ** 2) / S_k).item())

        xk_list.append(xk.copy())
        Pk_list.append(Pk.copy())
        Pk_pred_list.append(Pk_pred.copy())

        R_k = alpha * (v_k ** 2) + (1.0 - alpha) * R_val

    return (np.array(states), np.array(trends), np.array(seasonal),
            np.array(upper), np.array(lower), xk, Pk, np.array(innovations),
            xk_list, Pk_list, Pk_pred_list, np.array(nll_contribs))