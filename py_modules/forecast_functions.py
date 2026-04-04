import numpy as np
from py_modules.kf_core_functions import kf_update, kf_predict
from py_modules.seasonal_functions import build_fourier_features

def forecast(params, n_steps):
    A, Q, H, xk, Pk = params
    forecasts = []
    uppers = []
    lowers = []
    
    for k in range(n_steps):
        xk, Pk = kf_predict(A, xk, Pk, Q)
        forecast_val = (H @ xk).item()
        sigma = np.sqrt((H @ Pk @ H.T).item())
        forecasts.append(forecast_val)
        uppers.append(forecast_val + 2 * sigma)
        lowers.append(forecast_val - 2 * sigma)
    
    return np.array(forecasts), np.array(uppers), np.array(lowers)

def forecast_seasonal(params, weeks, n_steps):
    A, Q, R, xk, Pk = params
    fourier = build_fourier_features(weeks)
    
    forecasts, uppers, lowers = [], [], []
    
    for k in range(n_steps):
        xk, Pk = kf_predict(A, xk, Pk, Q)
        
        H_k = np.zeros((1, 8))
        H_k[0, 0] = 1
        H_k[0, 2:] = fourier[k]
        
        forecast_val = (H_k @ xk).item()
        sigma = np.sqrt((H_k @ Pk @ H_k.T).item())
        forecasts.append(forecast_val)
        uppers.append(forecast_val + 2 * sigma)
        lowers.append(forecast_val - 2 * sigma)
    
    return np.array(forecasts), np.array(uppers), np.array(lowers)
    