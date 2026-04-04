# kalman-filter

# A State-Space Approach to CDC Surveillance Data

Kalman filtering and smoothing applied to CDC flu positivity data for Illinois (October 2010 – March 2026). Three progressively richer state-space models are estimated via maximum likelihood, evaluated on held-out data, and used to generate 52-week-ahead forecasts.

## Motivation

CDC flu surveillance reports are noisy week-to-week. This project treats each weekly percent-positive reading as a noisy observation of a latent "true" flu signal and uses the Kalman filter to recover that signal, quantify uncertainty, and forecast future flu seasons.

## Data

Weekly flu percent-positive rates for Illinois sourced from the CDC's ILINet / NREVSS Clinical Labs system. The data spans roughly 800 weeks. An 80/20 chronological split places the test set starting in February 2023. A `log1p` transformation is applied to stabilize variance across seasons before filtering.

## Models

### Model 1: Scalar Random Walk

A single-state model where the hidden state evolves as $x_k = A \, x_{k-1} + w_k$. The autoregressive coefficient $A \approx 0.98$ is estimated via OLS; process noise $Q$ and observation noise $R$ are optimized by maximizing the Kalman filter log-likelihood with L-BFGS-B.

### Model 2: Local Linear Trend

Adds an explicit trend component so the state vector becomes $[\text{level}, \text{trend}]^\top$. MLE drives the trend variance to near zero, confirming that Illinois flu positivity does not exhibit a meaningful long-run drift. The model improves calibration over Model 1 (98.8% empirical coverage vs. 100% on a nominal 95% band).

### Model 3: Seasonal Structural Model

Extends the local linear trend with six Fourier harmonics (3 sine/cosine pairs at period = 52 weeks), giving an 8-dimensional state vector. Two additional features handle post-COVID regime change:

- **Split seasonal variance** — separate $Q_{\text{seasonal}}$ parameters before and after March 2020.
- **Adaptive observation noise** — $R_k = \alpha \, v_k^2 + (1 - \alpha) \, R$, allowing the filter to widen uncertainty during outbreak surges.

This model achieves the best-calibrated prediction intervals (98.1% coverage, 1.9% Ljung-Box failure rate) and is the only model that produces structurally informative forecasts with a visible seasonal shape.

## Forecasting

All three models generate 52-week-ahead forecasts by running the Kalman filter prediction equations beyond the last observation ($Z_t = 0$ for $t > n$). Only Model 3 produces a forecast that anticipates the timing and approximate magnitude of the next flu season.

## Rauch–Tung–Striebel Smoother

A backward pass (RTS smoother) is applied to the training data for each model, yielding smoothed state estimates with tighter uncertainty bands. Model 3's smoother produces the tightest bands, particularly during off-season periods, because the Fourier decomposition explains variance structurally.

## Project Structure

```
├── kalman_filter.ipynb          # Main notebook (analysis, plots, diagnostics)
├── py_modules/
│   ├── kf_core_functions.py     # Kalman predict & update (Numba-accelerated)
│   ├── baseline_functions.py    # Model 1: scalar random walk filter & MLE
│   ├── llt_functions.py         # Model 2: local linear trend filter & MLE
│   ├── seasonal_functions.py    # Model 3: seasonal structural filter & MLE
│   ├── forecast_functions.py    # Multi-step-ahead forecasting
│   └── coverage_analysis.py     # Empirical coverage calculation
```

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Numba
- scikit-learn
- statsmodels

Install with:

```bash
pip install numpy scipy pandas matplotlib numba scikit-learn statsmodels
```

## Usage

1. Place the CDC data CSVs (`ICL_NREVSS_Combined_prior_to_2015_16.csv` and `ICL_NREVSS_Clinical_Labs.csv`) in the project root.
2. Run the notebook end to end:

```bash
jupyter notebook kalman_filter.ipynb
```

The notebook trains on the first 80% of the data, evaluates on the remaining 20%, and generates 52-week forecasts beyond the test set.

## References

- Byrd, R.H., Lu, P., Nocedal, J., & Zhu, C. (1994). A Limited Memory Algorithm for Bound Constrained Optimization. *Northwestern University Technical Report*.
- Durbin, J., & Koopman, S.J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- Moré, J.J., & Thuente, D.J. (1990). On Line Search Algorithms with Guaranteed Sufficient Decrease. *Argonne National Laboratory Preprint MCS-P153-0590*.
- Miller, J.W. (2016). Kalman Filter and Smoother. *Lecture Notes, Duke University*.
- Rauch, H.E., Tung, F., & Striebel, C.T. (1965). Maximum Likelihood Estimates of Linear Dynamic Systems. *AIAA Journal*, 3(8), 1445–1450.
- Shumway, R.H., & Stoffer, D.S. (2017). *Time Series Analysis and Its Applications* (4th ed.). Springer.
- Welch, G., & Bishop, G. (2006). An Introduction to the Kalman Filter. *University of North Carolina at Chapel Hill*.
- Zhu, C., Byrd, R.H., Lu, P., & Nocedal, J. (1994). L-BFGS-B: Fortran Subroutines for Large-Scale Bound Constrained Optimization. *Northwestern University Technical Report*.
