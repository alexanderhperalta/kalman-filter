import numpy as np
import matplotlib.pyplot as plt

def acf_plot(confint, rho, plot_title):
    half_width = confint[:, 1] - rho
    lags = np.arange(len(rho))

    lags = lags[1:]
    rho = rho[1:]
    half_width = half_width[1:]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(lags, -half_width, half_width,
                    alpha=0.15, color='darkgreen', label='95% significance band')

    ax.vlines(lags, 0, rho, color='darkgreen', linewidth=1.5)
    ax.plot(lags, rho, 'o', color='darkgreen', markersize=4)

    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(plot_title)
    ax.legend()
    plt.show()
