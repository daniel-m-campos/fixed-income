import numpy as np
from scipy.stats import norm


def black_option(forward, strike, discount_factor, sigma, maturity, is_call=True):
    d1 = (np.log(forward / strike) + sigma ** 2 * maturity / 2) / (
        sigma * np.sqrt(maturity)
    )
    d2 = d1 - sigma * np.sqrt(maturity)
    if is_call:
        return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    else:
        return discount_factor * (-forward * norm.cdf(-d1) + strike * norm.cdf(-d2))
