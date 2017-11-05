import numpy as np
from scipy.optimize import minimize


def nelson_siegal(theta_0, theta_1, theta_2, kappa, maturities):
    inverse_maturities = (1 / maturities).replace(np.inf, 0)
    yields = np.zeros(maturities.shape)
    yields += theta_0
    yields += (theta_1 + theta_2) * (1 - np.exp(-maturities / kappa)) * inverse_maturities * kappa
    yields -= theta_2 * np.exp(-maturities / kappa)
    zeros = np.exp(-maturities * yields)
    return zeros


def price(cashflows, zeros):
    return (cashflows * zeros).sum(axis=1)


def price_error(real_prices, fitted_prices):
    return ((real_prices - fitted_prices) ** 2).dropna().sum()


def fit_error(x, real_prices, cashflows, maturities):
    zeros = nelson_siegal(*x, maturities=maturities)
    fitted_prices = price(cashflows, zeros)
    return price_error(real_prices, fitted_prices)


def fit(real_prices, cashflows, maturities, x0=None):
    x0 = x0 if x0 else [0.0, 0.0, 0.0, 1.0]
    return minimize(fit_error, x0, args=(real_prices, cashflows, maturities))
