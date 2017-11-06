import numpy as np
from scipy.optimize import minimize


def initialize_tree(maturity, time_step, is_zero):
    size = int(maturity / time_step)
    if is_zero:
        return np.zeros((size + 1, size + 1, size))
    return np.zeros((size, size))


def backfill(rate_tree, period, time_step):
    zero_tree = initialize_tree(period // time_step + 1, 1, False)
    zero_tree[:, -1] = 1
    pi = 0.5
    for j in range(period, 0, -1):
        discount = np.exp(-rate_tree[0:j+1, j] * time_step)
        zero_tree[0:j+1, j] = discount * pi * (zero_tree[0:j, j + 1] + zero_tree[1:j+1, j + 1])
    discount = np.exp(-rate_tree[0, 0] * time_step)
    zero_tree[0, 0] = discount * pi * (zero_tree[0,  1] + zero_tree[1, 1])
    return rate_tree, zero_tree


def ho_lee(theta, rate_tree, period, sigma, time_step):
    rate_tree[0, period] = rate_tree[0, period - 1] + theta * time_step + sigma * np.sqrt(time_step)
    for i in range(1, rate_tree.shape[0]):
        rate_tree[i, period] = rate_tree[i - 1, period - 1] + theta * time_step - sigma * np.sqrt(time_step)

    return backfill(rate_tree, period, time_step)


def black_derman_toy(theta, rate_tree, period, sigma, time_step):
    rate_tree[0, period] = rate_tree[0, period - 1] * np.exp(theta * time_step + sigma * np.sqrt(time_step))
    for i in range(1, rate_tree.shape[0]):
        rate_tree[i, period] = rate_tree[i - 1, period - 1] \
                               * np.exp(theta * time_step - sigma * np.sqrt(time_step))

    return backfill(rate_tree, period, time_step)


def error(theta, zero, model, implied_tree, period, sigma, time_step):
    _, zero_tree = model(theta, implied_tree, period, sigma, time_step)
    return (zero - zero_tree[0, 0]) ** 2


def fit(model, zeros, sigma, maturity, time_step):
    r0 = -1 / time_step * np.log(zeros[0])
    rate_tree = initialize_tree(maturity, time_step, False)
    rate_tree[0, 0] = r0

    zero_tree = initialize_tree(maturity, time_step, True)
    zero_tree[0, 0, 0] = np.exp(-r0 * time_step)
    zero_tree[:2, 1, 0] = 1

    thetas = np.nan(zeros.shape)
    fitted_zeros = np.nan(zeros.shape)

    for i, zero in enumerate(zeros[1:], start=1):
        result = minimize(error, [0], args=(zero, model, rate_tree, i, sigma, time_step))
        thetas[i - 1] = result.x

    return thetas, fitted_zeros, rate_tree
