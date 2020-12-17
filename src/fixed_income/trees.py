import numpy as np
from scipy.optimize import minimize

__all__ = ["ho_lee", "simple_bdt", "fit", "bond_price"]


def initialize_tree(maturity, time_step, is_zero):
    size = int(maturity / time_step)
    if is_zero:
        return np.zeros((size + 1, size + 1, size))
    return np.zeros((size, size))


def backfill(rate_tree, period, time_step):
    zero_maturity = period + 2
    zero_tree = np.zeros((zero_maturity, zero_maturity))
    zero_tree[:, -1] = 1
    pi = 0.5
    for j in range(period + 1, 0, -1):
        discount = np.exp(-rate_tree[:j, j - 1] * time_step)
        zero_tree[:j, j - 1] = (
            discount * pi * (zero_tree[:j, j] + zero_tree[1 : j + 1, j])
        )
    return zero_tree


def ho_lee(theta, rate_tree, period, sigma, time_step):
    rate_tree[0, period] = (
        rate_tree[0, period - 1] + theta * time_step + sigma * np.sqrt(time_step)
    )
    for i in range(1, period + 1):
        rate_tree[i, period] = (
            rate_tree[i - 1, period - 1]
            + theta * time_step
            - sigma * np.sqrt(time_step)
        )

    return rate_tree, backfill(rate_tree, period, time_step)


def simple_bdt(theta, rate_tree, period, sigma, time_step):
    rate_tree[0, period] = rate_tree[0, period - 1] * np.exp(
        theta * time_step + sigma * np.sqrt(time_step)
    )
    for i in range(1, rate_tree.shape[0]):
        rate_tree[i, period] = rate_tree[i - 1, period - 1] * np.exp(
            theta * time_step - sigma * np.sqrt(time_step)
        )

    return rate_tree, backfill(rate_tree, period, time_step)


def error(theta, zero, model, rate_tree, period, sigma, time_step):
    _, zero_tree = model(theta, rate_tree, period, sigma, time_step)
    return (zero - zero_tree[0, 0]) ** 2


def fit(model, zeros, sigma, time_step):
    r0 = -1 / time_step * np.log(zeros[0])
    maturity = len(zeros) * time_step
    rate_tree = initialize_tree(maturity, time_step, False)
    rate_tree[0, 0] = r0

    zero_tree = initialize_tree(maturity, time_step, True)
    zero_tree[0, 0, 0] = np.exp(-r0 * time_step)
    zero_tree[:2, 1, 0] = 1

    thetas = np.zeros(zeros.shape)
    errors = np.zeros(zeros.shape)

    for i, zero in enumerate(zeros[1:], start=1):
        result = minimize(
            error,
            x0=[0],
            args=(zero, model, rate_tree.copy(), i, sigma, time_step),
            method="powell",
        )
        thetas[i - 1] = result.x
        errors[i - 1] = result.fun
        rate_tree, z_tree = model(thetas[i - 1], rate_tree, i, sigma, time_step)
        zero_tree[: i + 2, : i + 2, i] = z_tree

    fitted_zeros = zero_tree[0, 0, :].squeeze()
    return thetas, fitted_zeros, rate_tree


def bond_price(rate_tree, coupon, maturity, time_step):
    size = int(maturity / time_step)
    bond_tree = np.zeros((size + 1, size + 1))
    bond_tree[:, -1] = 100
    pi = 0.5

    for j in range(size, 0, -1):
        discount = np.exp(-rate_tree[:j, j - 1] * time_step)
        bond_tree[:j, j - 1] = discount * (
            pi * (bond_tree[:j, j] + bond_tree[1 : j + 1, j]) + coupon * time_step
        )
    return bond_tree


def call_price(rate_tree, bond_tree, strike, maturity, time_step, first_time_call):
    size = int(maturity / time_step)
    call_tree = np.zeros((size + 1, size + 1))
    call_tree[:, -1] = bond_tree[:, -1] - strike
    pi = 0.5

    for j in range(size, 0, -1):
        discount = np.exp(-rate_tree[:j, j - 1] * time_step)
        call_tree[:j, j - 1] = (
            discount * pi * (call_tree[:j, j] + call_tree[1 : j + 1, j])
        )
        if (j - 1) * time_step >= first_time_call:
            call_tree[:j, j - 1] = np.max(
                [call_tree[:j, j - 1], bond_tree[:j, j - 1] - strike], axis=0
            )
    return call_tree
