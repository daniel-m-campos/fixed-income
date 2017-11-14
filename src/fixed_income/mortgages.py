import numpy as np
import pandas as pd

__all__ = ['payments']


def coupon(loan, maturity, mortgage_rate, freq=1):
    dt = 1 / freq
    periods = maturity * freq
    discount = 1 / (1 + mortgage_rate * dt)
    return loan * (1 / discount - 1) / (1 - pow(discount, periods))


def payments(loan, maturity, mortgage_rate, freq=1):
    dt = 1 / freq
    periods = np.arange(0, int(maturity / dt) + 1)
    payment_times = periods * dt

    discounts = (1 + mortgage_rate * dt) ** -periods
    discounts[0] = 0

    payment = coupon(loan, maturity, mortgage_rate, freq)

    balance = 0 * payment_times
    balance[0] = loan
    interest = 0 * payment_times
    amounts = 0 * payment_times
    for i in periods[1:]:
        amounts[i] = payment
        interest[i] = balance[i - 1] * mortgage_rate * dt
        balance[i] = balance[i - 1] - payment + interest[i]
    return (pd.DataFrame()
            .assign(time=payment_times)
            .assign(value=balance)
            .assign(payment=amounts)
            .assign(interest=interest)
            )
