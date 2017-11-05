import numpy as np
import pandas as pd
from scipy.optimize import minimize


def nelson_siegel(theta0, theta1, theta2, kappa, maturities):
    inverse_maturities = (1.0 / maturities).replace(np.inf, 0)
    yields = np.zeros(maturities.shape)
    yields += theta0
    yields += (theta1 + theta2) * (1 - np.exp(-maturities / kappa)) * inverse_maturities * kappa
    yields -= theta2 * np.exp(-maturities / kappa)
    return yields


def price(cashflows, zeros):
    return (cashflows * zeros).sum(axis=1)


def price_error(real_prices, fitted_prices):
    return ((real_prices - fitted_prices) ** 2).sum()


def fit_error(x, real_prices, cashflows, maturities):
    yields = nelson_siegel(*x, maturities=maturities)
    zeros = np.exp(-maturities * yields)
    fitted_prices = price(cashflows, zeros)
    return price_error(real_prices, fitted_prices)


def fit(real_prices, cashflows, maturities, x0=None):
    x0 = x0 if x0 is not None else [0.0, 0.0, 0.0, 1.0]
    return minimize(fit_error, x0, args=(real_prices, cashflows, maturities))


class NelsonSiegel:
    def __init__(self, theta0, theta1, theta2, kappa):
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
        self.kappa = kappa

    def __repr__(self):
        params = ",".join(f"{k}={v:.4f}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"

    def _forwards(self, df):
        df['Forward'] = -np.log(df.Zero).diff() / df.Maturity.diff()
        df.Forward.values[0] = df.Yield.values[0]
        return df

    def dataframe(self, maturities):
        yields = nelson_siegel(self.theta0, self.theta1, self.theta2, self.kappa, maturities)
        return (pd.DataFrame()
                .assign(Maturity=maturities)
                .assign(Yield=yields)
                .assign(Zero=np.exp(-maturities * yields))
                .pipe(self._forwards)
                .set_index('Maturity')
                )

    def price(self, cashflows, cashflow_maturities):
        yields = nelson_siegel(self.theta0, self.theta1, self.theta2, self.kappa, cashflow_maturities)
        zeros = np.exp(-cashflow_maturities * yields)
        return price(cashflows, zeros)

    @classmethod
    def from_fit(cls, real_prices, cashflows, cashflow_maturities, x0=None):
        result = fit(real_prices, cashflows, cashflow_maturities, x0)
        return cls(*result.x)
