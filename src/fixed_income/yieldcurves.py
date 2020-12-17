import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = ["NelsonSiegel", "Vasicek"]


def nelson_siegel(theta0, theta1, theta2, kappa, maturities):
    inverse_maturities = 1.0 / maturities
    inverse_maturities[inverse_maturities == np.inf] = 0
    yields = np.zeros(maturities.shape)
    yields += theta0
    yields += (
        (theta1 + theta2)
        * (1 - np.exp(-maturities / kappa))
        * inverse_maturities
        * kappa
    )
    yields -= theta2 * np.exp(-maturities / kappa)
    return yields


def price(cashflows, zeros):
    if isinstance(cashflows, pd.Series) and isinstance(zeros, pd.Series):
        return (cashflows * zeros).sum()
    return (cashflows * zeros).sum(axis=1)


def price_error(real_prices, fitted_prices):
    return ((real_prices - fitted_prices) ** 2).sum()


def ns_error(x, real_prices, cashflows, maturities):
    yields = nelson_siegel(*x, maturities=maturities)
    zeros = np.exp(-maturities * yields)
    fitted_prices = price(cashflows, zeros)
    return price_error(real_prices, fitted_prices)


def ns_fit(real_prices, cashflows, maturities, x0=None):
    x0 = x0 if x0 is not None else [0.0, 0.0, 0.0, 1.0]
    return minimize(
        ns_error, x0, args=(real_prices, cashflows, maturities), method="powell"
    )


def forwards(df):
    df["Forward"] = -np.log(df.Zero).diff() / df.Maturity.diff()
    df.Forward.values[0] = df.Yield.values[0]
    return df


class NelsonSiegel:
    def __init__(self, theta0, theta1, theta2, kappa):
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2
        self.kappa = kappa

    def __repr__(self):
        params = ",".join(f"{k}={v:.4f}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"

    def dataframe(self, maturities):
        yields = nelson_siegel(
            self.theta0, self.theta1, self.theta2, self.kappa, maturities
        )
        return (
            pd.DataFrame()
            .assign(Maturity=maturities)
            .assign(Yield=yields)
            .assign(Zero=np.exp(-maturities * yields))
            .pipe(forwards)
            .set_index("Maturity")
        )

    def price(self, cashflows, cashflow_maturities):
        return price(cashflows, self.zeros(cashflow_maturities))

    def yields(self, maturities):
        return nelson_siegel(
            self.theta0, self.theta1, self.theta2, self.kappa, maturities
        )

    def zeros(self, maturities):
        return np.exp(-maturities * self.yields(maturities))

    def delta(self, cashflows, cashflow_maturities):
        zeros = self.zeros(cashflow_maturities)
        return price(cashflows, zeros * cashflow_maturities)

    def duration(self, cashflows, cashflow_maturities):
        return self.delta(cashflows, cashflow_maturities) / self.price(
            cashflows, cashflow_maturities
        )

    def gamma(self, cashflows, cashflow_maturities):
        zeros = self.zeros(cashflow_maturities)
        return price(cashflows, zeros * pow(cashflow_maturities, 2))

    def convexity(self, cashflows, cashflow_maturities):
        return self.gamma(cashflows, cashflow_maturities) / self.price(
            cashflows, cashflow_maturities
        )

    @classmethod
    def from_fit(cls, real_prices, cashflows, cashflow_maturities, x0=None):
        result = ns_fit(real_prices, cashflows, cashflow_maturities, x0)
        return cls(*result.x)


def vasicek(eta, gamma, r0, sigma, maturities):
    inverse_maturities = (1.0 / maturities).replace(np.inf, 0)
    sigma2 = pow(sigma, 2)
    b = 1 / gamma * (1 - np.exp(-gamma * maturities))
    a = 1 / pow(gamma, 2) * (b - maturities) * (
        eta * gamma - 0.5 * sigma2
    ) - sigma2 * pow(b, 2) / (4 * gamma)
    return (-a + b * r0) * inverse_maturities


def vasicek_error(x, r0, sigma, real_prices, cashflows, maturities):
    yields = vasicek(*x, r0=r0, sigma=sigma, maturities=maturities)
    zeros = np.exp(-maturities * yields)
    fitted_prices = price(cashflows, zeros).replace(np.inf, 0)
    return price_error(real_prices, fitted_prices)


def vasicek_fit(r0, sigma, real_prices, cashflows, maturities, x0=None):
    x0 = x0 if x0 is not None else [0.1, 0.1]
    return minimize(
        vasicek_error,
        x0,
        args=(r0, sigma, real_prices, cashflows, maturities),
        method="Nelder-Mead",
        options={"maxiter": 1000},
        tol=1e-10,
    )


class Vasicek:
    def __init__(self, eta, gamma, r0, sigma):
        self.eta = eta
        self.gamma = gamma
        self.r0 = r0
        self.sigma = sigma

    def __repr__(self):
        params = ",".join(f"{k}={v:.5f}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"

    def dataframe(self, maturities):
        yields = vasicek(self.eta, self.gamma, self.r0, self.sigma, maturities)
        return (
            pd.DataFrame()
            .assign(Maturity=maturities)
            .assign(Yield=yields)
            .assign(Zero=np.exp(-maturities * yields))
            .pipe(forwards)
            .set_index("Maturity")
        )

    def price(self, cashflows, cashflow_maturities):
        yields = vasicek(self.eta, self.gamma, self.r0, self.sigma, cashflow_maturities)
        zeros = np.exp(-cashflow_maturities * yields)
        return price(cashflows, zeros)

    def yields(self, maturities):
        return vasicek(self.eta, self.gamma, self.r0, self.sigma, maturities)

    def zeros(self, maturities):
        return np.exp(-maturities * self.yields(maturities))

    def delta(self, cashflows, cashflow_maturities):
        zeros = self.zeros(cashflow_maturities)
        b = (1 - np.exp(-self.gamma * cashflow_maturities)) / self.gamma
        return price(cashflows, -b * zeros)

    @classmethod
    def from_fit(cls, r0, sigma, real_prices, cashflows, cashflow_maturities, x0=None):
        result = vasicek_fit(r0, sigma, real_prices, cashflows, cashflow_maturities, x0)
        return cls(*result.x, r0, sigma)
