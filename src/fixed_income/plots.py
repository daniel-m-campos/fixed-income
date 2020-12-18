import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd


def term_structure(
    yield_curve, prices, maturities, cashflows, cashflow_maturities, quote_date=None
):
    title = f"Treasuries @ {quote_date}" if quote_date else "Treasuries"

    fig, axes = _plt.subplots(3, 1, figsize=(8, 10))

    yc_df = yield_curve.dataframe(_pd.Series(_np.arange(0.5, 30.5, 0.5)))
    fitted_prices = yield_curve.price(cashflows, cashflow_maturities)

    ax = axes[0]
    ax.plot(maturities, prices, "*", c="b", label="Close Price")
    ax.plot(maturities, fitted_prices, "o", c="r", mfc="none", label="Fitted Price")
    ax.set_ylabel("Price ($)")
    ax.set_title(title)

    ax = axes[1]
    ax.plot(yc_df.Zero, "o-", c="b", label=None)
    ax.set_ylabel("Price ($)")
    ax.set_title("Nominal Discount")

    ax = axes[2]
    ax.plot(yc_df.Yield * 100, "o-", c="b", label="Zero")
    ax.plot(yc_df.Forward * 100, "o-", c="r", label="Forward")
    ax.set_ylabel("Yield (%)")
    ax.set_title("Nominal Yield")
    ax.legend()

    for ax in axes:
        ax.set_xlabel("Maturity")
        ax.grid()

    fig.tight_layout()
    return fig
