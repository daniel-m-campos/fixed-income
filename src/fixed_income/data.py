import datetime

import numpy as np
import pandas as pd
import requests

DATE_FORMAT = '%Y%m%d'


def _columns_of(table):
    return table.loc[0, :].values.tolist()


def _find_price(tables):
    return (t for t in tables if 'Bid' in _columns_of(t))


def _create_df(table):
    df = table.copy()
    df.columns = _columns_of(df)
    df = df.drop(df.index[0])
    return df


def _get_date(date):
    if isinstance(date, datetime.date):
        return date
    elif isinstance(date, str):
        return datetime.datetime.strptime(date, DATE_FORMAT)
    raise NotImplementedError(f'{type(date)} not supported.')


def wsj_treasury_prices(date=None):
    """Get US Treasury Bill, Note and Bond prices from www.wsj.com

    Parameters
    ----------
    date : str
        Optional, Date or date string of format %Y%m%d, e.g. 20170915

    Returns
    -------
    pandas.DataFrame
    """

    if date:
        date_string = date if isinstance(date, str) else date.strftime(DATE_FORMAT)
        url = f'http://www.wsj.com/mdc/public/page/2_3020-treasury-{date_string}.html?mod=mdc_pastcalendar'
    else:
        url = 'http://www.wsj.com/mdc/public/page/2_3020-treasury.html?mod=3D=#treasuryB'

    tables = pd.read_html(url)
    df = pd.concat(_create_df(t) for t in _find_price(tables))
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    df = df.sort_values(by=['Maturity', 'Coupon'])
    df.index = range(len(df))
    return df


def treasury_direct(date=None):
    """Get US Treasury prices from www.treasurydirect.gov

    Parameters
    ----------
    date : str
        Optional, Date or date string of format %Y%m%d, e.g. 20170915

    Returns
    -------
    pandas.DataFrame
    """

    if date is None:
        url = 'https://www.treasurydirect.gov/GA-FI/FedInvest/todaySecurityPriceDate.htm'
        table = pd.read_html(url)[0]
        clean_date = datetime.datetime.today()
    else:
        clean_date = _get_date(date)
        url = 'https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate.htm'
        data = {'priceDate.month': clean_date.month,
                'priceDate.day': clean_date.day,
                'priceDate.year': clean_date.year,
                'submit': 'Show Prices'}
        response = requests.post(url, data=data)
        assert response.ok
        table = pd.read_html(response.text)[0]

    df = _create_df(table)
    df['MATURITY DATE'] = pd.to_datetime(df['MATURITY DATE'])
    df['MATURITY DATE'] = pd.to_datetime(df['MATURITY DATE'])
    df['MATURITY'] = (df['MATURITY DATE'] - clean_date) / np.timedelta64(1, 'Y')
    df['COUPON'] = df['RATE'].str[:-1].astype(float)
    for c in ['BUY', 'SELL', 'END OF DAY']:
        df[c] = pd.to_numeric(df[c])
    return df


def cashflows_matrix(treasury_direct_df, quote_date):
    max_semi_periods = int(np.ceil(((treasury_direct_df['MATURITY DATE'] - quote_date) / np.timedelta64(6, 'M')).max()))
    maturities = np.zeros((len(treasury_direct_df), max_semi_periods))
    cashflows = maturities.copy()

    for i, row in treasury_direct_df.iterrows():
        semi_periods = int(np.ceil(row['MATURITY'] / 0.5))
        if semi_periods == 0:
            maturities[i - 1, 0] = row['MATURITY']
            cashflows[i - 1, 0] = 100
        else:
            maturities[i - 1, semi_periods - 1] = row['MATURITY']
            maturities[i - 1, :(semi_periods - 1)] = row['MATURITY'] - (0.5 * np.ones(
                (1, semi_periods - 1))).cumsum()[::-1]
            semi_coupon = row['COUPON'] / 2
            cashflows[i - 1, :semi_periods] = semi_coupon
            cashflows[i - 1, semi_periods - 1] += 100

    return cashflows, maturities
