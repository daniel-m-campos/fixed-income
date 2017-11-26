import datetime

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
    elif isinstance(date, datetime.date):
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

    return _create_df(table)
