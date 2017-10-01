import pandas as pd


def wsj_treasury_prices(date=None):
    """Get US Treasury Bill, Note and Bond prices from www.wsj.com

    Parameters
    ----------
    date : str
        Optional, Date string of format %Y%m%d, e.g. 20170915

    Returns
    -------
    pandas.DataFrame
    """

    def url(date_string):
        if date_string:
            return f'http://www.wsj.com/mdc/public/page/2_3020-treasury-{date_string}.html?mod=mdc_pastcalendar'
        return 'http://www.wsj.com/mdc/public/page/2_3020-treasury.html?mod=3D=#treasuryB'

    def columns_of(table):
        return table.loc[0, :].values.tolist()

    def find_price(tables):
        return (t for t in tables if 'Bid' in columns_of(t))

    def create_df(table):
        df = table.copy()
        df.columns = columns_of(df)
        df = df.drop(df.index[0])
        return df

    tables = pd.read_html(url(date))
    df = pd.concat(create_df(t) for t in find_price(tables))
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    df = df.sort_values(by=['Maturity', 'Coupon'])
    df.index = range(len(df))
    return df
