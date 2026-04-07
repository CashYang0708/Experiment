import numpy as np
import pandas as pd
from scipy.stats import rankdata


# region Auxiliary functions
def ts_sum(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series sum over the past 'window' days.
    """

    return df.rolling(window).sum()

def sum(df:pd.DataFrame, window):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series sum over the past 'window' days.
    """
    return df.rolling(window).sum()

def sma(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series mean over the past 'window' days.
    """
    return df.rolling(window).mean()


def stddev(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series standard deviation over the past 'window' days.
    """
    return df.rolling(window).std()


def rolling_rank(na:np.array):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na:np.array):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def rank(df:pd.DataFrame):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    # return df.rank(axis=1, pct=True)
    return df.rank(pct=True)

def delay(df:pd.DataFrame, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def correlation(x:pd.DataFrame, y:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param x,y: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def covariance(x:pd.DataFrame, y:pd.DataFrame, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def scale(df:pd.DataFrame, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def delta(df:pd.DataFrame, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def decay_linear(df:pd.DataFrame, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.values

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])
    # endregion


def ts_argmax(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame.
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df:pd.DataFrame, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame.
    """
    return df.rolling(window).apply(np.argmin) + 1

def returns(df:pd.DataFrame):
    """
    Wrapper function to estimate returns.
    :param df: a pandas DataFrame.
    :param period: the returns grade.
    :return: a pandas DataFrame with the returns.
    """
    return df.rolling(2).apply(lambda x: x.iloc[-1] / x.iloc[0]) - 1

def abs(df:pd.DataFrame):
    """
    Wrapper function to estimate absolute values.
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with the absolute values.
    """
    return df.abs()

def sum_if(df:pd.DataFrame, window, condition):
    """
    Wrapper function to estimate sum if condition is met.
    :param df: a pandas DataFrame.
    :param condition: the condition.
    :return: a pandas DataFrame with the sum of the values that met the condition.
    """
    df[~condition] = 0
    return df.rolling(window).sum()

def regbeta(df:pd.DataFrame, x):
    window=len(x)
    return df.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0])

def sequence(n):
    """
    Generate a sequence of numbers.
    :param n: the sequence length.
    :return: a numpy array with the sequence.
    """
    return np.arange(1,n+1)

def row_min(df:pd.DataFrame):
    """
    Get the row-wise minimum value.
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with the row-wise minimum value.
    """
    return df.min(axis=1)

def row_max(df:pd.DataFrame):
    """
    Get the row-wise maximum value.
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with the row-wise maximum value.
    """
    return df.max(axis=1)

def low_day(df:pd.DataFrame, window):
    """
    Get the number of days since the last low.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the number of days since the last low.
    """
    return df.rolling(window).apply(lambda x: len(x) - x.values.argmin())

def high_day(df:pd.DataFrame, window):
    """
    Get the number of days since the last high.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the number of days since the last high.
    """
    return df.rolling(window).apply(lambda x: len(x) - x.values.argmax())

def wma(df:pd.DataFrame, window):
    """
    Weighted moving average.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the weighted moving average.
    """
    weights = np.array(range(window-1,-1, -1))
    weights = np.power(0.9,weights)
    sum_weights = np.sum(weights)

    return df.rolling(window).apply(lambda x: np.sum(x*weights)/sum_weights)
