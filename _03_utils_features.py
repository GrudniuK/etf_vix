

import pandas as pd
import numpy as np
import ta

def sma_short_long(df: pd.DataFrame, prefix: str, sma_window_short: int, sma_window_long: int) -> pd.DataFrame:
    """
    generuje zmienne w oparciu o SMA

    Parameters
    ----------
    df: pd.DataFrame
        obiekt wejsciowy
    prefix: str
        prefix do generowanych zmiennych (zwykle ticker)
    sma_window_short: int
        parametr przekazywany do SMA (short)
    sma_window_long: int
        parametr przekazywany do SMA (long)
    Returns
    -------
    df: pd.DataFrame
        obiekt wejsciowy z dodatkowymi zmiennymi z prefixem
    """

    df[prefix+'_SMA_'+str(sma_window_short)] = ta.trend.sma_indicator(df['Close'], window=sma_window_short, fillna=False)
    df[prefix+'_SMA_'+str(sma_window_long)] = ta.trend.sma_indicator(df['Close'], window=sma_window_long, fillna=False)

    df[prefix+'_Close_pr_SMA_'+str(sma_window_short)] = df['Close'] / df[prefix+'_SMA_'+str(sma_window_short)]
    df[prefix+'_Close_pr_SMA_'+str(sma_window_long)] = df['Close'] / df[prefix+'_SMA_'+str(sma_window_long)]
    df[prefix+'_SMA_'+str(sma_window_short)+'_pr_SMA_'+str(sma_window_long)] = df[prefix+'_SMA_'+str(sma_window_short)] / df[prefix+'_SMA_'+str(sma_window_long)]

    #usuniecie niepotrzebnych zmiennych
    df = df.drop([prefix+'_SMA_'+str(sma_window_short), prefix+'_SMA_'+str(sma_window_long)], axis = 'columns')

    return df

def macd_add_features(df: pd.DataFrame, prefix: str, macd_window_slow: int, macd_window_fast: int, macd_window_sign: int) -> pd.DataFrame:
    """
    generuje zmienne w oparciu o MACD

    Parameters
    ----------
    df: pd.DataFrame
        obiekt wejsciowy
    prefix: str
        prefix do generowanych zmiennych (zwykle ticker)
    macd_window_slow: int
        parametr przekazywany do MACD
    macd_window_fast: int
        parametr przekazywany do MACD
    macd_window_sign: int
        parametr przekazywany do MACD
    Returns
    -------
    df: pd.DataFrame
        obiekt wejsciowy z dodatkowymi zmiennymi z prefixem
    """
    macd_object = ta.trend.MACD(df['Close'], macd_window_fast, macd_window_slow, macd_window_sign)
    df[prefix+'_MACD_'+str(macd_window_fast)+'_'+str(macd_window_slow)+'_'+str(macd_window_sign)+'_line'] = macd_object.macd()
    df[prefix+'_MACD_'+str(macd_window_fast)+'_'+str(macd_window_slow)+'_'+str(macd_window_sign)+'_diff'] = macd_object.macd_diff()
    df[prefix+'_MACD_'+str(macd_window_fast)+'_'+str(macd_window_slow)+'_'+str(macd_window_sign)+'_signal'] = macd_object.macd_signal()
    
    return df

def rsi_add_features(df: pd.DataFrame, prefix: str, rsi_window: int) -> pd.DataFrame:
    """
    generuje zmienne w oparciu o RSI

    Parameters
    ----------
    df: pd.DataFrame
        obiekt wejsciowy
    prefix: str
        prefix do generowanych zmiennych (zwykle ticker)
    rsi_window: int
        parametr przekazywany do RSI
    Returns
    -------
    df: pd.DataFrame
        obiekt wejsciowy z dodatkowymi zmiennymi z prefixem
    """
    
    df[prefix+'_RSI_'+str(rsi_window)] = ta.momentum.RSIIndicator(df['Close'], rsi_window).rsi()
    
    return df

def add_all_ta_features_extended(df: pd.DataFrame, prefix: str, open: str, high: str, low: str, close: str, volume: str, fillna: bool) -> pd.DataFrame:
    """
    generuje wszystkie dostepne wskazniki z TA plus ewentualnie dodatkowe zmienne

    Parameters
    ----------
    df: pd.DataFrame
        obiekt OHLCV wejsciowy
    prefix: str
        prefix do generowanych zmiennych (zwykle ticker)
    open: str
    high: str
    low: str
    close: str
    volume: str
    fillna: bool
    
    Returns
    -------
    df: pd.DataFrame
        obiekt wejsciowy z dodatkowymi zmiennymi z prefixem
    """
    df = ta.add_all_ta_features(df, open=open, high=high, low=low, close=close, volume=volume, fillna=fillna)
    #dodanie wlasnych zmiennych
    df['my_close_pr_trend_sma_fast'] = df[close] / df['trend_sma_fast']
    df['my_close_pr_trend_sma_slow'] = df[close] / df['trend_sma_slow']
    df['my_trend_sma_fast_pr_trend_sma_slow'] = df['trend_sma_fast'] / df['trend_sma_slow']

    sma_5 = ta.trend.sma_indicator(df[close], window=5, fillna=fillna)
    sma_25 = ta.trend.sma_indicator(df[close], window=25, fillna=fillna)
    df['my_close_pr_trend_sma_5'] = df[close] / sma_5
    df['my_close_pr_trend_sma_25'] = df[close] / sma_25
    df['my_trend_sma_5_pr_trend_sma_25'] = sma_5 / sma_25

    macd_object = ta.trend.MACD(df[close], window_fast=5, window_slow=35, window_sign=5)
    df['my_trend_macd_5_35_5'] = macd_object.macd()
    df['my_trend_macd_diff_5_35_5'] = macd_object.macd_diff()
    df['my_trend_macd_signal_5_35_5'] = macd_object.macd_signal()

    #usuniecie zbednych zmiennych
    df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis='columns')    
    #dodanie prefix
    df = df.add_prefix(prefix)
    return df




###########################################################

#generowanie zmiennych w oparciu o SMA
#pd_df_base_features = utils_features.sma_short_long(copy.copy(pd_df_base), 'base_SMA', 50, 200)
#pd_df_base_features = utils_features.sma_short_long(copy.copy(pd_df_base_features), 'base_SMA', 5, 25)

#generowanie zmiennych w oparciu o MACD
#pd_df_base_features = utils_features.macd_add_features(copy.copy(pd_df_base_features), 'base', 12, 26, 9)
#pd_df_base_features = utils_features.macd_add_features(copy.copy(pd_df_base_features), 'base', 5, 35, 5)

#generowanie zmiennych w oparciu o RSI
#pd_df_base_features = utils_features.rsi_add_features(copy.copy(pd_df_base_features), 'base', 14)


"""
Index(['base_volume_adi', 'base_volume_obv', 'base_volume_cmf',
       'base_volume_fi', 'base_volume_mfi', 'base_volume_em',
       'base_volume_sma_em', 'base_volume_vpt', 'base_volume_nvi',
       'base_volume_vwap', 'base_volatility_atr', 'base_volatility_bbm',
       'base_volatility_bbh', 'base_volatility_bbl', 'base_volatility_bbw',
       'base_volatility_bbp', 'base_volatility_bbhi', 'base_volatility_bbli',
       'base_volatility_kcc', 'base_volatility_kch', 'base_volatility_kcl',
       'base_volatility_kcw', 'base_volatility_kcp', 'base_volatility_kchi',
       'base_volatility_kcli', 'base_volatility_dcl', 'base_volatility_dch',
       'base_volatility_dcm', 'base_volatility_dcw', 'base_volatility_dcp',
       'base_volatility_ui', 'base_trend_macd', 'base_trend_macd_signal',
       'base_trend_macd_diff', 'base_trend_sma_fast', 'base_trend_sma_slow',
       'base_trend_ema_fast', 'base_trend_ema_slow', 'base_trend_adx',
       'base_trend_adx_pos', 'base_trend_adx_neg', 'base_trend_vortex_ind_pos',
       'base_trend_vortex_ind_neg', 'base_trend_vortex_ind_diff',
       'base_trend_trix', 'base_trend_mass_index', 'base_trend_cci',
       'base_trend_dpo', 'base_trend_kst', 'base_trend_kst_sig',
       'base_trend_kst_diff', 'base_trend_ichimoku_conv',
       'base_trend_ichimoku_base', 'base_trend_ichimoku_a',
       'base_trend_ichimoku_b', 'base_trend_visual_ichimoku_a',
       'base_trend_visual_ichimoku_b', 'base_trend_aroon_up',
       'base_trend_aroon_down', 'base_trend_aroon_ind', 'base_trend_psar_up',
       'base_trend_psar_down', 'base_trend_psar_up_indicator',
       'base_trend_psar_down_indicator', 'base_trend_stc', 'base_momentum_rsi',
       'base_momentum_stoch_rsi', 'base_momentum_stoch_rsi_k',
       'base_momentum_stoch_rsi_d', 'base_momentum_tsi', 'base_momentum_uo',
       'base_momentum_stoch', 'base_momentum_stoch_signal', 'base_momentum_wr',
       'base_momentum_ao', 'base_momentum_kama', 'base_momentum_roc',
       'base_momentum_ppo', 'base_momentum_ppo_signal',
       'base_momentum_ppo_hist', 'base_others_dr', 'base_others_dlr',
       'base_others_cr', 'base_my_close_pr_trend_sma_fast',
       'base_my_close_pr_trend_sma_slow',
       'base_my_trend_sma_fast_pr_trend_sma_slow',
       'base_my_close_pr_trend_sma_5', 'base_my_close_pr_trend_sma_25',
       'base_my_trend_sma_5_pr_trend_sma_25', 'base_my_trend_macd_5_35_5',
       'base_my_trend_macd_diff_5_35_5', 'base_my_trend_macd_signal_5_35_5'],
      dtype='object')
"""
