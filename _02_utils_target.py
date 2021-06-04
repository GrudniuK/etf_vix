

import pandas as pd
import numpy as np
import ta

def target_generate(df: pd.DataFrame, param_atr_base: int, param_atr_factor: float, param_window: int, param_smooth: int) -> pd.DataFrame:
    """
    Doklada zmienna 'target'

    wyliczenie ATR
    okreslenie zmiennej 'target' na podstawie spadku ponizej granicy wyznaczonej na podstawie ATR
    wygladzenie wartosc w celu unikniecia jednostkowych transakcji

    Parameters
    ----------
    df: pd.Dataframe
        obiekt OHLCV
    param_atr_base: int 
        parametr przekazywany do ATR
    param_atr_factor: float
        paramtr do wyznaczenia label = 0 jako df['Close'] - param_atr_factor * df['ATR']
    param_window: int 
        parametr okreslajacy na ile okresow patrzymy do przodu w celu wyznaczeni label 
    param_smooth: int
        parametr okreslajacy ile okresow do przodu i do tylu wykorzystuwanych jest do wygladzenia wynikow
    
    Returns
    -------
    df: pd.DataFrame
        ramka OHLCV + target
    """
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=param_atr_base, fillna=False)

    for i in range(1,(param_window+1)):
        df['lead_'+str(i)] = df['Close'].shift(-i)
        
    columns_lead = list()
    for i in range(1,(param_window+1)):
        df['lead_f_'+str(i)] = (df['lead_'+str(i)] <= df['Close'] - param_atr_factor * df['ATR'])
        columns_lead.append('lead_f_'+str(i))

    df['lead_f_all'] = df[columns_lead].sum(axis=1)
    df['target'] = np.where(df['lead_f_all'] > 0, 1, 0)
    #wygladzenie
    columns_smooth = list()
    #print(df.head())

    for i in range(1,(param_smooth+1)):
        df['label_p'+str(i)] = df['target'].shift(-i)
        columns_smooth.append('label_p'+str(i))
        df['label_m'+str(i)] = df['target'].shift(i)
        columns_smooth.append('label_m'+str(i))

    columns_smooth.append('target')
    df['label_all'] = df[columns_smooth].sum(axis=1) / (2 * param_smooth + 1)
    df['target'] = np.where(df['label_all'] > 0.5, 1, 0)

    #print(columns_smooth)
    #print(df.head())
    df = df[['Open', 'High', 'Low', 'Close', 'Volume','target']]
    #print(df.columns)

    return df



def target_evaluate (df: pd.DataFrame, target_variable: str, transaction_cost: float) -> pd.DataFrame:
    """
    funkcja oblicza liczbe dni, liczbe transakcji, skumulowana stope zwrotu dla benchmarku i strategii 

    Parameters
    ----------
    df: pd.DataFrame
        obiekt OHLCVT (wynik z target_generate())
    target_variable: str
        zmiennna okreslajaca target
    transaction_cost: float
        koszty transkacji (aktualnie 0.29%)
    Returns
    -------
    df_out: pd.DataFrame
        obiekt zawierajacy liczbe dni, liczbe transakcji, skumulowana stope zwrotu dla benchmarku i strategii (bez i z kosztami transakcji)
    df: pd.DataFrame
        obiekt zawierajacy CAGR dla benchmarku (na podstawie Close) oraz na podstawie 'target_variable' z uwzglednieniem kosztow transkacji
    """

    #df = copy.copy(pd_df_prediction)
    #target_variable = 'RegisticRegression_pred' 

    cnt_days = df.shape[0]

    #obliczenie liczby transakcji (zmiana targetu)
    df['target_change'] = np.where(df[target_variable].diff() != 0, 1, 0)

    cnt_change = df['target_change'].sum()-1

    #obliczenie stopy zwrotu
    df['log_Close'] = np.log(df.Close)
    df['log_return'] = df['log_Close'].diff()

    #benchmark
    cagr_benchmark = np.exp(df.log_return.sum()) - 1

    #strategia
    df['log_return_strategy'] = np.where(df[target_variable] == 1, 0, df['log_return'])
    df['log_return_strategy_with_costs'] = np.where(df[target_variable] == 1, 0, df['log_return'])
    df['log_return_strategy_with_costs'] = np.where(df['target_change'] != 1, df['log_return_strategy_with_costs'], df['log_return_strategy_with_costs'] + np.log(1-transaction_cost))

    df['cagr_benchmark'] = np.exp(df.log_return.cumsum()) - 1
    df['cagr_'+target_variable+'_with_costs'] = np.exp(df.log_return_strategy_with_costs.cumsum()) - 1
    df['cagr_'+target_variable+'_with_costs'] = np.where(df[target_variable] == 1, 0, df['cagr_'+target_variable+'_with_costs'])

    cagr_strategy_without_costs = np.exp(df.log_return_strategy.sum()) - 1
    cagr_strategy_with_costs = np.exp(df.log_return_strategy.sum() + cnt_change * np.log(1-transaction_cost)) - 1

    #df_out = pd.DataFrame(data = {
    #    'target_variable':[target_variable],
    #    'cnt_days':[cnt_days], 
    #    'cnt_change':[cnt_change], 
    #    'cagr_benchmark':[cagr_benchmark], 
    #    'cagr_'+target_variable+'_without_costs':[cagr_strategy_without_costs],
    #    'cagr_'+target_variable+'_with_costs':[cagr_strategy_with_costs]
    #    })

    df_out = pd.DataFrame(data = {
        'target_variable':[target_variable],
        'cnt_days':[cnt_days], 
        'cnt_change':[cnt_change], 
        'cagr_benchmark':[cagr_benchmark], 
        'cagr_target_without_costs':[cagr_strategy_without_costs],
        'cagr_target_with_costs':[cagr_strategy_with_costs]
    })
    
    return (df_out, df[['cagr_benchmark', 'cagr_'+target_variable+'_with_costs']])



def target_optimize(df: pd.DataFrame, param_atr_factor: int, param_window: int, param_smooth: int, transaction_cost: float) -> pd.DataFrame:
    """
    oblicza wyniki dla wszystkich mozliwych kombinacji

    Parametres
    ----------
    df: pd.DataFrame
        obiekt OHLCV
    param_atr_factor: int
        parametr okreslajacy maksymalna wartosc dla tego parametru
    param_window: int
        parametr okreslajacy maksymalna wartosc dla tego parametru
    param_smooth: int
        parametr okreslajacy maksymalna wartosc dla tego parametru
    transaction_cost: float
        koszty transkacji (aktualnie 0.29%)
    Returns
    -------
    pd_df_out: pd.DataFrame
    """
    pd_df_out = pd.DataFrame()
    for param_atr_factor_iter in range(1,param_atr_factor):
        for param_window_iter in range(1,param_window):
            for param_smooth_iter in range(1,param_smooth):
                print((param_atr_factor_iter, param_window_iter, param_smooth_iter))
                pd_df_labels = target_generate(df, 14, param_atr_factor_iter, param_window_iter, param_smooth_iter)
                pd_df_labels_eval = target_evaluate(pd_df_labels, 'target', transaction_cost)
                pd_df_labels_eval['param_atr_factor'] = param_atr_factor_iter
                pd_df_labels_eval['param_window'] = param_window_iter
                pd_df_labels_eval['param_smooth'] = param_smooth_iter
                pd_df_out = pd.concat([pd_df_out, pd_df_labels_eval])
    return pd_df_out

