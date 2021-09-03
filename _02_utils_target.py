

import pandas as pd
import numpy as np
import ta
import copy
from io import BytesIO
import matplotlib.pyplot as plt

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

    df_out = pd.DataFrame(data = {
        'target_variable':[target_variable],
        'cnt_days':[cnt_days], 
        'cnt_change':[cnt_change], 
        'cagr_benchmark':[cagr_benchmark], 
        'cagr_'+target_variable+'_without_costs':[cagr_strategy_without_costs],
        'cagr_'+target_variable+'_with_costs':[cagr_strategy_with_costs]
        })

    #df_out = pd.DataFrame(data = {
    #    'target_variable':[target_variable],
    #    'cnt_days':[cnt_days], 
    #    'cnt_change':[cnt_change], 
    #    'cagr_benchmark':[cagr_benchmark], 
    #    'cagr_target_without_costs':[cagr_strategy_without_costs],
    #    'cagr_target_with_costs':[cagr_strategy_with_costs]
    #})
    
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

    """
    #utils_target.target_optimize(copy.copy(pd_df_base), 5, 3, 3, transaction_cost)
    df = copy.copy(pd_df_base)
    param_atr_factor = 5
    param_window = 3
    param_smooth = 3
    transaction_cost = transaction_cost
    """

    pd_df_out = pd.DataFrame()
    for param_atr_factor_iter in range(1,param_atr_factor):
        for param_window_iter in range(1,param_window):
            for param_smooth_iter in range(1,param_smooth):
                """
                param_atr_factor_iter = 1
                param_window_iter = 1
                param_smooth_iter = 1
                """
                print((param_atr_factor_iter, param_window_iter, param_smooth_iter))
                pd_df_labels = target_generate(df, 14, param_atr_factor_iter, param_window_iter, param_smooth_iter)
                pd_df_labels_eval = target_evaluate(pd_df_labels, 'target', transaction_cost)[0]
                pd_df_labels_eval['param_atr_factor'] = param_atr_factor_iter
                pd_df_labels_eval['param_window'] = param_window_iter
                pd_df_labels_eval['param_smooth'] = param_smooth_iter
                pd_df_out = pd.concat([pd_df_out, pd_df_labels_eval])
    
    pd_df_out = pd_df_out.reset_index()
    return pd_df_out



def target_optimize_set(df: pd.DataFrame, cnt_changes_per_year: int, writer: object, pd_df_base: pd.DataFrame, transaction_cost: float) -> tuple:
    """
    ma za zadanie wybranie optymalnych parametrow przy zdefiniowanych ograniczeniach
    1. okreslenie maksymalnej liczby zmian
    2. wyfiltrowanie scenariuszy spelniajacych pkt. 1
    3. wybranie i zwrocenie parametrow dla najelszego scenariuasza pod katem cagr_target_with_costs

    Parametres
    ----------
    df: pd.DataFrame
        DataFrame (wynik z target_optimize)
    cnt_changes_per_year: int
        parametr okreslajacy maksymalna liczbę transakcji w jednym roku
    writer: object
        obiekt do zapisu XLSX
    pd_df_base: pd.DataFrame
        DataFrame do wyliczenia i evaluacji target przy okreslonych parametrach
    transaction_cost: float
        koszty transkacji (aktualnie 0.29%)

    Returns
    -------
    (param_atr_factor, param_window, param_smooth): tuple
        wybrane parametry definiujace target
    """

    #zapis do XLSX zbioru wejsciowego
    df.to_excel(writer, sheet_name='target_optimize', startrow=0, startcol=0)

    #wybor scenariuszy z ograniczeniem na maksymalna liczbe transakcji
    max_cnt_changes = int(df.iloc[0]['cnt_days']) / 250 * cnt_changes_per_year #250 to liczba dni pracujacych w roku (srednia/przyblizenie)
    df = df[df['cnt_change'] <= max_cnt_changes]
    df = df.sort_values(by = 'cagr_target_with_costs', ascending = False).reset_index()

    #zapis do XLSX 5 najlepszych scenariuszy
    df[0:5].to_excel(writer, sheet_name='target_optimize_top5', startrow=0, startcol=0)

    #generowanie wykresu dla 5 najlepszych scenariuszy

    #inicjalizacja listy
    list_plot_by_time = []
    for i in range(5):

        tuple_param = (
            df.iloc[i]['param_atr_factor'], 
            df.iloc[i]['param_window'], 
            df.iloc[i]['param_smooth']
            )
        
        df_target = target_generate(
            df=copy.copy(pd_df_base), 
            param_atr_base = 14, 
            param_atr_factor = tuple_param[0],
            param_window = tuple_param[1],
            param_smooth = tuple_param[2]
            )

        df_target_eval = target_evaluate(copy.copy(df_target), 'target', transaction_cost)[1]

        #zmiana nazwy zmiennej
        df_target_eval.rename(columns={'cagr_target_with_costs': str(tuple_param)}, inplace=True)

        #dodanie do listy
        list_plot_by_time.append(df_target_eval)

    #polaczenie df z listy
    df_plot_by_time = pd.concat(list_plot_by_time, axis=1)
    #usuniecie zduplikowanych kolumn
    df_plot_by_time= df_plot_by_time.loc[:,~df_plot_by_time.columns.duplicated()]

    #generowanie wykresow i zapis do XLSX
    image_data = BytesIO()
    fig, ax = plt.subplots(figsize=(12.8,4.8))
    df_plot_by_time.plot(ax=ax)
    #fig.show()
    fig.savefig(image_data)  
    worksheet = writer.sheets['target_optimize_top5']
    worksheet.insert_image(8, 0, 'tmp', {'image_data': image_data})

    return (int(df.iloc[0]['param_atr_factor']), int(df.iloc[0]['param_window']), int(df.iloc[0]['param_smooth']))
