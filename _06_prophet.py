

# https://facebook.github.io/prophet/docs/
# https://neuralprophet.com/
# https://pythondata.com/stock-market-forecasting-with-prophet/
# https://github.com/urgedata/pythondata/blob/master/fbprophet/fbprophet_part_one.ipynb



"""
[done] 1. zbudowanie modelu dla ln(Close)
[done] 2. horyzont prognozy zgodny z target
[done] 2. wyliczenie Close, low i high. Open jako prev(Close)
3. wyliczyć featers - to chyba nie ma sensu
[done] 4. wylicenie target
[done] 4. wyliczneie cagr dla Close, low i high
[done] 5. miara dopasowania na probie treningowej
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import resample, shuffle
from fbprophet import Prophet
import copy
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import _02_utils_target as utils_target

"""
#sprawdzenie czy pystan zostal poprawnie zainstalowany i dziala
#https://github.com/facebook/prophet/issues/1790
import pystan
model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = pystan.StanModel(model_code=model_code)  # this will take a minute
y = model.sampling(n_jobs=1).extract()['y']
y.mean()  # should be close to 0
"""



def prophet_features(df: pd.DataFrame, yyyymmdd: str, initial_no_of_days: int, future_no_of_days: int, tuple_target_param: tuple) -> pd.DataFrame:
    """
    Tworzy zmienne z predykcji modelem prophet

    Parameters
    ----------
    df: pd.DataFrame 
    yyyymmdd: str
        data dla ktorej chcemy pozyskac informacje w formamcie "yyyy-mm-dd"
    initial_no_of_days: int
        liczba sesji na ktorej budowany jest model 
    future_no_of_days: int
        liczba sesji dla ktorej robiona jest prognoza
    tuple_target_param: tuple
        (param_atr_factor, param_window, param_smooth)
        wybrane parametry definiujace target, wynik z utils_target.target_optimize_set

    Returns
    -------
    df: pd.DataFrame
        prognoza z modelu prophet wraz z miarami dopaskowania na ziorze train

    """

    df_ts_all = df[df.index <= yyyymmdd].tail(initial_no_of_days)

    #przygotowanie danych jako input do prophet
    df_ts = df_ts_all.reset_index()
    df_ts['log_Close'] = np.log(df_ts.Close)
    df_ts = df_ts[['Date','log_Close']]
    df_ts = df_ts.rename(columns={'Date':'ds', 'log_Close':'y'})

    model = Prophet(daily_seasonality=True)
    model.fit(df_ts)

    future = model.make_future_dataframe(periods=future_no_of_days)
    future = future[future['ds'].dt.dayofweek < 5] # usuniecie weekendow

    forecast = model.predict(future)

    forecast = forecast.set_index('ds')
    forecast['prophet_Close'] = np.exp(forecast['yhat'])
    forecast['prophet_Low'] = np.exp(forecast['yhat_lower'])
    forecast['prophet_High'] = np.exp(forecast['yhat_upper'])
    forecast['prophet_Open'] = forecast['prophet_Close'].shift(1)

    # polaczenie danych wejsciowych i wyjsciowych
    forecast = forecast[['prophet_Open','prophet_High','prophet_Low','prophet_Close']].join(df_ts_all[['Open','High','Low','Close']], how = 'left')
    forecast['Close'] = np.where(np.isnan(forecast['Close']), forecast['prophet_Close'], forecast['Close'])
    forecast['Open'] = np.where(np.isnan(forecast['Open']), forecast['prophet_Open'], forecast['Open'])
    forecast['High'] = np.where(np.isnan(forecast['High']), forecast['prophet_High'], forecast['High'])
    forecast['Low'] = np.where(np.isnan(forecast['Low']), forecast['prophet_Low'], forecast['Low'])
    forecast['Volume'] = None

    #majac predykcje z modelu mozemy na jej podstawie wyliczyc target
    prophet_target = utils_target.target_generate(
        df=copy.copy(forecast), 
        param_atr_base = 14, 
        param_atr_factor = tuple_target_param[0],
        param_window = tuple_target_param[1],
        param_smooth = tuple_target_param[2]    
        )

    #predykcja z modelu w okresie zgodnym z target
    prophet_target = prophet_target[prophet_target.index >= yyyymmdd].head(tuple_target_param[1]+1)

    df_out = prophet_target[prophet_target.index == yyyymmdd][['target']]
    df_out = df_out.reset_index()
    df_out = df_out.rename(columns={'ds':'Date', 'target':'prophet_target'})
    df_out['prophet_cagr_Close'] = prophet_target.iloc[-1]['Close'] / prophet_target.iloc[0]['Close']
    df_out['prophet_cagr_Low'] = prophet_target.iloc[-1]['Low'] / prophet_target.iloc[0]['Low']
    df_out['prophet_cagr_High'] = prophet_target.iloc[-1]['High'] / prophet_target.iloc[0]['High']
    df_out['prophet_Close_min_max'] = (prophet_target.iloc[-1]['Close'] - prophet_target.iloc[-1]['Low']) / (prophet_target.iloc[-1]['High'] - prophet_target.iloc[-1]['Low'])
    df_out['prophet_cagr_min_max'] = (df_out['prophet_cagr_Close'] - df_out['prophet_cagr_Low']) / (df_out['prophet_cagr_High'] - df_out['prophet_cagr_Low'])
    df_out['prophet_r2_score'] = r2_score(forecast[forecast.index <= yyyymmdd]['Close'], forecast[forecast.index <= yyyymmdd]['prophet_Close'])
    df_out['prophet_mape'] = mean_absolute_percentage_error(forecast[forecast.index <= yyyymmdd]['Close'], forecast[forecast.index <= yyyymmdd]['prophet_Close'])

    return df_out




