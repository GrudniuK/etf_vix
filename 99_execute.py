

import sys, getopt

argv = sys.argv[1:]
opts, args = getopt.getopt(argv,"t:")

if opts[0][1] == 'True': training_flag = True
elif opts[0][1] == 'False': training_flag = False
else: training_flag = '???'

#ponizsza flaga ustawiana jako paramatr w BAT
#training_flag = False #jezeli True to trenowane sa modele, jezeli nie to wykorzystywane sa poprzednie modele
optimize_target_flag = False #flaga mowiaca czy ma byc robiona optymalizacja targetu
#models_list = ['LogisticRegression','XGBoost','LightGBM'] #lista modeli
models_list = ['LightGBM'] #lista modeli



from typing import Counter
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import os
import ta
import matplotlib.pyplot as plt
import copy
from importlib import reload
import pickle
from sklearn.utils import resample, shuffle



import _00_utils as utils
import _02_utils_target as utils_target
import _03_utils_features as utils_features
import _04_utils_abt_dev_oot as utils_abt_dev_oot
import _05_utils_models as utils_models
import _06_prophet as utils_prophet
#reload(utils_features)



pd.options.mode.chained_assignment = None  # default='warn'
current_date = datetime.date(datetime.now()).strftime('%Y%m%d')
transaction_cost = 0.0029 #0.29%
random_state = 11235
my_n_splits = 5
my_scoring = 'f1'
xlsx_file = '_' + str(current_date) + '.xlsx'


##########################################################################################

#stworzenie katalogu
path_output = utils.prepare_directory(current_date)
print(path_output)
#path_output = './20210816'

#https://xlsxwriter.readthedocs.io/example_pandas_positioning.html
writer = pd.ExcelWriter(path_output + '/' + xlsx_file, engine='xlsxwriter')

#pobierz dane
#pd_df_base = yf.Ticker("^GSPC").history(period="max")
pd_df_base = yf.Ticker("^GSPC").history(start="1985-01-01")
#S&P 500 (^GSPC) SNP - SNP Real Time Price. Currency in USD 
#od 1927-12-30

#pd_df_base.to_pickle(path_output + '/pd_df_base.pickle')
#pd_df_base = pd.read_pickle(path_output + '/pd_df_base.pickle')

"""
Wybrany zostal ^GSPC poniewaz:
1. jest to odwzorowanie iindeksu wiec np. zawiera poprawny wolumen
2. jest w USD i w takiej walucie bede miec ETF w Bossa
3. ETF sa zgodne z ^SP500TR, ale ten nie ma wolumenu
4. roznie na moje potrzeby pomiedzy ^GSPC i ^SP500TR sa zaniedbywalne

GSPC = yf.Ticker("^GSPC").history(period="max") 
#S&P 500 (^GSPC) SNP - SNP Real Time Price. Currency in USD 
#od 1927-12-30
GSPC.tail(5)
GSPC.index.min()
GSPC['GSPC'] = GSPC['Close']
GSPC = GSPC[['GSPC']]

VUAA = yf.Ticker("VUAA.L").history(period="max") 
#Vanguard S&P 500 UCITS ETF acc USD (VUAA)
#od 2019-05-14
VUAA.tail(5)
VUAA.index.min()
VUAA['VUAA'] = VUAA['Close']
VUAA = VUAA[['VUAA']]

SP500TR = yf.Ticker("^SP500TR").history(period="max") 
#S&P 500 (TR) (^SP500TR) SNP - SNP Real Time Price. Currency in USD 
#od 1988-01-04
SP500TR.tail(5)
SP500TR.index.min()
SP500TR['SP500TR'] = SP500TR['Close']
SP500TR = SP500TR[['SP500TR']]

SPY = yf.Ticker("SPY").history(period="max") 
#SPDR S&P 500 ETF Trust (SPY) 
#od 1993-01-29
SPY.index.min()
SPY['SPY'] = SPY['Close']
SPY = SPY[['SPY']]

all = VUAA.join(GSPC, how="inner")
all = all.join(SPY, how="inner")
all = all.join(SP500TR, how="inner")

all.index.min()

all['SPY_roc'] = all['SPY'] / float(all.head(1)['SPY']) - 1
all['VUAA_roc'] = all['VUAA'] / float(all.head(1)['VUAA']) - 1
all['GSPC_roc'] = all['GSPC'] / float(all.head(1)['GSPC']) - 1
all['SP500TR_roc'] = all['SP500TR'] / float(all.head(1)['SP500TR']) - 1

corr_matrix = all.corr()
corr_matrix


all[['VUAA_roc','GSPC_roc','SPY_roc','SP500TR_roc']].plot()
plt.show()

"""

pd_df_vix = yf.Ticker("^VIX").history(period="max") #CBOE Volatility Index (^VIX) od 1990-01-02
#pd_df_vix.to_pickle(path_output + '/pd_df_vix.pickle')
#pd_df_vix = pd.read_pickle(path_output + '/pd_df_vix.pickle')

if optimize_target_flag == True:
    #optymalizacja parametrow dla wyznaczenia targetu
    pd_df_target_optimize = utils_target.target_optimize(
        copy.copy(pd_df_base), 
        param_atr_factor=5, 
        param_window=5, 
        param_smooth=5, 
        transaction_cost=transaction_cost
        )

    tuple_target_param = utils_target.target_optimize_set(copy.copy(pd_df_target_optimize), cnt_changes_per_year = 12, writer = writer, pd_df_base = copy.copy(pd_df_base), transaction_cost=transaction_cost)
else:
    tuple_target_param = (1,3,4)

#generowanie targetu dla optymalnych parametrow i zapis
pd_df_target = utils_target.target_generate(
    df=copy.copy(pd_df_base), 
    param_atr_base = 14, 
    param_atr_factor = tuple_target_param[0],
    param_window = tuple_target_param[1],
    param_smooth = tuple_target_param[2]    
    )[['target']]

#pd_df_target.to_pickle(path_output + '/pd_df_target.pickle')
#pd_df_target = pd.read_pickle(path_output + '/pd_df_target.pickle')
print(pd_df_target['target'].value_counts(normalize=True))
print(pd_df_target['target'].value_counts(normalize=False))

#generowanie featersow

### prophet

#wczytanie danych historycznych
prophet_output_df_old = pd.read_pickle('./_models/prophet_prediction.pickle')
prophet_output_df_old.columns
prophet_dates_list_old = list(prophet_output_df_old.reset_index()['Date'].dt.strftime('%Y-%m-%d'))
prophet_dates_list_old = prophet_dates_list_old[:-3] #usuniecie trzech ostatnich dat ma na celu zapewnienie ze bazujemy na calych dniach

#daty z bazowej tabeli
prophet_dates_list = list(pd_df_base.reset_index()['Date'].dt.strftime('%Y-%m-%d'))
prophet_dates_list = list(pd_df_base[pd_df_base.index >= '1990-01-01'].reset_index()['Date'].dt.strftime('%Y-%m-%d'))
#prophet_dates_list = [x for x in prophet_dates_list if x.startswith('199')]

#wygenerowanie danych dla brakujacych dat
prophet_dates_list_new = [x for x in prophet_dates_list if x not in prophet_dates_list_old]
prophet_output_list_new = [utils_prophet.prophet_features(df = copy.copy(pd_df_base), yyyymmdd = x, initial_no_of_days = 500, future_no_of_days = 30, tuple_target_param = tuple_target_param) for x in prophet_dates_list_new]
prophet_output_df_new = pd.concat(prophet_output_list_new).set_index('Date')

#polaczenie danych
prophet_output_df_all = pd.concat([prophet_output_df_old, prophet_output_df_new]).sort_index()

#zapis danych
pickle.dump(prophet_output_df_all, open('./_models/prophet_prediction.pickle', 'wb')) #zapis do folderu z modelami
pickle.dump(prophet_output_df_all, open(path_output + '/prophet_prediction.pickle', 'wb')) #zapis do folderu z data



#na podstawie pd_df_base
pd_df_base_features = utils_features.add_all_ta_features_extended(copy.copy(pd_df_base), prefix='base_', open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
#pd_df_base_features.to_pickle(path_output + '/pd_df_base_features.pickle')
#pd_df_base_features = pd.read_pickle(path_output + '/pd_df_base_features.pickle')
pd_df_base_features.shape
pd_df_base_features.columns

#na podstawie pd_df_vix
pd_df_vix_features = utils_features.add_all_ta_features_extended(copy.copy(pd_df_vix), prefix='vix_', open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
#pd_df_vix_features.to_pickle(path_output + '/pd_df_vix_features.pickle')11111111111
#pd_df_vix_features = pd.read_pickle(path_output + '/pd_df_vix_features.pickle')
pd_df_vix_features.shape

#Dollar Bars
pd_df_dollar = copy.copy(pd_df_base)
pd_df_dollar['Open_dollar'] = pd_df_dollar['Open'] * pd_df_dollar['Volume'] / 1000000
pd_df_dollar['High_dollar'] = pd_df_dollar['High'] * pd_df_dollar['Volume'] / 1000000
pd_df_dollar['Low_dollar'] = pd_df_dollar['Low'] * pd_df_dollar['Volume'] / 1000000
pd_df_dollar['Close_dollar'] = pd_df_dollar['Close'] * pd_df_dollar['Volume'] / 1000000
pd_df_dollar_features = utils_features.add_all_ta_features_extended(copy.copy(pd_df_dollar), prefix='dollar_', open="Open_dollar", high="High_dollar", low="Low_dollar", close="Close_dollar", volume="Volume", fillna=True)
pd_df_dollar_features.shape

#generowanie zmiennych na podstawie lag
pd_df_base_lag_pr = utils_features.lag_pr(df = copy.copy(pd_df_base[['Close']]), prefix = 'base_Close_lag_pr_', param_days = 7)
pd_df_dollar_lag_pr = utils_features.lag_pr(df = copy.copy(pd_df_dollar[['Close']]), prefix = 'dollar_Close_lag_pr_', param_days = 7)
pd_df_vix_lag_pr = utils_features.lag_pr(df = copy.copy(pd_df_vix[['Close']]), prefix = 'vix_Close_lag_pr_', param_days = 7)

#generowanei danych w oparciu o counter
pd_df_base_counter_5 = utils_features.counter_features(df=copy.copy(pd_df_base), prefix='base_5_', open='Open', high='High', low='Low', close='Close', window=5)
pd_df_base_counter_10 = utils_features.counter_features(df=copy.copy(pd_df_base), prefix='base_10_', open='Open', high='High', low='Low', close='Close', window=10)

pd_df_dollar_counter_5 = utils_features.counter_features(df=copy.copy(pd_df_dollar), prefix='dollar_5_', open='Open', high='High', low='Low', close='Close', window=5)
pd_df_dollar_counter_10 = utils_features.counter_features(df=copy.copy(pd_df_dollar), prefix='dollar_10_', open='Open', high='High', low='Low', close='Close', window=10)

pd_df_vix_counter_5 = utils_features.counter_features(df=copy.copy(pd_df_vix), prefix='vix_5_', open='Open', high='High', low='Low', close='Close', window=5)
pd_df_vix_counter_10 = utils_features.counter_features(df=copy.copy(pd_df_vix), prefix='vix_10_', open='Open', high='High', low='Low', close='Close', window=10)

#przygotwanie ABT
pd_df_vix_features.shape
pd_df_abt = pd_df_base_features.join(pd_df_vix_features, how="inner")
pd_df_abt.shape #tyle samo co pd_df_vix_features wiec ok

#ponizsze zalatwione przez laczenie "inner"
#pd_df_abt = pd_df_base_features.join(pd_df_vix_features, how="left")
#pd_df_abt = pd_df_abt[pd_df_abt.index >= min(pd_df_vix_features.index)] #to jeszcze nie wiadomo czy dziala
pd_df_abt = pd_df_abt.join(pd_df_dollar_features, how="left")

pd_df_abt = pd_df_abt.join(pd_df_base_lag_pr, how="left")
pd_df_abt = pd_df_abt.join(pd_df_dollar_lag_pr, how="left")
pd_df_abt = pd_df_abt.join(pd_df_vix_lag_pr, how="left")
pd_df_abt = pd_df_abt.join(prophet_output_df_all, how="left")

pd_df_abt = pd_df_abt.join(pd_df_base_counter_5, how="left")
pd_df_abt = pd_df_abt.join(pd_df_base_counter_10, how="left")
pd_df_abt = pd_df_abt.join(pd_df_dollar_counter_5, how="left")
pd_df_abt = pd_df_abt.join(pd_df_dollar_counter_10, how="left")
pd_df_abt = pd_df_abt.join(pd_df_vix_counter_5, how="left")
pd_df_abt = pd_df_abt.join(pd_df_vix_counter_10, how="left")

#pd_df_abt.shape
#pd_df_abt.columns

#usuniecie wysoce skorelowanych zmiennych
corr_matrix = pd_df_abt.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
pd_df_abt = pd_df_abt.drop(pd_df_abt[to_drop], axis=1)
pd_df_abt.shape

to_drop

pd_df_abt = pd_df_abt.join(pd_df_target, how="left")
#pd_df_abt.to_pickle(path_output + '/pd_df_abt.pickle')
#pd_df_abt = pd.read_pickle(path_output + '/pd_df_abt.pickle')

#wygenerowanie podsumowania dla ABT
pd_df_abt_describe = utils_abt_dev_oot.abt_summary(pd_df_abt)
pd_df_abt_describe.to_excel(writer, sheet_name='pd_df_abt_describe') 

pd_df_dev, pd_df_dev_upsample, pd_df_oot, pd_df_dev_oot_summary = utils_abt_dev_oot.generate_dev_oot(
    df = pd_df_abt, 
    break_date = pd_df_abt.tail(250).index.min(), #'2020-01-01', do OOT bierzemy ostatnie 250 sesji czyli okolo 1 rok
    upsample_target_share = 0.05, 
    random_state = random_state
)
#pd_df_dev.to_pickle(path_output + '/pd_df_dev.pickle')
#pd_df_dev_upsample.to_pickle(path_output + '/pd_df_dev_upsample.pickle')
#pd_df_oot.to_pickle(path_output + '/pd_df_oot.pickle')
pd_df_dev_oot_summary.to_excel(writer, sheet_name='pd_df_dev_oot_summary') 

#########################################################################################
#modele

if training_flag == True:

    for model_iter in models_list:
        (clf_final, pd_df_prediction, oot_score) = utils_models.create_model(
            model_name = model_iter, 
            df_dev = copy.copy(pd_df_dev), #jezeli bym chcial upsample, to trzeba oddzielna liste modeli
            df_oot = copy.copy(pd_df_oot), 
            random_state = random_state, 
            n_splits = my_n_splits, 
            scoring = my_scoring, 
            writer = writer
        )

        pickle.dump((clf_final, oot_score, str(current_date)), open(path_output + '/' + model_iter + '.model', 'wb'))
        pickle.dump((clf_final, oot_score, str(current_date)), open('./_models' + '/' + model_iter + '.model', 'wb')) #zapis do folderu z modelami
        pd_df_prediction.to_pickle(path_output + '/' + model_iter + '_prediction.pickle')
        utils_models.model_evaluate(pd_df_prediction=copy.copy(pd_df_prediction), pd_df_base=copy.copy(pd_df_base), model_name=model_iter, transaction_cost=transaction_cost,  writer=writer)

    #ensemble
    prediction_list = []
    for model_name in models_list:
        prediction_list.append(
                (
                    pickle.load(open(path_output + '/' + model_name + '_prediction.pickle', 'rb')),
                    pickle.load(open(path_output + '/' + model_name + '.model', 'rb'))[1],
                    pickle.load(open(path_output + '/' + model_name + '.model', 'rb'))[2]
                )
            )

    pd_df_ensemble = utils_models.ensemble_generate(prediction_list = prediction_list, models_list=models_list)
    
    utils_models.ensemble_evaluate(pd_df_ensemble=copy.copy(pd_df_ensemble), pd_df_base=copy.copy(pd_df_base), transaction_cost=transaction_cost, writer=writer)



#predyckja dla najnowsych danych
pd_df_pred = copy.copy(pd_df_oot.tail(25))

#wygenerowanie danych do sprawdzenia target_evaluate
#pd_df_pred = copy.copy(pd_df_oot)
#dopisanie  CLose
#pd_df_pred = pd_df_pred.join(pd_df_base[['Close']])
#tmp = utils_target.target_evaluate(copy.copy(pd_df_pred), 'target', transaction_cost)
#tmp_0 = tmp[0]
#tmp_1 = tmp[1]
#tmp_1 = tmp_1.join(pd_df_pred[['Close','target']])
#tmp_0.to_excel(writer, sheet_name='spr_0') 
#tmp_1.to_excel(writer, sheet_name='spr_1') 

prediction_list = []
for model_name in models_list:
    prediction_list.append(utils_models.prediction(df = pd_df_pred, model_name = model_name))

pd_df_prediction_new = utils_models.ensemble_generate(prediction_list = prediction_list, models_list=models_list)
pd_df_prediction_new = pd_df_prediction_new.join(pd_df_base, how="left")
pd_df_prediction_new.to_excel(writer, sheet_name='prediction') 

#generuj wersje modeli
model_version_list = []
for model_name in models_list:
    model_version_list.append(
        (model_name,
        pickle.load(open('./_models' + '/' + model_name + '.model', 'rb'))[2])
        )

pd_df_model_version = pd.DataFrame(model_version_list, columns=['model', 'version'])
pd_df_model_version.to_excel(writer, sheet_name='model_version') 

writer.save()

print(pd_df_prediction_new[['LightGBM_pred','LightGBM_prob_1']])

print('Koniec!')

#################################
#if __name__ == "__main__":

    