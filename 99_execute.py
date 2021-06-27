
#.venv\scripts\activate
#instalowanie pakietow:
#python -m pip install pandas
#pip install ta
#   https://github.com/bukosabino/ta
#   https://technical-analysis-library-in-python.readthedocs.io/en/latest/
#pip install yfinance
# https://github.com/ranaroussi/yfinance
#pip install matplotlib
#pip install -U scikit-learn
#pip install xgboost
#pip install lightgbm

#deactivate
#pip freeze

# pozyskanie danych OHCL:
#https://blog.quantinsti.com/historical-market-data-python-api/




from typing import Counter
import yfinance as yf
import pandas as pd
import numpy as np
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
#reload(utils_features)

optimize_target_flag = False #flaga mowiaca czy ma byc robiona optymalizacja targetu
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
#path_output = './20210606'

#https://xlsxwriter.readthedocs.io/example_pandas_positioning.html
writer = pd.ExcelWriter(path_output + '/' + xlsx_file, engine='xlsxwriter')

#pobierz dane
pd_df_base = yf.Ticker("SPY").history(period="max") #SPDR S&P 500 ETF Trust (SPY) od 1993-01-29
pd_df_base.to_pickle(path_output + '/pd_df_base.pickle')
#pd_df_base = pd.read_pickle(path_output + '/pd_df_base.pickle')

pd_df_vix = yf.Ticker("^VIX").history(period="max") #CBOE Volatility Index (^VIX) od 1990-01-02
pd_df_vix.to_pickle(path_output + '/pd_df_vix.pickle')
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

pd_df_target.to_pickle(path_output + '/pd_df_target.pickle')
#pd_df_target = pd.read_pickle(path_output + '/pd_df_target.pickle')
print(pd_df_target['target'].value_counts(normalize=True))
print(pd_df_target['target'].value_counts(normalize=False))

#generowanie featersow

#na podstawie pd_df_base
pd_df_base_features = utils_features.add_all_ta_features_extended(copy.copy(pd_df_base), prefix='base_', open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
pd_df_base_features.to_pickle(path_output + '/pd_df_base_features.pickle')
#pd_df_base_features = pd.read_pickle(path_output + '/pd_df_base_features.pickle')
pd_df_base_features.shape

#na podstawie pd_df_vix
pd_df_vix_features = utils_features.add_all_ta_features_extended(copy.copy(pd_df_vix), prefix='vix_', open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
pd_df_vix_features.to_pickle(path_output + '/pd_df_vix_features.pickle')
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

#przygotwanie ABT
pd_df_abt = pd_df_base_features.join(pd_df_vix_features, how="left")
pd_df_abt = pd_df_abt[pd_df_abt.index >= min(pd_df_vix_features.index)] #to jeszcze nie wiadomo czy dziala
pd_df_abt = pd_df_abt.join(pd_df_dollar_features, how="left")

pd_df_abt = pd_df_abt.join(pd_df_base_lag_pr, how="left")
pd_df_abt = pd_df_abt.join(pd_df_dollar_lag_pr, how="left")
pd_df_abt = pd_df_abt.join(pd_df_vix_lag_pr, how="left")

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
pd_df_abt.to_pickle(path_output + '/pd_df_abt.pickle')
#pd_df_abt = pd.read_pickle(path_output + '/pd_df_abt.pickle')

#wygenerowanie podsumowania dla ABT
pd_df_abt_describe = utils_abt_dev_oot.abt_summary(pd_df_abt)
pd_df_abt_describe.to_excel(writer, sheet_name='pd_df_abt_describe') 



pd_df_dev, pd_df_dev_upsample, pd_df_oot, pd_df_dev_oot_summary = utils_abt_dev_oot.generate_dev_oot(pd_df_abt, '2020-01-01', 0.05, random_state)
pd_df_dev.to_pickle(path_output + '/pd_df_dev.pickle')
pd_df_dev_upsample.to_pickle(path_output + '/pd_df_dev_upsample.pickle')
pd_df_oot.to_pickle(path_output + '/pd_df_oot.pickle')
pd_df_dev_oot_summary.to_excel(writer, sheet_name='pd_df_dev_oot_summary') 
#tutaj dorobic jeszcze wyniki dla oot

#########################################################################################
#modele

#LogisticRegression
(clf_final, pd_df_prediction) = utils_models.create_model(
    model_name = 'LogisticRegression', 
    df_dev = copy.copy(pd_df_dev_upsample), 
    df_oot = copy.copy(pd_df_oot), 
    random_state = random_state, 
    n_splits = my_n_splits, 
    scoring = my_scoring, 
    writer = writer
)

pickle.dump(clf_final, open(path_output + '/LogisticRegression.model', 'wb'))
pd_df_prediction.to_pickle(path_output + '/LogisticRegression_prediction.pickle')
#pd_df_prediction = pd.read_pickle(path_output + '/pd_df_prediction.pickle')

utils_models.model_evaluate(pd_df_prediction=copy.copy(pd_df_prediction), pd_df_base=copy.copy(pd_df_base), model_name='LogisticRegression', transaction_cost=transaction_cost,  writer=writer)

#XGBoost_upsample
(clf_final, pd_df_prediction) = utils_models.create_model(
    model_name = 'XGBoost_upsample', 
    df_dev = copy.copy(pd_df_dev_upsample), 
    df_oot = copy.copy(pd_df_oot), 
    random_state = random_state, 
    n_splits = my_n_splits, 
    scoring = my_scoring, 
    writer = writer
)

pickle.dump(clf_final, open(path_output + '/XGBoost_upsample.model', 'wb'))
pd_df_prediction.to_pickle(path_output + '/XGBoost_upsample_prediction.pickle')
utils_models.model_evaluate(pd_df_prediction=copy.copy(pd_df_prediction), pd_df_base=copy.copy(pd_df_base), model_name='XGBoost_upsample', transaction_cost=transaction_cost,  writer=writer)

#XGBoost
(clf_final, pd_df_prediction) = utils_models.create_model(
    model_name = 'XGBoost', 
    df_dev = copy.copy(pd_df_dev), 
    df_oot = copy.copy(pd_df_oot), 
    random_state = random_state, 
    n_splits = my_n_splits, 
    scoring = my_scoring, 
    writer = writer
)

pickle.dump(clf_final, open(path_output + '/XGBoost.model', 'wb'))
pd_df_prediction.to_pickle(path_output + '/XGBoost_prediction.pickle')
utils_models.model_evaluate(pd_df_prediction=copy.copy(pd_df_prediction), pd_df_base=copy.copy(pd_df_base), model_name='XGBoost', transaction_cost=transaction_cost,  writer=writer)

#LightGBM_upsample
(clf_final, pd_df_prediction) = utils_models.create_model(
    model_name = 'LightGBM_upsample', 
    df_dev = copy.copy(pd_df_dev_upsample), 
    df_oot = copy.copy(pd_df_oot), 
    random_state = random_state, 
    n_splits = my_n_splits, 
    scoring = my_scoring, 
    writer = writer
)

pickle.dump(clf_final, open(path_output + '/LightGBM_upsample.model', 'wb'))
pd_df_prediction.to_pickle(path_output + '/LightGBM_upsample_prediction.pickle')
utils_models.model_evaluate(pd_df_prediction=copy.copy(pd_df_prediction), pd_df_base=copy.copy(pd_df_base), model_name='LightGBM_upsample', transaction_cost=transaction_cost,  writer=writer)

#LightGBM
(clf_final, pd_df_prediction) = utils_models.create_model(
    model_name = 'LightGBM', 
    df_dev = copy.copy(pd_df_dev), 
    df_oot = copy.copy(pd_df_oot), 
    random_state = random_state, 
    n_splits = my_n_splits, 
    scoring = my_scoring, 
    writer = writer
)

pickle.dump(clf_final, open(path_output + '/LightGBM.model', 'wb'))
pd_df_prediction.to_pickle(path_output + '/LightGBM_prediction.pickle')
utils_models.model_evaluate(pd_df_prediction=copy.copy(pd_df_prediction), pd_df_base=copy.copy(pd_df_base), model_name='LightGBM', transaction_cost=transaction_cost,  writer=writer)

writer.save()


#https://xgboost.readthedocs.io/en/latest/index.html
https://towardsdatascience.com/getting-started-with-xgboost-in-scikit-learn-f69f5f470a97
https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn

gridsearchcv xgboost early stopping
https://www.kaggle.com/yantiz/xgboost-gridsearchcv-with-early-stopping-supported



#end?









#############################################























1




#######################


#pobranie danych yfinance
gspc = yf.Ticker("^GSPC").history(period="max") #S&P 500 (^GSPC) od 1927-12-30
spy = yf.Ticker("SPY").history(period="max") #SPDR S&P 500 ETF Trust (SPY) od 1993-01-29
vix = yf.Ticker("^VIX").history(period="max") #CBOE Volatility Index (^VIX) od 1990-01-02



plt.scatter(pd_df_out['cagr_strategy'],pd_df_out['cnt_change'],c=pd_df_out['param_atr_factor'], s = 0.5)    
plt.legend()
plt.show()

fig, ax = plt.subplots()
scatter = plt.scatter(pd_df_out['cagr_strategy'],pd_df_out['cnt_change'],c=pd_df_out['param_atr_factor'], s = 0.5)    
legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
ax.add_artist(legend)
plt.show()





#celem powinno byc maksymalizowanie st zwrotu i minimalizowanie liczby transakcji




plt.figure()
width=1
width2=0.1
pricesup=base_label[base_label.target_color=='green']
pricesdown=base_label[base_label.target_color=='red']

plt.bar(pricesup.index,pricesup.Close-pricesup.Open,width,bottom=pricesup.Open,color='g')
plt.bar(pricesup.index,pricesup.High-pricesup.Close,width2,bottom=pricesup.Close,color='g')
plt.bar(pricesup.index,pricesup.Low-pricesup.Open,width2,bottom=pricesup.Open,color='g')

plt.bar(pricesdown.index,pricesdown.Close-pricesdown.Open,width,bottom=pricesdown.Open,color='r')
plt.bar(pricesdown.index,pricesdown.High-pricesdown.Open,width2,bottom=pricesdown.Open,color='r')
plt.bar(pricesdown.index,pricesdown.Low-pricesdown.Close,width2, bottom=pricesdown.Close,color='r')
plt.grid()

plt.show()





if __name__ == "__main__":

writer.save()