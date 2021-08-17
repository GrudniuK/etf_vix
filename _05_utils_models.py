

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils import shuffle
import pickle
from io import BytesIO
import copy



import _02_utils_target as utils_target



def create_model(model_name: str, df_dev: pd.DataFrame, df_oot: pd.DataFrame, random_state: int, n_splits: int, scoring: str, writer: object) -> tuple:
    """
    Tworzy model i zwraca predykcje OOF i OOT

    Parameters
    ----------
    model_name: str
        nazwa algorytmu
    df_dev: pd.DataFrame
        zbior do trenowania modelu
    df_oot: pd.DataFrame
        zbior do oceny jakosci zbudowanego modelu
    random_state: int
        w celu zapewnienia powtarzalnosci
    n_splits: int
        liczba podzialow w StratifiedKFold
    scoring: str
        miara do oceny jaksci modelu np. 'f1', 'roc_auc'
    writer: object
        obiekt do zapisu XLSX
    
    df: pd.DataFrame
        obiekt wejsciowy
    xlsx_path_name: str
        sciezka + nazwa pliku XLSX
    sheet_name: str
        nazwa arkusza w XLSX
    Returns
    -------
    (clf_final, pd_df_prediction, pd_df_final_results['oot_score']): tuple
        clf_final: object
            finalny klasyfikator
        pd_df_prediction: pd.DataFrame
            wyniki OOF i OOT dla optymalnego modelu
        pd_df_final_results['oot_score']: float
            wynik score OOT (do wazenia)
    """

    """
    model_name = 'LightGBM' 
    df_dev = copy.copy(pd_df_dev_upsample)
    df_oot = copy.copy(pd_df_oot)
    random_state = 1
    n_splits = 2 
    scoring = 'f1'
    """

    #przygotowanie X i y dla dev
    pd_df_dev = shuffle(df_dev, random_state=random_state)

    scale_pos_weight = pd_df_dev['target'].value_counts(normalize=False)[0] / pd_df_dev['target'].value_counts(normalize=False)[1]

    X = df_dev.drop('target', axis='columns')
    y = df_dev.target

    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)

    if model_name == 'LogisticRegression_upsample':
        clf_GridSearchCV_input = make_pipeline(
            SimpleImputer(strategy='mean'), 
            StandardScaler(),
            LogisticRegression(penalty='elasticnet', solver='saga', random_state=random_state))

        #clf_GridSearchCV_input.get_params().keys()
        my_param_grid = {
            'logisticregression__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
            'logisticregression__C':[0.0001,0.1,1,10,10000]
            }

    if model_name == 'LogisticRegression':
        clf_GridSearchCV_input = make_pipeline(
            SimpleImputer(strategy='mean'), 
            StandardScaler(),
            LogisticRegression(penalty='elasticnet', solver='saga', random_state=random_state))

        #clf_GridSearchCV_input.get_params().keys()
        my_param_grid = {
            'logisticregression__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
            'logisticregression__C':[0.0001,0.1,1,10,10000]
            }

    if model_name == 'XGBoost_upsample':
        clf_GridSearchCV_input = make_pipeline(
            SimpleImputer(strategy='mean'), 
            StandardScaler(),
            XGBClassifier(eval_metric = 'logloss', seed = random_state, use_label_encoder = False))

        #clf_GridSearchCV_input.get_params().keys()
        #https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
        #https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
        ##https://xgboost.ai/
        ##https://xgboost.readthedocs.io/en/latest/python/python_api.html
        my_param_grid = {
            'xgbclassifier__n_estimators': list(range(50, 160, 10)),
            'xgbclassifier__max_depth': [3, 6, 9],
            'xgbclassifier__colsample_bytree': [0.7, 1],
            'xgbclassifier__learning_rate': [0.01, 0.1, 1.0]
            }

    if model_name == 'XGBoost':
        clf_GridSearchCV_input = make_pipeline(
            SimpleImputer(strategy='mean'), 
            StandardScaler(),
            XGBClassifier(eval_metric = 'logloss', seed = random_state, use_label_encoder = False, scale_pos_weight = scale_pos_weight))

        #clf_GridSearchCV_input.get_params().keys()
        #https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
        #https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
        ##https://xgboost.ai/
        ##https://xgboost.readthedocs.io/en/latest/python/python_api.html
        my_param_grid = {
            'xgbclassifier__n_estimators': list(range(50, 160, 10)),
            'xgbclassifier__max_depth': [3, 6, 9],
            'xgbclassifier__colsample_bytree': [0.7, 1],
            'xgbclassifier__learning_rate': [0.01, 0.1, 1.0]
            }
    
    if model_name == 'LightGBM_upsample':
        clf_GridSearchCV_input = make_pipeline(
            SimpleImputer(strategy='mean'), 
            StandardScaler(),
            LGBMClassifier(random_state = random_state))

        #clf_GridSearchCV_input.get_params().keys()
        #https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
        #https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/
        ##https://github.com/microsoft/LightGBM
        ##https://lightgbm.readthedocs.io/en/latest/Python-API.html
        my_param_grid = {
            'lgbmclassifier__n_estimators': list(range(50, 160, 10)),
            'lgbmclassifier__max_depth': [3, 6, 9],
            'lgbmclassifier__learning_rate': [0.01, 0.1, 1.0] 
            }
    
    if model_name == 'LightGBM':
        clf_GridSearchCV_input = make_pipeline(
            SimpleImputer(strategy='mean'), 
            StandardScaler(),
            LGBMClassifier(random_state = random_state, is_unbalance = True))

        #clf_GridSearchCV_input.get_params().keys()
        #https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/
        #https://machinelearningmastery.com/light-gradient-boosted-machine-lightgbm-ensemble/
        ##https://github.com/microsoft/LightGBM
        ##https://lightgbm.readthedocs.io/en/latest/Python-API.html
        my_param_grid = {
            'lgbmclassifier__n_estimators': list(range(50, 160, 10)),
            'lgbmclassifier__max_depth': [3, 6, 9],
            'lgbmclassifier__learning_rate': [0.01, 0.1, 1.0] 
            }

    clf_GridSearchCV_output = GridSearchCV(
        clf_GridSearchCV_input, 
        param_grid = my_param_grid, 
        scoring=scoring,
        refit=True,    #do zastanowienia czy nie powinienem refitowac na calym zbiorze bez upsample 
        cv=skf, 
        verbose=1 
        )

    clf_GridSearchCV_output.fit(X, y)

    #wyniki dla CV
    pd_df_cv_results = pd.DataFrame(clf_GridSearchCV_output.cv_results_)

    #finalne wyniki
    pd_df_final_results = pd.DataFrame(clf_GridSearchCV_output.best_params_, index=[0])
    pd_df_final_results['cv_best_index'] = clf_GridSearchCV_output.best_index_
    pd_df_final_results['cv_refit_time'] = clf_GridSearchCV_output.refit_time_
    pd_df_final_results['cv_n_splits'] = clf_GridSearchCV_output.n_splits_
    pd_df_final_results['scorer'] = str(clf_GridSearchCV_output.scorer_)
    pd_df_final_results['cv_best_score'] = clf_GridSearchCV_output.best_score_

    #istotnosc zmiennych
    if model_name == 'LogisticRegression_upsample':
        pd_df_feature_importance = pd.DataFrame(
            list(clf_GridSearchCV_output.best_estimator_[2].coef_[0]),
            index = list(X.columns),
            columns = ['importance_cv']
        )
        pd_df_feature_importance['importance_cv_abs']=abs(pd_df_feature_importance['importance_cv'])
        pd_df_feature_importance = pd_df_feature_importance.sort_values('importance_cv_abs', axis=0, ascending=False)

    if model_name == 'LogisticRegression':
        pd_df_feature_importance = pd.DataFrame(
            list(clf_GridSearchCV_output.best_estimator_[2].coef_[0]),
            index = list(X.columns),
            columns = ['importance_cv']
        )
        pd_df_feature_importance['importance_cv_abs']=abs(pd_df_feature_importance['importance_cv'])
        pd_df_feature_importance = pd_df_feature_importance.sort_values('importance_cv_abs', axis=0, ascending=False)

    if model_name == 'XGBoost_upsample':
        pd_df_feature_importance = pd.DataFrame(
            list(clf_GridSearchCV_output.best_estimator_[2].feature_importances_),
            index = list(X.columns),
            columns = ['importance_cv']
        )
        pd_df_feature_importance['importance_cv_abs']=abs(pd_df_feature_importance['importance_cv'])
        pd_df_feature_importance = pd_df_feature_importance.sort_values('importance_cv_abs', axis=0, ascending=False)

    if model_name == 'XGBoost':
        pd_df_feature_importance = pd.DataFrame(
            list(clf_GridSearchCV_output.best_estimator_[2].feature_importances_),
            index = list(X.columns),
            columns = ['importance_cv']
        )
        pd_df_feature_importance['importance_cv_abs']=abs(pd_df_feature_importance['importance_cv'])
        pd_df_feature_importance = pd_df_feature_importance.sort_values('importance_cv_abs', axis=0, ascending=False)

    if model_name == 'LightGBM_upsample':
        pd_df_feature_importance = pd.DataFrame(
            list(clf_GridSearchCV_output.best_estimator_[2].feature_importances_),
            index = list(X.columns),
            columns = ['importance_cv']
        )
        pd_df_feature_importance['importance_cv_abs']=abs(pd_df_feature_importance['importance_cv'])
        pd_df_feature_importance = pd_df_feature_importance.sort_values('importance_cv_abs', axis=0, ascending=False)

    if model_name == 'LightGBM':
        pd_df_feature_importance = pd.DataFrame(
            list(clf_GridSearchCV_output.best_estimator_[2].feature_importances_),
            index = list(X.columns),
            columns = ['importance_cv']
        )
        pd_df_feature_importance['importance_cv_abs']=abs(pd_df_feature_importance['importance_cv'])
        pd_df_feature_importance = pd_df_feature_importance.sort_values('importance_cv_abs', axis=0, ascending=False)

    #predykcja na OOF (out of fold)
    X_oof = df_dev.drop('target', axis='columns')
    y_oof = df_dev.target

    y_oof_predict = cross_val_predict(
        clf_GridSearchCV_output.best_estimator_, 
        X_oof, 
        y=y_oof, 
        cv=skf, 
        fit_params=None, 
        method='predict'
    )

    y_oof_predict_proba = cross_val_predict(
        clf_GridSearchCV_output.best_estimator_, 
        X_oof, 
        y=y_oof, 
        cv=skf, 
        fit_params=None, 
        method='predict_proba'
    )

    pd_df_oof_prediction = pd.DataFrame(y_oof)
    pd_df_oof_prediction['source'] = 'out_of_fold'
    pd_df_oof_prediction[model_name + '_pred'] = y_oof_predict
    pd_df_oof_prediction[model_name + '_prob_0'] = y_oof_predict_proba[:,0]
    pd_df_oof_prediction[model_name + '_prob_1'] = y_oof_predict_proba[:,1]

    #predykcja na OOT
    X_oot = df_oot.drop('target', axis='columns')
    y_oot = df_oot.target
    pd_df_final_results['oot_score'] = clf_GridSearchCV_output.score(X_oot, y_oot) #oddanie do finalnych wynikow

    y_oot_predict_proba = clf_GridSearchCV_output.predict_proba(X_oot)
    pd_df_oot_prediction = pd.DataFrame(y_oot)
    pd_df_oot_prediction['source'] = 'out_of_time'
    pd_df_oot_prediction[model_name + '_pred'] = clf_GridSearchCV_output.predict(X_oot)
    pd_df_oot_prediction[model_name + '_prob_0'] = y_oot_predict_proba[:,0]
    pd_df_oot_prediction[model_name + '_prob_1'] = y_oot_predict_proba[:,1]

    pd_df_prediction = pd.concat([pd_df_oof_prediction, pd_df_oot_prediction]).sort_index()

    #grupowanie - istotne przy upsample
    pd_df_prediction = pd_df_prediction.groupby(pd_df_prediction.index).agg({
        'target':'mean',
        'source':'max',
        model_name + '_pred':'mean',
        model_name + '_prob_0':'mean',
        model_name + '_prob_1':'mean',
    })
    pd_df_prediction[model_name + '_pred'] = np.where(pd_df_prediction[model_name + '_pred'] > 0.5, 1, 0)

    #budowa modelu na calym DEV+OOT z optymalnymi parametrami
    df_all = pd.concat([df_dev, df_oot])
    X_all = df_all.drop('target', axis='columns')
    y_all = df_all.target
    clf_final = clf_GridSearchCV_output.best_estimator_.fit(X_all, y_all)

    #zapis wynikow do XLSX
    start_row = 10 #zostawienie miejsca dla wynikow strategii, ktore beda wyliczone na zewnatrz
    pd_df_final_results.to_excel(writer, sheet_name=model_name, startrow=start_row, startcol=0) 
    pd_df_feature_importance.to_excel(writer, sheet_name=model_name, startrow=start_row + pd_df_final_results.shape[0] + 2, startcol=0)
    pd_df_cv_results.to_excel(writer, sheet_name=model_name, startrow=start_row + pd_df_final_results.shape[0] + 2, startcol=pd_df_feature_importance.shape[1] + 2)

    return (clf_final, pd_df_prediction, float(pd_df_final_results['oot_score']))


def model_evaluate(pd_df_prediction: pd.DataFrame, pd_df_base: pd.DataFrame, model_name: str, transaction_cost: float,  writer: object) -> None:
    """
    Tworzy podsumowania w XLSX dla predykcji z modelu

    Parameters
    ----------
    pd_df_prediction: pd.DataFrame
        wyniki OOF i OOT dla optymalnego modelu
    pd_df_base: pd.DataFrame
        OHCL na potrzeby dodania ceny Close
    model_name: str
        nazwa algorytmu
    transaction_cost

    writer: object
        obiekt do zapisu XLSX
    
    Returns
    -------
    None
    """
    
    #dopisanie  CLose
    pd_df_prediction = pd_df_prediction.join(pd_df_base[['Close']])

    #przygotowanie podsumowania i wykresow
    df_model_summary = pd.DataFrame()
    for sample, row in zip([['out_of_fold','out_of_time'], ['out_of_fold'], ['out_of_time']], [0, 25, 50]):
        #print(type(str(sample)), row)

        df_tmp = pd_df_prediction[pd_df_prediction['source'].isin(sample)]

        df_tmp_target = utils_target.target_evaluate(copy.copy(df_tmp), 'target', transaction_cost)
        df_tmp_target_0 = df_tmp_target[0]
        df_tmp_target_0['sample'] = str(sample)

        df_tmp_model = utils_target.target_evaluate(copy.copy(df_tmp), model_name + '_pred', transaction_cost)
        df_tmp_model_0 = df_tmp_model[0]
        df_tmp_model_0['sample'] = str(sample)

        df_model_summary = pd.concat([df_model_summary, df_tmp_target_0, df_tmp_model_0]) # do XLSX

        df_plot_by_time = df_tmp_target[1].join(df_tmp_model[1].drop('cagr_benchmark', axis='columns'))

        #generowanie wykresow i zapis do XLSX
        image_data = BytesIO()
        fig, ax = plt.subplots(figsize=(12.8,4.8))
        df_plot_by_time.plot(ax=ax)
        fig.savefig(image_data)
        worksheet = writer.sheets[model_name]
        worksheet.insert_image(row, 19, 'tmp', {'image_data': image_data})

    #zapisanie wynikow do XLSX
    df_model_summary.to_excel(writer, sheet_name=model_name, startrow=0, startcol=0)
    #plt.close("all")
    return None



def prediction(df: pd.DataFrame, model_name: str) -> tuple:
    """
    Tworzy model i zwraca predykcje OOF i OOT

    Parameters
    ----------
    df: pd.DataFrame
        ABT w celu predykcji dla najnowyszch dancyh
    model_name: str
        nazwa algorytmu

    Returns
    -------
    (df_prediction, oot_score): tuple
        df_prediction: pd.DataFrame
            predykcja dla ABT
        oot_score: float
            score z OOT - do stackingu (waga)
        model_version: str
            wersja modelu
    """

    model_obj = pickle.load(open('./_models' + '/' + model_name + '.model', 'rb'))
    X = df.drop('target', axis='columns')

    df_predict_proba = model_obj[0].predict_proba(X)
    df_prediction = pd.DataFrame(X.index)
    df_prediction[model_name + '_pred'] = model_obj[0].predict(X)
    df_prediction[model_name + '_prob_1'] = df_predict_proba[:,1]
    df_prediction = df_prediction.set_index('Date').sort_index()
    
    return(df_prediction, model_obj[1], model_obj[2])



def ensemble_generate(prediction_list: list, models_list: list) -> pd.DataFrame:
    """
    Tworzy pd.DataFrame z zebranymi predykcjami i generuje stacking

    Parameters
    ----------
    prediction_list: list
        lista obiektow. Kazdy obiekt zawiera
            df_prediction: pd.DataFrame
                predykcja dla ABT
            oot_score: float
                score z OOT - do stackingu (waga)
            model_version: str
                wersja modelu

    models_list: list
        lista modeli

    Returns
    -------
    pd_df_pred: pd.DtaFrame
        predykcje dla wszystkich modeli i stckingu
    """

    pd_df_list = []
    oot_score_list = []
    version_list = []
    for i in range(len(prediction_list)):
        pd_df_list.append(prediction_list[i][0])
        oot_score_list.append(prediction_list[i][1])
        version_list.append(prediction_list[i][2])

    pd_df_pred = pd.concat(pd_df_list, axis = 1)

    #stacking
    models_list_pred = [e + '_pred' for e in models_list]
    models_list_prob_1 = [e + '_prob_1' for e in models_list]

    #srednia wazona pred_1 gdzie waga to oot_score
    pd_df_pred['weight_prob_1'] = (pd_df_pred[models_list_prob_1] * oot_score_list).sum(axis=1) / sum(oot_score_list)
    pd_df_pred['weight_pred_1'] = np.where(pd_df_pred['weight_prob_1'] > 0.5, 1, 0)

    #srednia arytmetyczna prob_1
    pd_df_pred['average_prob_1'] = pd_df_pred[models_list_prob_1].sum(axis=1) / len(models_list)
    pd_df_pred['average_pred_1'] = np.where(pd_df_pred['average_prob_1'] > 0.5, 1, 0)

    #glosowanie na podstawie pred_1
    pd_df_pred['voting_prob_1'] = pd_df_pred[models_list_pred].sum(axis=1) / 3
    pd_df_pred['voting_pred_1'] = np.where(pd_df_pred['voting_prob_1'] > 0.5, 1, 0)

    #wygladzenie predykcji dla momdeli podstawowych
    for model in models_list:
        pd_df_pred[model + '_smooth_pred'] = np.where(
            (pd_df_pred[model + '_pred'] == 1) & (pd_df_pred[model + '_pred'].shift(1) == 1) & (pd_df_pred[model + '_pred'].shift(2) == 1),
            1, 
            0
        )

    return pd_df_pred



def ensemble_evaluate(pd_df_ensemble: pd.DataFrame, pd_df_base: pd.DataFrame, transaction_cost: float,  writer: object) -> None:
    """
    Tworzy podsumowania w XLSX dla stackingu

    Parameters
    ----------
    pd_df_ensemble: pd.DataFrame
        wyniki z ensemble_generate
    pd_df_base: pd.DataFrame
        OHCL na potrzeby dodania ceny Close
    transaction_cost

    writer: object
        obiekt do zapisu XLSX
    
    Returns
    -------
    None
    """

    pd_df_ensemble = pd_df_ensemble.T.drop_duplicates().T #to chyba można zrobić lepiej!
    pd_df_ensemble = pd_df_ensemble[pd_df_ensemble['source'].isin(['out_of_time'])]
    
    ensemlbe_list = [s for s in pd_df_ensemble.columns if "pred" in s] #lista kolumn dla ktorych bedzie generowane podsumowanie

    #dopisanie CLose na potrzeby target_evaluate
    pd_df_ensemble = pd.merge(pd_df_ensemble, pd_df_base, how='left', left_index=True, right_index=True)

    #generowanie dla kazdego modelu
    ensemble_summary_list = []
    ensemble_cagr_list = []
    for target_variable in ensemlbe_list:
        tmp = utils_target.target_evaluate(df = pd_df_ensemble, target_variable = target_variable, transaction_cost = transaction_cost)
        ensemble_summary_list.append(tmp[0]) #podsumowanie
        ensemble_cagr_list.append(tmp[1]) #time series

    df_ensemble_summary = pd.concat(ensemble_summary_list)
    df_ensemble_cagr = pd.concat(ensemble_cagr_list, axis = 1)

    #usuniecie duplikatow kolumn
    df_ensemble_cagr = df_ensemble_cagr.T.drop_duplicates().T #to chyba można zrobić lepiej!

    #zapisanie wynikow do XLSX
    df_ensemble_summary.to_excel(writer, sheet_name="ensemble", startrow=0, startcol=0)
    
    #generowanie wykresow i zapis do XLSX
    image_data = BytesIO()
    fig, ax = plt.subplots(figsize=(12.8,4.8))
    df_ensemble_cagr.plot(ax=ax)
    fig.savefig(image_data)
    worksheet = writer.sheets["ensemble"]
    worksheet.insert_image(0, 19, 'tmp', {'image_data': image_data})
    
    #plt.close("all")
    return None

    