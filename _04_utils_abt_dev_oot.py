

import pandas as pd
import numpy as np
from sklearn.utils import resample, shuffle

def abt_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    generuje podsumowanie dla df

    Parameters
    ----------
    df: pd.DataFrame
        obiekt wejsciowy
    Returns
    -------
    df_describe: pd.DataFrame
        obiekt wejsciowy zawierajacy podsumowanie dla df
    """
    df_describe = df.describe().transpose()
    df_describe['nunique'] = df.nunique()
    df_describe['nunique_with_NaN'] = df.nunique(dropna=False)
    df_describe['cnt_NaN'] = df.isnull().sum()
    
    return df_describe



def generate_dev_oot(df: pd.DataFrame, break_date: str, upsample_target_share: float, random_state: int) -> tuple:
    """
    generuje zbiory dev, dev_upsample, oot oraz zapisuje krotkie podsumowanie do XLSX

    Parameters
    ----------
    df: pd.DataFrame
        obiekt wejsciowy
    break_date: str
        data podzialu dev i oot 
    upsample_target_share: float
        parametr okreslajacy udzial targetu po upsample
    random_state: int
        
    Returns
    -------
    tuple
        (df_dev, df_dev_upsample, df_oot, df_summary)
    """

    #out of time
    df_oot = df[df.index >= break_date]

    #development bez upsammple
    df_dev = df[df.index < break_date]

    #development z upsammple
    df_dev_0 = df_dev[df_dev['target']==0]
    df_dev_1 = df_dev[df_dev['target']==1]

    df_dev_1_upsample = resample(df_dev_1, 
        replace=True,                               # losowanie ze zwracaniem
        n_samples=round(df_dev_0.shape[0] * upsample_target_share),  # target ma snatnowic 5%
        random_state=random_state
        )

    df_dev_upsample = pd.concat([df_dev_0, df_dev_1_upsample])

    #przygotowanie podsumowania
    df_summary = pd.DataFrame(
        {
            'row_cnt': [df.shape[0], df_dev.shape[0], df_dev_upsample.shape[0], df_oot.shape[0]],
            'col_cnt': [df.shape[1], df_dev.shape[1], df_dev_upsample.shape[1], df_oot.shape[1]],
            'target_cnt': [df.target.sum(), df_dev.target.sum(), df_dev_upsample.target.sum(), df_oot.target.sum()],
            'target_share': [None, None, None, None],
            'date_min': [df.index.min(), df_dev.index.min(), df_dev_upsample.index.min(), df_oot.index.min()],
            'date_max': [df.index.max(), df_dev.index.max(), df_dev_upsample.index.max(), df_oot.index.max()],
            'df_name': ['pd_df_abt', 'df_dev', 'df_dev_upsample', 'df_oot']
        }, 
        index = ['total','dev','dev_upsample','oot']
        )

    df_summary['target_share'] = df_summary['target_cnt'] / df_summary['row_cnt']
    return (df_dev, df_dev_upsample, df_oot, df_summary)
