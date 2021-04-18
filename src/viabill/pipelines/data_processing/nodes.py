from typing import Any, Dict

import re
from unidecode import unidecode
import numpy as np
import pandas as pd
import time
import logging
import re

log = logging.getLogger(__name__)


def _process_customers(df_cust: pd.DataFrame, parameters: Dict):

    d_sex = {1: 'male', 2: 'female', 0: 'other'}
    df_cust.sex = df_cust.sex.apply(lambda x: d_sex[x])

    df_cust['residentialAddress_clean'] = df_cust.residentialAddress.apply(
        lambda x: re.sub(r'[0-9]+', '', x))
    df_cust['postalAddress_clean'] = df_cust.postalAddress.apply(
        lambda x: re.sub(r'[0-9]+', '', x) if x == x else '')
    df_cust['same_address'] = (
        df_cust.residentialAddress == df_cust.postalAddress).astype(int)

    return df_cust


def _process_transactions(df_trans: pd.DataFrame, parameters: Dict):

    # sample for debuging
    # df_trans = df_trans.sample(5000, random_state=1)

    df_trans['late'] = df_trans.filter(regex='payment').apply(
        lambda x: x == 1).any(axis=1).astype(int)

    df_trans['default'] = df_trans.filter(regex='payment').apply(
        lambda x: x == 2).any(axis=1).astype(int)

    # mark transactions from shop 113 as a fraud
    df_trans['fraud'] = 0
    df_trans.loc[df_trans.shopID == 113, 'fraud'] = 1

    # drop transaction marked as a fraud and transaction without customerID
    df_trans = df_trans.loc[df_trans.fraud == 0].dropna(
        subset=['customerID']).reset_index(drop=True)

    # rank customer transactions
    df_trans['rank'] = df_trans.groupby("customerID")["transactionID"].rank(
        "dense", ascending=True)

    # create default lag - infroamtion about default from previous cutomer
    # transaction
    df_trans['default_lag1'] = df_trans. \
        set_index('rank'). \
        groupby('customerID')[['default']]. \
        shift(1).reset_index(drop=True)

    # create late payment lag - infroamtion about late payment from previous
    # cutomer transaction
    df_trans['late_lag1'] = df_trans. \
        set_index('rank'). \
        groupby('customerID')[['late']]. \
        shift(1).reset_index(drop=True)

    # create price lag
    df_trans['price_lag1'] = df_trans. \
        set_index('rank'). \
        groupby('customerID')[['price']]. \
        shift(1).reset_index(drop=True)

    # aggregate data about previouse customer transactions using window
    # functions

    start_time = time.time()

    results = []
    for i in parameters["trans_back_aggreg_list"]:

        log.info(f"starting aggregation for #{i} transactions back")
        if len(results) == 0 or len(parameters["trans_back_aggreg_list"]) <= 1:
            results += [pd.concat([
                df_trans.set_index('rank').groupby('customerID')[['default_lag1', 'late_lag1', 'price_lag1']].rolling(i, min_periods=1).sum().reset_index(),
                df_trans.set_index('rank').groupby('customerID')[['default_lag1', 'late_lag1', 'price_lag1']].rolling(i, min_periods=1).mean().reset_index().iloc[:, 2:]
                ], axis=1)]

            results[-1].columns = ['customerID', 'rank',
                                   f'default_lst_{str(i).zfill(2)}_sum',
                                   f'late_lst_{str(i).zfill(2)}_sum',
                                   f'price_lst_{str(i).zfill(2)}_sum',
                                   f'default_lst_{str(i).zfill(2)}_avg',
                                   f'late_lst_{str(i).zfill(2)}_avg',
                                   f'price_lst_{str(i).zfill(2)}_avg']
        else:
            results += [pd.concat([
                df_trans.set_index('rank').groupby('customerID')[['default_lag1', 'late_lag1', 'price_lag1']].rolling(i, min_periods=1).sum().reset_index().iloc[:, 2:],
                df_trans.set_index('rank').groupby('customerID')[['default_lag1', 'late_lag1', 'price_lag1']].rolling(i, min_periods=1).mean().reset_index().iloc[:, 2:]
                ], axis=1)]

            results[-1].columns = [f'default_lst_{str(i).zfill(2)}_sum',
                                   f'late_lst_{str(i).zfill(2)}_sum',
                                   f'price_lst_{str(i).zfill(2)}_sum',
                                   f'default_lst_{str(i).zfill(2)}_avg',
                                   f'late_lst_{str(i).zfill(2)}_avg',
                                   f'price_lst_{str(i).zfill(2)}_avg']

        log.info("--- %s seconds ---" % (time.time() - start_time))

    defaults = pd.concat(results, axis=1)
    df_trans_all = df_trans.merge(defaults, on=['customerID', 'rank'],
                                  how='left')

    # treat last customer transaction as current credit application
    df_trans['rank_reverse'] = df_trans.groupby("customerID")["transactionID"]. \
        rank("dense", ascending=False)

    # split transactions into current transaction and historical
    df_trans_newest = df_trans.loc[df_trans.rank_reverse == 1]. \
        reset_index(drop=True)
    df_trans_history = df_trans.loc[df_trans.rank_reverse > 1].\
        reset_index(drop=True)

    # aggregate transaction history
    df_trans_history_aggr = df_trans_history. \
        groupby('customerID'). \
        agg(
            hist_trans_count=('default', 'count'),
            hist_default_sum=('default', 'sum'),
            hist_default_avg=('default', 'mean'),
            hist_late_sum=('late', 'sum'),
            hist_late_avg=('late', 'mean'),
            hist_price_sum=('price', 'sum'),
            hist_price_avg=('price', 'mean')
        ).reset_index(drop=False)

    # join historical transaction features created by aggregation all values
    df_trans_newest = df_trans_newest.merge(
        df_trans_history_aggr, on='customerID', how='left')

    # join historical transaction features created by window functions
    df_trans_newest = df_trans_newest.merge(
        defaults, on=['customerID', 'rank'], how='left')

    return df_trans_newest


def create_master_table(df_cust: pd.DataFrame,
                        df_trans: pd.DataFrame,
                        parameters: Dict) -> pd.DataFrame:
    """Combines all data to create a master table.
    """

    df_cust = _process_customers(df_cust, parameters)
    df_trans = _process_transactions(df_trans, parameters)

    # join data
    master_table = df_cust.merge(df_trans, on=['customerID'],
                                 how='left')

    # create geo risk ranking
    # temporary solution, if used in final solution, need to prepare in fit/transform maner
    bins = [-np.inf, 0.049, 0.071, 0.088, 0.107, 0.137, np.inf]
    geo_risk_rank = master_table.groupby('residentialAddress_clean')[['hist_default_sum', 'hist_trans_count']]. \
        sum().reset_index(). \
        assign(geo_risk_rank=lambda x: pd.cut(x['hist_default_sum']/x['hist_trans_count'], bins).cat.codes)

    master_table = master_table.merge(geo_risk_rank[['residentialAddress_clean', 'geo_risk_rank']], on='residentialAddress_clean', how='left')

    # drop clients without transactions
    master_table = master_table.dropna(subset=['default'])

    return master_table
