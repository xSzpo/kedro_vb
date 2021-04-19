from typing import Any, Dict, Tuple

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

    df_trans["defualted_payment"] = df_trans.replace({
        'paymentStatus1': {1: 0, 2: 1},
        'paymentStatus2': {1: 0, 2: 2},
        'paymentStatus3': {1: 0, 2: 3},
        'paymentStatus4': {1: 0, 2: 4},
    }).filter(regex='payment').replace({0: np.nan}).min(axis=1).fillna(0)

    df_trans['money_lost'] = df_trans['defualted_payment']/4 * df_trans['price']

    df_trans["late_payment_first"] = df_trans.replace({
        'paymentStatus1': {1: 1, 2: 0},
        'paymentStatus2': {1: 2, 2: 0},
        'paymentStatus3': {1: 3, 2: 0},
        'paymentStatus4': {1: 4, 2: 0},
    }).filter(regex='payment').replace({0: np.nan}).min(axis=1).fillna(0)

    # mark transactions from shop 113 as a fraud
    df_trans['fraud'] = 0
    df_trans.loc[df_trans.shopID == 113, 'fraud'] = 1

    # drop transaction marked as a fraud and transaction without customerID
    df_trans = df_trans.loc[df_trans.fraud == 0].dropna(
        subset=['customerID']).reset_index(drop=True)

    # rank customer transactions
    df_trans['rank'] = df_trans.groupby("customerID")["transactionID"].rank(
        "dense", ascending=True)

    # create default lag - informtion about default from previous cutomers
    # transaction
    df_trans['default_lag1'] = df_trans. \
        set_index('rank'). \
        groupby('customerID')[['default']]. \
        shift(1).reset_index(drop=True)

    # create defualted_payment lag - informtion about number of defaulte
    # payment from previous cutomers transaction
    df_trans['defualted_payment_lag1'] = df_trans. \
        set_index('rank'). \
        groupby('customerID')[['defualted_payment']]. \
        shift(1).reset_index(drop=True)

    # create lost lag - informtion about lost from previous cutomers
    # transaction
    df_trans['money_lost_lag1'] = df_trans. \
        set_index('rank'). \
        groupby('customerID')[['money_lost']]. \
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

    # create late_payment_first lag - informtion about number payment which
    # was late as first from previous cutomers transaction
    df_trans['late_payment_first_lag1'] = df_trans. \
        set_index('rank'). \
        groupby('customerID')[['late_payment_first']]. \
        shift(1).reset_index(drop=True)

    # aggregate data about previouse customer transactions using window
    # functions

    start_time = time.time()

    aggreate_list = df_trans.filter(regex="lag").columns.to_list()

    results = []
    for i in parameters["trans_back_aggreg_list"]:

        log.info(f"starting aggregation for #{i} transactions back")
        names_sum = [f'{k}_lst_{str(i).zfill(2)}_sum' for k in list(map(lambda x: x.replace('_lag1',''), aggreate_list))]
        names_avg = [f'{k}_lst_{str(i).zfill(2)}_avg' for k in list(map(lambda x: x.replace('_lag1',''), aggreate_list))]

        if len(results) == 0 or len(parameters["trans_back_aggreg_list"]) <= 1:
            results += [pd.concat([
                df_trans.set_index('rank').groupby('customerID')[aggreate_list].rolling(i, min_periods=1).sum().reset_index(),
                df_trans.set_index('rank').groupby('customerID')[aggreate_list].rolling(i, min_periods=1).mean().reset_index().iloc[:, 2:]
                ], axis=1)]

            results[-1].columns = ['customerID', 'rank']+names_sum+names_avg
        else:
            results += [pd.concat([
                df_trans.set_index('rank').groupby('customerID')[aggreate_list].rolling(i, min_periods=1).sum().reset_index().iloc[:, 2:],
                df_trans.set_index('rank').groupby('customerID')[aggreate_list].rolling(i, min_periods=1).mean().reset_index().iloc[:, 2:]
                ], axis=1)]

            results[-1].columns = names_sum+names_avg

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


def split_data(df: pd.DataFrame,
               parameters: Dict) -> Tuple[pd.DataFrame, pd.DataFrame,
                                          pd.DataFrame, pd.DataFrame,
                                          pd.DataFrame, pd.DataFrame,
                                          pd.DataFrame]:
    """Splits data into features and targets training, test, valid sets.
    Args:
        df: Data containing features and target.
        df_aggr: Data containing edditional features, join on=['GC_addr_suburb','market','date_offer_w']
        parameters: Parameters defined in parameters.yml.
    Returns:
        df_oot   - data set that will allow to check model performance on the
                   latest available data (out of time) (EXISTING CUSTOMER),
        df_train - dataset for model training (EXISTING CUSTOMER),
        df_test  - dataset for model performance assessing (out of sample) (EXISTING CUSTOMER),
        df_valid - dataset for hp tuning (out of sample) (EXISTING CUSTOMER),
        df_train_new - dataset for model training (NEW CUSTOMER),
        df_test_new  - dataset for model performance assessing (out of sample) (NEW CUSTOMER)
        df_valid_new - dataset for hp tuning (out of sample) (NEW CUSTOMER),
    """

    df_new_customer = df.loc[df['rank'] == 1].reset_index(drop=True)
    df_existing_customer = df.loc[df['rank'] > 1].reset_index(drop=True)

    # EXISTING CUSTOMER
    df_oot = df_existing_customer.sort_values('customerID').iloc[-parameters['oot']:].sort_index()
    df = df_existing_customer.sort_values('customerID').iloc[:-parameters['oot']]

    size = df_existing_customer.shape[0]
    train_size = int(size * parameters['train'])
    test_size = int(size * parameters['test'])

    df_existing_customer = df_existing_customer.sample(frac=1, random_state=1).reset_index(drop=True)
    df_train = df_existing_customer.iloc[:train_size].sort_index()
    df_test = df_existing_customer.iloc[train_size:(train_size+test_size)].sort_index()
    df_valid = df_existing_customer.iloc[(train_size+test_size):].sort_index()

    # NEW CUSTOMER
    size = df_new_customer.shape[0]
    train_size = int(size * parameters['train'])
    test_size = int(size * parameters['test'])

    df_new_customer = df_new_customer.sample(frac=1, random_state=1).reset_index(drop=True)
    df_train_new_customer = df_new_customer.iloc[:train_size].sort_index()
    df_test_new_customer = df_new_customer.iloc[train_size:(train_size+test_size)].sort_index()
    df_valid_new_customer = df_new_customer.iloc[(train_size+test_size):].sort_index()
    return df_oot, df_train, df_test, df_valid, df_train_new_customer, df_test_new_customer, df_valid_new_customer
