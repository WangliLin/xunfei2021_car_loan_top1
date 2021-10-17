#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: heiye
@time: 2021/9/20 13:13
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
from sklearn.preprocessing import StandardScaler, QuantileTransformer, KBinsDiscretizer, LabelEncoder, MinMaxScaler, PowerTransformer


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def gen_thres_new(df_train, oof_preds):
    df_train['oof_preds'] = oof_preds
    quantile_point = df_train['loan_default'].mean()
    thres = df_train['oof_preds'].quantile(1 - quantile_point)

    _thresh = []
    for thres_item in np.arange(thres - 0.2, thres + 0.2, 0.01):
        _thresh.append(
            [thres_item, f1_score(df_train['loan_default'], np.where(oof_preds > thres_item, 1, 0), average='macro')])

    _thresh = np.array(_thresh)
    best_id = _thresh[:, 1].argmax()
    best_thresh = _thresh[best_id][0]

    print("阈值: {}\n训练集的f1: {}".format(best_thresh, _thresh[best_id][1]))
    return best_thresh


def gen_submit_file(df_test, test_preds, thres, save_path):
    df_test['test_preds_binary'] = np.where(test_preds > thres, 1, 0)
    df_test_submit = df_test[['customer_id', 'test_preds_binary']]
    df_test_submit.columns = ['customer_id', 'loan_default']
    print(f'saving result to: {save_path}')
    df_test_submit.to_csv(save_path, index=False)
    print('done!')
    return df_test_submit


def train_lgb_kfold(X_train, y_train, X_test, n_fold=5):
    '''train lightgbm with k-fold split'''
    gbms = []
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    oof_preds = np.zeros((X_train.shape[0],))
    test_preds = np.zeros((X_test.shape[0],))

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
        logging.info(f'############ fold {fold} ###########')
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[val_index], y_train[train_index], y_train[val_index]
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 64,
            'learning_rate': 0.02,
            'min_data_in_leaf': 150,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'n_jobs': -1,
            'seed': 1024
        }

        gbm = lgb.train(params,
                        dtrain,
                        num_boost_round=1000,
                        valid_sets=[dtrain, dvalid],
                        verbose_eval=50,
                        early_stopping_rounds=20)

        oof_preds[val_index] = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        test_preds += gbm.predict(X_test, num_iteration=gbm.best_iteration) / kfold.n_splits
        gbms.append(gbm)

    return gbms, oof_preds, test_preds



def train_xgb_kfold(X_train, y_train, X_test, n_fold=5):
    '''train xgboost with k-fold split'''
    gbms = []
    kfold = StratifiedKFold(n_splits=10, random_state=1024, shuffle=True)
    oof_preds = np.zeros((X_train.shape[0],))
    test_preds = np.zeros((X_test.shape[0],))

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
        logging.info(f'############ fold {fold} ###########')
        X_tr, X_val, y_tr, y_val = X_train.iloc[train_index], X_train.iloc[val_index], y_train[train_index], y_train[val_index]
        dtrain = xgb.DMatrix(X_tr, y_tr)
        dvalid = xgb.DMatrix(X_val, y_val)
        dtest = xgb.DMatrix(X_test)

        params={
            'booster':'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 8,
            'subsample':0.9,
            'min_child_weight': 10,
            'colsample_bytree':0.85,
            'lambda': 10,
            'eta': 0.02,
            'seed': 1024
        }

        watchlist = [(dtrain, 'train'), (dvalid, 'test')]

        gbm = xgb.train(params,
                        dtrain,
                        num_boost_round=1000,
                        evals=watchlist,
                        verbose_eval=50,
                        early_stopping_rounds=20)

        oof_preds[val_index] = gbm.predict(dvalid, iteration_range=(0, gbm.best_iteration))
        test_preds += gbm.predict(dtest, iteration_range=(0, gbm.best_iteration)) / kfold.n_splits
        gbms.append(gbm)

    return gbms, oof_preds, test_preds