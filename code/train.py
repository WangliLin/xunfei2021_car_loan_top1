#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: heiye
@time: 2021/9/20 13:03
"""

from utils import *
from gen_feats import *


def train_xgb(train, test, feat_cols, label_col, n_fold=10):
    '''训练xgboost'''
    for col in ['sub_Rate', 'main_Rate', 'outstanding_disburse_ratio']:
        train[col] = train[col].apply(lambda x: 1 if x > 1 else x)
        test[col] = test[col].apply(lambda x: 1 if x > 1 else x)

    X_train = train[feat_cols]
    y_train = train[label_col]
    X_test = test[feat_cols]
    gbms_xgb, oof_preds_xgb, test_preds_xgb = train_xgb_kfold(X_train, y_train, X_test, n_fold=n_fold)

    if not os.path.exists('../user_data/gbms_xgb.pkl'):
        save_pkl(gbms_xgb, '../user_data/gbms_xgb.pkl')

    return gbms_xgb, oof_preds_xgb, test_preds_xgb


def train_lgb(train, test, feat_cols, label_col, n_fold=10):
    '''训练lightgbm'''
    X_train = train[feat_cols]
    y_train = train[label_col]
    X_test = test[feat_cols]
    gbms_lgb, oof_preds_lgb, test_preds_lgb = train_lgb_kfold(X_train, y_train, X_test, n_fold=n_fold)

    if not os.path.exists('../user_data/gbms_lgb.pkl'):
        save_pkl(gbms_lgb, '../user_data/gbms_lgb.pkl')

    return gbms_lgb, oof_preds_lgb, test_preds_lgb


if __name__ == '__main__':
    # 读取原始数据集
    logging.info('data loading...')
    train = pd.read_csv('../xfdata/车辆贷款违约预测数据集/train.csv')
    test = pd.read_csv('../xfdata/车辆贷款违约预测数据集/test.csv')

    # 特征工程
    logging.info('feature generating...')
    train, test = gen_new_feats(train, test)
    train, test = gen_target_encoding_feats(train, test, TARGET_ENCODING_FETAS, target_col='loan_default', n_fold=10)
    train, test = gen_neighbor_feats(train, test)

    train['asset_cost_bin'] = train['asset_cost_bin'].astype(int)
    test['asset_cost_bin'] = test['asset_cost_bin'].astype(int)
    train['loan_to_asset_ratio_bin'] = train['loan_to_asset_ratio_bin'].astype(int)
    test['loan_to_asset_ratio_bin'] = test['loan_to_asset_ratio_bin'].astype(int)
    train['asset_cost_bin_mean_target'] = train['asset_cost_bin_mean_target'].astype(float)
    test['asset_cost_bin_mean_target'] = test['asset_cost_bin_mean_target'].astype(float)

    # 模型训练：linux和mac的xgboost结果会有些许不同，以模型文件结果为主
    gbms_xgb, oof_preds_xgb, test_preds_xgb = train_xgb(train.copy(), test.copy(),
                                                        feat_cols=SAVE_FEATS,
                                                        label_col='loan_default')
    gbms_lgb, oof_preds_lgb, test_preds_lgb = train_lgb(train, test,
                                                        feat_cols=SAVE_FEATS,
                                                        label_col='loan_default')
    xgb_thres = gen_thres_new(train, oof_preds_xgb)
    lgb_thres =  gen_thres_new(train, oof_preds_lgb)

    # 结果聚合
    df_oof_res = pd.DataFrame({'customer_id': train['customer_id'],
                               'oof_preds_xgb': oof_preds_xgb,
                               'oof_preds_lgb': oof_preds_lgb})

    # 模型融合
    df_oof_res['xgb_rank'] = df_oof_res['oof_preds_xgb'].rank(pct=True)
    df_oof_res['lgb_rank'] = df_oof_res['oof_preds_lgb'].rank(pct=True)
    df_oof_res['preds'] = 0.31 * df_oof_res['xgb_rank'] + 0.69 * df_oof_res['lgb_rank']
    thres = gen_thres_new(df_oof_res, df_oof_res['preds'])

    '''
    df_test_res = pd.DataFrame({'customer_id': test['customer_id'],
                                'test_preds_xgb': test_preds_xgb,
                                'test_preds_lgb': test_preds_lgb})

    df_test_res['xgb_rank'] = df_test_res['test_preds_xgb'].rank(pct=True)
    df_test_res['lgb_rank'] = df_test_res['test_preds_lgb'].rank(pct=True)
    df_test_res['preds'] = 0.31 * df_test_res['xgb_rank'] + 0.69 * df_test_res['lgb_rank']

    # 结果产出
    df_submit = gen_submit_file(df_test_res, df_test_res['preds'], thres,
                                save_path='../prediction_result/result.csv')
    '''



