#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: heiye
@time: 2021/9/20 13:03
"""

from utils import *
from gen_feats import *


def predict_xgb_kfold(model_path, train, test, feat_cols, label_col):
    for col in ['sub_Rate', 'main_Rate', 'outstanding_disburse_ratio']:
        train[col] = train[col].apply(lambda x: 1 if x > 1 else x)
        test[col] = test[col].apply(lambda x: 1 if x > 1 else x)

    X_train = train[feat_cols]
    y_train = train[label_col]
    X_test = test[feat_cols]

    gbms = load_pkl(model_path)
    n_fold = len(gbms)
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)

    oof_preds = np.zeros((X_train.shape[0],))
    test_preds = np.zeros((X_test.shape[0],))

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
        logging.info(f'############ xgb fold {fold} ###########')
        X_val, y_val = X_train.iloc[val_index], y_train[val_index]
        dvalid = xgb.DMatrix(X_val, y_val)
        dtest = xgb.DMatrix(X_test)

        gbm = gbms[fold]

        oof_preds[val_index] = gbm.predict(dvalid, iteration_range=(0, gbm.best_iteration))
        test_preds += gbm.predict(dtest, iteration_range=(0, gbm.best_iteration)) / kfold.n_splits

    return oof_preds, test_preds


def predict_lgb_kfold(model_path, train, test, feat_cols, label_col):
    X_train = train[feat_cols]
    y_train = train[label_col]
    X_test = test[feat_cols]

    gbms = load_pkl(model_path)
    n_fold = len(gbms)
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    oof_preds = np.zeros((X_train.shape[0],))
    test_preds = np.zeros((X_test.shape[0],))

    for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
        logging.info(f'############ lgb fold {fold} ###########')
        X_val, y_val = X_train.iloc[val_index], y_train[val_index]

        gbm = gbms[fold]

        oof_preds[val_index] = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        test_preds += gbm.predict(X_test, num_iteration=gbm.best_iteration) / kfold.n_splits

    return oof_preds, test_preds


if __name__ == '__main__':
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
    oof_preds_xgb, test_preds_xgb = predict_xgb_kfold(model_path='../user_data/gbms_xgb.pkl',
                                                      train=train.copy(),
                                                      test=test.copy(),
                                                      feat_cols=SAVE_FEATS,
                                                      label_col='loan_default')

    oof_preds_lgb, test_preds_lgb = predict_lgb_kfold(model_path='../user_data/gbms_lgb.pkl',
                                                      train=train,
                                                      test=test,
                                                      feat_cols=SAVE_FEATS,
                                                      label_col='loan_default')
    xgb_thres = gen_thres_new(train, oof_preds_xgb)
    lgb_thres = gen_thres_new(train, oof_preds_lgb)

    # 结果聚合
    df_oof_res = pd.DataFrame({'customer_id': train['customer_id'],
                               'oof_preds_xgb': oof_preds_xgb,
                               'oof_preds_lgb': oof_preds_lgb,
                               'loan_default': train['loan_default']
                               })

    # 模型融合
    df_oof_res['xgb_rank'] = df_oof_res['oof_preds_xgb'].rank(pct=True)
    df_oof_res['lgb_rank'] = df_oof_res['oof_preds_lgb'].rank(pct=True)
    df_oof_res['preds'] = 0.31 * df_oof_res['xgb_rank'] + 0.69 * df_oof_res['lgb_rank']
    thres = gen_thres_new(df_oof_res, df_oof_res['preds'])

    df_test_res = pd.DataFrame({'customer_id': test['customer_id'],
                                'test_preds_xgb': test_preds_xgb,
                                'test_preds_lgb': test_preds_lgb})

    df_test_res['xgb_rank'] = df_test_res['test_preds_xgb'].rank(pct=True)
    df_test_res['lgb_rank'] = df_test_res['test_preds_lgb'].rank(pct=True)
    df_test_res['preds'] = 0.31 * df_test_res['xgb_rank'] + 0.69 * df_test_res['lgb_rank']

    # 结果产出
    df_submit = gen_submit_file(df_test_res, df_test_res['preds'], thres,
                                save_path='../prediction_result/result.csv')