#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: heiye
@time: 2021/9/20 13:11
"""

from utils import *

TARGET_ENCODING_FETAS = [
                            'employment_type',
                             'branch_id',
                             'supplier_id',
                             'manufacturer_id',
                             'area_id',
                             'employee_code_id',
                             'asset_cost_bin'
                         ]

SAVE_FEATS = [
                 'customer_id',
                 'neighbor_default_prob',
                 'disbursed_amount',
                 'asset_cost',
                 'branch_id',
                 'supplier_id',
                 'manufacturer_id',
                 'area_id',
                 'employee_code_id',
                 'credit_score',
                 'loan_to_asset_ratio',
                 'year_of_birth',
                 'age',
                 'sub_Rate',
                 'main_Rate',
                 'loan_to_asset_ratio_bin',
                 'asset_cost_bin',
                 'employment_type_mean_target',
                 'branch_id_mean_target',
                 'supplier_id_mean_target',
                 'manufacturer_id_mean_target',
                 'area_id_mean_target',
                 'employee_code_id_mean_target',
                 'asset_cost_bin_mean_target',
                 'credit_history',
                 'average_age',
                 'total_disbursed_loan',
                 'main_account_disbursed_loan',
                 'total_sanction_loan',
                 'main_account_sanction_loan',
                 'active_to_inactive_act_ratio',
                 'total_outstanding_loan',
                 'main_account_outstanding_loan',
                 'Credit_level',
                 'outstanding_disburse_ratio',
                 'total_account_loan_no',
                 'main_account_tenure',
                 'main_account_loan_no',
                 'main_account_monthly_payment',
                 'total_monthly_payment',
                 'main_account_active_loan_no',
                 'main_account_inactive_loan_no',
                 'sub_account_inactive_loan_no',
                 'enquirie_no',
                 'main_account_overdue_no',
                 'total_overdue_no',
                 'last_six_month_defaulted_no'
            ]


def gen_new_feats(train, test):
    '''生成新特征：如年利率/分箱等特征'''
    # Step 1: 合并训练集和测试集
    data = pd.concat([train, test])

    # Step 2: 具体特征工程
    # 计算二级账户的年利率
    data['sub_Rate'] = (data['sub_account_monthly_payment'] * data['sub_account_tenure'] - data[
        'sub_account_sanction_loan']) / data['sub_account_sanction_loan']

    # 计算主账户的年利率
    data['main_Rate'] = (data['main_account_monthly_payment'] * data['main_account_tenure'] - data[
        'main_account_sanction_loan']) / data['main_account_sanction_loan']

    # 对部分特征进行分箱操作
    # 等宽分箱
    loan_to_asset_ratio_labels = [i for i in range(10)]
    data['loan_to_asset_ratio_bin'] = pd.cut(data["loan_to_asset_ratio"], 10, labels=loan_to_asset_ratio_labels)
    # 等频分箱
    data['asset_cost_bin'] = pd.qcut(data['asset_cost'], 10, labels=loan_to_asset_ratio_labels)
    # 自定义分箱
    amount_cols = [
                   'total_monthly_payment',
                   'main_account_sanction_loan',
                   'main_account_disbursed_loan',
                   'sub_account_sanction_loan',
                   'sub_account_disbursed_loan',
                   'main_account_monthly_payment',
                   'sub_account_monthly_payment',
                   'total_sanction_loan'
                ]
    amount_labels = [i for i in range(10)]
    for col in amount_cols:
        total_monthly_payment_bin = [-1, 5000, 10000, 30000, 50000, 100000, 300000, 500000, 1000000, 3000000, data[col].max()]
        data[col + '_bin'] = pd.cut(data[col], total_monthly_payment_bin, labels=amount_labels).astype(int)

    # Step 3: 返回包含新特征的训练集 & 测试集
    return data[data['loan_default'].notnull()], data[data['loan_default'].isnull()]


def gen_target_encoding_feats(train, test, encode_cols, target_col, n_fold=10):
    '''生成target encoding特征'''
    # for training set - cv
    tg_feats = np.zeros((train.shape[0], len(encode_cols)))
    kfold = StratifiedKFold(n_splits=n_fold, random_state=1024, shuffle=True)
    for _, (train_index, val_index) in enumerate(kfold.split(train[encode_cols], train[target_col])):
        df_train, df_val = train.iloc[train_index], train.iloc[val_index]
        for idx, col in enumerate(encode_cols):
            target_mean_dict = df_train.groupby(col)[target_col].mean()
            df_val[f'{col}_mean_target'] = df_val[col].map(target_mean_dict)
            tg_feats[val_index, idx] = df_val[f'{col}_mean_target'].values

    for idx, encode_col in enumerate(encode_cols):
        train[f'{encode_col}_mean_target'] = tg_feats[:, idx]

    # for testing set
    for col in encode_cols:
        target_mean_dict = train.groupby(col)[target_col].mean()
        test[f'{col}_mean_target'] = test[col].map(target_mean_dict)

    return train, test


def gen_neighbor_feats(train, test):
    '''产生近邻欺诈特征'''
    if not os.path.exists('../user_data/neighbor_default_probs.pkl'):
        # 该特征需要跑的时间较久，因此将其存成了pkl文件
        neighbor_default_probs = []
        for i in tqdm(range(train.customer_id.max())):
            if i >= 10 and i < 199706:
                customer_id_neighbors = list(range(i - 10, i)) + list(range(i + 1, i + 10))
            elif i < 199706:
                customer_id_neighbors = list(range(0, i)) + list(range(i + 1, i + 10))
            else:
                customer_id_neighbors = list(range(i - 10, i)) + list(range(i + 1, 199706))

            customer_id_neighbors = [customer_id_neighbor for customer_id_neighbor in customer_id_neighbors if
                                     customer_id_neighbor in train.customer_id.values.tolist()]
            neighbor_default_prob = train.set_index('customer_id').loc[customer_id_neighbors].loan_default.mean()
            neighbor_default_probs.append(neighbor_default_prob)

        df_neighbor_default_prob = pd.DataFrame({'customer_id': range(0, train.customer_id.max()),
                                                 'neighbor_default_prob': neighbor_default_probs})
        save_pkl(df_neighbor_default_prob, '../user_data/neighbor_default_probs.pkl')
    else:
        df_neighbor_default_prob = load_pkl('../user_data/neighbor_default_probs.pkl')
    train = pd.merge(left=train, right=df_neighbor_default_prob, on='customer_id', how='left')
    test = pd.merge(left=test, right=df_neighbor_default_prob, on='customer_id', how='left')

    return train, test


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

    # 特征工程 一些后处理
    for col in ['sub_Rate', 'main_Rate', 'outstanding_disburse_ratio']:
        train[col] = train[col].apply(lambda x: 1 if x > 1 else x)
        test[col] = test[col].apply(lambda x: 1 if x > 1 else x)
    train['asset_cost_bin'] = train['asset_cost_bin'].astype(int)
    test['asset_cost_bin'] = test['asset_cost_bin'].astype(int)
    train['loan_to_asset_ratio_bin'] = train['loan_to_asset_ratio_bin'].astype(int)
    test['loan_to_asset_ratio_bin'] = test['loan_to_asset_ratio_bin'].astype(int)

    # 存储包含新特征的数据集
    logging.info('new data saving...')
    cols = SAVE_FEATS + ['loan_default', ]
    train[cols].to_csv('../user_data/train_final.csv', index=False)
    test[cols].to_csv('../user_data/test_final.csv', index=False)
