# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV

def cal_ks_auc(score, y):
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    ks = max(tpr - fpr)
    return auc, ks

def auto_binning(df, target,max_bin,method,alg_method):
    """
    :param df: dataframe 要求dtype = float
    :param target: Y标签
    :param max_bin: 最大分箱箱数
    :param method: mono模型分箱方法; tree决策树分箱; widthequal 等距; depthequal 等频率
    :param alg_method: 计算指标 'iv','gini','entropy'信息熵
    :return: 
    """
    from woe_bin import woe_bin
    wbin = woe_bin(df, target, 0.05, max_bin, method,alg_method) #tree 决策树分箱;widthequal;depthequal;mono
    mapiv = wbin.split_data()
    df_woe = wbin.apply_woetab(df, mapiv)

    # print(df_woe['pb_cretra_bas_n18_reloancaccnum'])
    print("test") 
    def beautifulMapiv(mp):
        # if set(['varname', 'll', 'ul', 'bin', 'total', 'p1', 'iv']) > set(mp.columns.tolist()):
        if {'varname', 'll', 'ul', 'bin', 'total', 'p1', 'iv'} > set(mp.columns.tolist()):
            raise Exception("mapiv columns is not correct !")
        mapiv = mp.copy()
        # print(mapiv)
        mapiv['Bucket'] = mapiv.apply(lambda x: "(" + str(x['ll']) + ", " + str(x['ul']) + "]", axis=1)
        mapiv['Bucket'] = mapiv.apply(lambda x: "missing" if x['bin'] == 0 else x['Bucket'], axis=1)

        mapiv['cnt'] = mapiv['total']
        mapiv['cnt_distri'] = mapiv['total'] / (mapiv['total'].sum() / len(set(mapiv['varname'])))
        mapiv['good_cnt'] = mapiv['total'] - mapiv['p1']
        mapiv["bad_cnt"] = mapiv['p1']
        mapiv["badrate"] = mapiv['p1'] / mapiv['total']
        mapiv["lift"] = mapiv['p1'] / mapiv['total'] / (mapiv['p1'].sum() / mapiv['total'].sum())
        # 按不同的变量计算ks值
        # mapiv["ks"] = abs(mapiv['good_cnt'].cumsum() / mapiv['good_cnt'].sum() - mapiv['bad_cnt'].cumsum() / mapiv['bad_cnt'].sum()) 
        df_index = []
        for i in range(len(mapiv['varname'])):
            df_index.append(i)
        mapiv['ID'] = df_index
        
        df_ks = pd.DataFrame(columns=['ID', 'ks'])
        var_list = mapiv['varname'].unique()
        for var in var_list:
            df1 = pd.DataFrame(columns=['ID', 'ks'])
            df_mapiv = mapiv[mapiv['varname'] == var]
            df1['ID'] = df_mapiv['ID']
            df1['ks'] = abs(df_mapiv['good_cnt'].cumsum()/ df_mapiv['good_cnt'].sum() - df_mapiv['bad_cnt'].cumsum()/ df_mapiv['bad_cnt'].sum()) 
            df_ks = pd.concat([df_ks, df1], ignore_index=True)
        mapiv = pd.merge(mapiv, df_ks, left_on='ID', right_on='ID', how='inner')
        mapiv.rename(columns={'iv': 'totoal_iv'}, inplace=True)

        return mapiv[['varname','ll','ul', 'Bucket', 'cnt' , 'cnt_distri', "bad_cnt", 'badrate', "lift", 'totoal_iv', 'ks', 'bin']]
    return df_woe, beautifulMapiv(mapiv)

def lrModel(df, target):
    df_woe, mapiv = auto_binning(df, target)

    logreg = LogisticRegression(random_state=1234, fit_intercept=True, penalty='l2', solver='newton-cg',
                                class_weight='balanced', n_jobs=1)
    lr_param = {'C': np.arange(0.1, 1, 0.01)}
    lr_gsearch = GridSearchCV(estimator=logreg, param_grid=lr_param, cv=3, scoring='f1')
    # 执行超参数优化
    lr_gsearch.fit(df_woe, df[target])
    print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
    LR_model_fit = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='newton-cg',
                                    class_weight='balanced')
    LR_model_fit.fit(df_woe, df[target])
    pred_score = LR_model_fit.predict_proba(df_woe)[:, 1]
    # C_best = list(LR_model_fit.intercept_)[0]
    # x_copy = df_woe
    # x_copy["intercept"] = C_best
    # LR = sm.Logit(df[target], x_copy[df_woe.columns.tolist() + ['intercept']]).fit(method="bfgs")
    return pred_score, mapiv

def main():
    paramas={'id':['app_no'],
             'target':'def_fpd15'
             }
    df_mainscore = pd.read_csv('main.csv')
    df_thirdscore = pd.read_csv('test.csv')

    m_name_l = [x for x in df_mainscore if 'score' in x.lower()]
    t_name_l = [x for x in df_thirdscore if 'score' in x.lower()]
    if not m_name_l or not t_name_l:
        print('检查分数名称, 是否用score表示分数！！')
    else:
        df_merge = pd.merge(df_mainscore, df_thirdscore, how='inner', on=paramas.get('id'))
        # assert paramas.get('target') not in df_merge.columns,  'target不在数据集中'
        m_name, t_name = m_name_l[0], t_name_l[0]
        target = paramas.get('target')
        # print(m_name, t_name, target)
        m_auc, m_ks = cal_ks_auc(df_merge[m_name], df_merge[target])
        t_auc, t_ks = cal_ks_auc(df_merge[t_name], df_merge[target])
        print("主模型auc:{0}, ks:{1}; \n 三方分数的auc:{2}, ks:{3}".format(m_auc, m_ks, t_auc, t_ks))
        print('----------------开始模型融合-------------------')
        df_merge_f = df_merge[[m_name, t_name, target]]
        preditscore, mapiv = lrModel(df_merge_f, target)
        con_auc, con_ks = cal_ks_auc(preditscore, df_merge[target])
        print("融合模型auc:{0}, ks:{1} ".format(con_auc, con_ks))
        mapiv.to_csv('分数分箱.csv', index=False)

if __name__ == '__main__':
    main()

