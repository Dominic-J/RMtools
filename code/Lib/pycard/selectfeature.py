#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:15:32 2021

@author: fm_yangon
"""

import numpy as np
import pandas as pd
#from minepy import MINE
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif ,f_regression

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def list_diff(list1 ,list2):
    """return ：两个list之间的差集"""
    if len(list1) > 0 and len(list2) > 0:
        return list(np.setdiff1d(list1, list2))
    else:
        print('list_diff:len <= 0 !!')


class SelectFeatures():
    """
    X:pandas.DataFrame
    y:pandas.Serise or nparray
    n_feature_to_select:选择特征的数
    only_get_index:是否返回选择特征的索引
    """
    def __init__(self ,X ,y ,n_feature_to_select = None ,only_get_index = True):
        self.cols = X.columns.to_list()
        self.X = np.array(X)
        self.y = np.array(y)
        self.x_index = range(self.X.shape[1])
        self.only_get_index = only_get_index
        self.n_feature_to_select = n_feature_to_select

        if n_feature_to_select is None:
            self.n_feature_to_select = np.ceil(2 /3 * self.X.shape[1])
            print('self.n_feature_to_select:',self.n_feature_to_select)
        self.removed = []

    def _log(self ,index ,method):
        print('***{}:'.format(method))
        print('  remain feature index:\n  {}'.format(index))
        rmvd = list_diff(self.x_index ,index)
        self.removed += rmvd
        print('  removed feature index:\n  {}'.format(rmvd))

    def _return(self ,ret ,method):
        # true代表该特征被选中
        index = ret.get_support(indices = True)
        self._log(index ,method)
        if self.only_get_index == True:
            return index
        else:  #返回筛选之后的变量
            return ret.transform(self.X)

    def _by_kbest(self ,func ,method):
        '''filter方法'''
        ret = SelectKBest(func ,k = self.n_feature_to_select).fit(self.X ,self.y)
        return self._return(ret, method)

    def _by_RFE(self ,mm ,method ,step = 1):
        '''wrapper 方法'''
        ret = RFE(estimator = mm ,n_features_to_select=self.n_feature_to_select ,
                  step = step).fit(self.X ,self.y)
        return self._return(ret ,method)

    def _by_model(self ,mm ,method):
        '''embedded 方法'''
        ret = SelectFromModel(mm).fit(self.X ,self.y)
        return self._return(ret ,method)

    def by_var(self ,threshold = 0.16):
        '''start'''
        ret = VarianceThreshold(threshold=threshold).fit(self.X)
        return self._return(ret ,'by_var')

    def by_chi2(self):
        return self._by_kbest(chi2, 'by_chi2')

    def by_pearson(self):
        '''相关系数'''
        _pp = lambda X,Y:np.array(list(map(lambda x:pearsonr(x ,Y) ,X.T))).T[0]
        return self._by_kbest(_pp ,'by_pearson')

    """
    def by_max_info(self):
        def _mic(x ,y):
            m = MINE()
            m.compute_score(x ,y)
            return (m.mic() ,0.5)
        _pp = lambda X ,Y:np.array(list(map(lambda x: _mic(x ,Y), X.T))).T[0]
        return self._by_kbest(_pp ,'by_max_info')
    """

    def by_f_regression(self):
        '''
        F values of features
        p-values of F-scores
        '''
        ret = f_regression(self.X ,self.y)
        print('Feature importance by f_regression:{}'.fotmat(ret))
        return ret

    def by_f_classif(self):
        ret = f_classif(self.X, self.y)
        print('Feature importance by f_regression:{}'.format(ret))
        return ret

    def by_RFE_lr(self ,args = None):
        return self._by_RFE(LogisticRegression(random_state=42), 'by_RFE_lr')

    def by_RFE_svm(self ,args = None):
        return self._by_RFE(LinearSVC(random_state=42), 'by_RFE_svm')

    def by_gbdt(self):
        return self._by_model(GradientBoostingClassifier(random_state=42) ,'by_gbdt')

    def by_rf(self):
        return self._by_model(RandomForestClassifier(random_state=42) ,'by_rf')

    def by_et(self):
        return self._by_model(ExtraTreesClassifier(random_state=42) ,'by_et')

    def by_lr(self ,C=0.1):
        return self._by_model(LogisticRegression(penalty='l2' ,C=C ,random_state=42) ,'by_lr')

    def by_svm(self ,C=0.01):
        return self._by_model(LinearSVC(penalty = 'l2' ,C=C ,dual = False ,random_state=42) ,'by_svm')

    def select_10_methods(self):
        name = ['by_var' ,'by_RFE_svm' ,'by_RFE_lr' ,
                'by_svm' ,'by_lr' ,'by_et' ,'by_rf' ,'by_gbdt']
        map_index_cols = dict(zip(range(len(self.cols)) ,self.cols))

        method_dict = {}
        method_dict['by_var'] = self.by_var()
#        method_dict['by_max_info'] = self.by_max_info()
#        method_dict['by_pearson'] = self.by_pearson()
        method_dict['by_RFE_svm'] = self.by_RFE_svm()
        method_dict['by_RFE_lr'] = self.by_RFE_lr()
        method_dict['by_svm'] = self.by_svm()
        method_dict['by_lr'] = self.by_lr()
        method_dict['by_et'] = self.by_et()
        method_dict['by_rf'] = self.by_rf()
        method_dict['by_gbdt'] = self.by_gbdt()

        selected = [j for i in list(method_dict.values()) for j in i]

        dicts01 = {}
        for nm in name:
            dicts01[nm] = [1 if i in list(method_dict[nm]) else 0 for i in range(len(self.cols))]


        statf = pd.Series(selected).value_counts().reset_index()
        statf.columns = ['col_idx' ,'count']
        statf['feature'] = statf.col_idx.map(map_index_cols)

        statf.sort_values(by = 'col_idx' ,ascending=True ,inplace = True)
        for i in name:
            statf[i] = dicts01[i]

        statf.sort_values(by = ['count' ,'col_idx'] ,ascending=[False,True] ,inplace = True)

        selected = statf['feature'][:self.n_feature_to_select.astype(int)].to_list()
        print('*'*10 + 'remains columns:\n{}'.format(selected))

        return selected ,statf





