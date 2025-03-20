#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:07:03 2021

@author: fm_yangon
"""

"""
XGBoost 自动化调参工具
1、支持网格搜索和随机搜索
2、支持增量搜索，前一个参数确定后不再变化，继续下一个参数的调优，节省时间，但放弃多种参数的组合
3、每组参数有缺省值，支持自定义超参范围，方便精细化控制
优势：
1、提供用户自定义参数的自由组合
2、支持参数随时修改、随时清空
3、记录调参历史（参数和学习器）
"""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
from sklearn.model_selection import KFold ,StratifiedKFold
from scipy.stats import halfnorm ,randint as sp_randint ,uniform

class TuneXGB():
    """estimator:保留中间参数的模型，手动调整"""
    cur_params = {
        'colsample_bytree': 1,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 100,
        'reg_lambda': 1,
        'reg_alpha': 0,
        'scale_pos_weight': 1,
        'subsample': 1
        }
    init_params = cur_params.copy()
    #记录历史
    history_estimator = []
    history_paras = []

    #1、缺省的调优顺序流程
    param_grids_list = [
        #集成机构--解决偏差
        { 'n_estimators': range(100,1000,5)
            } ,
        { 'learning_rate':[0.01 ,0.015 ,0.025 ,0.05 ,0.1]
            } ,
        #树结构调参--解决偏差
        { 'max_depth':[3,5,7,9,12,15,17,25]
            } ,
        { 'min_child_weight':[1,3,5,7]
            } ,
        #树结构--叶子节点
        { 'gamma':[0,0.05,0.1,0.3,0.5,0.7,0.9,1.0]
            } ,
        #样本参数--解决方差
        { 'subsample':[0.5,0.6,0.7,0.8,0.9,1.0]
            } ,
        { 'colsample_bytree':[0.4,0.6,0.7,0.8,0.9,1.0]
            } ,
        #正则参数--解决方差
        { 'reg_alpha':[0,0.1,0.5,1.0,10,100,200,1000]
            } ,
        { 'reg_lambda':[0.01,0.1,1.0,10,100,200,1000]
            }
        ]

    #用户随意组合待调参的参数
    def __init__(self,
                 x ,y ,objective = 'binary:logistic' ,random_state = 42 ,cv_folds = 5,
            metric = 'roc_auc'):

        self.x = x
        self.y = y
        self.objective = objective
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.estimator = TuneXGB.get_estimator_class(self.objective)
        self.kfold = None
        self.metric = metric
        self._init_base_param()

        @classmethod
        def show_default_para(cls):
            print(TuneXGB.init_params)

        @classmethod
        def show_default_order(cls):
            count = 1
            for vv in TuneXGB.param_grids_list:
                print('{:2} step: {}'.format(count ,vv))
                count +=1

        @classmethod
        def restore(cls):
            '''清空信息，便于调参控制'''
            TuneXGB.history_estimator = []
            TuneXGB.history_paras = []

        @classmethod
        def get_estimator_class(cls ,objective):
            '''
            自定义estimator_map内容：
            'binary':xgb.XGBClassifier ,
            'count':xgb.XGBRegressor ,
            'multi':xgb.XGBClassifier ,
            'rank':xgb.XGBRegressor ,
            'reg':xgb.XGBRegressor
            '''
            estimator_map = {
                'binary':xgb.XGBClassifier
                }
            return estimator_map[objective.split(':')[0]]

        @classmethod
        def update_cur_params(cls ,params):
            TuneXGB.cur_params.update(params)
            print(TuneXGB.cur_params)

        def _init_base_param(self):
            '''根据数据更新参数'''
            if self.objective.startswith('multi'):
                TuneXGB.cur_params['num_class'] = len(self.y.unique())
            else:
                TuneXGB.cur_params['base_score'] = np.mean(self.y)
            if self.random_state is not None:
                TuneXGB.cur_params['random_state'] = self.random_state

        def get_cur_estimator(self):
            '''获取当前类中最新参数的估计器'''
            return self.estimator(**TuneXGB.cur_params)

        def _get_folds(self ,cv_folds):
            '''使用sklearn中的StaratifiedKFold 和 KFold'''
            if self.kfold is not None:
                return self.kfold
            if 'binary' in self.objective or 'multi' in self.objective:
                self.kfold = StratifiedKFold(n_splits=cv_folds,random_state=self.random_state)
            elif 'count' in self.objective or 'reg' in self.objective:
                self.kfold = KFold(n_splits=cv_folds ,random_state=self.random_state)
            else:
                raise ValueError('Invalid objective: {}'.format(objective))
            return self.kfold

        def _print_grid_results(self ,gs):
            bs = gs.best_score_
            if gs.scoring == 'neg_mean_squared_error':
                bs = abs(gs.best_score_)**0.5
            elif gs.scoring == 'neg_log_loss':
                bs = abs(gs.best_score_)

            print("   Best: {0:0.5} using {1}".format(bs ,gs.best_params_))

            means = gs.cv_results_['mean_test_score']
            stds = gs.cv_results_['std_test_score']
            params = gs.cv_results_['params']
            print('mean ,stdev ,param:')
            for mean ,stdev ,param in zip(means ,stds ,params):
                print("  {:0.5f} ({:0.5f}) with: {}".format(mean ,stdev ,param))

        def grid_search(self ,params ,n_jobs = 4 ,metric = None ,folds = None ,model = None ,verbose = 0):
            """
                支持用户输入模型，fit数据
                统一使用GridSearchCV，也可以使用xgb.cv
            """
            if model is None:
                model = self.get_cur_estimator()
            f = folds if folds is not None else self.cv_folds
            m = metric if metric is not None else self.metric
            gs = GridSearchCV(model, params ,scoring=m ,cv=self._get_folds(f) ,
                              n_jobs = n_jobs ,verbose=verbose)
            gs.fit(self.x ,self.y)
            return gs

        def random_search(self ,params ,n_iter = 20 ,n_jobs = 4 ,metric = None ,
                          folds = None ,model = None ,verbose = 0):
            '''正态参数uniform ，半正态halfnorm'''
            if model is None:
                model = self.get_cur_estimator()
            f = folds if folds is not None else self.cv_folds
            m = metric if metric is not None else self.metric

            param_distributions = {
                'colsample_bytree':
                    uniform(params.get('colsample_bytree_loc', 0.2),
                            params.get('colsample_bytree_scale', 0.5)) ,
                'gamma':
                    uniform(params.get('gamma_loc',0),params.get('gamma_scale',0.9)) ,
                'max_depth':
                    sp_randint(params.get('max_depth_low',2),params.get('max_depth_high',11)),
                'min_child_weight':
                    sp_randint(params.get('min_child_weight_low' ,1),
                               params.get('min_child_weight_high' ,11)),
                'reg_alpha':
                    halfnorm(params.get('reg_alpha_loc',0),params.get('reg_alpha_scale',5)) ,
                'reg_lambda':
                    halfnorm(params.get('reg_lambda_loc',0),params.get('reg_lambda_scale',5)) ,
                'subsample':
                    uniform(params.get('subsample_loc',0.1) ,params.get('subsanple_scale',0.8))
                }

            rs = RandomizedSearchCV(estimator = model, param_distributions=param_distributions ,
                                    cv = self._get_folds(f) ,n_iter=n_iter ,n_jobs=n_jobs,
                                    scoring=m,verbose=verbose)
            rs.fit(self.x ,self.y)
            return rs

        def tune_sequence(self):
            """
            独立、增量式搜索，相比全量参数的网格搜索效率高，但是最终效果会打折扣
            """
            print('Tunning xgboost parameters ....')
            for pp in TuneXGB.param_grids_list:
                self.tune_step(pp)

        def tune_step(self ,params ,n_jobs=4 ,method='grid' ,metric=None ,folds = None ,verbose = 1):
            """
            开放给用户自定义调优
            params：待优化的字典参数
            folds：整数，fold的数量
            返回：
                1、params中最优的参数字典
                2、最优的估计器（未fit数据）
            """
            _metric = metric if metric is not None else self.metric
            _folds = folds if folds is not None else self.cv_folds

            gs = self.grid_search(params ,n_jobs = n_jobs ,metric = _metric ,folds=_folds ,verbose=verbose)

            print('-'*60)
            print('Tunning{}'.format(params))
            print('  use metric:{} ,folds:{}'.format(_metric ,_folds))

            if verbose:
                self._print_grid_results(gs)

            opt = {}
            for kk in params.keys():
                opt[kk] = gs.best_params_[kk]
                print('Best params:\n  {}\n'.format(opt))

            #更新全局
            TuneXGB.cur_params.update(opt)
            #保存这一步的估计器
            print('Save param as:{}'.format(len(TuneXGB.history_paras)))
            TuneXGB.history_paras.append(TuneXGB.cur_params.copy())

            print('Save estimator as:{}'.format(len(TuneXGB.history_estimator)))
            TuneXGB.history_estimator.append(gs.beset_estimator_)

            return opt