# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from .tools import re_search
from .core import cols_report ,cross_woe ,NumBin ,vif
import sys

class GetWoeDf(object):
    """
    建模变量筛选、存储、删除等
    以探索性数据分析结果 、 IV 、排序性 、相关性 、稳定性 、共线性 筛选入模 或者 策略变量
    -----------------------------
    建模 及 策略仍需要很多人工干预，建议适当使用
    """
    def __init__(self, df ,target = None ,export = None):
        """
        参数：
        --------------------------------
        df：数据集
        target：目标变量
        export：不参与分析的变量名称
        ---------
        内参：
        --------------------------
        self.group_cols_: 经过数据分析之后得到的变量分类字典
        self.final_bin_: 探索性数据分析之后的分箱和woe数据
        """
        self.df = df.reset_index(drop = True)
        self.target = target
        self.export = export
        self.cols = df.columns.to_list()
        if isinstance(self.target,str) and self.target in self.cols: self.cols.remove(self.target)
        self.group_cols_ = None
        self.__cols = None
        self.__df = pd.DataFrame()
        self.__toexcel = None
        self.final_bin_ = pd.DataFrame()
        if len(self.df.columns.to_list()) > len(set(self.df.columns.to_list())):
            print("数据表变量存在重复，请及时处理！")

    def run_begin(self, woedf = False ,inplace = True ):
        """
        初始化 参数
        --------------------
        woedf：是否对数值型变量执行 排序性合并操作
        inplace：是否执行get_woedf 函数
        
        """
        if inplace:
            self.get_edacols(export = self.export)
            if woedf:
                self.get_woedf(woedf)
            else:
                self.get_woedf()
        else:
            self.get_edacols(export = self.export)
            
    def get_edacols(self, export = None):
        """
        得到建模需要的变量，探索性数据分析的结果
        -------------------------------
        参数：
        df :dataframe
        export:不需要参与数据分析的变量 ，但是可以是list，把不需要使用的变量都排除掉
        """
        if isinstance(export ,str):self.export = [export]
        if isinstance(export ,list):self.export = export
        if isinstance(self.export,list):
            for i in self.export:
                if i in self.cols: self.cols.remove(i)
        cols_nan = self.df.nan_rate().loc[self.df.nan_rate().nanRate == 1].index.to_list()
        if len(cols_nan):
            for i in cols_nan:
                self.cols.remove(i)
        dfi = cols_report(self.df[self.cols])
        cols_num = dfi.loc[(dfi.infered_type == 'num')&(dfi.unique > 5)&(dfi.nanRate <= 0.8)].index.to_list()
        cols_cate = dfi.loc[(dfi.infered_type == 'cate')&(dfi.unique >= 2)&(dfi.unique <= 10)&(dfi.nanRate <= 0.8)].index.to_list()
        cols_cate.extend(dfi.loc[(dfi.infered_type == 'num')&(dfi.unique <= 5)&(dfi.unique >= 2)&(dfi.nanRate <= 0.8)].index.to_list())
        cols_drop = dfi.index.to_list()
        for i in cols_cate:
            cols_drop.remove(i)
        for i in cols_num:
            cols_drop.remove(i)
        self.group_cols_ = {'cols_num':cols_num ,'cols_cate':cols_cate ,'eda_drop': cols_drop ,'cols_nan': cols_nan}
        self.__cols = {'cols_num':dfi.loc[dfi.infered_type == 'num'].index.to_list() ,'cols_cate':dfi.loc[dfi.infered_type == 'cate'].index.to_list()}
        
    def get_woedf(self ,inplace = False):
        """
        得到woe_df的相关数据
        --------------------------
        参数：
        inplace：表示是否对数值型变量进行合并输出具有排序的分箱结果  
        output: 结果是否为输出模式 ，输出模式会删掉 woe、IV_i 、Gini
        """
        cols_group = self.group_cols_
        if isinstance(self.target ,str): target = self.target
        self.final_bin_['target'] = self.df[target]
        for i in cols_group.get('cols_cate'):
            if i is not None: 
                self.__df = self.__df.append(cross_woe(self.df[i] ,self.df[target] ,output=False))
                self.final_bin_[ i + '_bin'] = self.df[i].copy()
        clf = NumBin()
        for i in cols_group.get('cols_num'):
            if i is not None:               
                clf.fit(self.df[i] ,self.df[target] ,output=False ,inplace=inplace)
                self.final_bin_[ i + '_bin'] = clf.bin_var()[i + '_bin'].copy()
                self.__df = self.__df.append(clf.woe_df_)
        self.__df['colName'] = self.__df['colName'].apply(lambda x:x + '_bin')
    
    def get_woe_final(self, var_names=None, patten=None ,output=True):
        """__正常输出策略分析结果 ，查询指定变量的 woe_df

        参数:
        -------
        patten: str,
            用正则表达式描述的变量名模式。若传入此参数，则优先返回所查到的所有符合该模式的所有变量的 woe_df
        var_names : str or list_of_str,
            需要查询的变量名/变量列表. 默认值 None 表示返回所有变量的 woe_df
        output: 结果是否为输出模式 ，输出模式会删掉 woe、IV_i 、Gini

        返回:
        -------
        woe_df: dataframe, 指定变量的 woe_df"""
        
        woe_df = self.__df.copy()
        if patten is not None:
            var_names = re_search(patten, self.all_vars())
        if isinstance(var_names, str):var_names = [var_names]
        if isinstance(var_names ,list):
            logic = self.__df.colName == var_names
            woe_df = self.__df[logic]
        if output:
            woe_df = woe_df.drop_col(['IV_i' ,'woe' ,'Gini'])
        return woe_df
    
    def get_woe_df(self ,iv = 0.03, corr = 0.7 ,sorting = [1.5 ,0.01] ,vif_n = 10 ,
                   psi_col = None ,psi = [4 ,0.1] ,inplace = False ,output = True):
        """
        得到策略分析的结果 ，建议 第一次做分析的时候 inplace为 False
        ------------------------------
        参数：
        ---------------------------------------
        iv：计算IV的阈值
        corr：计算相关性的阈值
        sorting：计算排序性的阈值
        vif_n：计算共线性的阈值
        psi_col：划分计算psi的数据集的变量
        psi：计算psi的阈值
        var_names：str or list_of_str or index_of_str,需要返回的变量名称
        inplace：是否操作  group_cols_
        output: 结果是否为输出模式 ，输出模式会删掉 woe、IV_i 、Gini
        --------------------------------------------------
        sorting 和 psi 是list类型，其他的都是 单变量
        psi_col：是字符型的，如果没有，这默认不输出，一般以时间作为验证集划分
        返回参数：
        -------------------
        返回计算的所有结
        """
        woe_df = self.__df.copy()
        woe_df = woe_df.merge(self.get_ivcols(iv = iv ,inplace = inplace) ,how = 'left' ,on = 'colName')
        woe_df = woe_df.merge(self.get_corrcols(corr = corr,inplace = inplace) ,how = 'left' ,on = 'colName')     
        woe_df = woe_df.merge(self.get_sortcols(bad_size = sorting[0] ,per = sorting[1],inplace = inplace) ,how = 'left' ,on = 'colName') 
        woe_df = woe_df.merge(self.get_vifcols(vif_n = vif_n,inplace = inplace) ,how = 'left' ,on = 'colName') 
        if isinstance(psi_col ,str):
            woe_df = woe_df.merge(self.get_psicols(psi_col = psi_col ,per = psi[0] ,psi = psi[1],inplace = inplace).drop_duplicates() ,how = 'left' ,on = 'colName')
        self.__toexcel = woe_df.sort_values(re_search('_keep' ,woe_df)).reset_index(drop=True)
        if output:
            woe_df = woe_df.drop_col(['IV_i' ,'woe' ,'Gini'])
        return woe_df.sort_values(re_search('_keep' ,woe_df)).reset_index(drop=True)
        
    def woe_dict(self, var_names=None):
        """woe_dict 字典包含了把分箱变量映射成woe变量的所有信息，用 DataFrame 表示. \n

        参数:
        ----------
        var_names: str or list_of_str,
            单个或多个分箱变量。\n

        返回值:
        ----------
        woe_map: DataFrame, colName 列是分箱变量名，woe 列是 woe 值， index是分箱值，分箱值可以是 np.nan ."""
        woe_map = self.__df.set_index('colBins')[['colName', 'woe']].copy()
        return woe_map
    
    def bin2woe(self, var_names=None, inplace = False):
        """把 detail_sr 表中的分箱变量，映射成 woe 变量。

        参数:
        -----------
        var_names: 要转换的分箱变量列表 ，尾部带 '_bin'
        inplace: bool,
            是否把映射的 woe 变量，直接添加为 detail_sr 的列

        返回:
        -----------
        woe_detail_df: DataFrame,
            若 inplace 参数为 True，则无返回值
            所有变量映射成了 woe 变量的明细数据表，index 与 detail_sr.index 是对齐的.
            若某一 woe 列有-9999的值出现，说明有分箱变量值，没有在 woe_map 中找到对应 woe 值，一般暗示了某种错误"""
        woe_map = self.woe_dict(var_names)
        woe_series = []
        for var in woe_map.colName.unique():
            map_i = woe_map.loc[woe_map.colName == var, 'woe']  # map_i 是个 series
            woe_sr_i = self.final_bin_[var].map(lambda bin_value: map_i.get(bin_value, -9999))            
            woe_sr_i.name = var[:-4] + '_woe'   # woe变量统一以 '_woe' 结尾
            woe_series.append(woe_sr_i)

        if inplace:
            self.df = self.df.same_col_merged(self.final_bin_ ,how = 'left' ,left_index = True ,right_index = True)
            for sr in woe_series:
                self.df[sr.name] = sr
        else:
            for sr in woe_series:
                self.final_bin_[sr.name] = sr

    def append(self ,woe_df ,inplace = True):
        """增加一个变量的 woe_df 到本对象中. 要添加多个变量的 woe_df,请用 extend

        参数:
        -----------
        woe_df: dataframe, woe 表，格式由 cal_woe 函数返回值的形式确定。"""
        if inplace:
            woe_df['colName'] = woe_df['colName'].apply(lambda x:x+'_bin')
        col_name = woe_df.colName.unique()
        if len(col_name) == 0:
            raise Exception("参数 woe_df 的 colName 列中没有值,很可能是数据格式不对")
        elif len(col_name) > 1:
            raise Exception("参数 woe_df 中有多个变量，请使用 extend 方法")
        else:
            if col_name[0] in self.__df.colName:
                print("提醒：{}已经存在于此WoeDf对象中，忽略 append 操作。\n"
                      "若要更新变量{}的woe_df数据，请使用update_var".format(col_name[0]))
            else:
                self.__df = self.__df.append(woe_df)
                
    def extend(self, woe_dfs ,inplace = True):
        """追加多个变量的 woe_df 到本对象中

        参数:
        ----------
        woe_dfs: 多个 woe_df 表拼在一起组成的 DataFrame
        """
        if inplace:
            woe_dfs['colName'] = woe_dfs['colName'].apply(lambda x:x+'_bin')
        col_name = woe_dfs.colName.unique()  # woe_dfs 中的变量名有重复时，程序无法识别和处理，也不处理
        cols_in = self.all_vars()
        cols_extend = []
        for col in col_name:
            if col in cols_in:
                print("提醒：{}已经存在于此WoeDf对象中，忽略 extend 操作。\n"
                      "若要更新变量{}的woe_df数据，请使用update_var".format(col))
            else:
                cols_extend.append(col)
        df = woe_dfs[woe_dfs.colName.isin(cols_extend)]
        self.__df = self.__df.append(df)
        
    def var_ivs(self, var_names=None, ascending=None, by='iv'):
        """返回所有记录变量的 IV 值 \n

        参数:
        -------
        by: str,
            按哪一指标进行排序，可选值有 'iv', 'ks'
        ascending: bool,
            默认值 None 表示不排序, True 表示升序，False表示降序
        var_names: str or list_of_str or index_of_str,
            返回哪些变量的 IV，默认值 None 表示返回所有变量的 IV

        返回:
        -------
        iv_df: dataframe,
            所有变量的 IV 值，按指定方式排序"""
        assert by.lower() in ('iv', 'ks')
        df = self.get_woe_final(var_names=var_names)
        ivs = df[['colName', 'IV', 'KS']].drop_duplicates().set_index('colName')
        col = 'IV' if by.lower() == 'iv' else 'ks'

        if ascending is None:
            return ivs
        else:
            return ivs.sort_values(by=col, ascending=ascending)

    def del_var(self, var_names):
        """删除指定变量的 woe_df

        参数:
        ------
        var_names : str or list_of_str, 需要删除的变量名/变量列表 ,以_bin结尾

        返回:
        -------
        无返回值，会修改对象自身的数据"""
        if isinstance(var_names ,str): var_names = [var_names]
        var_names = [i + '_bin' for i in var_names if i.find('_bin') < 0]
        logic = self.__df.colName.isin(var_names)
        logic = ~logic

        self.__df = self.__df[logic]

    def update_var(self, woe_df):
        """更新变量的 woe_df。当变量的分箱逻辑有变动时经常需要这样做 \n

        参数:
        ------
        woe_df : dataframe,
            变量的woe_df
        """
        var_name = woe_df.colName.iloc[0]
        self.del_var(var_name)
        self.append(woe_df)

    def all_vars(self):
        """以列表格式，返回所有保存了 woe_df 的列名"""
        return self.__df.colName.drop_duplicates().tolist()
    
    def generate_scorecard(self, betas, A=427.01, B=-57.7078):
        """根据逻辑回归的系数，把入模变量各箱的woe值转换成各箱的得分，输出标准评分卡和打分映射字典。  \n

        参数:
        ----------
        betas : series or dict, 逻辑回归得到的各变量的系数, 应包括常数项 Intecept 或 const.  \n
            betas 中，各变量名是分箱变量（以_bin结尾），而非woe变量（以_woe结尾）
        A: float, 分数转换的参数A   \n
        B: float, 分数转换的参数B   \n

        返回:
        ----------
        scorecard_df : dataframe, 标准的评分卡 \n"""
        if isinstance(betas, dict):
            betas = pd.Series(betas)
        betas = betas.rename({'Intercept': 'const'}).rename({'intercept': 'const'})

        var_names = {i: i.replace('_woe', '_bin') for i in betas.index.drop('const')}
        betas = betas.rename(var_names)
        woe_df = self.get_woe_df(betas.index.drop('const'))  # 入模变量的 woe_df
        const_df = pd.DataFrame(columns=woe_df.columns, index=[0])
        const_df.loc[0, 'colName'] = 'const'
        const_df.loc[0, 'woe'] = 1
        scorecard_df = pd.concat([const_df, woe_df])
        idx = scorecard_df.index

        betas.index.name = 'colName'
        betas = betas.reset_index().rename(columns={0: 'betas'})  # 0这一列是模型的参数值，colName列是模型的变量名
        scorecard_df = scorecard_df.merge(betas, on='colName')  # merge 拼表后丢失了 index
        scorecard_df.index = idx
        scorecard_df['score'] = B * scorecard_df['betas'] * scorecard_df['woe']  # 各变量的各箱得分

        base_score = A + B * betas.loc[0, 'betas']  # 基础分
        scorecard_df.loc[scorecard_df.colName == 'const', 'score'] = base_score

        g = scorecard_df.groupby('colName')
        max_score = g['score'].max().sum()
        min_score = g['score'].min().sum()
        print("scorecard range: [{0:.2f}, {1:.2f}".format(min_score, max_score))

        return scorecard_df

    def to_excel(self, file_name, var_names=None):
        """把 woe_df 对象的数据，保存为 excel 表文件

        参数:
        ----------
        file_path: str,
            excel文件的绝对路径， 不需要带扩展名
        var_names: str or list_of_str, 可选
            只保存指定变量的 woe_df 数据到 excel 中。None 表示保存所有变量的 woe_df 数据"""
        if isinstance(self.__toexcel ,pd.DataFrame):
            woe_df = self.__toexcel.copy()
        else:
            woe_df = self.__df.copy()
            
        if 'colBins' not in woe_df.columns.to_list():
            woe_df = woe_df.reset_index().rename_col({'index':'colBins'})

        if sys.platform == 'linux':
            file_path = r'//data/result_data/excel/'
        elif sys.platform == 'win32':
            file_path = r'D:/Result_data/excel/'
        else:
            pass
        with pd.ExcelWriter(file_path + file_name + '.xlsx') as writer:  # doctest: +SKIP
            woe_df.to_excel(writer ,index = False)
            print('write to {}, done!'.format(file_name + '.xlsx'))
        
    def get_ivcols(self, iv = 0.03 , inplace = False):
        """
        根据IV值删选需要的变量 ，一般删除IV 值小于0.03的变量
        --------------------------------
        参数：
        iv：IV的阈值
        inplace: 是否删除IV较低的变量
        """
        dfi_iv = self.var_ivs()
        cols_num = self.group_cols_.get('cols_num')
        cols_cate = self.group_cols_.get('cols_cate')
        eda_drop = self.group_cols_.get('eda_drop') 
        cols_nan = self.group_cols_.get('cols_nan') 
        if dfi_iv.loc[dfi_iv['IV'] < iv].empty == False:
            cols_iv = dfi_iv.loc[dfi_iv['IV'] < iv].reset_index()['colName'].to_list()
            cols_iv = [i[:-4] for i in cols_iv]
            for i in cols_iv:
                if i in cols_num: cols_num.remove(i)
                elif i in cols_cate: cols_cate.remove(i)
                else: pass                
            dfi = pd.DataFrame({'colName': [i + '_bin' for i in cols_num] + [i + '_bin' for i in cols_cate]})
            dfi['iv_keep'] = 1
            return dfi.sort_values('iv_keep',ascending = False)
        else:
            dfi = pd.DataFrame({'colName': [i + '_bin' for i in cols_num] + [i + '_bin' for i in cols_cate]})
            dfi['iv_keep'] = 1
            return dfi.sort_values('iv_keep',ascending = False)             
            cols_iv = []
        if inplace:
            self.group_cols_ = {'cols_num':cols_num ,'cols_cate':cols_cate ,'eda_drop': 
                                eda_drop ,'cols_nan':cols_nan ,'iv_drop': cols_iv}
        
    def get_corrcols(self, corr = 0.7 ,inplace = False):
        """
        计算相关性的函数
        --------------------------
        参数：
        corr: 阈值
        inplace: 如果是 True ，则根据阈值 对 group_cols_做更新，新增corr_drop ，以IV 排序，取IV 最高的变量
        -------------------------
        ----返回的结果中第一列为列名  已经打标尾缀 _bin
        第二列为 相关性分组，超过这个阈值的相关性分为一组
        """
        cols_iv = self.__df.sort_values(['IV','colName'] ,ascending=False).drop_duplicates('colName').colName.to_list()
        cols_bin = [i[:-4] + '_woe' for i in cols_iv]
        cols_woe = re_search('woe' ,self.final_bin_)
        if len(cols_woe) == False:
            self.bin2woe()
            dfi_corr = self.final_bin_[cols_bin + cols_iv].copy()
            cols_woe = re_search('woe' ,dfi_corr)
            cols_drop = re_search('woe' ,dfi_corr)
        else:
            dfi_corr = self.final_bin_[cols_bin + cols_iv].copy()
            cols_woe = re_search('woe' ,dfi_corr)
            cols_drop = re_search('woe' ,dfi_corr)
        dfi_final = dfi_corr[cols_woe].corr_tri()   
        corr_final = pd.DataFrame()        
        j = 1
        for i in cols_drop:
            if i in cols_woe:
                cols_l = [i]
                dfi_final = dfi_final[cols_woe].copy()           
                cols_m = dfi_final.loc[dfi_final[i] > corr].index.to_list()
                cols_k = []
                if len(cols_m):
                    for k in cols_m:
                        if k in cols_woe:
                            cols_k.extend(dfi_final.loc[dfi_final[k] > corr].index.to_list())
                    cols_m.extend(cols_k)
                    cols_m = [k for i ,k in enumerate(cols_m) if k not in cols_m[:i]]
                cols_l.extend(cols_m)
                for k in cols_l:
                    if k in cols_woe:
                        cols_woe.remove(k)
                cols_l = [i[:-4] + '_bin' for i in cols_l]
                cols_m = [str(j) + '_' + str(i+1) for i in range(len(cols_l))]
                corr_final = corr_final.append(pd.DataFrame({'colName':cols_l ,'corr_keep':cols_m }))
                j+=1
        if inplace:
            corr_keep = corr_final.loc[corr_final.corr_keep.apply(lambda x: x[-2:]) == '_1'].colName.to_list()
            corr_drop = corr_final.loc[corr_final.corr_keep.apply(lambda x: x[-2:]) != '_1'].colName.to_list()
            if len(corr_keep):corr_keep = [i[:-4] for i in corr_keep]
            if len(corr_keep):corr_drop = [i[:-4] for i in corr_drop]
            self.group_cols_['cols_num'] = [i for i in self.group_cols_.get('cols_num') if i not in corr_drop]
            self.group_cols_['cols_cate'] = [i for i in self.group_cols_.get('cols_cate') if i not in corr_drop]
            self.group_cols_['corr_drop'] = corr_drop
        return corr_final
    
    def get_sortcols(self, bad_size = 1.5 ,per = 0.01, inplace = False):
        """
        判断变量的排序性。
        一般情况下  只针对  数值型的变量做一些标记 ，不建议直接删除，可以等确定后再进行删除
        参数：
        -----------------
        bad_size:坏样本的占比 或者 好样本的占比倍数
        per:好样本末分箱占比
        inplace ：是否删除排序性较差的变量
        """
        def in_order(nums):
            a = [nums[i-1] <= nums[i] for i in range(1, len(nums))]
            return all(x == a[0] for x in a)
        mean_value = round(self.df[self.target].value_counts()[1]/self.df[self.target].count(),2)
        sort_dfi = self.__df.reset_index().rename_col({'index':'colBins'}).copy()
        cols_name = sort_dfi.drop_duplicates('colName')['colName'].to_list()
        cols_or = self.group_cols_.get('cols_num')
        cols_or.extend(self.group_cols_.get('cols_cate'))
        lift_keep = list()
        sort_keep = list()
        for i in cols_name:
            df = sort_dfi.loc[(sort_dfi['colName'] == i)].copy()
            if len(df) > 3 and i[:-4] in cols_or:
                if max(df['badRate']/mean_value) >= bad_size and df.loc[df['badRate'] == max(df['badRate'])].reset_index()['binPct'].iloc[0] >= per:
                    lift_keep.append(i[:-4])
                if in_order(df['badRate'].to_list()):
                    sort_keep.append(i[:-4])
                    
        lift_final = pd.DataFrame({'colName':[i + '_bin' for i in lift_keep]})
        lift_final['lift_keep'] = 1
        sort_final = pd.DataFrame({'colName':[i + '_bin' for i in sort_keep]})
        sort_final['sort_keep'] = 1
        sort_final = sort_final.merge(lift_final ,how = 'outer' ,on = 'colName')
        if inplace:
            lift_keep = [i for i in cols_or if i not in lift_keep]
            sort_keep = [i for i in cols_or if i not in sort_keep]
            self.group_cols_['cols_num'] = [i for i in self.group_cols_.get('cols_num') if i not in lift_keep]
            self.group_cols_['cols_cate'] = [i for i in self.group_cols_.get('cols_cate') if i not in lift_keep]
            self.group_cols_['lift_drop'] = lift_keep         
            self.group_cols_['cols_num'] = [i for i in self.group_cols_.get('cols_num') if i not in sort_keep]
            self.group_cols_['cols_cate'] = [i for i in self.group_cols_.get('cols_cate') if i not in sort_keep]
            self.group_cols_['sort_keep'] = sort_keep       
        return sort_final.sort_values('sort_keep')                    
    
    def get_psicols(self, psi_col ,per = 4 ,psi = 0.1 ,inplace = False):
        """
        根据psi ，删除不稳定的变量
        参数：
        ------------------------------------------
        psi_col:作为数据集划分的变量名称  ，一般是时间  date类型
        pei:划分多少个数据集，默认为4个 ，查看数据的稳定性，等频进行 分组
        -------------------
        返回：变量名称加  psi的相关数据
        """
        if len(re_search(psi_col ,self.final_bin_)):
            self.final_bin_.drop_col([psi_col,'psi_id'] ,inplace=True)           
        self.final_bin_ = self.final_bin_.merge(self.df[[psi_col]] ,how = 'left' ,left_index = True ,right_index = True)
        psi_c = sorted(self.final_bin_.drop_duplicates(psi_col)[psi_col].to_list())
        psi_l = [i+1 for i in range(len(psi_c))]
        df = pd.DataFrame({psi_col:psi_c ,'psi_id':psi_l})
        self.final_bin_ = self.final_bin_.merge(df ,how = 'left' ,on = psi_col)
        psi_x = pd.qcut(self.final_bin_['psi_id'] ,per).value_counts().index
        cols_i = []
        cols_i.extend([psi_x[i].left for i in range(len(psi_x))])
        cols_i.extend([psi_x[i].right for i in range(len(psi_x))])
        cols_i = sorted(list(set(cols_i)))
        cols_i = [cols_i[i] for i in range(len(cols_i)) if i not in [0,len(cols_i) -1]]
        cols_psi = self.group_cols_.get('cols_num')
        cols_psi.extend(self.group_cols_.get('cols_cate'))
        
        def psi_split(x):
            for i in range(len(cols_i)):
                if x <= cols_i[i]: 
                    tmp = i + 1
                    break
                else:tmp = i + 2 
            return tmp
        self.final_bin_['psi_id'] = self.final_bin_['psi_id'].apply(psi_split)
        dfi = self.final_bin_.copy()
        
        psi_final = pd.DataFrame()        
        for i in cols_psi:            
            psi_df = dfi.loc[dfi.psi_id == 1][i+'_bin'].value_counts().sort_index().to_frame()
            for k in dfi.drop_duplicates('psi_id')['psi_id'].to_list():
                if k != 1:
                    psi_df = pd.concat([psi_df ,dfi.loc[dfi.psi_id == k][i+'_bin'].value_counts().sort_index().to_frame().rename_col({i+'_bin': i + '_' +str(k)})] ,axis = 1)
            psi_df = psi_df / psi_df.sum()
            base = psi_df.iloc[:, 0]
            col_compare = psi_df.columns[1:]
            for col in col_compare:
                psi_i = (psi_df[col] - base) * np.log(psi_df[col] / base)
                psi_df[str(col) + '_psi'] = psi_i.sum()
            
            psi_final = psi_final.append(pd.DataFrame({'colName': [i +'_bin'] ,
                                          'psi_max': [round(max(psi_df.reset_index()[re_search('_psi',psi_df)].iloc[0]),2)] ,
                                          'psi_min': [round(min(psi_df.reset_index()[re_search('_psi',psi_df)].iloc[0]),2)] ,
                                          'psi_avg': [round(np.mean(psi_df.reset_index()[re_search('_psi',psi_df)].iloc[0]),2)] }))
        if inplace:
            psi_drop = psi_final.loc[psi_final.psi_avg > psi].colName.to_list()
            psi_drop = [i[:-4] for i in psi_drop]
            self.group_cols_['cols_num'] = [i for i in self.group_cols_.get('cols_num') if i not in psi_drop]
            self.group_cols_['cols_cate'] = [i for i in self.group_cols_.get('cols_cate') if i not in psi_drop]
            self.group_cols_['psi_drop'] = psi_drop                
        return psi_final.sort_values('psi_avg')
    
    def get_vifcols(self, vif_n = 10 ,inplace = False):
        """
        根据共线性筛选变量
        参数：
        ------------------------
        vif：阈值，以这个为
        
        """
        cols_woe = re_search('woe' ,self.final_bin_)
        if len(cols_woe) == False:
            self.bin2woe()
            dfi = self.final_bin_.copy()
        else:
            dfi = self.final_bin_.copy()
        vif_final = vif(dfi[re_search('woe',dfi)]).to_frame().reset_index().rename_col({'VIF':'vif_keep'})
        vif_final['colName'] = vif_final['colName'].apply(lambda x:x[:-4] + '_bin')
        vif_final['vif_keep'] = vif_final['vif_keep'].apply(lambda x:round(x,2))
        if inplace:
            vif_drop = vif_final.loc[vif_final.vif_keep >= vif_n].colName.to_list()
            vif_drop = [i[:-4] for i in vif_drop]
            self.group_cols_['cols_num'] = [i for i in self.group_cols_.get('cols_num') if i not in vif_drop]
            self.group_cols_['cols_cate'] = [i for i in self.group_cols_.get('cols_cate') if i not in vif_drop]
            self.group_cols_['vif_drop'] = vif_drop 
        return vif_final.sort_values('vif_keep')
    
    def update_cols(self, col_type ,cols ,up_type = 'append'):
        """
        更改变量的位置
        -------------------------------------
        参数：
        col_type：字典的 键值
        cols：需要更改的变量名称 ，可以是 str 或者 list
        up_type：更新类型，是新增 还是 删除
        ---------------------------------
        结果：
        返回的是更新之后的字典类型
        """
        cols_k = []
        cols_v = []
        cols_not = []
        if isinstance(cols ,str):cols = [cols]
        
        for k ,v in self.group_cols_.items():
            cols_k.append(k)
            cols_v.extend(v)
        for i in cols:
            if i not in cols_v:
                cols.remove(i)
                cols_not.append(i)
        if len(cols):
            if up_type == 'append':
                if col_type in cols_k:
                    for i in cols:
                        if i in self.group_cols_.get(col_type):
                            pass
                        else:
                            for k in cols_k:
                                self.group_cols_[k] = [n for n in self.group_cols_[k] if n != i]
                            self.group_cols_[col_type].append(i)
                else:
                    for k in cols_k:
                        self.group_cols_[k] = [n for n in self.group_cols_.get(k) if n not in cols]
                    self.group_cols_[col_type] = cols
            elif up_type == 'drop':
                if col_type in ['cols_num' ,'cols_cate']:
                    self.group_cols_[col_type] = [n for n in self.group_cols_.get(col_type) if n not in cols]
                    self.group_cols_['num_drop'] = cols
                elif col_type in [i for i in cols_k if i not in ['cols_num' ,'cols_cate']]:
                    self.group_cols_[col_type] = [n for n in self.group_cols_.get(col_type) if n not in cols]
                    for i in cols:
                        if i in self.__cols.get('cols_num'):
                            self.group_cols_.get('cols_num').append(i)
                        else:
                            self.group_cols_.get('cols_cate').append(i)
                else:
                    print(col_type + "not in group_cols_ de keys !!!")
        else:
            print("no cols need to update !!!")
        if len(cols_not):
            print(str(cols_not) + "not in df ,pleace check first !!!")