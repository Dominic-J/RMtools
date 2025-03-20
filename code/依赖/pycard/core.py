# -*- coding: utf-8 -*-
"""Created on Thu May 28 15:06:51 2015 @author: 左词
评分卡包的主模块, 包含了数据清洗、转换相关的所有工具"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from sklearn.metrics import auc, normalized_mutual_info_score  # 标准化互信息，度量变量a包含变量b的信息量的重合度
from . import config, tools
from os.path import join
from .tools import re_search

def infer_col_type(df):
    """
    输出dataframe中的数据类型：
    ----------------------------------
    一般是 date 、datetime 、int 、float 、str
    """
    dfi = df.dtypes
    col_type = dfi.copy()
    col_dtype = dfi.copy()
    for col_name ,dtype in zip(dfi.index ,dfi):
        if np.issubdtype(dtype,np.integer):
            col_type[col_name] = 'int64'
            col_dtype[col_name] = 'num'
        elif np.issubdtype(dtype,np.floating):
            col_type[col_name] = 'float64'
            col_dtype[col_name] = 'num'
        elif np.issubdtype(dtype,np.datetime64):
            col_type[col_name] = 'datetime64'
            col_dtype[col_name] = 'dtime'
        elif np.issubdtype(dtype ,np.object0):
            if len(df.loc[pd.notnull(df[col_name])].reset_index()[col_name]):
                if isinstance(df.loc[pd.notnull(df[col_name])].reset_index()[col_name].iloc[0] ,str):
                    col_type[col_name] = 'object64'
                    col_dtype[col_name] = 'cate'
                else:
                    col_type[col_name] = 'date64'
                    col_dtype[col_name] = 'dtime'
            else:
                col_type[col_name] = 'object64'
                col_dtype[col_name] = 'cate'                
        else:
            pass
        dfi_type = pd.DataFrame({'dtype':col_type.to_list(),'infered_type':col_dtype.to_list()} ,index=col_type.index.to_list())
    return dfi_type

def cols_report(detail_df):
    """对明细表的各列的描述性统计.
    参数:
    --------
    detail_sr: dataframe, 最细粒度的数据表
    返回值:
    --------
    cols_describe: dataframe, 数据表各列的描述性统计，如缺失率、非重复值、变异系数、中位数等"""
    type_df = infer_col_type(detail_df)  # 列的数据类型
    nan_df = detail_df.nan_rate()  # 列的缺失率
    unique_df = detail_df.apply(pd.Series.nunique)  # 列的非重复值个数
    unique_df.name = 'unique'
    cols_describe = pd.concat([type_df, unique_df, nan_df], axis=1).reindex(type_df.index)

    # 数值型列的描述统计
    sub_df_num = detail_df.select_dtypes(include=np.number)
    if sub_df_num.shape[1] > 0:
        desc_df = sub_df_num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose().drop_col('count')
        desc_df['cv'] = desc_df['std'] / desc_df['mean']
        cols_describe = pd.concat([cols_describe, desc_df], axis=1).reindex(type_df.index)

    # 非数值型列的描述统计
    sub_df_other = detail_df.select_dtypes(exclude=np.number)
    if sub_df_other.shape[1] > 0:
        desc_df = sub_df_other.describe(datetime_is_numeric=True).transpose().drop_col(['count', 'unique'])
        cols_describe = pd.concat([cols_describe, desc_df], axis=1)
    return cols_describe


class ColsGroup:
    """用于维护特征分类的数据结构，可分门别类地保存类别型、序数型、数值型、主外键、日期类特征名，\n
    并可随着分析的推进，不断地记录剔除的列名。\n

    初始化参数:
    -----------
    keys=() : iterable, 主键/外键类列名和目标变量的集合\n
    dtime=() : iterable, 日期/时间型列名的集合\n
    drop=() : iterable, 已剔除的特征的集合\n

    属性列表：
    ----------
    keys: list, 保存主键/外键类的特征。此类特征一般不会入模。\n
    dtime： list, 保存日期/时间型的特征。此类特征一般不会入模。\n
    dropped: property, 保存已删除的特征。\n

    unq8： property, 保存尚未被剔除的、非重复值小于8个的特征，可能是类别/数值/序数型。此类特征最易分析，应最先找它们来。\n
    cate： property, 保存尚未被剔除的类别型的特征。\n
    num： property, 保存尚未被剔除的数值型的特征。\n
    ordi： property, 保存尚未被剔除的序数型的特征。\n
    onehot: property, 保存独热型的特征。\n

    all_vars: property, 所有的特征。\n
    analysis_vars: property, 所有分析类的特征，包括 unq8, cate, num, ordi, onehot。keys, dtime不是分析类的特征。\n

    方法列表：
    ----------
    add: 用来添加特征到指定的类别中去。\n
    drop: 用来记录要剔除的特征。\n
    undo_drop: 用来撤回剔除操作，找回已剔除的特征。\n
    print_detail: 打印出各类变量的详细清单，每 5 个一行。\n
    """

    def __init__(self, keys=(), dtime=(), drop=()):
        def remove_dup(iters):
            """返回原顺序的iters，但去除了重复的元素"""
            no_dup = []
            d = set()
            for i in iters:
                if i not in d:
                    d.add(i)
                    no_dup.append(i)
            return no_dup

        self.keys = remove_dup(keys)  # 保存主键、外键类的变量和目标变量
        self.dtime = remove_dup(dtime)  # 保存日期时间类的变量
        self.__drop = remove_dup(drop)  # 保存被删除的变量

        self.__unq8 = []  # 保存不同取值数小于8个的变量
        self.__cate = []  # 保存无序分类变量
        self.__num = []  # 保存数值变量
        self.__ordi = []  # 保存序数型变量
        self.__onehot = []  # 保存独热型变量
        self.__colstore = {} # 保存其他类型的变量
        self.__collist = [] # 临时存储变量列表

    @property
    def dropped(self):
        """返回已被剔除的所有变量组成的列表"""      
        return self.__drop
    @property
    def colstore(self):
        return self.__colstore
    
    @property
    def unq8(self):
        """返回尚未被剔除的序数型变量组成的列表"""
        return [i for i in self.__unq8 if i not in self.dropped]

    @property
    def cate(self):
        """尚未被剔除的类别型变量组成的列表"""
        return [i for i in self.__cate if i not in self.dropped]

    @property
    def num(self):
        """尚未被剔除的数值型变量组成的列表"""
        return [i for i in self.__num if i not in self.dropped]

    @property
    def ordi(self):
        """返回尚未被剔除的序数型变量组成的列表"""
        return [i for i in self.__ordi if i not in self.dropped]

    def dtime(self):
        """返回尚未被剔除的时间型变量组成的列表"""
        return [i for i in self.__dtime if i not in self.dropped]

    @property
    def onehot(self):
        """返回尚未被剔除的onehot型变量组成的列表"""
        return [i for i in self.__onehot if i not in self.dropped]

    @property
    def all_vars(self, include_drop=True):
        """所有的变量，包括 keys, dtime类，分析类、drop类，返回一个 list

        参数:
        ----------
        include_drop: bool, 是否要包括已标记为剔除的变量"""
        vars = self.keys + self.dtime + self.unq8 + self.cate + self.num + self.ordi + self.onehot
        if include_drop:
            vars = vars + self.dropped
        return vars

    @property
    def analysis_vars(self):
        """所有的分析类变量： unq8, cate, num, ordi, onehot。返回一个 list"""
        return self.unq8 + self.cate + self.num + self.ordi + self.onehot

    def add(self, col_names, dtype='num'):
        """添加 unq8|cate|num|ordi|onehot 类型的特征名。
        本方法只是把相关的特征名，添加到对应的类型中去，并不会更改数据表的任何数据。
        即使重复添加变量也没有影响。add方法会自动处理好重复问题。
        函数无返回值 \n

        参数:
        ----------
        col_names : 需要添加的列名。如果是单个列名, 只能是字符串类型。如果是多列，可以是任意 iterable。\n
        dtype: str, 把 col_names 添加到哪个类别中去，可选值有'unq8','cate','num','ordi','onehot' ，传入其他值会报错。"""

#        assert dtype in ('unq8', 'cate', 'num', 'ordi', 'onehot' ), "dtype not in ('unq8', 'cate', 'num', 'ordi', 'onehot')"
        single_col = isinstance(col_names, str)# 列名只能是字符串类型，若是其它类型，此处检查不出来
        #
        if single_col:
            if dtype == 'unq8' and col_names not in (self.__unq8 + self.keys + self.dtime):
                self.__unq8.append(col_names)
            elif dtype == 'cate' and col_names not in (self.__cate + self.keys + self.dtime):
                self.__cate.append(col_names)
            elif dtype == 'num' and col_names not in (self.__num + self.keys + self.dtime):
                self.__num.append(col_names)
            elif dtype == 'ordi' and col_names not in (self.__ordi + self.keys + self.dtime):
                self.__ordi.append(col_names)
            elif dtype == 'ordi' and col_names not in (self.__onehot + self.keys + self.dtime):
                self.__onehot.append(col_names)
            else:
                if len(self.__colstore) >= 1:
                    if dtype in self.__colstore.keys():
                        if isinstance(col_names,str):
                            self.__collist = self.__colstore.get(dtype)         
                            self.__collist.append(col_names)
                            self.__colstore[dtype] = list(set(self.__collist))
                        else:
                            for i in col_names:
                                self.__collist = self.__colstore.get(dtype)         
                                self.__collist.append(i)
                            self.__colstore[dtype] = list(set(self.__collist))                                
                    else:
                        self.__colstore[dtype] = list(set([col_names]))
                else:
                    self.__colstore[dtype] = list(set([col_names]))
                    
        else:
            for namei in col_names:
                self.add(namei, dtype=dtype)

    def drop(self, name, dtype):
        """添加需要剔除的列名. 无返回值 \n
        本方法只是记录要剔除的特征名到 dropped 中去，并不会删除数据表的任何一列数据。
        即使重复剔除变量也没有影响。drop 方法会自动处理好重复问题。

        参数:
        ----------
        name : 如果是单个列名, 只能是字符串类型。如果是多列，可以是任意 iterable。"""
        if isinstance(name, str):  # 列名应该只用字符串来表示，若用了其他类型的值，此处检测不出来
            if name not in self.__drop:  # 不重复添加已经删除的变量
                self.__drop.append(name)  # 如果是单个列。单列只能是字符串，不能是其他类型
                self.__drop.append(dtype)    
        else:
            for namei in name:
                self.drop(namei)

    def undo_drop(self, name):
        """把列名 name 从已剔除集合中移除。\n

        参数:
        ----------
        name : 需要删除的列名。如果是单个列名, 只能是字符串类型。如果是多列，可以是任意 iterable。"""
        if isinstance(name, str):  # 列名应该只用字符串来表示，若用了其他类型的值，此处检测不出来
            try:
                self.__drop.remove(name)
            except ValueError:
                pass
        else:
            for namei in name:
                self.undo_drop(namei)

    def print_detail(self):
        """打印出各类变量的详细清单，每 5 个一行"""
        for key in ('unq8', 'cate', 'ordi', 'num', 'onehot', 'dropped'):
            cols = getattr(self, key)
            if len(cols):
                print(key, ":")
                for col in [cols[i: i + 5] for i in range(0, len(cols), 5)]:
                    print("    {},".format(', '.join(col)))

    def __str__(self):
        a = []
        for i in ('unq8', 'cate', 'ordi', 'num', 'onehot', 'dropped', 'keys', 'dtime'):
            cols = getattr(self, i)
            if len(cols):
                a.append('%-6s 个数: %4d' % (i, len(cols)))
        a = '\n'.join(a)
        return a

    def __repr__(self):
        return self.__str__()


def outlier(detail_df):
    """数值型变量的异常值检测：超过 boxplot 图的上、下边界的值，认为是异常值\n

    参数:
    -----------
    detail_sr: dataframe, 明细数据表

    返回:
    -----------
    des_with_outlier: dataframe, 各个数值型列的描述性统计，并计算上、下边界值，及超过边界值的行占比"""
    des = detail_df.describe([0.25, 0.75]).transpose()
    q3 = des['75%']
    q1 = des['25%']
    delta_q = q3 - q1

    des['left_edge'] = q1 - 1.5 * delta_q  # 下边界
    des['right_edge'] = q3 + 1.5 * delta_q  # 上边界
    des['left_outlier_pct'] = 0.0  # 极小值的占比
    des['right_outlier_pct'] = 0.0   # 极大值的占比
    cols_has_too_low = des.index[des['min'] < des['left_edge']]
    for col in cols_has_too_low:
        logic = detail_df[col] < des.at[col, 'left_edge']
        des.at[col, 'left_outlier_pct'] = logic.mean()

    cols_has_too_high = des.index[des['max'] > des['right_edge']]
    for col in cols_has_too_high:
        logic = detail_df[col] > des.at[col, 'right_edge']
        des.at[col, 'right_outlier_pct'] = logic.mean()

    return des


def outlier_replace(detail_df, outlier_df):
    """对样本中的异常值进行替换(哪些是异常值，需要人来判断，并将此信息传给 outlier_df)：
         对于数值型变量，极大极小值替换为边界值；
         对于离散型变量，出现频数极小的值统一归到'other'组中

    参数:
    -----------
    detail_sr : 训练样本明细数据
    outlier_df : dataframe, 以列名为index, 以 ['left_edge', 'right_edge', 'other'] 为列名的异常值表, 三个列均为可选，不要求齐全
        index中的每个列，均表示 detail_sr 中的该列中有异常值。若该列是数值型的列，则把极小/极大值的边界值放在
         outlier_df 的 left/right 列中；若该列是离散型的列，则把异常值放在 outlier_df 的 other 列中，且组织成list

    返回:
    ------------
    replaced_df : dataframe, 替换了异常值之后的样本数据"""
    replaced_df = detail_df.copy()
    for col in outlier_df.index:
        row = outlier_df.loc[col]
        if hasattr(row, 'left_edge') and pd.notnull(row.left_edge):
            logic = detail_df[col] <= row.left_edge
            replaced_df.loc[logic, col] = row.left_edge
        if hasattr(row, 'right_edge') and pd.notnull(row.right_edge):
            logic = detail_df[col] >= row.right_edge
            replaced_df.loc[logic, col] = row.right_edge
        if hasattr(row, 'other') and pd.notnull(row.other):
            logic = detail_df[col].isin(row.other)
            replaced_df.loc[logic, col] = 'OutlierOther'
    return replaced_df


def group_to_other(cate_sr, k=30, other_value='OutlierOther'):
    """类别型的列，当某些类别值出现的次数低于阈值时，合并到“Other”分组中.
    不处理缺失值。

    参数:
    -----------
    cate_sr: series, 一列类别型的明细数据
    k: int, 设定的阈值，低于此阈值的类别值会合并分组
    other_value: str, 合并后的分组的取值，默认'Other'。当类别型的列中本身就有一种类别值叫'Other'时，会造成混淆。此时需要给这些
        合并的分组赋一个不同的取值

    返回值:
    ------------
    cate_grouped: series, 与 cate_sr 等长的明细数据列，其中频数小于 k 的类别值，替换成了 other_value
    other_group: list, 列表的元素值，就是被合并到其他分组中的类别值"""
    val = cate_sr.value_counts()
    other_group = list(val[val < k].index)   # 有可能是个空列表
    cate_grouped = cate_sr.copy()
    if len(other_group):
        logic = cate_sr.isin(other_group)
        cate_grouped = cate_sr.copy()
        cate_grouped[logic] = other_value
    return cate_grouped, other_group


def category_outlier_replace(detail_df, k=30, other_value='OutlierOther'):
    """把类别型的列中，出现频次太小的取值，合并到其他分组中。
    不处理缺失值。

    参数:
    -----------
    detail_sr: dataframe, 明细数据, 函数会自动检测类别型变量
    k: int, 设定的阈值，低于此阈值的类别值会合并分组
    other_value: str, 合并后的分组的取值，默认'Other'。当类别型的列中本身就有一种类别值叫'Other'时，会造成混淆。此时需要给这些
        合并的分组赋一个不同的取值

    返回值:
    ------------
    cate_grouped: dataframe, 明细数据列，仅包括合并后的同名列，其中频数小于 k 的类别值，替换成了 other_value
    other_group: Series, 元素值就是被合并到其他分组中的类别值, index是对应的列名
    """
    other_group = pd.Series()
    cate_grouped = []
    for col in detail_df:
        sr_i = detail_df[col]
        if sr_i.is_category():   # 仅对类别型变量做合并
            cate_group_i, other_group_i = group_to_other(sr_i, other_value=other_value, k=k)
            if len(other_group_i):   # 可能是个空列表
                other_group[col] = other_group_i
                cate_grouped.append(cate_group_i)
    cate_grouped = pd.concat(cate_grouped, axis=1)
    return cate_grouped, other_group


def discretize(detail_sr, bins=256, method='width'):
    """离散化数值型变量, 到给定的有限组离散区间，区间是左开右闭的，并返回区间的右边界值。
    离散化之后，仍然是数值型变量。
    若数值型变量本身的取值个数小于给定的值，则不会离散化它。
    离散化可以钝化数值型变量的变动对模型的影响，亦可大幅提高部分算法的求解速度。
    dtype 为 np.number 的子类的列都会被识别为数值型的列

    参数:
    -----------
    detail_sr: Series, 明细数据列
    bins: int, 离散化成多少分组
    method: str, 以什么方法来离散化，可选有2种：
        width(等宽，每个组的取值间距相等)，
        depth(等深，每个组的样本占比相等)

    返回:
    -----------
    dis: Series, 离散化后的数据列，其取值代表离散化区间的右端点值
    """
    cut_fun = pd.cut if method == 'width' else pd.qcut

    if detail_sr.is_numeric():
        n_unique = len(detail_sr.unique())
        if n_unique > bins:
            dis = cut_fun(detail_sr, bins=bins)
            dis = dis.apply(lambda interval: interval.right)
            dis = dis.astype(detail_sr.dtype)

    return dis


def cal_woe(tab_df, dtype='num'):
    """根据分箱变量和目标变量的列联表，计算 woe, iv, gini。tab_df 应包含分箱变量的所有箱。  \n
    本模块把这个函数计算的结果表，称作woe_df。很多函数的返回值，都是woe_df。  \n

    参数:
    ----------
    tab_df：分箱变量和目标变量的列联表   \n
    dtype: str, 分箱变量的类型，取值有 num, cate, ordi, 分别表示数值型、类别型、序数型变量。 \n

    返回值:
    ---------
    woe_df: dataframe, 包含的列及其解释如下 \n
        colName: 列名, \n
        0: good, \n
        1: bad, \n
        total: 行汇总(good + bad), \n
        binPct: total 的列占比（All_i / total.sum()）, \n
        badRate: 行 bad 占比（每一行中，bad/total） \n
        woe: 分箱的 woe 值, \n
        IV_i: 分箱的 IV 值, \n
        IV: 变量的 IV 值, \n
        Gini: 变量的 Gini 系数值
    """
    assert dtype in ('num', 'cate', 'ordi')

    woe_df = tab_df.copy()
    woe_df['total'] = woe_df[1] + woe_df[0]
    woe_df['goods'] = woe_df[0].cumsum()
    woe_df['bads'] = woe_df[1].cumsum()
    dfsum = woe_df.sum()
    woe_df['binPct'] = round(woe_df['total'] / dfsum.total ,4)
    woe_df['badRate'] = round(woe_df[1] / woe_df['total'],4)

    good_pct = woe_df[0].values / dfsum[0]
    bad_pct = woe_df[1].values / dfsum[1]
    while not bad_pct.all():
        i = bad_pct.argmin()
        # 若箱内 bad 为 -1，则调整为 bad_pct_i = 0.5 / bad_all, good_pct_i = (good_i + 0.5) / good_all
        bad_pct[i] = 0.5 / dfsum[1]
        good_pct[i] = (woe_df[0].iloc[i] + 0.5) / dfsum[0]
        #print("警告：箱 {} 中的坏客户数为0，将调整计算woe值".format(woe_df.index[i]))
    woe_df['woe'] = np.log((bad_pct + 1e-5)/ (good_pct + 1e-5))
    woe_df['IV_i'] = (bad_pct - good_pct) * woe_df['woe']
    woe_df['IV'] = round(woe_df['IV_i'].sum(),4)
    # 计算KS
    woe_df['KS'] = round(max(abs((woe_df['goods']/dfsum[0] - woe_df['bads']/dfsum[1]))),4)
    woe_df = woe_df.drop(columns = ['goods','bads'])
    # 对 woe_df 排序，不同类型的变量，排序方式不同
    if dtype == 'cate':
        woe_df = woe_df.sort_values(by='woe')
    elif dtype in ('num', 'ordi'):  # 数值型变量、序数型变量，需要按变量值排序后计算
        try:
            woe_df['num'] = woe_df.index.map(lambda bin_value: int(bin_value.split('_')[0]))
            woe_df = woe_df.sort_values(by='num')
            del woe_df['num']
        except:
            pass

    # 按分箱值本身排序时，缺失值会排在最后，把它移到第一位去
    if pd.isnull(woe_df.index[-1]):
        na_row = woe_df.iloc[-1:, :]
        woe_df = pd.concat([na_row, woe_df.iloc[:-1, :]])

    # 计算 gini
    dfCum = pd.DataFrame([[0, 0]], columns=[0, 1])
    dfCum = pd.concat([dfCum ,woe_df[[0, 1]].cumsum() / dfsum.loc[[0, 1]]])  # 累计占比
    area = 0
    for i in range(1, len(dfCum)):
        area += 1 / 2 * (dfCum[1].iloc[i - 1] + dfCum[1].iloc[i]) * (dfCum[0].iloc[i] - dfCum[0].iloc[i - 1])
    woe_df['Gini'] = round(2 * (area - 0.5),4)
    
    return woe_df.rename_col({0:'good',1:'bad'})


def join_row(tab_df, n, iadd=1, dtype='num'):
    """把第 n 行与其他行相加合并，iadd 一般为 1 或 -1，即相加相邻行到第 n 行，同时删除相邻行 \n
    合并分箱时，经常会用到此函数。

    参数:
    ----------
    tab_df : 需要操作的数据表\n
    n : 需要合并的行的序号\n
    iadd: 相对于n的行号偏移量。\n

    返回值:
    ----------
    new_df: 新的 dataframe, 不修改原 df"""
    if 'colName' in tab_df:
        colName = tab_df.colName.iloc[0]

    tab_df = tab_df.copy()
    tab_df.iloc[n, :] = tab_df.iloc[n, :] + tab_df.iloc[n + iadd, :]
    tab_df = tab_df.idrop_row(n + iadd)

    if 'colName' in tab_df:
        tab_df.colName = colName

    return cal_woe(tab_df, dtype=dtype)


def join_as_other(tab_df, k=100):
    """样本量小于 k 的箱，统一合并到other组中。因为某个箱的样本量太小的话，受随机扰动的影响就很大。 \n
    其他箱的名字就叫做 "other", 因为除了 "other"，其余箱的分箱逻辑都知道 \n

    参数:
    ----------
    tab_df : 列联表，表的每一行是一个箱 \n
    k: int, 样本量的阈值

    返回值:
    ----------
    tab_k: dataframe, 相比输入参数tab, 删除了样本量小于 k 的行，并增加了一个 "other" 箱
    """
    All = tab_df[1] + tab_df[0]
    logic = All < k
    if logic.any():
        other = tab_df[logic].sum().to_frame(name='other').transpose()
        other[[0, 1]] = other[[0, 1]].astype(tab_df[1].dtype)
        tab_k = tab_df[~logic]
        tab_k = pd.concat([other, tab_k])
    return tab_k


def crosstab(index, column):
    """求列联表时，把缺失也统计进来. pandas 自带的 crosstab 会把缺失值忽略掉  \n
    参数:
    ----------
    index : series, 用作列联表行维的series  \n
    column : series, 用作列联表列维的series, 应与 index 列等长   \n
    返回值:
    ----------
    tab_df: dataframe, 两个输入列的列联表，列中的缺失值也会统计进来"""

    tab_df = pd.crosstab(index, column, dropna=False)
    logic = pd.isnull(index)
    if logic.any():
        na_row = column[logic].value_counts().to_frame(name=np.nan).transpose()
        tab_df = pd.concat([na_row, tab_df])

    logic = column.isnull()
    if logic.any():
        na_col = index[logic].value_counts(dropna=False).to_frame(name=np.nan)
        tab_df = pd.concat([tab_df, na_col], axis=1)

    return tab_df


def crosstab_pct(index, column, dropna=False):
    """把列联表的个数及占比均求出来 \n
    参数:
    ----------
    index : series, 用作交叉表行维的列  \n
    column : series, 用作交叉表列维的列  \n
    dropna : bool, 是否要把缺失值丢弃 \n
    返回值:
    ---------
    tab_with_pct: dataframe, 其中列联表个数在上半部分，列联表的占比在下半部分
    """
    if dropna:
        a = pd.crosstab(index, column, margins=True)
    else:
        a = crosstab(index, column)
        a['All'] = a.sum(axis=1)
        a.loc['All'] = a.sum()
    b = a / a.iloc[-1, -1]
    return pd.concat([a, b])


def cross_woe(var, y, dtype='num', dropna=False ,output=True):
    """以分箱变量var和目标变量y，计算woe_df。\n
    参数:
    ----------
    var: sr, 分好箱的离散特征变量   \n
    y: sr, 模型的目标变量  \n
    dtype: str, 分箱变量的类型，取值有num, cate, ordi。 \n
    dropna: 是否要丢弃 var 变量的 nan 分组，默认为False \n
    返回值:
    ---------
    woe_df: dataframe, 包含的列及其解释如下 \n
        colName: 列名, 0: good, 1: bad, All: 行汇总(good+bad), binPct: All列的列占比, badRate: 行bad占比 \n
        woe: 分箱的woe值, IV_i: 分箱的IV值, IV: 变量的IV值, Gini: 变量的Gini系数值  \n
    参见 cal_woe, cross_woe 会调用 cal_woe 函数。"""
    var, y = na_split_y(var.replace(np.NaN ,'0_nan'), y)
    if not dropna:
        tab_df = crosstab(var, y)
    else:
        tab_df = pd.crosstab(var, y)

    try:
        tab_df['colName'] = var.name
    except AttributeError:
        tab_df['colName'] = 'var'
    df = cal_woe(tab_df.reindex(columns=['colName', 0, 1]), dtype=dtype)
    df['lift'] = round(df['badRate']/(df['bad'].sum()/df['total'].sum()),2)
    df.reset_index(inplace=True)
    df = df.rename_col({i:'colBins' for i in ['index',var.name]})
    if output:
        df = df.drop_col(['woe','IV_i' ,'Gini'])
    return df


def binning(sr_numeric, bins):
    """把 sr_numeric 离散化成区间型的列，分数最低的在第 1 组，最高的在 len(bins)+1 组.\n
    如果 sr_numeric 有缺失，缺失被分在第 0 组。区间均为左开右闭型：(low,high] \n

    参数:
    ----------
    sr_numeric: Series , 需要离散化的数值型或序数型变量  \n
    bins: list, 对 sr_numeric 进行离散化的分割点。bins 的分割点可以乱序，函数会自动排序分割点。
        n 个点会把 sr_numeric 离散化成 n+1 个组, 函数会自动为 bins 添加 -np.inf, np.inf, 第一
        个区间为 (-np.inf, 最小分割点], 最后一个区间为 （最大分割点, np.inf]

    返回值:
    ----------
    group: 1darray, 离散化后的一维数组"""
    bins = sorted(bins)
    if -np.inf not in bins:
        bins = [-np.inf] + bins
    if np.inf not in bins:
        bins = bins + [np.inf]

    def base_map(x):
        for i in range(len(bins) - 1):
            start = bins[i]
            end = bins[i + 1]
            if start < x <= end:
                return "{0}_({1}, {2}]".format(i + 1, start, end)  # 分箱字符串格式： i_(start, end]
        return '0_nan'
    return sr_numeric.map(base_map)


def monotony_type(woe_sr):
    """判断分箱变量的 woe 的单调类型：单调上升、单调下降、U形、倒U形  \n

    参数:
    ----------
    woe_sr: 分箱变量的各个箱的woe。sr 应按箱的业务逻辑排好序了再传给此函数。  \n

    返回值:
    ----------
    mono: str, 可能的取值有('单调上升', '单调下降, 'U形', 'n形', '无序')  """

    def arising(sr):
        """单调上升"""
        diff = sr.diff() > 0
        return diff[1:].all()

    def derising(sr):
        """单调下降"""
        return arising(-sr)

    def u_like(sr):
        """U形"""
        idx = sr.idxmin()
        left = sr[:idx]
        right = sr[idx:]
        if len(left) == 0 or len(right) == 0:
            return False
        return derising(left) and arising(right)

    def n_like(sr):
        """倒U形"""
        return u_like(-sr)

    if arising(woe_sr):
        mono = '单调上升'
    elif derising(woe_sr):
        mono = '单调下降'
    elif u_like(woe_sr):
        mono = 'U形'
    elif n_like(woe_sr):
        mono = 'n形'
    else:
        mono = '无序'
    return mono


def ord2int(detail_df, ord_dict):
    """把序数型变量转换为int型，int的大小代表了序数变量的排序 \n
    参数:
    ----------
    detail_sr: dataframe, 明细数据表 \n
    ord_dict: dict，以列名为键，以按序排列的类别值组成的list为值\n
    返回值：
    ----------
    ord_df: 转换成int型的明细数据表, 后缀'_ord'的列是映射成的 int 型列 \n
    ord_map: dict, 各列的转换映射, 即每一列中，{类别值: 序号} 的映射字典"""
    ord_df = pd.DataFrame(index=detail_df.index)
    ord_map = {}
    for col, value in ord_dict.items():
        ord_map[col] = {i: num for i, num in zip(value, range(1, len(value) + 1))}
        ord_df[col + '_ord'] = detail_df[col].map(lambda x: ord_map[col][x])
    return ord_df, ord_map


def cate2int(detail_df, cols=None):
    """把类别型变量的类别值，映射成integer.\n
    参数:
    ----------
    detail_sr: dataframe, 包含了类别型变量的明细数据框 \n
    cols: array-like, 类别型变量名组成的列表，None 表示 df_cate 中所有列都是类别型变量，需要映射。\n
    返回值：
    ----------
    cate_df: dataframe, 对每一列，其带后缀'_cate'的列是映射的int型列 \n
    cate_map: 映射字典， 即每一列中，{类别值: 序号} 的映射字典"""
    if cols is None:
        cols = detail_df.columns
    if isinstance(cols, str):
        cols = [cols]

    cate_df = pd.DataFrame(index=detail_df.index)
    cate_map = {}
    for col in cols:
        cates = detail_df[col].unique()
        cate_map[col] = dict(zip(cates, range(len(cates))))
        cate_df[col + '_cate'] = detail_df[col].map(lambda x: cate_map[col].get(x))
    return cate_df, cate_map


def cate2onehot(cate_df, cols=None):
    """把类别型变量的类别值，转换成独热编码.\n  pandas.get_dummies()函数即可实现此功能
    参数:
    ----------
    cate_df: dataframe, 包含了类别型变量的明细数据框 \n
    cols: array-like, 类别型变量名组成的列表，None 表示 df_cate 中所有列都是类别型变量，需要映射。\n
    返回值：
    ----------
    df_onehot: dataframe, onehot数据框  \n"""

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    if cols is None: cols = cate_df.columns

    La = LabelEncoder()
    cates_type = []
    df_int = pd.DataFrame(index=cate_df.index)
    for col in cols:
        sr_i = cate_df[col]
        ar_i = La.fit_transform(sr_i)
        df_int[col + '_int'] = ar_i  # 映射变量
        cates_type.extend([col + '_' + str(i) for i in La.classes_])  # 记录类别到数字的映射

    enc = OneHotEncoder(sparse=False)
    df_onehot = pd.DataFrame(enc.fit_transform(df_int),
                             columns=cates_type, index=df_int.index)
    return df_onehot


class NumBin:
    """数值型变量的（单变量）分箱算法，使用的是 gini 不纯度分割算法。\n

    参数:
    ----------
    max_depth: int(default=3)
        最多分割到第几层
    max_bins_num: int(default=8)
        箱的最大个数，此参数决定最多分成多少个箱。
    min_bin_samples : int(default=30)
        箱的最小样本容量。如果一个箱的样本量小于此值，会被自动合并。
    min_impurity_decrease: float(default=0)
        一个箱会被继续分割成 2 箱，如果分割后使基尼不纯度下降的幅度 >= min_impurity_decrease"""

    def __init__(self, max_bins_num=8, max_depth=3,
                 min_bin_samples= 150, min_impurity_decrease=0):
        super().__init__()
        from sklearn.tree import DecisionTreeClassifier
        self.clf_ = DecisionTreeClassifier(max_leaf_nodes=max_bins_num,
                                           max_depth=max_depth,
                                           min_samples_leaf=min_bin_samples,
                                           min_impurity_decrease=min_impurity_decrease)
        self.__bin_code = {}  # 用于保存已经用该分箱算法实例 fit 过的分箱变量的分箱代码
        self.__bin_var = pd.DataFrame()  # 保存已经 fit 过的原始变量的分箱变量
        self.bins_ = None
        self.woe_df_ = None

    def fit(self, sr, y ,inplace = False ,output=True):
        """数值型/序数型变量的分箱，按决策树算法寻找最优分割点。\n

        参数:
        ----------
        sr : Series
            数值型预测变量。
        y : Series or ndarray
            目标变量，取值 {0, 1}
        inplace: 表示是否合并占比较小的分箱，合并使得分箱具有排序性

        返回值:
        ----------
        无返回值，但会更新 self 对象的如下属性：\n
            clf_: 训练好的决策树分类器 \n
            bins_: 算法找出的最佳分割点 \n
            woe_df_: 根据最佳分割点计算出来的 woe_df  \n
        另外，此对象会保存所有它 fit 过的变量的分箱变量和分箱转换代码，参见方法 bin_code、bin_var"""

        def tree_split_point(fitted_tree):
            """解析并返回树的各个分割节点"""
            bins = sorted(set(fitted_tree.tree_.threshold))
            return [round(i, 4) for i in bins]

        na_group, var_no, y_no = na_split(sr, y)
        self.clf_.fit(var_no.to_frame(), y_no)
        bins = tree_split_point(self.clf_)
        self.bins_ = bins[1:]
        if str(sr.name).find('_bin') == -1:col_name = str(sr.name) + '_bin'
        binvar = pd.DataFrame({col_name:self.transform(sr)})  # 包含有对缺失值的处理, 生成 woe_df
        binvar_cross = self.transform(sr)
        self.__bin_var = pd.concat([self.__bin_var,binvar] ,axis=1)
        tab_ = cross_woe(binvar_cross, y ,output=output)
        tab_['colName'] = sr.name

        if sr.name is None:
            print("Warning: 该数据列的 name 属性为 None")
        self.__bin_code[sr.name] = self.__generate_transform_fun(sr.name)
        
        if inplace:
            dfi = tab_.reset_index().rename_col({sr.name:'colBins'}).copy()
            dfi = dfi.loc[dfi.colBins != '0_nan'].reset_index(drop=True)

            if dfi['badRate'].iloc[0] > dfi['badRate'].iloc[len(dfi) - 1]:
                cols_b = [i - 1 for i ,k in enumerate(dfi['badRate'].to_list()) if i > 0 and k > min(dfi['badRate'].to_list()[:i+1])]
            elif dfi['badRate'].iloc[0] < dfi['badRate'].iloc[len(dfi) - 1]:
                cols_b = [i - 1 for i ,k in enumerate(dfi['badRate'].to_list()) if i > 0 and k < max(dfi['badRate'].to_list()[:i+1])]
            else:
                cols_b = self.bins_
            self.bins_ = [k for i,k in enumerate(self.bins_) if i not in cols_b]
            #重新跑数据
            if str(sr.name).find('_bin') == -1:col_name = str(sr.name) + '_bin'			
            binvar = pd.DataFrame({col_name:self.transform(sr)})  # 包含有对缺失值的处理, 生成 woe_df
            binvar_cross = self.transform(sr)
            self.__bin_var = pd.concat([self.__bin_var,binvar] ,axis=1)

            tab_ = cross_woe(binvar_cross, y ,output=output)
            tab_['colName'] = sr.name            

        col_i = tab_.sort_values('badRate').index.to_list()
        if '0_nan' in tab_['colBins'].to_list() and col_i.index(0) > 0 and col_i.index(0) < len(col_i)-1:
            col_m = col_i[col_i.index(0)-1]
            col_n = col_i[col_i.index(0)+1]
            dfn=pd.DataFrame()
            if col_m > col_n:
                dfn = pd.concat([tab_.iloc[1:col_m] ,tab_.iloc[:1:] ,tab_.iloc[col_m:]])
            else:
                dfn = pd.concat([tab_.iloc[1:col_n] ,tab_.iloc[:1:] ,tab_.iloc[col_n:]])
            dfn.reset_index(inplace=True ,drop=True)
            dfn.reset_index(inplace=True)
            dfn['colBins'] = dfn.apply(lambda x: str(x['index']) + '_' +x['colBins'].split('_')[1] ,axis = 1)
            dfn.drop_col('index' ,inplace=True)           
        else: dfn = tab_.copy()
        if dfn.lift.iloc[0] > 1:
            dfn['badCum'] = dfn['bad'].cumsum()
            dfn['totalCum'] = dfn['total'].cumsum()
            dfn['badCrate'] = round(dfn['badCum']/dfn['totalCum'] ,4)
            dfn['totalCrate'] = dfn['binPct'].cumsum()
        else:
            dfn.sort_values('colBins' ,ascending=False ,inplace=True)
            dfn['badCum'] = dfn['bad'].cumsum()
            dfn['totalCum'] = dfn['total'].cumsum()
            dfn['badCrate'] = round(dfn['badCum']/dfn['totalCum'] ,4)
            dfn['totalCrate'] = dfn['binPct'].cumsum()   
            dfn.sort_values('colBins' ,inplace=True)                               
        self.woe_df_ = dfn  
    def transform(self, sr):
        """根据训练好的最优分割点，把原始变量转换成分箱变量 \n

        参数:
        ------------
        sr: series, 原始变量，与训练时的变量含义相同时，做此转换才有意义  \n

        返回值:
        ------------
        binvar: ndarray, 转换成分箱变量，离散化成了有限组的一维数组"""
        binvar = binning(sr, self.bins_)
        return binvar

    def __generate_transform_fun(self, varname):
        """生成分箱映射函数的代码. """
        bins = [-np.inf] + self.bins_ + [np.inf]
        fun_code = """
def {varname}_trans(x): 
    # {varname} 连续型特征的分箱转换函数 
    inf = np.inf
    bins = {bins}
    for i in range(len(bins)-1):
        start = bins[i]
        end = bins[i+1]
        if start < x <= end:
            return {binstr}  # 分箱字符串格式： i_(start, end]
    return '0_nan'""".format(varname=varname, bins=bins,
                             binstr='"{0}_({1}, {2}]".format(i+1, start, end)')
        return fun_code

    def bin_var(self, var_names=None):
        """返回已 fit 过的变量所对应的的分箱变量

        参数:
        -----------
        var_names: str or list_of_str or index_of_str
            要访问的分箱变量名（或对应的原始变量名亦可），单个列名或多个列，列名只支持用字符串命名。
            分箱变量的命名规则是：原始变量名 + '_bin'
            默认值 None 表示返回所有已拟合过的变量的分箱变量

        返回:
        -----------
        binned_vars: series or dataframe
            对应的分箱变量，若输入是单个列名，则返回 series; 若输入是多个列名，则返回 dataframe"""
        if isinstance(var_names, str):  # 只能识别字符串格式的列名
            if not var_names.endswith('_bin'):
                var_names = var_names + '_bin'
            return self.__bin_var[var_names]
        elif var_names is None:
            return self.__bin_var  # None 表示返回所有分箱变量
        else:
            var_names = [var if var.endswith('_bin') else var + '_bin' for var in var_names]
            return self.__bin_var[var_names]

    def bin_code(self, var_name):
        """打印给定变量的分箱转换代码。

        参数:
        ------------
        var_name: str, 给定的变量名（特征名），该特征必须是经过本分箱对象 fit 过的。"""
        print(self.__bin_code.get(var_name, "变量 {} 没有被本对象 fit 过".format(var_name)))

    def clear(self, target='all'):
        """清空本对象所保存的、已 fit 过的变量的数据。这些数据大多是临时性质的，一直不清空，会占用大量内存空间。

        参数:
        ------------
        target: str,
            要清空哪种数据目标，可选值有 'code', 'var', 'all', 分别指清空保存的分箱代码、分箱变量、代码和变量"""

        assert target in ('code', 'var', 'all')
        if target == 'code':
            self.__bin_code = {}
        elif target == 'var':
            self.__bin_var = pd.DataFrame()
        else:
            self.__bin_code = {}
            self.__bin_var = pd.DataFrame()


class OrdiBin:
    """序数型变量的分箱，使用的是 gini 不纯度分割算法。\n

    参数:
    ----------
    max_depth: int(default=3)
        最多分割到第几层
    max_bins_num: int(default=8)
        箱的最大个数，此参数决定最多分成多少个箱。
    min_bin_samples : int(default=30)
        箱的最小样本容量。如果一个箱的样本量小于此值，会被自动合并。
    min_impurity_decrease: float(default=0)
        一个箱会被继续分割成2箱，如果分割后使基尼不纯度下降的幅度 >= 此参数值"""

    def __init__(self, max_bins_num=8, max_depth=3,
                 min_bin_samples=30, min_impurity_decrease=0):
        super().__init__()
        from sklearn.tree import DecisionTreeClassifier
        self.clf_ = DecisionTreeClassifier(max_leaf_nodes=max_bins_num,
                                           max_depth=max_depth,
                                           min_samples_leaf=min_bin_samples,
                                           min_impurity_decrease=min_impurity_decrease)
        self.__bin_code = {}  # 用于保存已经用该分箱算法实例 fit 过的分箱变量的分箱代码
        self.__bin_var = pd.DataFrame()  # 保存已经 fit 过的原始变量的分箱变量
        self.bins_ = None
        self.woe_df_ = None

    def fit(self, sr, y):
        """序数型变量的分箱，按决策树算法寻找最优分割点。\n

        参数:
        ----------
        sr : Series
            序数型预测变量。
        y : Series or ndarray
            目标变量，取值 {0, 1}

        返回值:
        ----------
        无返回值，但会更新 self 对象的如下属性：\n
            clf_: 训练好的决策树分类器 \n
            bins_: 算法找出的最佳分割点 \n
            woe_df_: 根据最佳分割点计算出来的 woe_df  \n"""

        def tree_split_point(fitted_tree):
            """解析并返回树的各个分割节点"""
            bins = sorted(set(fitted_tree.tree_.threshold))
            return [round(i, 4) for i in bins]

        na_group, var_no, y_no = na_split(sr, y)
        var_categories = list(var_no.cat.categories)
        var_code = var_no.cat.codes   # 序数型变量的整数编码
        self.clf_.fit(var_code.to_frame(), y_no)

        bins = tree_split_point(self.clf_)
        # bins 保存了整数编码的分割点, 分割区间是左闭右开的,以便与分片索引一致，取值范围[1, n-1]
        bins = [int(i)+1 for i in bins[1:]]
        bins_contain_start_end = [0] + bins + [len(var_categories)]
        bin_dict = pd.Series()   # 分箱名: 箱内离散取值列表的映射关系
        for idx in range(0, len(bins_contain_start_end)-1):  # 1 ~ len(bins)+1 个分箱
            start = bins_contain_start_end[idx]
            end = bins_contain_start_end[idx+1]
            bin_name = '{bin_no}_{bin_value}'.format(bin_no=idx+1, bin_value=var_categories[start:end])
            bin_dict[bin_name] = var_categories[start:end]
        self.bins_ = bin_dict

        binvar = self.transform(sr)  # 包含有对缺失值的处理, 生成 woe_df
        self.__bin_var[str(sr.name) + '_bin'] = binvar

        tab_ = cross_woe(binvar, y ,output=output)
        tab_['colName'] = sr.name
        self.woe_df_ = tab_

        if sr.name is None:
            print("Warning: 该数据列的 name 属性为 None")
        self.__bin_code[sr.name] = self.__generate_transform_fun(sr.name)

    def transform(self, sr):
        """根据训练好的最优分割点，把原始变量转换成分箱变量 \n

        参数:
        ------------
        sr: series, 原始变量，与训练时的变量含义相同时，做此转换才有意义  \n

        返回值:
        ------------
        binvar: ndarray, 转换成分箱变量，离散化成了有限组的一维数组"""
        sr_bin = pd.Series(index=sr.index)  #
        for bin_name in self.bins_.index:
            bin_list = self.bins_[bin_name]
            logic = sr.isin(bin_list)    # bin_list 中有缺失值时， sr的缺失值进行 isin 判断也能识别为 True
            sr_bin.loc[logic] = bin_name
        return sr_bin

    def __generate_transform_fun(self, varname):
        """生成分箱映射函数的代码. """
        fun_code = """
def {varname}_trans(x):
    # {varname} 序数型特征的分箱转换函数
    labels = {index}
    bins = {bins}
    for label, bin_list in zip(labels, bins):
        if pd.isnull(x):  # x 是缺失时，x in bin_list 并不能判断出真实结果
            if pd.isnull(label):
                return '0_nan'
            else:
                if any(pd.isnull(i) for i in bin_list):
                    return label
        else:
            if x in bin_list:
                return label""".format(varname=varname,
                                       index=list(self.bins_.index),
                                       bins=list(self.bins_))
        return fun_code

    def bin_var(self, var_names=None):
        """返回已 fit 过的变量所对应的的分箱变量

        参数:
        -----------
        var_names: str or list_of_str or index_of_str
            要访问的分箱变量名（或对应的原始变量名亦可），单个列名或多个列，列名只支持用字符串命名。
            分箱变量的命名规则是：原始变量名 + '_bin'
            默认值 None 表示返回所有已拟合过的变量的分箱变量

        返回:
        -----------
        binned_vars: series or dataframe
            对应的分箱变量，若输入是单个列名，则返回 series; 若输入是多个列名，则返回 dataframe"""
        if isinstance(var_names, str):  # 只能识别字符串格式的列名
            if not var_names.endswith('_bin'):
                var_names = var_names + '_bin'
            return self.__bin_var[var_names]
        elif var_names is None:
            return self.__bin_var  # None 表示返回所有分箱变量
        else:
            var_names = [var if var.endswith('_bin') else var + '_bin' for var in var_names]
            return self.__bin_var[var_names]

    def bin_code(self, var_name):
        """打印给定变量的分箱转换代码。

        参数:
        ------------
        var_name: str, 给定的变量名（特征名），该特征必须是经过本分箱对象 fit 过的。"""
        print(self.__bin_code.get(var_name, "变量 {} 没有被本对象 fit 过".format(var_name)))

    def clear(self, target='all'):
        """清空本对象所保存的、已 fit 过的变量的数据。这些数据大多是临时性质的，一直不清空，会占用大量内存空间。

        参数:
        ------------
        target: str,
            要清空哪种数据目标，可选值有 'code', 'var', 'all', 分别指清空保存的分箱代码、分箱变量、代码和变量"""

        assert target in ('code', 'var', 'all')
        if target == 'code':
            self.__bin_code = {}
        elif target == 'var':
            self.__bin_var = pd.DataFrame()
        else:
            self.__bin_code = {}
            self.__bin_var = pd.DataFrame()


class CateBin:
    """
    类别型变量的分箱，用卡方检验把相近的箱合并成一个箱。\n

    参数:
    ----------
    tol : float, optional(default=0.1)
        卡方独立性检验的 p 值阈值，大于等于 corr_tol 的相邻组会合并。p越小，最终分组间的
        差异越显著。可选值有5个：0.2, 0.15、0.1、0.05、0.01\n
    k : int, optional(default=100)
        样本容量的阈值，若某个类别值的样本量 < k, 则此类别归入 "other" 类。"""

    def __init__(self, tol=0.1, k=30):
        # 四个置信度对应的卡方值。卡方值小于阈值的组需要合并
        chi2_level = {0.2: 1.6424, 0.15: 2.0723, 0.1: 2.7055, 0.05: 3.8415, 0.01: 6.6349}
        self.tol = tol
        self.chi2_tol = chi2_level[tol]
        self.k = k
        super().__init__()
        self.__bin_code = {}  # 用于保存已经用该分箱算法实例 fit 过的分箱变量的分箱代码
        self.__bin_var = pd.DataFrame()  # 保存已经 fit 过的原始变量的分箱变量
        self.bins_ = None
        self.woe_df_ = None

    @staticmethod
    def cal_chi(tab_df):
        """四格列联表计算卡方值的公式"""
        a = tab_df.iloc[0, 0]
        b = tab_df.iloc[0, 1]
        c = tab_df.iloc[1, 0]
        d = tab_df.iloc[1, 1]

        # 为防直接计算，值过大溢出，更改了计算顺序，
        chi_2 = (a * d - b * c) / ((a + b) * (c + d)) * (a * d - b * c) / ((a + c) * (b + d)) * (a + b + c + d)
        return chi_2

    def fit(self, sr, y):
        """
        类别型变量的分箱。采用的是卡方合并法，2个类别的卡方独立检验值小于阈值时，合并它们。\n

        参数:
        ----------
        sr : Series or iterable
            分箱的自变量，一般是Series。
        y : Series or iterable
            分箱的目标变量，必须与 sr 等长。

        返回:
        ----------
        无返回值，但会更新 self 的如下属性：\n
        bins_: 算法找出的合并分箱的逻辑 \n
        woe_df_: 根据最佳合并分箱计算出来的 woe_df  \n"""

        def cross_var(sr, y):
            """对离散变量做交叉列联表"""
            tab_df = crosstab(sr, y)
            tab_df['badRate'] = tab_df[1] / (tab_df[0] + tab_df[1])
            tab_df['bins'] = [[i] for i in tab_df.index]  # bins用来记录分组合并的细节
            tab_df = tab_df.sort_values(by='badRate', ascending=0)
            return tab_df

        chi2_tol = self.chi2_tol
        k = self.k
        tab = cross_var(sr, y)
        tab = join_as_other(tab, k=k)
        if 'other' in tab.index:
            ind = tab.index.drop('other')
        else:
            ind = tab.index

        # 计算两两类别组合而成的四格列联表的卡方值
        # tol_df保存计算好的卡方值.
        tol_df = pd.DataFrame(index=ind, columns=ind)
        for i in range(1, len(tol_df)):
            for j in range(i):  # j = 0,1, ..., i-1
                idx_i = ind[i]
                idx_j = ind[j]
                tol_df.loc[idx_i, idx_j] = self.cal_chi(tab.loc[[idx_i, idx_j]])

        def min_chi2(tab, tol_df, loop_i):
            """递归地合并卡方值最小的2个组，直到所有组的卡方值都大于等于阈值，或除了other组外只剩2个组。
            每次合并后，都检查和修改 tol_df 的值"""
            minc = tol_df.min().min()
            if minc >= chi2_tol or len(tol_df) == 2:
                return tab
            else:
                logic = tol_df.min() == minc
                idx_j = logic.index[logic][0]
                logic = tol_df.min(axis=1) == minc
                idx_i = logic.index[logic][0]

                label = 'merge_' + str(loop_i)
                tab.loc[label, :] = tab.loc[idx_j] + tab.loc[idx_i]
                tab = tab.drop([idx_i, idx_j], axis=0)
                tol_df = tol_df.drop([idx_i, idx_j], axis=0)
                tol_df = tol_df.drop([idx_i, idx_j], axis=1)

                for idx in tol_df:
                    tol_df.loc[label, idx] = self.cal_chi(tab.loc[[label, idx]])
                tol_df[label] = np.nan

                return min_chi2(tab, tol_df, loop_i + 1)

        tab = min_chi2(tab, tol_df, 1)
        tab.index = tab['bins']
        self.bins_ = tab['bins']

        tab['colName'] = sr.name
        tab = cal_woe(tab.reindex(columns=['colName', 0, 1]), dtype='cate')
        self.woe_df_ = tab

        if sr.name is None:
            print("Warning: 该数据列的 name 属性为 None")
        self.__bin_code[sr.name] = self.__generate_transform_fun(sr.name)

        self.__bin_var[str(sr.name) + '_bin'] = self.transform(sr)

    def transform(self, sr):
        """根据训练好的分箱逻辑，把分类变量转换成分箱变量. \n

        参数:
        ------------
        sr: series, 原始变量，与训练时的变量含义相同时，做此转换才有意义  \n

        返回值:
        ------------
        binvar: series, 转换成的分箱变量"""

        def trans(x):
            for label, bin_list in zip(self.bins_.index, self.bins_):
                if pd.isnull(x):  # x 是缺失时，x in bin_list 并不能判断出真实结果
                    if pd.isnull(label):
                        return '0_nan'
                    else:
                        if any(pd.isnull(i) for i in bin_list):
                            return label
                else:
                    if x in bin_list:
                        return label

        return sr.map(trans)

    def __generate_transform_fun(self, varname):
        """生成转换函数的代码。"""
        fun_code = """
def {varname}_trans(x):
    # {varname} 分类特征的分箱转换函数
    labels = {index}
    bins = {bins}
    for label, bin_list in zip(labels, bins):
        if pd.isnull(x):  # x 是缺失时，x in bin_list 并不能判断出真实结果
            if pd.isnull(label):
                return '0_nan'
            else:
                if any(pd.isnull(i) for i in bin_list):
                    return label
        else:
            if x in bin_list:
                return label""".format(varname=varname,
                                       index=list(self.bins_.index),
                                       bins=list(self.bins_))
        return fun_code

    def bin_var(self, var_names=None):
        """返回已 fit 过的变量所对应的的分箱变量

        参数:
        -----------
        var_names: str or list_of_str or index_of_str
            要访问的分箱变量名（或对应的原始变量名亦可），单个列名或多个列，列名只支持用字符串命名。
            分箱变量的命名规则是：原始变量名 + '_bin'
            默认值 None 表示返回所有已拟合过的变量的分箱变量

        返回:
        -----------
        binned_vars: series or dataframe
            对应的分箱变量，若输入是单个列名，则返回 series; 若输入是多个列名，则返回 dataframe"""
        if isinstance(var_names, str):  # 只能识别字符串格式的列名
            if not var_names.endswith('_bin'):
                var_names = var_names + '_bin'
            return self.__bin_var[var_names]
        elif var_names is None:
            return self.__bin_var  # None 表示返回所有分箱变量
        else:
            var_names = [var if var.endswith('_bin') else var + '_bin' for var in var_names]
            return self.__bin_var[var_names]

    def bin_code(self, var_name):
        """打印给定变量的分箱转换代码。

        参数:
        ------------
        var_name: str, 给定的变量名（特征名），该特征必须是经过本分箱对象 fit 过的。"""
        print(self.__bin_code.get(var_name, "变量 {} 没有被本对象 fit 过".format(var_name)))

    def clear(self, target='all'):
        """清空本对象所保存的、已 fit 过的变量的数据。这些数据大多是临时性质的，一直不清空，会占用大量内存空间。

        参数:
        ------------
        target: str,
            要清空哪种数据目标，可选值有 'code', 'var', 'all', 分别指清空保存的分箱代码、分箱变量、代码和变量"""

        assert target in ('code', 'var', 'all')
        if target == 'code':
            self.__bin_code = {}
        elif target == 'var':
            self.__bin_var = pd.DataFrame()
        else:
            self.__bin_code = {}
            self.__bin_var = pd.DataFrame()



class WoeDf(object):
    """
    用来保存 woe_df 的类，并实现了常见的查询功能，如查变量的IV值排名、查某个变量的 woe_df 等

    初始楷参数:
    ---------------------------
    woe_df: dataframe,
        woe_df 表，格式由 cal_woe 函数返回值的形式确定。默认值 None 表示初始化时不添加任何数据
    df: 数据集
    target：目标变量
    get_bin2woe：实现变量的IV KS的计算
    """

    def __init__(self ,df = None,target = None,woe_df=None):
        columns = ['colName', 'good', 'bad', 'total', 'binPct', 'badRate', 'woe', 'IV_i', 'IV', 'KS','Gini']
        if woe_df is None:
            self.__df = pd.DataFrame(columns=['colName', 'good', 'bad', 'total', 'binPct', 'badRate', 'woe', 'IV_i', 'IV', 'KS', 'Gini'])
        elif isinstance(woe_df, pd.DataFrame) and set(woe_df.columns) == set(columns):
            self.__df = woe_df
        else:
            raise Exception("只支持用 woe_df 表来初始化本对象")
        if self.df is not None:
            self.df = df.reset_index(drop=True).copy()
            self.target = target
            if len(self.df.columns.to_list()) > len(set(self.df.columns.to_list())):
                print("数据表变量存在重复，请及时处理！")        
        
    def run_bin2woe(self ,cols = None ,drop_na = None ,iv = 0):
        """
        该函数可以直接输出KS、IV等指标
        参数：
        ------------------------------------
        cols：选择需要计算输出的变量名称 可以是str 或者 list ,None 表示全部输出
        drop_na:需要删除的空值的列 ，可以是str 或者 list,all表示全部删除
        iv: 保留IV 大于多少的结果
        """
        self.df.get_float()
        self.df = self.df.replace('',np.NaN)
        cols_m = cols
        if isinstance(cols,str) or cols is None: cols = self.df.columns.to_list()
        if isinstance(drop_na,str) and drop_na != 'all': drop_na = [drop_na]
        if isinstance(drop_na,str) and drop_na == 'all': drop_na = self.df.columns.to_list()
        data_report = cols_report(self.df[cols])
        data_report.reset_index(inplace=True)
        cols_str = list()
        cols_int = list()
        cols_str = data_report.loc[(data_report['unique'] >= 2)&(data_report['unique'] <= 10)&(data_report['infered_type'] != 'dtime')]['index'].to_list()
        cols_int = data_report.loc[(data_report['infered_type'] == 'num')&(data_report['unique'] > 5)]['index'].to_list()
        for i in cols_int:
            if i in cols_str:cols_str.remove(i)
        cols_str.remove(self.target)
        clf = NumBin()
        for i in cols_int:
            if isinstance(drop_na,list) and 'all' in drop_na:
                clf.fit(self.df.loc[pd.notnull(self.df[i])][i] ,self.df.loc[pd.notnull(self.df[i])][self.target])
            elif isinstance(drop_na,list) and 'all' not in drop_na:
                if i in drop_na:
                    clf.fit(self.df.loc[pd.notnull(self.df[i])][i] ,self.df.loc[pd.notnull(self.df[i])][self.target])
                else:
                    clf.fit(self.df[i] ,self.df[self.target])
            else:
                clf.fit(self.df[i] ,self.df[self.target])
            if clf.woe_df_.IV.iloc[0] >= iv:
                self.append(clf.woe_df_)
        for i in cols_str:
            if cross_woe(self.df[i] ,self.df[self.target]).IV.iloc[0] >= iv:
            
                if isinstance(drop_na,list) and 'all' in drop_na:
                    dfi = cross_woe(self.df.loc[pd.notnull(self.df[i])][i] ,self.df.loc[pd.notnull(self.df[i])][self.target])
                    self.append(dfi)
                elif isinstance(drop_na,list) and 'all' not in drop_na:
                    if i in drop_na:
                        dfi = cross_woe(self.df.loc[pd.notnull(self.df[i])][i] ,self.df.loc[pd.notnull(self.df[i])][self.target])
                        self.append(dfi)
                    else:
                        dfi = cross_woe(self.df[i] ,self.df[self.target])
                        self.append(dfi)
                else:
                    dfi = cross_woe(self.df[i] ,self.df[self.target])
                    self.append(dfi)
                
        if isinstance(cols_m,str): 
            return self.get_woe_df(var_names=cols)  
        else:
            return self.get_woe_df()  

    def append(self, woe_df ,inplace = True):
        """增加一个变量的 woe_df 到本对象中. 要添加多个变量的 woe_df,请用 extend

        参数:
        -----------
        woe_df: dataframe, woe 表，格式由 cal_woe 函数返回值的形式确定。"""
        if inplace:
            woe_df['colName'] = woe_df['colName'].apply(lambda x:x+'_bin' if x.find('_bin') == -1 else x)
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
                self.__df = pd.concat([self.__df,woe_df])
        
    def extend(self, woe_dfs):
        """追加多个变量的 woe_df 到本对象中

        参数:
        ----------
        woe_dfs: 多个 woe_df 表拼在一起组成的 DataFrame"""
        woe_dfs['colName'] = woe_dfs['colName'].apply(lambda x:x+'_bin' if x.find('_bin') == -1 else x)
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
        self.__df = pd.concat([self.__df ,df])
        
    def del_var(self, var_names):
        """删除指定变量的 woe_df

        参数:
        ------
        var_names : str or list_of_str, 需要删除的变量名/变量列表

        返回:
        -------
        无返回值，会修改对象自身的数据"""
        if isinstance(var_names, str):
            logic = self.__df.colName != var_names
        else:
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

    def get_woe_df(self, var_names=None, patten=None ,reset_index = True):
        """查询指定变量的 woe_df

        参数:
        -------
        patten: str,
            用正则表达式描述的变量名模式。若传入此参数，则优先返回所查到的所有符合该模式的所有变量的 woe_df
        var_names : str or list_of_str,
            需要查询的变量名/变量列表. 默认值 None 表示返回所有变量的 woe_df

        返回:
        -------
        woe_df: dataframe, 指定变量的 woe_df"""
        
        woe_df = self.__df.copy()
        if patten is not None:
            var_names = re_search(patten, self.all_vars())
        if isinstance(var_names, str):
            logic = self.__df.colName == var_names
            woe_df = self.__df[logic]
        
        if reset_index == True and 'colBins' not in woe_df.columns.to_list():
            return woe_df.reset_index().rename_col({'index':'colBins'})
        else:
            return woe_df

    def head(self, n=5):
        """查看头部 n 个变量的 woe_df"""
        col_names = self.__df.colName.unique()
        head_n = col_names[:n]
        logic = self.__df.colName.isin(head_n)
        return self.__df[logic]

    def tail(self, n=5):
        """查看尾部 n 个变量的 woe_df"""
        col_names = self.__df.colName.unique()
        tail_n = col_names[-n:]
        logic = self.__df.colName.isin(tail_n)
        woe_df = self.__df[logic]
        if 'colBins' not in woe_df.columns.to_list():
            woe_df = woe_df.reset_index().rename_col({'index':'colBins'})
        return woe_df

    def to_excel(self, file_name, var_names=None):
        """把 woe_df 对象的数据，保存为 excel 表文件

        参数:
        ----------
        file_path: str,
            excel文件的绝对路径， 不需要带扩展名
        var_names: str or list_of_str, 可选
            只保存指定变量的 woe_df 数据到 excel 中。None 表示保存所有变量的 woe_df 数据"""
        if 'colBins' not in self.get_woe_df(var_names).columns.to_list():
            woe_df = self.get_woe_df(var_names).reset_index().rename_col({'index':'colBins'})
        with pd.ExcelWriter(r'//data/result_data/excel/'+ file_name + '.xlsx') as writer:  # doctest: +SKIP
            woe_df.to_excel(writer)
            print('write to {}, done!'.format(file_name + '.xlsx'))

    def var_ivs(self, var_names=None, ascending=None, by='iv'):
        """返回所有记录变量的 IV 值 \n

        参数:
        -------
        by: str,
            按哪一指标进行排序，可选值有 'iv', 'gini'
        ascending: bool,
            默认值 None 表示不排序, True 表示升序，False表示降序
        var_names: str or list_of_str or index_of_str,
            返回哪些变量的 IV，默认值 None 表示返回所有变量的 IV

        返回:
        -------
        iv_df: dataframe,
            所有变量的 IV 值，按指定方式排序"""
        assert by.lower() in ('iv', 'gini')
        df = self.get_woe_df(var_names=var_names)
        ivs = df[['colName', 'IV', 'Gini']].drop_duplicates().set_index('colName')
        col = 'IV' if by.lower() == 'iv' else 'Gini'

        if ascending is None:
            return ivs
        else:
            return ivs.sort_values(by=col, ascending=ascending)
        
    def woe_dict(self, var_names=None):
        """woe_dict 字典包含了把分箱变量映射成woe变量的所有信息，用 DataFrame 表示. \n

        参数:
        ----------
        var_names: str or list_of_str,
            单个或多个分箱变量。\n

        返回值:
        ----------
        woe_map: DataFrame, colName 列是分箱变量名，woe 列是 woe 值， index是分箱值，分箱值可以是 np.nan ."""
        woe_map = self.get_woe_df(var_names)[['colName', 'woe']]
        return woe_map

    def bin2woe(self, detail_df, var_names=None, inplace=True):
        """把 detail_sr 表中的分箱变量，映射成 woe 变量。

        参数:
        -----------
        detail_sr: 明细数据表，一般是训练样本或测试样本数据, 并包含有所有的分箱变量。
            本模块要求，所有的分箱变量的变量名，均以 _bin 结尾
        var_names: 要转换的分箱变量列表
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
            woe_sr_i = detail_df[var].map(lambda bin_value: map_i.get(bin_value, -9999))            
            woe_sr_i.name = var[:-4] + '_woe'   # woe变量统一以 '_woe' 结尾
            woe_series.append(woe_sr_i)

        if inplace:
            for sr in woe_series:
                detail_df[sr.name] = sr
        else:
            return pd.concat(woe_series, axis=1)

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
    def var2bin(self ,detail_df ,target ,inplace=True):
        '''
        将使用到的Var变量转化为bin变量
        detail_df:  数据表
        target:  y变量
        '''
        mm = set()
        bin_data = self.get_woe_df().reset_index()
        bin_data = bin_data.rename_col({'index':'col'})
        for i in range(len(bin_data)):
            if isinstance(bin_data.col.iloc[i],str):
                if  bin_data.col.iloc[i][-1]== ']':
                    mm.add(bin_data.colName.iloc[i])
        mm = list(mm)
        nn = list(set(bin_data.colName.to_list()))
        for i in mm:
            nn.remove(i)
        if inplace:
            for i in nn:
                detail_df[i] = detail_df[i[:-4]]
            clf = NumBin() 
            for i in mm:
                clf.fit(detail_df[i[:-4]] ,detail_df[target])
                detail_df[i] = clf.bin_var()[i]
        if inplace == False:
            df = pd.DataFrame()
            for i in nn:
                df[i] = detail_df[i[:-4]]
            clf = NumBin() 
            for i in mm:
                clf.fit(detail_df[i[:-4]] ,detail_df[target])
                df[i] = clf.bin_var()[i] 
            return pd.merge(detail_df,df,how='left',left_index=True ,right_index=True)

def iv_filter(woe_df, k_iv=0.02, k_gini=0.04):
    """ 把 IV、Gini 均小于阈值的变量剔除，返回大于阈值的变量  \n

    参数:
    ----------
    woe_df: WoeDf, 包含多个分箱变量的 woe_df \n
    k_iv: float, 变量IV的阈值 \n
    k_gini: float, 变量Gini的阈值 \n

    返回值:
    ----------
    var_filtered: dataframe, 剔除了IV和Gini均小于阈值的变量，包含colName, IV, Gini 三列"""
    rank = woe_df.get_ivs(woe_df)
    logic = (rank.IV >= k_iv) & (k_gini >= k_gini)
    return rank[logic].colName.tolist()


def corr_filter(detail_df, vars_iv, corr_tol=0.9, iv_diff=0.01):
    """相关性系数 >= tol的两个列, 假设 var1 的 IV 比 var2 的 IV 更高:
        若 var1_iv - var2_iv > iv_diff，则将其中 IV 值更低的列删除 \n

    参数:
    ----------
    detail_sr: dataframe, 需要计算相关性的明细数据框 \n
    vars_iv: dict or series, 各个变量的 IV 指标 \n
    corr_tol: float, 线性相关性阈值，超过此阈值的两个列，则判断两个列相关性过高，进而判断 IV 之差是否足够大 \n
    iv_diff: float, 两个列的 IV 差值的阈值，自动删除 IV 更低的列

    返回值:
    ----------
    corr_df: dataframe, 相关性矩阵，并删除了相关性超过阈值的列 \n
    dropped_col: list, 删除的列"""
    cols = vars_iv.colName
    corr_df = detail_df.corr_tri()
    corr_df = corr_df.abs()
    dropped_col = []
    while True:
        row, col = corr_df.argmax()
        if corr_df.loc[row, col] >= corr_tol:
            drop_label = row if vars_iv[row] < vars_iv[col] else col
            dropped_col.append(drop_label)
            corr_df = corr_df.drop(drop_label).drop(drop_label, axis=1)
            if len(corr_df) == 1:
                break
        else:
            break
    return corr_df, dropped_col


def bin2score(scorecard_df, detail_df, model_version='', postfix='scr'):
    """把各变量的分箱映射成得分值。  \n

    参数:
    -------------
    scorecard_df: 标准的评分卡，通过 WoeDf.generate_scorecard 方法得到 \n
    detail_sr: dataframe, 明细数据框，应包含有所有分箱变量 \n
    version: str, 评分卡的版本。一般一张评分卡会不定期迭代，每次迭代应有一个版本号 \n
    postfix: str, 此函数会修改 detail_sr, 为其添加上转换的 打分变量。此参数含义是 \n
        打分变量名 = 分箱变量名[:-3] + '_' + postfix，即为原变量名添加后缀 \n

    返回:
    -------------
    无返回值，会原地修改 detail_sr，增加各变量得分列和总分列。  \n
    """
    const = scorecard_df.at[0, 'score']

    logic = scorecard_df.colName != 'const'
    score_map = scorecard_df[logic]
    vars = score_map.colName.unique()
    scored_varnames = []
    for var in vars:
        map_i = score_map.loc[score_map.colName == var, 'score']  # map_i 是个 series, 分箱值不可以是 nan
        col_score = var[:-3] + postfix
        score_fun = lambda bin_value: map_i.get(bin_value, -9999)  # 得分若出现-9999，暗示分箱变量出现错误
        detail_df[col_score] = detail_df[var].apply(score_fun)
        scored_varnames.append(col_score)

    total_score = 'score_' + model_version
    detail_df[total_score] = detail_df[scored_varnames].sum(axis=1) + const
    detail_df[total_score] = detail_df[total_score].apply(lambda x: int(round(x, 0)))  # 总分四舍五入取整

    # 检查有无打分异常的变量
    err = detail_df[scored_varnames].min()
    logic = err == -9999
    if logic.any():
        print("以下变量的打分有异常值:")
        print(err[logic].index.tolist())


def psi(pct_base, pct_actual):
    """计算变量的稳定性指标 psi。\n

    参数:
    -------------
    pct_base : series, 是基准分布。 \n
    pct_actual : series or dataframe, 是要比较的分布。如果是df，则每一列应该是一个分布。\n
        pct_actual.index 应与 pct_base.index 对齐  \n

    返回值:
    -------------
    psi_df: dataframe，带psi结尾的列是各列相对于基准的psi"""
    psi_df = pd.concat([pct_base, pct_actual], axis=1)
    psi_df = psi_df / psi_df.sum()
    base = psi_df.iloc[:, 0]
    col_compare = psi_df.columns[1:]
    for col in col_compare:
        psi_i = (psi_df[col] - base) * np.log(psi_df[col] / base)
        psi_df[str(col) + '_psi'] = psi_i.sum()
    return psi_df


def csi(pct_base, pct_actual):
    """计算变量的分数迁移性指标。\n

    参数:
    -------------
    pct_base : dataframe, 包含2列：binPct列是基准分布，score列是基准得分。   \n
    pct_actual : sr or df, 是要比较的分布。如果是df，则每一列应该是一个分布。
        pct_actual.index 应与 pct_base.index 对齐  \n

    返回值:
    -------------
    csi_df: dataframe，带vsi结尾的列是各列的分数迁移"""
    csi_df = pd.concat([pct_base, pct_actual], axis=1)
    base_pct = csi_df['binPct']
    col_compare = csi_df.columns[2:]
    csi_df[col_compare] = csi_df[col_compare] / csi_df[col_compare].sum()
    for col in col_compare:
        shifti = (csi_df[col] - base_pct) * csi_df['sr_numeric']
        csi_df[str(col) + '_vsi'] = shifti.sum()
    return csi_df


def woe_trend(detail_df, bin_cols, by, y="fpd10"):
    """在时间维度上看变量的woe变化趋势，即每个周期内都计算一次分箱变量的woe值，按周期看woe值的波动情况 \n

    参数:
    ---------
    detail_sr: 明细数据表，至少应包含分箱变量、时间变量、目标变量  \n
    bin_cols: iterable, 分箱变量  \n
    by: str, 时间维度的变量名，detail_df将按此变量的值做groupby，每一组内计算一次各变量的woe值。  \n
    y: str, 目标变量的名字

    返回值:
    ---------
    woe_period: dataframe, columns是时间的各个周期，row是各个分箱变量的分箱值，values是各个箱在各个周期内的woe值
    """
    group = detail_df.groupby(by)
    badRate = group[y].mean()  # 各周期内的全局 badrate, 是各周期内计算woe值的基准
    df_woe = []
    for var in bin_cols:
        var_badrate = group.apply(lambda x: x.pivot_table('fpd10', var, aggfunc='mean')).transpose()
        for j in var_badrate:
            badrate_base = badRate[j]
            var_badrate[str(j) + '_woe'] = var_badrate[j].apply(bad2woe, args=(badrate_base,))
        var_badrate['colName'] = var
        df_woe.append(var_badrate)
    return pd.concat(df_woe)


def matrix_rule(matrix_df, row_name, col_name):
    """矩阵细分：分别以2个变量的取值为行、列索引，在 matrix_df 矩阵中查找值，此值便是矩阵细分的返回值。 \n

    参数:
    ---------
    matrix_df: dataframe, 交叉规则矩阵。index，column代表输入变量的取值条件，value代表此条件下对应的输出。 \n
        此矩阵表达了矩阵细分的逻辑. 
        缺失值在 index, column 中用 'nan' 表示
    row_name: str, 行方向的输入变量的名字，列名 \n
    col_name: str, 列方向的输入变量的名字，cols 参数是输入条件之二 \n
    
    返回值:
    ---------
    apply_fun: 函数，以 row 为输入，以 matrix_df的values中的值为返回值。\n
        函数用法：df.apply(apply_fun, axis=1), 即以 df 的各行作为 apply_fun 的输入参数。 \n
    
    示例:
    ----------
    matrix_df = pd.DataFrame(data=[[1,2],[3,4]], index=['Male', 'Female'], columns=['Married', 'UnMarried']) \n
    detail_sr = pd.DataFrame({'gender':['Male', 'Female','Male', 'Female','Male', 'Female','Male', 'Female'],
                            'Marry': ['Married','Married','Married','Married','UnMarried','UnMarried','UnMarried','UnMarried']}) \n
    detail_sr.apply(matrix_rule(matrix_df, 'gender', 'Marry'), axis=1)  \n
    以上代码的返回值如下： \n
    pd.Series([1,3,1,3,2,4,2,4])
    """

    def apply_fun(row):
        index_value = row[row_name]
        if pd.isnull(index_value): index_value = 'nan'

        column_value = row[col_name]
        if pd.isnull(column_value): column_value = 'nan'

        return matrix_df.loc[index_value, column_value]

    return apply_fun


def bad2woe(badrate_i, badrate_base):
    """把 badrate 转换成 woe 值。\n

    参数:
    ---------
    badrate_i: float or 1darray or series, 各个箱的 badrate  \n
    badrate_base: float, 全局的基准 badrate  \n

    返回值:
    ----------
    woe_i: 类型同 badrate_i, 各个箱对应的 woe 值。"""
    return np.log(1 / badrate_i - 1) - np.log(1 / badrate_base - 1)


def bad2woe_base(badrate_base):
    """生成带全局基准 badrate 的转换函数，此函数能把 badrate 转换成 woe 值。\n

    参数:
    ---------
    badrate_base: float, 全局的基准 badrate  \n

    返回值:
    ----------
    bad2woe_fun: 函数，以 badrate_i（各个箱对应的badrate为输入值），返回各箱对应的 woe 值。"""
    return lambda bad_i: np.log(1 / bad_i - 1) - np.log(1 / badrate_base - 1)


def woe2bad(woe_i, badrate_base):
    """bad2woe 的逆函数，把 woe 值转换成 badrate 值。 \n

    参数:
    ---------
    woe_i: float or 1darray or series, 各个箱对应的 woe 值  \n
    badrate_base: float, 全局的基准 badrate  \n

    返回值:
    ----------
    badrate_i: 类型同 woe_i, 各个箱的 badrate。
    """
    return 1 / (1 + ((1 - badrate_base) / badrate_base) * np.exp(woe_i))


def corr_heatmap(detail_df, img_path=None, figsize=(16, 14), font_size=12):
    """计算数值型 df 的各列的相关性矩阵，并画 heatmap 图

    参数：
    -------------
    detail_sr: dataframe, 表的所有列，都应是数值型的，否则无法计算相关性，或计算结果无意义
    img_path: str, 保存图像文件的路径和文件名，保存成png格式，扩展名不用输入。默认 None 表示不存盘。
    font_size: 热力图的标注字体大小
    figsize: int, 绘图的尺寸大小

    返回值：
    -------------
    corr_matrix: dataframe, 相关性矩阵
    """
    from seaborn import heatmap

    corr_matrix = detail_df.corr_tri()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax = heatmap(corr_matrix, cmap='RdBu', annot=True, center=0, fmt='.2f',
                 annot_kws={'size': font_size}, ax=ax)

    if img_path is not None:
        fig.savefig(img_path + '.png', format='png')

    return corr_matrix


def na_split(var, y):
    """把目标变量缺失值对应的数据丢弃、特征变量的缺失值对应的数据单独分离出来作为一组 \n

    参数:
    ---------
    var: series, 特征变量 \n
    y: series, 目标变量 \n

    返回值:
    ---------
    na_group: 如果var没有缺失值，np_group返回值为0；否则返回一dataframe  \n
    var_notNa: 分离缺失值后的特征变量  \n
    y_notNa: 分离缺失值后的目标变量   \n"""
    logic = pd.notnull(y)
    var = var[logic]
    y = y[logic]
    logic = pd.isnull(var)
    if logic.any():
        na_grp = pd.DataFrame(columns=[0, 1, 'All', 'badRate', 'binPct'], index=[np.nan])
        var_no = var[~logic]
        y_na, y_no = y[logic], y[~logic]
        na_grp[1] = y_na.sum()
        na_grp['All'] = len(y_na)
        na_grp[0] = na_grp['All'] - na_grp[1]
        na_grp['badRate'] = na_grp[1] / na_grp['All']
        na_grp['binPct'] = na_grp['All'] / len(y)
        return na_grp, var_no, y_no
    return 0, var, y


def na_split_y(var, y):
    """把目标变量缺失值对应的数据丢弃。很多场景下都要剥离出不可观测的数据来。  \n

    参数:
    ----------
    var: series or dataframe, 特征变量  \n
    y: series, 与 var 观测数相等，目标变量.   \n

    返回值:
    ----------
    var_notna: 类型同 var，去掉了目标变量缺失值对应的观测   \n
    y: series, 去掉了缺失值"""
    logic = pd.notnull(y)
    var_notna = var[logic]
    y_notna = y[logic]
    return var_notna, y_notna


def roc(score, y, detail=False, sample_weight=None):
    """计算gini系数，可加权重。\n

    参数:
    ---------
    sr_numeric: series or ndarray, 模型的得分、概率或决策函数值，值越低表示越差。  \n
    y: series or ndarray, 模型的目标变量，取值 {0, 1}  \n
    detail: bool, 是否返回ROC曲线数据。当 detail 为 False 时只返回 gini, ks 系数;   \n
        当 detail 为 True 时返回 gini 系数和用于绘制 ROC 曲线的 fpr, tpr 数组  \n
    sample_weight: series or ndarray, 与 sr_numeric 长度相等。各个观测的样本权重  \n

    返回值:
    ----------
    gini: float, 模型的基尼系数  \n
    ks: float, 模型的KS系数   \n
    fpr: 假阳率，ROC 曲线的 x 轴数据。仅当 detail 参数为 True 时返回。  \n
    tpr: 真阳率，ROC 曲线的 y 轴数据。仅当 detail 参数为 True 时返回。"""
    score, y = na_split_y(score, y)
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    df = pd.DataFrame({'sr_numeric': score, 'bad': y, 'weight': sample_weight})  # .sort_values('sr_numeric')
    df['good_w'] = (1 - df['bad']) * df['weight']
    df['bad_w'] = df['bad'] * df['weight']
    All = np.array(df.groupby('bad')['weight'].sum())
    df_gini = df.groupby('sr_numeric')[['good_w', 'bad_w']].sum().cumsum() / All

    score_min = 2 * df_gini.index[0] - df_gini.index[1]  # 比实际的最小值略小，score_min = min0 - (min1 - min0)
    df_0 = pd.DataFrame([[0, 0]], columns=['good_w', 'bad_w'], index=[score_min])
    df_gini = pd.concat([df_0, df_gini])

    A = auc(df_gini.good_w, df_gini.bad_w)
    Gini = (A - 0.5) / 0.5

    diff = df_gini['bad_w'] - df_gini['good_w']
    KS = abs(diff.max())

    if detail:
        return Gini, KS, df_gini.good_w, df_gini.bad_w
    return Gini, KS


def gini_ks_groupby(score, y, by):
    """按指定维度分组，计算每组中的 Gini, KS 指标  \n

    参数:
    ---------
    sr_numeric : series or ndarray, 模型得分、概率或决策函数值，值越低表示越差。  \n
    y : 目标变量, 0 为 good, 1 为 bad    \n
    by : series, ndarray or list_of_them . 分组的维度变量。   \n

    返回值:
    ----------
    df_metrics: dataframe, 每组的 Gini, KS 值。"""
    df_score = pd.DataFrame({'sr_numeric': score, 'y': y})

    def gini_ks(df_prob):
        d = pd.Series()
        g, k = roc(df_prob['sr_numeric'], df_prob['y'])
        d['Gini'] = g
        d['KS'] = k
        return d

    df_metrics = df_score.groupby(by).apply(gini_ks)
    return df_metrics


class ModelEval:
    """模型评估对象。用于从不同角度评估模型效果。\n
    同一个 sr_numeric 和 y 的数据，可以绘制Gini图、KS图、Lift图、bins分组等。\n
    def __init__(self, sr_numeric, y, score_name=None, plot=True)\n

    初始化参数:
    ------------
        sr_numeric : 模型得分或概率，越高越好。 \n
        y : 模型的真实label，0表示good，1表示bad. \n
        score_name: 模型的名字，默认值 None 表示以 sr_numeric.name 为名  \n
        plot : 是否在实例化时自动绘制 GiniKS 曲线图。\n

    属性说明：
    ----------
        所有属性均以'_'结尾，所有方法则以小写字母开头   \n
        score_: 模型的得分、概率或决策函数值，剔除了目标变量为缺失值的行  \n
        y_: 模型的真实目标变量，剔除了其中的缺失值   \n
        na_group_: 如果 sr_numeric 中有缺失值，缺失值会单独分离成 na_group_, 以免影响 Gini, KS 值的计算结果   \n
        gini_: 模型的 gini 指数  \n
        ks_: 模型的 KS 值   \n
        ks_score_: 模型取得 KS 值时的 sr_numeric 值。   \n
        good_curve_: 假阳率。信贷领域中，“假阳”就是 good，因此“假阳率”就是 good 累计占比。   \n
        bad_curve_: 真阳率。信贷领域中，“真阳”就是bad，因此“真阳率”就是 bad 累计占比。"""

    def __init__(self, score, y, score_name=None, plot=True):
        if score_name is not None:
            self.score_name_ = score_name
        else:
            try:
                self.score_name_ = score.name
            except AttributeError:
                self.score_name_ = 'train'
        self.na_group_, self.score_, self.y_ = na_split(score, y)  # 把得分缺失的样本剔除
        self.gini_, self.ks_, self.good_curve_, self.bad_curve_ = roc(self.score_, self.y_, detail=True)
        self.ks_score_ = (self.bad_curve_ - self.good_curve_).abs().idxmax()
        if plot:
            self.giniks_plot()

    def gini_plot(self, img_path=None):
        """绘制 ROC 曲线图, 并计算 Gini 系数。无返回值  \n

        参数:
        ------------
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        g, good, bad = self.gini_, self.good_curve_, self.bad_curve_
        plt.figure(figsize=(7, 7))
        plt.plot(good, bad, [0, 1], [0, 1], 'r')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('Good')
        plt.ylabel('Bad')
        plt.title('Gini = %.4f (%s)' % (g, self.score_name_))
        plt.show()
        if img_path is not None:
            plt.savefig(join(img_path, self.score_name_) + '_Gini.png', format='png')

    def ks_plot(self, img_path=None):
        """绘制 KS 曲线图，并计算 KS 系数。无返回值   \n

        参数:
        ------------
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        ks, good, bad = self.ks_, self.good_curve_, self.bad_curve_
        good_score = good[self.ks_score_]
        bad_score = bad[self.ks_score_]
        plt.figure(figsize=(7, 7))
        plt.plot(good.index, good, 'g', bad.index, bad, 'r')
        plt.plot([self.ks_score_, self.ks_score_], [good_score, bad_score], 'b')
        plt.xlabel('Score')
        plt.ylabel('CumPct')
        plt.title('KS = %.4f (%s)' % (ks, self.score_name_))
        plt.legend(['Good', 'Bad'], loc='lower right')
        plt.ylim([0, 1])
        plt.show()
        if img_path is not None:
            plt.savefig(join(img_path, self.score_name_) + '_KS.png', format='png')

    def giniks_plot(self, img_path=None):
        """绘制 GiniKS 曲线图。它有两个子图，一个子图显示 ROC 曲线，一个子图显示 KS 曲线。  \n

        参数:
        ------------
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        f = plt.figure(figsize=(12, 5.5))

        # Gini 图
        ax1 = f.add_subplot(1, 2, 1)
        ax1.plot(self.good_curve_, self.bad_curve_, [0, 1], [0, 1], 'r')
        ax1.set_xlabel('Good')
        ax1.set_ylabel('Bad')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Gini: {0:.4f} ({1})'.format(self.gini_, self.score_name_))

        # KS 图
        ax2 = f.add_subplot(1, 2, 2)
        ks, good, bad = self.ks_, self.good_curve_, self.bad_curve_
        good_score = good[self.ks_score_]
        bad_score = bad[self.ks_score_]
        ax2.plot(good.index, good, 'g', bad.index, bad, 'r')
        ax2.plot([self.ks_score_, self.ks_score_], [good_score, bad_score], 'b')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('CumPst')
        ax2.set_title('KS = %.4f (%s)' % (ks, self.score_name_))
        ax2.set_ylim([0, 1])
        ax2.legend(['Good', 'Bad'], loc='lower right')

        f.show()
        if img_path is not None:
            plt.savefig(join(img_path, self.score_name_) + '_GiniKS.png', format='png')

    def lift_plot(self, bins=10, gain=False, img_path=None):
        """绘制模型的提升图/增益图, 图中点的 x 坐标值是各 sr_numeric 各区间的右端点。 \n
        提升图：随着分数cutoff 点的增大，预测为 bad 的样本中的 badrate 相对于基准 badrate 的提升倍数 \n
        增益图：随着分数cutoff 点的增大，预测为 bad 的样本中的 badrate。增益图为提升图的绝对值版本，只是纵轴不同。

        参数:
        ------------
        bins : int, 把分数分为多少个等深区间，默认10个。\n
        gain : bool, 默认为False, 绘制提升图。如果设为True，将绘制增益图。\n
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        score, y, name = self.score_, self.y_, self.score_name_
        df = pd.DataFrame({'sr_numeric': score, 'y': y})
        badRate = y.mean()
        group = pd.qcut(score, bins)
        bad = df.groupby(group).y.agg(['sum', 'count'])  # 此处 bad 会自动按 sr_numeric 升序排列
        bad = bad.cumsum()  # lift 是累积正确率，不是各分组内的正确率
        if not gain:
            chart_name = 'Lift'
            bad['lift'] = bad['sum'] / bad['count'] / badRate  # 相对于基准的提升倍数
        else:
            chart_name = 'Gain'
            bad['lift'] = bad['sum'] / bad['count']  # 绝对提升值，即预测为 bad 的样本中的badrate

        # 传统提升图的 x 轴是 depth = 预测为bad的个数/总样本观测数，对于等深区间就是[1/bins, 2/bins, ...]，很好推算
        # 此处用 sr_numeric 各个分组区间的右端点作为 x 轴，能在 lift 图中展示各点对应的score值。
        barL = [eval(i.replace('(', '[')) for i in bad.index]
        x = [round(i[1], 4) for i in barL]

        plt.figure(figsize=(7, 7))
        plt.plot(x, bad.lift, 'b.-', ms=10)
        plt.xlabel('sr_numeric')
        plt.ylabel('{1} (base: {0:.2f}%)'.format(100 * badRate, chart_name))
        plt.title('{1} ({0})'.format(name, chart_name))
        if img_path is not None:
            plt.savefig(join(img_path, name) + '_{}.png'.format(chart_name), format='png')

    def divg_plot(self, bins=30, img_path=None):
        """绘制分数的散度图，bad, good分组的直方图均做了归一化。\n

        参数:
        ----------
        bins : int, 把分数分为多少个等宽区间。\n
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        score, y, name = self.score_, self.y_, self.score_name_
        df = pd.DataFrame(data={'sr_numeric': score, 'y': y})
        score0 = df.score[df.y == 0]
        score1 = df.score[df.y == 1]
        u0 = score0.mean()
        v0 = score0.var()
        u1 = score1.mean()
        v1 = score1.var()
        div = 2 * (u0 - u1) ** 2 / (v0 + v1)
        plt.figure(figsize=(7, 7), facecolor='w')
        plt.hist(score0, bins=bins, alpha=0.3, color='g', normed=True, label='Good')
        plt.hist(score1, bins=bins, alpha=0.3, color='r', normed=True, label='Bad')
        plt.legend(loc='upper left')
        plt.xlabel('Score')
        plt.ylabel('Freq')
        plt.title('Divergence = %.4f (%s)' % (div, name))
        plt.show()
        if img_path is not None:
            plt.savefig(join(img_path, name) + '_divg.png', format='png')

    def cutoff(self, q=None, step=10):
        """把评分等分为若干bins，并计算 badRate 和 recall 表格.\n

        参数:
        ----------
        q : int, 等深分组的组数. \n
        step : 等宽分组的步长，当 q 为 None 时才使用 step 参数  \n

        返回值:
        ----------
        cutoff_df: dataframe, 计算了以各个点作为cutoff点时，通过的bad，good占比、拒绝的bad, good占比。
        """
        score, y = self.score_, self.y_
        if q is not None:
            bins = pd.qcut(score, q, duplicates='drop')
        else:
            if step == 0:
                raise Exception("step 步长不能为0")
            bins = score // step * step
        df = pd.crosstab(bins, y).sort_index()
        df['colName'] = self.score_name_
        df = df.reindex(columns=['colName', 0, 1])
        df['All'] = df[0] + df[1]
        df['binPct'] = df['All'] / df['All'].sum()
        df['badRate'] = df[1] / df['All']
        reject_df = df[[0, 1, 'All']].cumsum() / df[[0, 1, 'All']].sum()
        reject_df['rejBadRate'] = df[1].cumsum() / df['All'].cumsum()
        reject_df = reject_df.shift().fillna(0)
        approve_df = 1 - reject_df[[0, 1, 'All']]
        approve_badrate = {}
        for i in df.index:
            approve_badrate[i] = df[1][i:].sum() / df['All'][i:].sum()
        approve_df['passBadRate'] = pd.Series(approve_badrate)
        reject_df = reject_df.rename(columns={0: 'rejGood', 1: 'rejBad', 'All': 'rejAll'})
        approve_df = approve_df.rename(columns={0: 'passGood', 1: 'passBad', 'All': 'passAll'})
        df = pd.concat([df, approve_df, reject_df], axis=1)
        return df

    def compare_to(self, *thats, img_path=None):
        """与其他 ModelEval 对象比较,把它们的 ROC 曲线画在同一个图上。\n

        参数:
        ----------
        thats : 1~5 个 ModelEval 对象。\n
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片\n"""
        that, *args = thats
        if img_path is not None:
            img_path = join(img_path, self.score_name_) + '_'
        gini_compare(self, that, *args, img_path=img_path)

    def __setattr__(self, name, value):
        if name in self.__dict__:  # 一旦赋值，便不可修改、删除属性
            raise AttributeError("can't set readonly attribute {}".format(name))
        self.__dict__[name] = value

    def __delattr__(self, name):
        raise AttributeError("can't del readonly attribute {}".format(name))

    def __str__(self):
        return "ModelEval {0}: Gini={1:.4f}, KS={2:.4f}\n".format(self.score_name_, self.gini_, self.ks_)

    def __repr__(self):
        return self.__str__()


def gini_compare(model_eval1, model_eval2, *args, img_path=None):
    """对比不同的ModelEval对象的性能，把它们的 ROC 曲线画在同一张图上。注意：最多可同时比较 6 个对象。 \n

    参数:
    ----------
    model_eval1: ModelEval 对象。  \n
    model_eval2: ModelEval 对象。  \n
    args: 0 ~ 4 个 ModelEval 对象。  \n
    img_path : 若提供路径名，将会在此路径下，以gini_compare为名保存图片\n"""
    plt.figure(figsize=(7, 7))
    plt.axis([0, 1, 0, 1])
    plt.title('Gini Compare')
    plt.xlabel('Good')
    plt.ylabel('Bad')
    score_list = [model_eval1, model_eval2] + list(args)
    k = len(score_list)
    if k > 6:  # 最多只支持6个模型的比较
        k = 6
        score_list = score_list[:6]
        print("Waring: 最多只支持6个模型的比较,其余模型未比较")
    color = ['b', 'r', 'c', 'm', 'k', 'y'][:k]
    lable = []
    for mod, cor in zip(score_list, color):
        plt.plot(mod.good_curve_, mod.bad_curve_, cor)
        lable.append('{0}: {1:.4f}'.format(mod.Score_name, mod.Gini))
    plt.plot([0, 1], [0, 1], color='grey')
    plt.legend(lable, loc='lower right')
    plt.show()
    if img_path is not None:
        plt.savefig(img_path + '_gini_compare.png', format='png')


def vif(detail_df):
    """计算各个变量的方差膨胀系数 VIF  \n

    参数:
    ----------
    detail_sr: dataframe, 明细数据框，一般由清洗好、转换好的各个woe变量组成   \n

    返回值:
    ----------
    vif_sr: series, 各个变量的方差膨胀系数"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    col = dict(zip(detail_df.columns, range(detail_df.shape[1])))
    vif_df = {}
    df_arr = np.array(detail_df)
    for i in col:
        vif_df[i] = variance_inflation_factor(df_arr, col[i])
    vif_sr = pd.Series(vif_df)
    vif_sr.index.name = 'colName'
    vif_sr.name = 'VIF'
    return vif_sr


def fscore(X, y, cols=None):
    """计算数值型单变量的 fscore 指标，适用2分类或多分类。fscore可用来作为特征选择的标准，值越大，表示此变量越有用。\n

    参数:
    ----------
    X: dataframe, 特征变量的数据框, X 的特征都是  \n
    y: series or 1darray, 目标变量，与 X 观测数相等  \n
    cols: list or Index, X 中，特征变量的列名，X 中可以包括额外的列，而 cols 参数指定了需要计算 fscore 的列。  \n
        默认值 None 表示使用 X 中的所有列。  \n

    返回值:
    ----------
    cols_score: series, 各个特征变量的 fscore """

    y = np.array(y)  # 防止 y 的index 与 X 的未对齐
    g = X.groupby(y)
    if cols is None:
        cols = X.columns
    avg = g[cols].mean()
    avg_overall = X[cols].mean()
    var = g[cols].var()
    # 计算各分类中心与全局中心的距离和，与各类内部方差的和的比值
    f_score = ((avg - avg_overall) ** 2).sum() / var.sum()
    return f_score


def prob2odds(prob_bad):
    """把概率转换成 log(odds)

    参数:
    -----------
    prob_bad: float, series or 1darray, bad 的概率.  \n

    返回值:
    ----------
    log_odds: 类型同 prob_bad, 对应的 log(odds)"""
    return np.log((1 - prob_bad) / prob_bad)


def prob2score(prob_bad, A=427.01, B=57.7078):
    """把概率转换成评分卡的得分  \n

    参数:
    -----------
    prob_bad: float, series or 1darray, bad 的概率.  \n
    A: float, 分数转换的参数A   \n
    B: float, 分数转换的参数B   \n

    返回值:
    ----------
    sr_numeric: 类型同 prob_bad, 对应的评分卡分数"""
    return A + B * prob2odds(prob_bad)


def offset_prob_transform(prob_model, badrate_actual, badrate_dev):
    """offset 法 adjusting ：进行了 oversampling 抽样的模型，把模型的预测概率转化为实际预测概率  \n

    参数:
    -----------
    prob_model: 1darray or series, 基于 oversampled 的样本训练出来的模型预测的 bad 概率  \n
    badrate_actual: float, 原始样本的 badrate  \n
    badrate_dev: float, oversampled 的用于开发模型的样本 badrate   \n

    返回值:
    ----------
    prob_actual: 1darray or series, 实际预测 bad 概率"""
    good_dev = 1 - badrate_dev
    good_actual = 1 - badrate_actual
    prob_actual = prob_model * good_dev * badrate_actual / \
                  ((1 - prob_model) * badrate_dev * good_actual + prob_model * good_dev * badrate_actual)
    return prob_actual


def offset_adjust(model_const, badrate_actual, badrate_develop):
    """offset 法 adjusting ：进行了 oversampling 抽样的模型，把模型的截矩参数转化为实际的截距参数 \n
    因为理论上，非等比抽样并不会改变模型的入模变量的参数值（即beta1 ~ betan），只会改变截距项 beta0

    参数:
    -----------
    model_const: float, 基于 oversampled 的样本训练出来的模型的截矩参数  \n
    badrate_actual: float, 原始样本的 badrate  \n
    badrate_develop: float, oversampled 的用于开发模型的样本 badrate   \n

    返回值:
    ----------
    actual_const: float, 真实样本中, 模型的截矩参数。真实样本模型的其他参数，与抽样模型的其他参数相同"""
    good_dev = 1 - badrate_develop
    good_actual = 1 - badrate_actual
    offset = np.log(good_dev * badrate_actual / (badrate_develop * good_actual))
    actual_const = model_const - offset
    return actual_const


def sample_weight_adjust(badrate_actual, badrate_dev):
    """sample weight 法 adjusting: 进行了 oversampling 抽样的模型, 计算 bad 和 good 样本的权重  \n

    参数:
    -----------
    badrate_actual: float, 原始样本的 badrate  \n
    badrate_dev: float, oversampled 的用于开发模型的样本 badrate   \n

    返回值:
    ----------
    weights: dict, bad 和 good 样本的权重。"""
    good_dev = 1 - badrate_dev
    good_actual = 1 - badrate_actual
    weights = {0: good_actual / good_dev, 1: badrate_actual / badrate_dev}
    return weights


def chi_test(x, y):
    """皮尔逊卡方独立检验: 衡量特征的区分度  \n

    参数:
    -----------
    x: array-like, 一维，离散型特征变量  \n
    y: array-like，一维，另一个离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n

    返回值:
    ----------
    chi_result: dict, 卡方检验结果, 其中:  \n
        键'Value'是卡方值,  \n
        键'Prob'是检验的 p 值，值越小， x 与 y 之间的关联性越强/ 区分度越大   \n
        键'LLP'是 -log(Prob)，值越大， x 与 y 之间的关联性越强   \n
        键'DF'是卡方值的自由度.
    """
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(x, y).fillna(0)
    chi_value, p_value, def_free, _ = chi2_contingency(tab)
    return {'DF': def_free, 'Value': chi_value, 'Prob': p_value, 'LLP': -np.log(p_value)}


def likelihood_ratio_test(x, y):
    """多项分布的似然比检验： 衡量特征的区分度. 皮尔逊卡方独立检验是似然比检验的近似  \n

    参数:
    -----------
    x: array-like, 一维，离散型特征变量  \n
    y: array-like，一维，二分类离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n

    返回值:
    ----------
    likelihood_result: dict, 卡方检验结果, 其中:  \n
        键'Value'是似然比的值,  \n
        键'Prob'是检验的 p 值，值越小， x 与 y 之间的关联性越强/ 区分度越大   \n
        键'LLP'是 -log(Prob)，值越大， x 与 y 之间的关联性越强   \n
        键'DF'是似然比的自由度."""
    from scipy.stats import chi2_contingency  # 似乎没有找到直接计算似然比的函数, 用此函数计算出期望个数
    tab = pd.crosstab(x, y).fillna(0)
    chi, p, df_free, e_of_tab = chi2_contingency(tab)  # e_of_tab 即列联表中各项的期望个数
    likelihood = (2 * tab * np.log(tab / e_of_tab)).sum().sum()
    p_value = chi2.sf(likelihood, df_free)
    return {'Value': likelihood, 'Prob': p_value, 'DF': df_free, 'LLP': -np.log(p_value)}


def odds_ratio_test(x, y):
    """优势比检验: 仅适用于 x, y 均是二分类随机变量的情形. 优势比的值越大, 表示特征 x 的区分度越强  \n
    对 x, y 中特征的取值顺序无关，函数会统一把小于 1 的概率比转换为其倒数。  \n

    参数:
    -----------
    x: array-like, 一维，离散型特征变量  \n
    y: array-like，一维，另一个离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n

    返回值:
    ----------
    odds_result: dict, 优势比及其 95% 的置信区间, 置信区间不包含1表明存在显著差异. 其中:  \n
        键'Value'是优势比的值,  \n
        键'left'是优势比的 95% 置信区间的左端点   \n
        键'right'是优势比的 95% 置信区间的右端点."""
    tab = pd.crosstab(x, y).fillna(0)
    n11 = tab.iloc[0, 0]
    n12 = tab.iloc[0, 1]
    n21 = tab.iloc[1, 0]
    n22 = tab.iloc[0, 1]
    if all([n11, n12, n21, n22]):
        ratio = n11 * n22 / (n21 * n12)
        var = 1 / n11 + 1 / n12 + 1 / n21 + 1 / n22
    else:
        ratio = (n11 + 0.5) * (n22 + 0.5) / ((n12 + 0.5) * (n21 + 0.5))
        var = 1 / (n11 + 0.5) + 1 / (n12 + 0.5) + 1 / (n21 + 0.5) + 1 / (n22 + 0.5)
    if ratio < 1:
        ratio /= 1  # 概率比统一转换成大于1的情况，这样概率比越大，说明区分度越大

    z = 1.96  # 正态分布 97.5% 分位数
    confidence_left = ratio * np.exp(-z * np.sqrt(var))
    confidence_right = ratio * np.exp(z * np.sqrt(var))
    return {'Value': ratio, 'left': confidence_left, 'right': confidence_right}


def f_test(x, y):
    """F检验: 衡量一个连续变量和一个离散变量之间的关联性.  \n

    参数:
    -----------
    x: array-like, 一维，连续型特征变量  \n
    y: array-like，一维，离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n

    返回值:
    ----------
    odds_result: dict, F 统计量的值及其 p 值. 其中:  \n
        键'Value'是 F 统计量的值,  \n
        键'Prob'是检验的 p 值，值越小， x 与 y 之间的关联性越强/ 区分度越大   \n
        键'LLP'是 -log(Prob)，值越大， x 与 y 之间的关联性越强   \n
        键'dfn', 'dfd' 是 F 统计量的自由度."""
    r = len(pd.unique(y))
    n = len(y)
    df = pd.DataFrame({'x': x, 'y': y})
    group = df.groupby('y')
    mean_i = group['x'].mean()
    mean_all = df['x'].mean()
    n_i = group[x].count()
    sstr = (n_i * (mean_i - mean_all) ** 2).sum()  # 组之间的平方和
    sstr_mean = sstr / (r - 1)

    df['mean_i'] = df['y'].map(mean_i)  # 以 y 的取值为分组, 各个组内的 x 的平均值
    sse = ((df['x'] - df['mean_i']) ** 2).sum()  # 组内的平方和
    sse_mean = sse / (n - r)

    f_value = sstr_mean / sse_mean
    from scipy.stats import f
    p_value = f(f_value, r - 1, n - r)

    return {'Value': f_value, 'Prob': p_value, 'dfn': r - 1, 'dfd': n - r, 'LLP': -np.log(p_value)}


def gini_variance(x, y):
    """基尼方差/熵方差: 衡量两个变量之间的关联性. 基尼方差越大，区分度越大。两个变量可以是以下三种情况： \n
        一个连续变量和一个类别变量
        一个连续变量和一个序数变量
        两个类别变量
        两个顺序变量

    参数:
    -----------
    x: array-like, 一维，连续型特征变量  \n
    y: array-like，一维，离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n

    返回值:
    ----------
    var: float, 基尼方差/熵方差, 取值范围 0-1，var 越大，特征 x 的区分度越大
    """
    df = pd.DataFrame({'x': x, 'y': y})
    group = df.groupby('y')
    mean_i = group['x'].mean()
    df['mean_i'] = df['y'].map(mean_i)  # 以 y 的取值为分组, 各个组内的 x 的平均值
    sse = ((df['x'] - df['mean_i']) ** 2).sum()  # 组内的平方和

    mean_all = df['x'].mean()
    std = ((df['x'] - mean_all) ** 2).sum()  # 总体方差之和

    # 同见 f_test:  std = sse + sstr, 总体方差 = 组内误差 + 组间误差
    var = 1 - sse / std  # 组内平方和占总体误差的比例越小，说明组间差异越大。因此 var 越大，区分度越大
    return var


def straitified_sample(detail_df, frac, by):
    """分层随机抽样。  \n

    参数:
    -----------
    detail_sr: dataframe, 明细数据表   \n
    frac: float or dict, 抽样比例。如果是 dict, 则指定的是各个层的抽样比例   \n
    by: str or list_of_str, 分层的特征名，可以是多个特征，但这些特征必须在 detail_sr 中  \n

    返回值:
    ----------
    sampled_df: dataframe, 抽样的样本数据  \n
    weight: float or dict, 样本权重。如果 frac 是 dict, 则 weight 也是 dict, 反映了各层的样本权重
    """
    group = detail_df.groupby(by, as_index=False)
    sub_dfs = []

    if isinstance(frac, float):
        weight = 1 / frac
        for layer in group.groups:
            sample_layer = detail_df.get_group(layer)
            sub_dfs.append(sample_layer.sample(frac=frac))
    else:
        weight = {}
        for layer in group.groups:
            sample_layer = detail_df.get_group(layer)
            sub_dfs.append(sample_layer.sample(frac=frac[layer]))
            weight[layer] = 1 / frac[layer]

    sampled_df = pd.concat(sub_dfs)
    return sampled_df, weight


def parcel_assign(rejected_df, group_name, group_badrate, factor=2):
    """打包法拒绝推断：按各评分组内的 badrate, 随机等比地给拒绝样本赋值 0/1  \n

    参数:
    -----------
    rejected_df: dataframe, 被评分卡拒绝的申请客户样本，并已经被新开发的、 \n
        未做拒绝推断的评分卡打了分数  \n
    group_name: str, rejected_df 中，分数分组的列的名字  \n
    group_badrate: series or dict, 已知表现的开发样本中，各分数分组内的badrate  \n
    factor: float or series/dict, badrate的乘积因子。若为series/dict, 则表示每个组一个因子。\n
        被拒绝样本的各组badrate 比开发样本的对应组更差，badrate_rej = factor * badrate_dev

    返回值:
    ----------
    assigned_y: series, 推断的目标变量
    """
    group = rejected_df.groupby(group_name, as_index=False)
    random_assign = lambda badrate: lambda: 1 if np.random.rand() < badrate else 0
    assigned_y = []
    if isinstance(factor, dict):
        factor = pd.Series(factor)
    factored_badrate = factor * group_badrate

    for group_value in group.groups:
        sub_df = group.get_group(group_value)
        sub_badrate = factored_badrate[group_value]
        sub_y = sub_df.iloc[0].map(random_assign(sub_badrate))
        assigned_y.append(sub_y)
    assigned_y = pd.concat(assigned_y)
    return assigned_y


def dist_compare(sr1, sr2, legend=(), title=None, figsize=(10, 6),
                 fig_path=None, auto_show=True):
    """把两个列的分布画在同一个图上，方便对比, 并计算两个分布的 JS 散度距离。

    参数:
    ---------
    sr1: array like, 推荐用 pd.Series 类型，
        列1（随机变量1的取值序列）
    sr2: array like, 推荐用 pd.Series 类型，
        列1（随机变量2的取值序列）
    legend: tuple, 
        两个列的名称，用于标记并区分两个分布
        若不传，则默认使用 sr1.name, sr2.name，需确保 sr1, sr2 有 name 属性
    title: str, 绘图的标题
    figsize: tuple, 绘图的大小
    fig_path: str, 
        保存所绘图片的绝对路径，不需要带扩展名，自动保存为 png 格式
    auto_show: bool, 
        是否自动显示所绘图片。当批量绘制大量图片时，可能不想自动显示图片
    """
    from seaborn import distplot
    if legend == ():
        legend = (sr1.name, sr2.name)
        
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.subplots_adjust(wspace=0.2)
    distplot(sr1, ax=ax, label=legend[0]).set_alpha(0.2)
    distplot(sr2, ax=ax, label=legend[1]).set_alpha(0.2)
    ax.legend()  # 显示图例
    
    if title:
        ax.set_title(title)
    if auto_show:        
        fig.show()
    if fig_path:
        fig.savefig(fig_path + '.png', dpi=400)


def js_divergence(sr1, sr2, bins=None):
    """计算两个分布的 JS 散度，量化的是两个分布的距离

    参数:
    ---------
    sr1: series, 
        随机变量 x1 的分布（或明细数据列，取决于bins参数值）。
        若是分布，x1 的取值保存在 sr.index 中，其概率 p 则保存在 sr.values 中
        x1 的分布最好提前离散化，以免有些取值的概率 p 不稳定
    sr2: series, 同 sr1
    bins: list or None
        当 bins 为 None 时，函数认为 sr1, sr2 是两个计算好的分布;
        当 bins 为 list 时，函数认为 sr1, sr2 是两个明细数据列，会按 bins 中的分割点离散化数据列，
        然后再计算两个分布的 JS 散度;
        当 bins 为 int 值时，函数认为 sr1, sr2 是两个明细数据列，并将数据列离散化为 bins 个离散区间，
        然后再计算两个分布的 JS 散度；
        
    返回值:
    ---------
    js_div: float, 两个分布的 JS 散度"""
    
    def ks_divergence(sr1, sr2, bins=None):
        """计算两个分布的 KS 散度"""
        p1 = sr1.values
        p2 = sr2.values
        # 定义：当 p1=0 时，p1 * log(p1/p2) = 0
        ks_div = sum([i * np.log(i / j) for i, j in zip(p1, p2) if i != 0])
        return ks_div
    
    if isinstance(bins, list):
        sr1_cate = pd.cut(sr1, bins=bins)
        sr2_cate = pd.cut(sr2, bins=bins)
        sr1 = sr1_cate.distribution().iloc[:,1]  # 取其分布列
        sr2 = sr2_cate.distribution().iloc[:,1]
        
    elif isinstance(bins, int):
        def common_inteval():
            """计算公共的离散化区间"""
            sr = pd.concat([sr1, sr2])
            q = sr.describe([0.25, 0.75])
            q3 = q['75%']
            q1 = q['25%']
            delta_q = q3 - q1
            upper = q3 + 1.5 * delta_q
            lower = q1 - 1.5 * delta_q
            bin_list = np.linspace(lower, upper, bins+1)
            return list(bin_list)
        
        def distribution(sr):
            """计算分布"""
            def discritilize():  # [0,1,3,5,9]
                def base_dis(x):
                    "对标量做离散化映射"
                    bins = [-np.inf] + common_bin + [np.inf]
                    start_list = bins[:-1]
                    end_list = bins[1:]
                    
                    for start, end in zip(start_list, end_list):
                        if start < x <= end:
                            return '({start}, {end}]'.format(start=start, end=end)
                    return 'nan'
                return sr.map(base_dis)
            
            sr_dis = discritilize()
            sr_cnt = sr_dis.value_counts() 
            p = sr_cnt / sr_cnt.sum()
            return p
        
        common_bin = common_inteval()
        sr1 = distribution(sr1)
        sr2 = distribution(sr2)
        
    elif bins is None:
        pass
    else:
        raise Exception("非法输入，bins参数只接受 list, int 或 None ")

    df = pd.concat([sr1, sr2], axis=1).fillna(0)
    sr1, sr2 = df.iloc[:, 0], df.iloc[:, 1]
    sr3 = (sr1 + sr2) / 2   
    js_div = (ks_divergence(sr1, sr3) + ks_divergence(sr2, sr3)) / 2
    return js_div

