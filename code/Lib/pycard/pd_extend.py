# -*- coding: utf-8 -*-



"""Created on Thu Nov 23 20:53:19 2017 @author: 左词
扩展了 pandas 模块，添加了开发评分卡经常用到的方法"""
import pandas as pd
from .tools import re_search, flatten_multi_index ,GetBirthAge
from . import config
from .core import normalized_mutual_info_score
from .core import cross_woe ,infer_col_type ,cols_report
import numpy as np
from datetime import date ,datetime
import hashlib
from gmssl import sm2 , sm3 ,func
from gmssl.sm3 import sm3_hash
    
def default_param_decorator(func, **my_kwargs):
    """默认参数装饰器，带有默认参数值的函数和方法，修改其默认参数的值, 以更符合常用场景  \n

    参数:
    ---------
    func: 函数，其中带有若干个默认参数，但默认参数设定的值不是最常用的值，期望装饰器更改它  \n
    my_kwargs: 若干个关键字参数，其默认值可以设定成所期望的值   \n

    返回值:
    ---------
    func_param: 函数，与 func 功能相同，只是由my_kwargs设定的关键字参数的默认值不同"""
    def func_param(*args, **kwargs):
        kwargs.update(my_kwargs)
        return func(*args, **kwargs)
    return func_param


pd.DataFrame.to_csv_idx = default_param_decorator(pd.DataFrame.to_csv, index=False)
pd.DataFrame.to_csv_idx.__doc__ = """
same as df.to_csv(path_or_buf,index=False,**kwargs)，不保存 df 的 index。\n

Parameters:
-----------
path_or_buf : string or file handle, default None, will return the result as a string.\n
其余所有参数，参见df.to_csv方法"""

pd.DataFrame.to_sql_idx = default_param_decorator(pd.DataFrame.to_sql, index=False,
                                                  if_exists='append', flavor='mysql')
pd.DataFrame.to_sql_idx.__doc__ = """
same as df.to_sql(name, con, flavor='mysql', index=False, if_exists='append',**kwargs)\n
把 df 中的数据写入到数据库的指定表中。\n
Parameters:
-----------
name : str, 数据库表名。若此表已存在，会把数据追加在表的尾部。\n
con : SQLAlchemy engine，数据库连接对象。\n
其余所有参数，参见df.to_sql方法"""


# def add_merge_keys():
#     """为 DataFrame 添加 merge 相关的方法"""
#     for col_name in config.MERGE_KEYS:
#         for how in ['left', 'inner']:
#             fun_name = 'merge_on_{on}_{how}'.format(on=col_name, how=how)
#             setattr(pd.DataFrame,
#                     fun_name,
#                     default_param_decorator(pd.DataFrame.merge, on=col_name, how=how))
#             setattr(getattr(pd.DataFrame, fun_name),
#                     '__doc__',
#                     "same as df.merge(other,on='{on}',how='{how}')".format(on=col_name, how=how))
# add_merge_keys()


def default_keyparam_decorator(func, *ex_args):
    """默认关键字参数装饰器，此装饰器把传入的参数自动赋值给args指定的关键字参数名。\n

    参数:
    ----------
    func: 函数，其中有些关键字参数，经常需要给它们传值。  \n
    ex_args: str, 关键字参数的名字，此装饰器自动把传入的值，传递给func的这些关键字参数  \n

    返回值:
    ----------
    func_key: 函数，逻辑同 func, 只是把 func 中由ex_args指定的那些关键字参数当作位置参数来使用。  \n

    Example:
    --------
    pd.DataFrame.rename_col = defaultKeyParam(pd.DataFrame.rename,'columns')
    d = {'oldname1':'newname1','oldname2':'newname2'}
    df.rename_col(d)  # 等价于 df.rename(columns=d), 懒得在每次调用 rename 方法时敲 columns="""
    def func_key(*args, **kwargs):
        n = len(ex_args)
        args_for_key = args[-n:]  # 传给默认关键字的参数，要按顺序放在位置参数的后面，这样args的最后n个参数，
        default_key = dict(zip(ex_args, args_for_key))  # 就能与ex_args_key里面的默认关键字参数对应起来
        kwargs.update(default_key)  
        if len(args) > n:  # 如果除了默认关键字参数，还有其他位置参数
            args = args[:-n]  # 提取 args 里面的位置参数
            return func(*args, **kwargs)
        else:
            return func(**kwargs)
    return func_key
pd.DataFrame.rename_col = default_keyparam_decorator(pd.DataFrame.rename, 'columns')
pd.DataFrame.rename_col.__doc__ = """
重命名columns, df.rename_col(some_dict) 等价于 df.rename(columns=some_dict)\n
Parameters:
-----------
some_dict : dict of {oldname: newname}, 需要重命名列名的字典。\n"""


def idrop_row(self, ints):
    """
    以整数值下标来指定需要删除的行。返回删除指定行之后的df。\n
    drop 方法以 labels 来指定需要删除的行，无法以labels的序号来指定需要删除的行.\n

    参数:
    ----------  
    ints: int or list_of_int, 需要删除的行的序号\n
    see also: idrop, idrop_col, drop, drop_col
    """
    idx = list(range(len(self)))
    if isinstance(ints, int): ints = [ints]
    for i in ints:
        idx.remove(i)
    return self.iloc[idx, :]
pd.DataFrame.idrop_row = idrop_row


def idrop_col(self, ints):
    """
    以整数值下标来指定需要删除的列。返回删除指定列之后的df。\n

    参数:
    ----------    
    ints: int or list_of_int, 需要删除的列的序号 \n
    see also: idrop, idrop_row, drop, drop_col \n
    """
    idx = list(range(len(self.columns)))
    if isinstance(ints, int): ints = [ints]
    for i in ints:
        idx.remove(i)
    return self.iloc[:, idx]
pd.DataFrame.idrop_col = idrop_col


def drop_col(self, cols, inplace=False):
    """删除某一列或某些列

    参数:
    ----------
    cols : 单个列名，或列名组成的 ,不能包含 数值型 sequnce \n
    inplace : 是否要原地修改 df。 默认值False表示返回新 df \n
    """
    if isinstance(cols, (str, int, float)): cols = [cols]
    cols1 = cols.copy()
    for i in cols1:
        if i not in re_search(i,self):
            cols.remove(i)
    if len(cols) > 0:
        return self.drop(columns=cols, inplace=inplace)
    else:
        return self
pd.DataFrame.drop_col = drop_col


def corr_tri(self):
    """返回相关性矩阵的下三角矩阵, 上三角部分的元素赋值为 nan \n"""
    from numpy import ones
    corr_df = self.corr()
    n = len(corr_df)
    logic = ones((n, n))
    for i in range(n):
        for j in range(n):
            if i > j:
                logic[i, j] = 0
    return corr_df.mask(logic == 1)
pd.DataFrame.corr_tri = corr_tri


def nan_rate(self):
    """计算数据列的缺失率

    返回:
    ----------
    nan_sr: 数据列，该列的各个 key 说明：
        len: 数据列的元素个数
        count: 数据列的非空元素个数
        nanRate: 数据列的缺失率"""
    d = pd.Series([],dtype = 'float64')
    d['len'] = len(self)
    d['count'] = self.count()
    d['nanRate'] = 1 - d['count'] / d['len']
    return d
pd.Series.nan_rate = nan_rate


def nan_rate_df(self, col=None):
    """计算各列的缺失率\n

    参数:
    ----------    
    cols: 可以是单个列字符串，或列的集合. 需要计算缺失率的列，None表示计算所有列 \n

    返回值:
    ----------    
    当col只有一列时，返回序列；当col是多列时，返回df
    参见 pd.Series.nan_rate"""
    if col is None: col = self.columns
    if isinstance(col, (str, int, float)):  # col只有一列的话
        col = [col]

    d = []
    for i in col:
        nan_sr = self[i].nan_rate()
        nan_sr.name = i
        d.append(nan_sr)
    return pd.concat(d, axis=1).transpose()
pd.DataFrame.nan_rate = nan_rate_df


def dup_info(self):
    """对数据列的元素值进行重复性检查，并统计重复值信息

    返回:
    ----------
    DupInfo对象，该自定义对象有以下属性：
        rows:  重复的元素值列表
        logic: 所有有重复的行，标记为 True, 值唯一的行，标记为 False 的逻辑列
        count: 有重复的元素值的个数
        sum:   重复的总行数
        value_counts: 每个有重复的元素值，其重复次数的统计"""
    class DupInfo:
        def __init__(self, name):
            self.name = name
            self.count, self.sum = 0, 0

        def __str__(self):
            return 'name: %s\ncount: %s\nsum: %s\n' % (self.name, self.count, self.sum)

        def __repr__(self):
            return self.__str__()

    dup = DupInfo(self.name)
    logic = self.duplicated()
    if logic.sum():
        dup.rows = self[logic].unique() 
        dup.logic = self.isin(dup.rows)
        dup.count = len(dup.rows)
        dup.value_counts = self[dup.logic].value_counts()
        dup.sum = dup.value_counts.sum()
    return dup
pd.Series.dup_info = dup_info
pd.Index.dup_info = dup_info


def dup_cnt(self ,cols = None):
    """
    统计dataframe中重复的个数并输出
    参数：
    ------------
    cols：可以是 str ，也可以是list ，也可以是元组
    """
    if isinstance(self ,pd.DataFrame):
        if isinstance(cols ,str): cols = [cols]
        for i in cols:
            if cols.index(i) == 0:
                data_frame = self.groupby(i)[i].count().to_frame(i + '_cnts')
                data_frame.sort_values([i+'_cnts' ,i] ,inplace=True ,ascending=False)
                data_frame.reset_index(inplace=True)
            else:
                df = self.groupby(i)[i].count().to_frame(i + '_cnts').reset_index()
                if df.empty == False:
                    df.sort_values([i+'_cnts',i] ,inplace=True ,ascending=False)
                    df.reset_index(inplace=True ,drop=True)
                    data_frame=data_frame.merge(df ,how = 'outer' ,left_index=True ,right_index=True)
    else:
        data_frame = self.value_counts().to_frame(self.name + '_cnts').reset_index()
        data_frame[self.name] = data_frame['index']
        data_frame.drop(columns = 'index' ,inplace=True)
        data_frame = data_frame[[self.name ,self.name + '_cnts']].copy()
        data_frame.sort_values([self.name+'_cnts' ,self.name],inplace=True ,ascending=False)
        data_frame.reset_index(inplace=True ,drop=True)
    return data_frame
pd.DataFrame.dup_cnt = dup_cnt
pd.Series.dup_cnt = dup_cnt
pd.Index.dup_cnt = dup_cnt

def same_col_merged(self, df, left_index = False, right_index = False, left_on = None, right_on = None,
                    on = None, how = 'left' ):
    """如果两个表中有同名的列，则拼表后，同名的列会被自动重命名为colname_x,colname_y 
    该函数表示可以直接删除被合并的dataframe中的相同名字的变量

    ----------
    重命名后的df，不修改原数据框
    """
    cols_s = self.columns.to_list()
    cols_d = df.columns.to_list()    
    cols_d = [i for i in cols_d if i not in cols_s]
    if isinstance(on,str):cols_d.append(on)
    dfi = self.merge(df[cols_d] ,how = how ,left_index = left_index ,right_index = right_index ,left_on = left_on ,
                      right_on = right_on ,on = on)
    return dfi
pd.DataFrame.same_col_merged = same_col_merged

def distribution(self, sort='index'):
    """计算单个变量的分布, 返回的数据框有两列：cnt(个数)、binPct(占比) \n

    参数:
    ----------
    sort: str, 'index' 表示按变量值的顺序排序，其他任意值表示变量值的个数占比排序"""
    a = self.value_counts()
    b = a / a.sum()
    df = pd.DataFrame({'cnt': a, 'binPct': b})
    if sort == 'index':
        df = df.sort_index()
    return df.reindex(columns=['cnt', 'binPct'])
pd.Series.distribution = distribution


def distribution_df(self, col=None):
    """计算df的各个变量的频率分布。\n

    参数:
    ----------
    cols: str or list_of_str, 需要计算频率分布的列，None表示所有列。col不应包含\n
        连续型的列，只应包含离散型的列。
    """
    if col is None:
        col = self.columns
    if isinstance(col, str): col = [col]
    var_cnts = pd.DataFrame()
    for i in col:
        di = self[i].destribution()
        di['colName'] = i
        var_cnts = var_cnts.append(di)
    return var_cnts
pd.DataFrame.distribution = distribution_df


def cv(self):
    """计算变异系数。"""
    m = self.mean()
    s = self.std()
    if m == 0:
        print("warning, avg of input is 0")
    return s / m
pd.Series.cv = cv


def argmax(self):
    """计算 df 的最大值所对应的行、列索引，返回 (row, cols) 元组"""
    m0 = self.max()
    m1 = self.max(axis=1)
    row = m1.idxmax()
    col = m0.idxmax()
    return row, col
pd.DataFrame.argmax = argmax


def argmin(self):
    """计算 df 的最小值所对应的行、列索引，返回 (row, cols) 元组"""
    m0 = self.min()
    m1 = self.min(axis=1)
    row = m1.idxmin()
    col = m0.idxmin()
    return row, col
pd.DataFrame.argmin = argmin


def pivot_tables(self, values, index, columns):
    """定制的透视表，把badRate、样本数、样本占比一次全求出来

    参数:
    ----------
    所有参数，参见 df.pivot_table 的同名参数"""
    tb = []    
    for fun in ['mean', 'count']:
        a = self.pivot_table(values=values, index=index, columns=columns, margins=True, aggfunc=fun)
        tb.append(a)
    tb.append(tb[1]/tb[1].iloc[-1, -1])
    return pd.concat(tb)
pd.DataFrame.pivot_tables = pivot_tables


def ranges(self):
    """计算最小值、最大值"""
    minv, maxv = self.min(), self.max()
    if isinstance(self, pd.DataFrame):
        return pd.DataFrame({'min': minv, 'max': maxv})
    else:
        return pd.Series({'min': minv, 'max': maxv})
pd.DataFrame.range = ranges
pd.Series.range = ranges


def value_counts(self, cols=None ,inplace = False):
    """df 对多个列的 value_counts，返回一个 df.\n

    参数:
    ----------
    cols: iterable of col_names, 默认值 None 表示对 df 的所有列计算
    inplace: 表示  是否是针对每个变量单独计算 count
    ------------------------
    返回：
    计算好的count的dataframe
    """
    if inplace:
        if cols is None:
            cols = self.columns
        vc = []
        for i in cols:
            vci = self[i].value_counts().to_frame(name='col_cnts')
            vci['col_name'] = i
            vci.reset_index(inplace=True)
            vci['col_bins'] = vci['index']
            vci.drop(columns = 'index' ,inplace=True)
            vci['all_cnts'] = vci['col_cnts'].sum()
            vci['col_rate'] = vci['col_cnts']/ vci['col_cnts'].sum()
            vci = vci.sort_values('col_bins')
            vci['col_rate'] = [str(round(float(i)*100,2))+'%' if float(i) > -np.inf else '0%' for i in vci['col_rate'].to_list()]
            vcf = pd.DataFrame({'col_name':[i]})
            vcf = vcf.append(vci)            
            vc.append(vcf[['col_name','col_bins','col_cnts','all_cnts','col_rate']])
        dfi = pd.concat(vc, axis=0).reset_index(drop = True)      
        
    else:
        if isinstance(cols ,(str,int,float)):cols = [cols]
        dfi = self.groupby(cols)[cols[-1]].count().to_frame('col_cnts')
        dfi = dfi.merge(dfi.groupby(level=0)['col_cnts'].sum().to_frame().rename(columns = {'col_cnts':'section_cnts'}) , how = 'left' ,left_index = True ,right_index =True)
        dfi['section_rate'] = dfi['col_cnts']/ dfi['section_cnts']
        dfi['section_rate'] = [str(round(float(i)*100,2))+'%' if float(i) > -np.inf else '0%' for i in dfi['section_rate'].to_list()]       
        dfi['total_cnts'] = dfi['col_cnts'].sum()
        dfi['total_rate'] = dfi['col_cnts']/ dfi['total_cnts']
        dfi['total_rate'] = [str(round(float(i)*100,2))+'%' if float(i) > -np.inf else '0%' for i in dfi['total_rate'].to_list()]
    return dfi
pd.DataFrame.value_counts = value_counts

def value_counts_series(self, sort = 'index'):
    """
    针对系列性数据进行 value_counts
    sort：按照 index 或者 values 排序，只会是升序
    """
    dfi = self.value_counts().to_frame().reset_index().rename(columns = {'index':'colBins'})
    cols_n = dfi.columns.to_list()
    dfi = dfi.rename(columns={cols_n[1]:'colCnts'})
    cols_n = dfi.columns.to_list()
    dfi['totalCnts'] = dfi.iloc[:,1:2].sum()[cols_n[1]]
    dfi['colRate'] = dfi.iloc[:,1:2] / dfi.iloc[:,1:2].sum()
    dfi['colRate'] = [str(round(float(i)*100,2))+'%' if float(i) > -np.inf else '0%' for i in dfi['colRate'].to_list()]
    if sort == 'index':
        return dfi.sort_values('colBins').reset_index(drop = True)
    elif sort == 'values':
        return dfi.sort_values(cols_n[1]).reset_index(drop = True)
    else:
        print("参数 输入错误")
pd.Series.value_count = value_counts_series 

def unique_dtype(self):
    """计算 Series 非重复的数据类型个数"""
    return self.map(type).value_counts()
pd.Series.unique_dtype = unique_dtype


def find_idx(self, label):
    """找出 label 在 index 中的位置 \n

    参数:
    ----------
    label: 需要在 index 中查找的值 \n

    返回值:
    ----------
    idx: int, 若查中，则返回 label 在 index 中的下标；若未查中，返回-1"""

    def op1(labeli):
        if pd.isnull(label):  # label是缺失的话，用==判断找不出来
            return pd.isnull(labeli)  # is np.nan 判断缺失不可靠，np.isnan函数判断才可靠, 因为 is 是判断同一性
        else:
            return labeli == label

    for i, labeli in enumerate(self):
        if op1(labeli):
            return i
    return -1
pd.Index.find = find_idx


def merge_after_dup_check(self, other, on=None, how='inner', left_on=None, right_on=None, 
                          left_index=False, right_index=False):
    """拼表功能同内置的 merge方法，但在 merge 之前检查两个表的联接键是否有重复。若有重复，打印出来以提醒"""
    if on:  # 若联接键传给了 on 参数
        dup_check1 = self[on].duplicated().sum()
        dup_check2 = other[on].duplicated().sum()
    else:
        dup_check1 = self[left_on].duplicated().sum() if left_on else self.index.duplicated().sum()
        dup_check2 = other[right_on].duplicated().sum() if right_on else other.index.duplicated().sum()
    
    if dup_check1 > 0: print("提示：主表的联接键存在重复，重复数为{}".format(dup_check1))
    if dup_check2 > 0: print("提示：副表的联接键存在重复，重复数为{}".format(dup_check2))

    return self.merge(other, on=on, how=how, left_on=left_on, right_on=right_on,
                      left_index=left_index, right_index=right_index)
pd.DataFrame.merge_after_dupcheck = merge_after_dup_check


def as_ordinal(self, categories):
    """将列转换为序数型的列，不修改自身

    参数:
    ----------
    categories: list, 序数型变量的枚举值，应按次序排放在categories中

    返回:
    ----------
    ordinal_sr: series, 序数型的列。序数型的列，其数据类型为CategorialDtype, 且是有序的"""
    cat_dtype = pd.CategoricalDtype(categories, ordered=True)
    ordinal_sr = self.astype(cat_dtype)
    return ordinal_sr
pd.Series.as_ordinal = as_ordinal


def as_category(self, other_group=None):
    """把列转换为类别型的列，不修改自身

    参数:
    ----------
    other_group: str or None, 是否额外定义一组，用于代表“其他”。
        类别型变量的枚举值不需要传入，通过列的取值来推断。
        但类别型的某些取值若频数极低，常常需要合并到“其他”组中去。因此类别的枚举值仅仅包含出现过的那些值，往往不够，需要再增加一组

    返回:
    -----------
    category_sr: series, 类别型的列"""
    categories = self.unique()
    cat_type = pd.CategoricalDtype(categories, ordered=False)
    category_sr = self.astype(cat_type)
    if other_group is not None:
        category_sr.cat.add_categories(other_group, inplace=True)
    return category_sr
pd.Series.as_category = as_category


def is_numeric(sr):
    """是否为数值型的列。"""
    from numpy import dtype
    srtype = sr.dtype
    for dtype in [dtype('i1'), dtype('i2'), dtype('i4'), dtype('i8'),
                  dtype('u1'), dtype('u2'), dtype('u4'), dtype('u8'),
                  dtype('f2'), dtype('f4'), dtype('f8')]:
        if srtype == dtype:
            return True
    for cls in [pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
                pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype]:
        if isinstance(srtype, cls):
            return True

    return False
pd.Series.is_numeric = is_numeric


def is_ordinal(sr):
    """是否为序数型的列。
    序数型列的数据类型为 pd.CategoricalDtype, 且其 ordered 属性为 True

    参数:
    ------------
    sr: Series, 数据列

    返回值:
    ------------
    flag: boolean, 取值为 True 或 False"""
    if isinstance(sr.dtype, pd.CategoricalDtype):
        if sr.cat.ordered:
            return True
    return False
pd.Series.is_ordinal = is_ordinal


def is_category(sr):
    """是否是类别型的列。
    类别型列的数据类型为 pd.CategoricalDtype, 且其 ordered 属性为 False

    参数:
    ------------
    sr: Series, 数据列

    返回值:
    ------------
    flag: boolean, 取值为 True 或 False"""
    if isinstance(sr.dtype, pd.CategoricalDtype):
        if not sr.cat.ordered:
            return True
    return False
pd.Series.is_category = is_category


def rename_multi_cols(self, join_str='_', inplace=True):
    """当 dataframe 的列是 multi_index 时，把其重命名成普通的 index, 始终保持二维表的形态

    参数：
    --------
    join_str: str, 连接 multi_index 的第一层和第二层 label 的连接符
    inplace: bool, 是否原地修改。默认为 True

    返回:
    ---------
    flattened_df: dataframe, 当 inplace 参数为 False 时，返回重命名后的新表
    """
    flattened = flatten_multi_index(self.columns, join_str=join_str)
    if inplace:
        self.columns = flattened
    else:
        df = self.copy()
        df.columns = flattened
        return df
pd.DataFrame.rename_multi_index_cols = rename_multi_cols


def mutual_info(self, cols=None):
    """计算各列的标准化互信息矩阵。互信息是衡量两个随机变量之间信息重合度的度量指标，比线性相关性指数
    更全面。

    参数:
    ----------
    cols: str or list_of_str
        要计算哪些列的标准化互信息，至少应包含 2列。默认值 None 表示计算所有列之间的互信息
        计算互信息，需要输入的列均是离散化的列，取值数较多的数值型列，需要先离散化，计算的互信息才有意义

    返回:
    -----------
    mutaul_info_matrix: dataframe,
        给定列之间的标准化互信息矩阵。由于 I(x,y) = I(y,x), 因此只返回下三角方阵。"""
    if cols is None:
        cols = self.columns

    mutaul_info_matrix = pd.DataFrame(index=cols, columns=cols, dtype='f')
    for col_idx in range(len(cols)-1):
        for row_idx in range(col_idx+1, len(cols)):
            col_name = cols[col_idx]
            row_name = cols[row_idx]
            mutaul_info_matrix.loc[row_name, col_name] = normalized_mutual_info_score(self[col_name], self[row_name])

    return mutaul_info_matrix
pd.DataFrame.mutual_info = mutual_info

def binning(self, cols, bins, inplace=True , target = None, woe = False):
    """把 cols 离散化成区间型的列，分数最低的在第 1 组，最高的在 len(bins)+1 组.\n
    如果 cols 有缺失，缺失被分在第 0 组。区间均为左开右闭型：(low,high] \n

    参数:
    ----------
    cols: Series , 需要离散化的数值型或序数型变量  \n
    bins: list, 对 cols 进行离散化的分割点。bins 的分割点可以乱序，函数会自动排序分割点。
        n 个点会把 cols 离散化成 n+1 个组, 函数会自动为 bins 添加 -np.inf, np.inf, 第一
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
    if woe == False:
        if inplace == True:
            self.loc[:,cols + '_bin'] = self[cols].map(base_map)
            self.loc[:,cols + '_bin'].value_counts()
        else:
            self[cols].map(base_map).value_counts()
    if woe == True: 
        if inplace == True:
            self.loc[:,cols + '_bin'] = self[cols].map(base_map)
            return cross_woe(self[cols + '_bin'] ,self[target]) 
        else:
            dfi = self[[cols ,target]].copy()
            dfi.loc[:,cols + '_bin'] = dfi[cols].map(base_map)
            return cross_woe(dfi[cols + '_bin'] ,dfi[target])         
pd.DataFrame.binning = binning

def get_float(self ,cols = None ,inplace = True):
    """
    可以将建模需要的变量从str类型转化为float或者int类型 ，注意此过程不可逆
    参数：
    -------------------------------------
    cols: 参与数据类型转换的变量,可以是一个str 也可以是一个list
    inplace: 如果等于True 表示在原变量上修改 ，否则 删除原变量，在原变量上后缀 新增加 _str
    """
    if isinstance(cols ,str) : cols = [cols]
    if isinstance(cols ,list) == False:cols = self.columns.to_list()
    for i in cols:
        if inplace:
            try:
                self[i] = self[i].astype(int)
            except Exception:
                try:
                    self[i] = self[i].astype(float)
                except Exception:
                    pass
        else:
            try:
                self[i+'_str'] = self[i].astype(int)
                self.drop(columns = [i] ,inplace=True)
            except Exception:
                try:
                    self[i + '_str'] = self[i].astype(float)
                    self.drop(columns = [i] ,inplace=True)
                except Exception:
                    pass
pd.DataFrame.get_float = get_float

def get_json(self, cols = None ,keys = None):
    """
    前期需要把cols 处理成str或者dict类型的
    参数：
    -------------------
    cols ：字段名，可以是str 或者 list
    keys ：JSON的键，需要解析的键，默认全部解析，可以是str 或者 list
    """
    if isinstance(self ,pd.DataFrame):
        if isinstance(cols ,(str,int,float)): cols = [cols]         
        for i in cols:
            self[i] = self[i].fillna('none')
            if isinstance(keys ,(str,int,float)): 
                keys = [keys]
            else:
                for k in range(len(self)):
                    if self[i].iloc[k].find('{') > -1:
                        keys = list(eval(str(self[i].iloc[k])).keys())
                        break
            if cols.index(i) == 0:
                for j in keys:
                    self[i] = self[i].apply(lambda x: x if str(x).find(j) > -1 else  '{}"{}":{}{}'.format('{', j ,'{}','}'))
                    dict_l = self[i].apply(lambda x:eval(str(x))[j] if isinstance(x,str) else (x[j] if isinstance(x,dict) else None)).to_list()
                    if keys.index(j) == 0:
                        data_frame = pd.DataFrame(dict_l,index=[self.index]).reset_index(drop=True)
                    else:
                        dfi = pd.DataFrame(dict_l,index=[self.index]).reset_index(drop=True)
                        if dfi.empty == False:
                            data_frame = data_frame.merge(dfi ,how = 'left' ,left_index = True ,right_index = True)
            else:
                for j in keys:
                    self[i] = self[i].apply(lambda x: x if str(x).find(j) > -1 else  '{}"{}":{}{}'.format('{', j ,'{}','}'))
                    dict_l = self[i].apply(lambda x:eval(str(x))[j] if isinstance(x,str) else (x[j] if isinstance(x,dict) else None)).to_list()
                    dfi = pd.DataFrame(dict_l,index=[self.index]).reset_index(drop=True)
                    data_frame = data_frame.merge(dfi ,how = 'left' ,left_index = True ,right_index = True)

    else:
        if isinstance(keys ,(str,int,float)): 
            keys = [keys]
        else:
            for k in range(len(self)):
                if self[i].iloc[k].find('{') > -1:
                    keys = list(eval(str(self[i].iloc[k])).keys())
                    break
        for j in keys:
            dict_l = [eval(str(i))[j] if str(i).find(j) > -1 else '{}"{}":{}{}'.format('{', j ,'{}','}') for i in self.fillna('none')]
            if keys.index(j) == 0:
                data_frame = pd.DataFrame(dict_l,index=[self.index]).reset_index(drop=True)
            else:
                dfi = pd.DataFrame(dict_l,index=[self.index]).reset_index(drop=True)
                if dfi.empty == False:
                    data_frame = data_frame.merge(dfi ,how = 'left' ,left_index = True ,right_index = True)
    return data_frame

pd.DataFrame.get_json = get_json
pd.Series.get_json = get_json
pd.Index.get_json = get_json

def get_2per(self , cols = None,decimal = 2 ,inplace=False):
    """
    将dataframe中小数转化为百分比
    参数：
    -------------------------
    cols：dataframe中的变量名称，可以是str 和 list
    inplace：false 新增一列命名为 变量名称 + per 否则在原列进行修改
    """
    if isinstance(self ,(list,pd.Series)):       
        return [str(round(float(i)*100,decimal))+'%' if float(i) > -np.inf else '0%' for i in self]
    else:
        if isinstance(cols,(str,int,float)):cols = [cols]
        for i in cols:
            if inplace==True:
                self[i] = [str(round(float(i)*100,decimal))+'%' if float(i) > -np.inf else '0%'  for i in self[i]]
            else:
                self[i+'per'] = [str(round(float(i)*100,decimal))+'%' if float(i) > -np.inf else '0%'  for i in self[i]]
        
pd.Series.get_2per = get_2per
pd.DataFrame.get_2per = get_2per

def get_idcard(self ,cols = None ,apply_date = None ,get = 'All' ,inplace=True):
    """
    将dataframe 或者  series中身份证解析为 生日、年龄、性别
    参数：
    -----------------------------------
    cols：变量名称，只能是  str
    apply_date: 申请日期 ,只能是str
    get：需要解析的参数  默认为All ，所有都解析，birthday ，age ，sex
    inplace：是否在原dataframe中添加，默认为在原dataframe中添加解析的数据
    """   
    if isinstance(self ,pd.DataFrame):
        if apply_date is not None:
            kh_idcard = GetBirthAge(self[cols] ,self[apply_date])
        else: 
            kh_idcard = GetBirthAge(self[cols])
                  
        birthday = kh_idcard.get_birthday()
        age = kh_idcard.get_age()      
        sex = kh_idcard.get_sex()
        if get == 'All':
            if inplace == True:
                self['birthday'] = birthday
                self['age'] = age
                self['sex'] = sex
            else:
                dfi = pd.DataFrame({ 'idcard':self[cols] ,'birthday':birthday ,'age':age ,'sex':sex})
                return dfi
        else:
            if inplace == True:
                if get == 'birthday':
                    self['birthday'] = birthday
                elif get == 'age':
                    self['age'] = age
                elif get == 'sex':
                    self['sex'] = sex
                else:
                    print("请输入正确参数！！！")
            else:
                dfi = pd.DataFrame({ 'idcard':self[cols] ,'birthday':birthday ,'age':age ,'sex':sex})
                return dfi    
    else:
        kh_idcard = GetBirthAge(self)                
        birthday = kh_idcard.get_birthday()
        age = kh_idcard.get_age()      
        sex = kh_idcard.get_sex() 
        if get == 'All':
            dfi = pd.DataFrame({ 'idcard':self ,'birthday':birthday ,'age':age ,'sex':sex})
        elif get == 'birthday':
            dfi = pd.DataFrame({ 'idcard':self ,'birthday':birthday })
        elif get == 'age':
            dfi = pd.DataFrame({ 'idcard':self ,'age':age})
        elif get == 'sex':
            dfi = pd.DataFrame({ 'idcard':self ,'sex':sex})
        else:
            print("请输入正确参数！！！")        
        return dfi
pd.DataFrame.get_idcard = get_idcard
pd.Series.get_idcard = get_idcard      

def get_date(self ,cols = None ,inplace = True):
    """把日期字符串转换成 datetime.date对象。  \n

     参数:
     ------------
     date_str: str, 表示日期的字符串。  \n
        只要date_str的前 8/10 位表示日期就行，多余的字符串不影响解析； \n
        日期格式只能是下列之一： \n
            '2017-06-01' 格式的日期,   \n
            '2017/06/01' 格式的日期；  \n
            '20170601' 格式的日期。   \n

    返回值:
    -----------
    date_obj: datetime.date 对象。"""
    def str2date(x):
        if len(str(x)) >= 10 and x is not None:
            tmp = date(int(str(x)[:4]) , int(str(x)[5:7]) , int(str(x)[8:10]))        
        elif len(str(x)) == 8 and x is not None:
            tmp = date(int(str(x)[:4]) , int(str(x)[4:6]) , int(str(x)[6:8]))
        else:
            tmp = pd.NaT
        return tmp
    if isinstance(self,pd.DataFrame):
        if isinstance(cols ,str):cols = [cols]
        if inplace:
            for i in cols:
                self.loc[:,i] = self[i].apply(str2date)
        else:
            for i in range(len(cols)):
                if i == 0:
                    dfi = pd.DataFrame({cols[i]: self[cols[i]].apply(str2date)})
                else:
                    dfi = dfi.merge(pd.DataFrame({cols[i]: self[cols[i]].apply(str2date)}) ,how = 'left' ,left_index = True ,right_index = True)
            return dfi
                
    else:
        return(self.apply(str2date))
pd.DataFrame.get_date = get_date
pd.Series.get_date = get_date

def get_datetime(self ,cols = None ,inplace = True):
    """把字符串形式的datetime, 转换成 datetime 类型。\n

    参数:
    -----------
    str_datetime: str, 2018-03-12 12:28:32 格式的时间
    
    返回值:
    -----------
    dtime: datetime, 转换后的 datetime 对象"""
    def str2date(x):
        if len(str(x)) >= 19 and x is not None:
            tmp = datetime.strptime(str(x)[:19], '%Y-%m-%d %H:%M:%S')
        else:
            tmp = pd.NaT
        return tmp
    if isinstance(self,pd.DataFrame):
        if isinstance(cols ,str):cols = [cols]
        
        if inplace:      
            for i in cols:
                self.loc[:,i] = self[i].apply(str2date)                
        else:
            for i in range(len(cols)):
                if i == 0:
                    dfi = pd.DataFrame({cols[i]: self[cols[i]].apply(str2date)})
                else:
                    dfi = dfi.merge(pd.DataFrame({cols[i]: self[cols[i]].apply(str2date)}) ,how = 'left' ,left_index = True ,right_index = True)
            return dfi
    else:
        return(self.apply(str2date))
pd.DataFrame.get_datetime = get_datetime
pd.Series.get_datetime = get_datetime

def get_qcut(self ,cols ,bins = 5):
    """
    等距分箱
    """
    if isinstance(cols,str) : cols = [cols]
    for i in cols:
        cols_m = self.loc[pd.notnull(self[i])][i].to_list()
        cols_m = sorted(cols_m)
        bins_c = int(len(cols_m)/bins)
        a = [i * bins_c for i in range(bins)]
        a.remove(0)
        fin_bins = list()
        cols_f = [cols_m[i] for i in a]
        cols_f = list(set(cols_f))
        self.binning( i ,cols_f)
pd.DataFrame.get_qcut = get_qcut

def re_d(self , patten):
    """
    参数：
    --------------------------
    patten：需要查找的字符串，不在乎大小写
    ------------
    返回的是 ：dataframe
    """
    from re import compile
    patten = patten.lower()
    patten = compile(patten)
    match = []
    cols_name = self.columns.to_list()
    cols_low = [i.lower() for i in cols_name]
    for i in cols_low:
        tmp = patten.search(i)
        if tmp is not None:
            match.append(tmp.string)
    cols_re = []
    if len(match):
        for i in match:
            cols_re.append(cols_name[cols_low.index(i)])
        return self[cols_re]
    else:
        print("no re_search in dataframe!!!")
pd.DataFrame.re_d = re_d

def re_c(self , patten):
    """
    参数：
    --------------------------
    patten：需要查找的字符串，不在乎大小写
    ------------
    返回的是 ：变量名  columns
    """
    from re import compile
    patten = patten.lower()
    patten = compile(patten)
    match = []
    cols_name = self.columns.to_list()
    cols_low = [i.lower() for i in cols_name]
    for i in cols_low:
        tmp = patten.search(i)
        if tmp is not None:
            match.append(tmp.string)
    cols_re = []
    if len(match):
        for i in match:
            cols_re.append(cols_name[cols_low.index(i)])
        return cols_re
    else:
        return match
pd.DataFrame.re_c = re_c

def cols_report(self , cols = None):
    """对明细表的各列的描述性统计.
    参数:
    --------
    cols: 变量名称，参与描述性统计分析的变量名称
    可以是str 和 list 
    如果没有输入变量名，则返回整个的描述性统计分析
    返回值:
    --------
    cols_describe: dataframe, 数据表各列的描述性统计，如缺失率、非重复值、变异系数、中位数等"""
    if isinstance(cols,str):cols = [cols]
    if isinstance(cols,list):dfi = self[cols].copy()
    else: dfi = self.copy()
    type_df = infer_col_type(dfi)  # 列的数据类型
    nan_df = dfi.nan_rate()  # 列的缺失率
    unique_df = dfi.apply(pd.Series.nunique)  # 列的非重复值个数
    unique_df.name = 'unique'
    cols_describe = pd.concat([type_df, unique_df, nan_df], axis=1).reindex(type_df.index)

    # 数值型列的描述统计
    sub_df_num = dfi.select_dtypes(include=np.number)
    if sub_df_num.shape[1] > 0:
        desc_df = sub_df_num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose().drop_col('count')
        desc_df['cv'] = desc_df['std'] / desc_df['mean']
        cols_describe = pd.concat([cols_describe, desc_df], axis=1).reindex(type_df.index)

    # 非数值型列的描述统计
    sub_df_other = dfi.select_dtypes(exclude=np.number)
    if sub_df_other.shape[1] > 0:
        desc_df = sub_df_other.describe(datetime_is_numeric=True).transpose().drop_col(['count', 'unique'])
        cols_describe = pd.concat([cols_describe, desc_df], axis=1)
    return cols_describe
pd.DataFrame.cols_report = cols_report
 
def md5(self, cols):
    """
    md5加密
    参数：
    ----------------------------
    cols：可以是str 或者 list，不区分大小写
    """
    if isinstance(cols,str): cols = [cols]
    cols = [i.lower() for i in cols]
    cols_n = self.columns.to_list()
    cols_m = [i.lower() for i in cols_n]
    for i in cols:
        n = cols_n[cols_m.index(i)]
        self[n+'_md5'] = self[n].apply(lambda x: hashlib.md5(bytes(str(x) ,encoding='utf-8')).hexdigest() if x is not None and pd.notnull(x) else x)
    return "MD5 加密完成，原变量名称+'_md5'"
pd.DataFrame.md5 = md5

def sha256(self, cols):
    """
    sha256加密
    参数：
    ----------------------------
    cols：可以是str 或者 list，不区分大小写
    """
    if isinstance(cols,str): cols = [cols]
    cols = [i.lower() for i in cols]
    cols_n = self.columns.to_list()
    cols_m = [i.lower() for i in cols_n]
    for i in cols:
        n = cols_n[cols_m.index(i)]
        self[i+'_sha'] = self[i].apply(lambda x: hashlib.sha256(bytes(str(x) ,encoding='utf-8')).hexdigest() if x is not None and pd.notnull(x) else x)
    return "MD5 加密完成，原变量名称+'_sha'"
pd.DataFrame.sha256 = sha256

def sha3_256(self, cols):
    """
    sha3_256 加密
    参数：
    ----------------------------
    cols：可以是str 或者 list，不区分大小写
    """
    if isinstance(cols,str): cols = [cols]
    cols = [i.lower() for i in cols]
    cols_n = self.columns.to_list()
    cols_m = [i.lower() for i in cols_n]
    for i in cols:
        n = cols_n[cols_m.index(i)]
        self[i+'_sha3'] = self[i].apply(lambda x: hashlib.sha3_256(bytes(str(x) ,encoding='utf-8')).hexdigest() if x is not None and pd.notnull(x) else x)
    return "MD5 加密完成，原变量名称+'_sha3'"
pd.DataFrame.sha3_256 = sha3_256

def sm3(self, cols):
    """
    国密加密
    参数：
    ----------------------------
    cols：可以是str 或者 list，不区分大小写
    """
    if isinstance(cols,str): cols = [cols]
    cols = [i.lower() for i in cols]
    cols_n = self.columns.to_list()
    cols_m = [i.lower() for i in cols_n]
    for i in cols:
        n = cols_n[cols_m.index(i)]
        self[i+'_sm3'] = self[i].apply(lambda x: sm3_hash(func.bytes_to_list(bytes(str(x),encoding='utf-8'))) if x is not None and pd.notnull(x) else x)
    return "MD5 加密完成，原变量名称+'_sm3'"
pd.DataFrame.sm3 = sm3

def col_list(self ,cols = None):
    """
    返回dataframe的columns
    """
    if isinstance(cols,str):cols = [cols]
    if isinstance(cols,list):
        columns = self[cols].columns.to_list()
    else:
        columns = self.columns.to_list()
    return columns
pd.DataFrame.col_list = col_list

def rs_loc(self ,patten ,col = None):  
    """
    全量筛选变量的一个函数，该函数只针对字符串类型的数据生效：
    pattern：需要查找的值
    col：变量名称，如果为空则全量查询
    """
    def get_patten(self ,patten , col= None):
        from re import compile
        patten = patten.lower()
        patten = compile(patten)
        cols_rec = []
        cols_cate = cols_report(self)
        cols_cate = cols_cate.loc[(cols_cate.infered_type == 'cate')&(cols_cate.nanRate < 1)].index.to_list()
        if col is None:
            cols_fg = []
            for i in cols_cate:
                match = []
                cols_list = list(set(self[i].to_list()))
                cols_low = [i.lower() if isinstance(i,str) else 'fm_yangon' for i in cols_list]
                for j in cols_low:
                    tmp = patten.search(j)
                    if tmp is not None:
                        match.append(tmp.string)
                cols_re = []
                if len(match):
                    for k in match:             
                        cols_re.append(cols_list[cols_low.index(k)]) 
                        df01 = self.loc[self[i].isin(cols_re)].copy()
                        df01['col_name'] = i
                        col = df01.pop('col_name')
                        df01.insert(loc = 0 ,column = 'col_name' ,value = col)
                        cols_fg.append(df01)                    
                    cols_rec.append(i)
            if len(cols_rec):
                return pd.concat(cols_fg).drop_duplicates()
            else:
                print("no re_search in dataframe!!!")
        else:
            if col not in self.columns.to_list():
                print('no col_name in dataframe')
            else:
                match = []
                cols_list = self[col].to_list()
                cols_low = [i.lower() if isinstance(i,str) else i for i in cols_list]
                for i in cols_low:
                    tmp = patten.search(i)
                    if tmp is not None:
                        match.append(tmp.string)
                cols_re = []
                if len(match):
                    for i in match:
                        cols_re.append(cols_list[cols_low.index(i)])
                    return self.loc[self[col].isin(cols_re)]
                else:
                    print("no re_search in dataframe!!!")  
    pd.DataFrame.get_patten = get_patten
    if patten.find("&") == -1:
        return self.get_patten(patten ,col)
    else:
        col_patten = patten.split("&")
        for i in range(len(col_patten)):
            if i == 0:
                dfi = self.get_patten(col_patten[i] ,col)
            else:
                if isinstance(dfi,str): pass
                else: dfi = dfi.get_patten(col_patten[i] ,col)
        return dfi
pd.DataFrame.rs_loc = rs_loc


def append(self ,df ,inplace = True):
    """
    两个表之间的拼接 
    ------------------------------
    df: 拼接的表
    inplace: 是否在原表上做拼接
    """
    if inplace == True:
        dfi = pd.concat([self,df])
    else:
        dfi = pd.concat([self,df])
    return dfi
pd.DataFrame.append = append


def appl_result(self ,col_name = None , cols = None, func = None):
    """
    简化函数调用程序
    --------------------------------------
    col_name: 新增列名称
    cols: 旧变量名称，单变量的时候填写。当参数为None时，多变量的时候，只需要两个参数
    func: 自定义函数，当多变量的时候，该参数可以为None
    """
    if isinstance(cols,str):
        self[col_name] = self[cols].apply(func)
    else:
        self[col_name] = self.apply(cols ,axis = 1)
pd.DataFrame.appl_result = appl_result


def rs_copy(self ,drop = True):
    if drop:
        dfi = self.copy().reset_index(drop=True)
    else:
        dfi = self.copy().reset_index()
    return dfi
pd.DataFrame.rs_copy = rs_copy

#%% 删除多余的函数
del pivot_tables, distribution, same_col_merged, dup_info, dup_cnt,ranges, corr_tri, value_counts
del unique_dtype, nan_rate, nan_rate_df, idrop_row, idrop_col
del default_param_decorator, default_keyparam_decorator
del cv, distribution_df, as_ordinal, as_category, is_numeric, is_ordinal, is_category
del argmax, argmin, find_idx, drop_col, binning, get_float, merge_after_dup_check, rename_multi_cols
del mutual_info ,get_json ,get_2per ,get_idcard ,get_date ,get_datetime ,get_qcut ,re_c ,re_d
del md5 ,sha256 ,sha3_256 ,sm3 ,col_list ,rs_loc ,append ,appl_result ,rs_copy

