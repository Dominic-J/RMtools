# -*- coding: utf-8 -*-
"""Created on Mon Mar  5 20:58:25 2018 @author: 左词
模块的配置文件. 修改这里的配置文件后，需要重新加载整个模块以使修改生效"""


# 拼表中，最常用的列名，比如身份证号码。pycard将以此列名定制2个拼表方法，新增给DataFrame:
# pd.DataFrame.merge_{col_name}_left : 以 col_name 为拼接键，左连接
# pd.DataFrame.merge_{col_name}_inner ：以 col_name 为拼接键，内连接
MERGE_KEYS = ['account_no', 'apply_id', 'putoutno']


# 用到最多的最主要的主键，评分卡一般在合同级别或人的级别上建模，因此最主要的主键一般是合同ID或人的ID。
PRIM_KEY = 'account_no'


# 数据库账号 URL，MysqlTools使用此配置信息以读写指定的库
MYSQL_DICT = {
       '示例':   'mysql+pymysql://username:passwd@ip_address:port/database_name?charset=utf8',
       'local_test':   'mysql+pymysql://root:420625@127.0.0.1:3306/test?charset=utf8',
       'local_dm':   'mysql+pymysql://root:420625@127.0.0.1:3306/dm?charset=utf8',
       'inner_xarule': 'mysql+pymysql://root:Xiaoan2019!@10.10.0.181:3306/xiaoanrule?charset=utf8',
       'inner_riskdb': 'mysql+pymysql://root:Xiaoan2019!@10.10.0.181:3306/riskdb?charset=utf8',
       'pboc': 'mysql+pymysql://root:Xiaoan2019!@10.10.0.181:3306/pboc?charset=utf8',
       'yfm_modeldb': 'mysql+pymysql://root:Xiaoan2019!@10.10.0.181:3306/yfm_modeldb?charset=utf8',
       'clx_asset_front': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/clx_asset_front?charset=utf8',
       'clx_loan': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/clx_loan?charset=utf8',
       'dev_risk_analysis': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/dev_risk_analysis?charset=utf8',
       'dev_risk_bi': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/dev_risk_bi?charset=utf8',
       'dev_risk_tmp': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/dev_risk_tmp?charset=utf8',
       'riskdata': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/riskdata?charset=utf8',
       'rulengine': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/rulengine?charset=utf8',
       'temp_data_1': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/temp_data_1?charset=utf8',
       'temp_data_2': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/temp_data_2?charset=utf8',
       'temp_data_3': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/temp_data_3?charset=utf8',
       'dev_risk36': 'mysql+pymysql://yangfengming:select9@10.101.2.36:3306/dev_risk_tmp?charset=utf8',
       'risk_variable': 'mysql+pymysql://yangfengming:select9@10.101.2.32:3306/risk_variable?charset=utf8',
       }


# Postgresql数据库的账号配置
PG_DICT = {
    'pg_risk': 'postgresql+psycopg2://risk_dev:G8XpjZ4rDeRN8dRwM@10.10.0.233:5432/postgres'}


# 数据库表中常见的 {字段:数据类型} ，配置此字典后，infer_mysql_dtype函数对df的对应列的
# 数据类型便是准确的，不必再推测。
MYSQL_COL_DTYPE = {
       'contract_id': 'char(16)', 
       'idcard_no':'char(18)',
       'user_id': 'char(13)',
       'bank_card_no':'char(19)',
       'mobile': 'char(11)',
       'aplDate':'date'}


# 做模型需要先把变量分为（类别型,数字型,主键,目标变量,日期时间）几类，此字典配置常见字段
# 所属的类，infer_col_type函数会利用此信息
MODEL_TYPE = {
'cate':{},  # 类别型
'num':{},  # 数值型
'id':{'contract_id', 'idcard_no', 'user_id', 'bank_card_no', 'bankcard_no',  # 主键类
       'customer_id', 'certid', 'seq_id', 'card_id', 'mobile', 'phone', 'cell'},
'y': {},  # 目标变量
'datetime':{}  # 日期时间
}

