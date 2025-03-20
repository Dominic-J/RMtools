# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pymysql 
#from pyhive.presto import connect
from sqlalchemy import *
from sqlalchemy import types
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *
from . import config
import os
from datetime import datetime,date
import time
import sys
import os
import warnings
import vertica_python as vp
from urllib import parse
import re


pwd = parse.quote_plus("Sasphqry@123")
pwd_o = parse.quote_plus("!QAZ+1234")
SQL_DICT = {
    'local': 'mysql+pymysql://root:root123@localhost:3306/',
    '31': 'mysql+pymysql://yangfengming:select2022@10.101.2.31:3306/',
    'hive':'hive://root:root123@10.101.3.29:10000/',
    'presto':'presto://10.101.3.29:8881/',
    'oracle':f'oracle+cx_oracle://ana_fx:{pwd_o}@180.2.48.161:9300/?service_name=sasapp',
    'vertica':f'vertica+vertica_python://sasphqry:{pwd}@'
}

class Mysql2Tools(object):
    __doc__ = """创建连接Hive的不同数据库的引擎, 方便读写数据库。\n
    
    参数:
    ----------
    name: str, 
        数据库连接名称。查看已配置了哪些数据库连接：
        MysqlTools.print_connects()
    
        数据库连接可在 config 模块中配置 \n
        database:直接输入MySQL的数据库就好了
        sql_server:mysql服务器地址的最后小数点后面的数字
    """
    def __init__(self, sql_server ,database):
        # config.CONNECT_DICT 在实例化时才会被调用，实时获取CONNECT_DICT的值。
        # 若把此函数体的第一行赋值移到函数外，则在编译时CONNECT_DICT的值就已传了进来，
        # CONNECT_DICT就相当于 MysqlTools 的闭包常数，起不到动态配置作用。
        
        # 此函数编译的字节码，仅定义了调用config.CONNECT_DICT，在实例化时，才传入config.CONNECT_DICT
        # 的真实值。这样，config.CONNECT_DICT做了修改后，会反映到MysqlTools上来
        self.__connect_dict = SQL_DICT
        assert sql_server in self.__connect_dict, 'Unknown host'
        self.Name = sql_server
        self.database = database
        self.__tb_name = None
        
    def __con(self):
        """定义专门的私有方法，用于创建引擎"""
        url = f"{self.__connect_dict[self.Name]}{self.database}?charset=utf8"
        return create_engine(url)

    def query(self, sql):
        """执行查询操作，返回查询的结果表df \n

        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        # 调用 query 方法时才创建连接，而非实例化MysqlTools时就创建连接。这样连接不会因为空闲过久而丢失
        # 函数调用完之后，会自动清理内部变量，释放连接
        con = self.__con()
        dfi = pd.read_sql_query(sql, con)
        
        return dfi

    def to_sql(self, table_name, df, drop = None, chunksize=1000):
        """把df表中的数据存入数据库中、以table_name为名的表中。若此表已在库中，则会把数据追加在尾部。

        参数:
        ----------
        df: dataframe, 要写到数据库中的数据表
        table_name: str, 数据库中的写入目标表
        chunksize: int, 每写入多少行就提交一次，算一次事务操作"""
        con = self.__con()
        try:
            describe = self.query(f"describe {table_name}")
            cols_type = describe.Type.to_list()
            cols_colm = describe.Field.to_list()
            df_cols = df.columns.to_list()
            for i in range(len(cols_type)):
                if cols_type[i].lower().find('int') > -1:
                    if cols_colm[i] in df_cols: df.loc[:,cols_colm[i]] = df[cols_colm[i]].astype(int)
                elif cols_type[i].lower().find('double') > -1 or cols_type[i].lower().find('float') > -1:
                    if cols_colm[i] in df_cols: df.loc[:,cols_colm[i]] = df[cols_colm[i]].astype(float)
                elif cols_type[i].lower().find('char') > -1 or cols_type[i].lower().find('text') > -1:
                    if cols_colm[i] in df_cols: df.loc[:,cols_colm[i]] = df[cols_colm[i]].apply(lambda x: None if pd.isnull(x) else str(x))
                elif cols_type[i].lower() == 'date': 
                    if cols_colm[i] in df_cols: df.get_date([cols_colm[i]])
                elif cols_type[i].lower().find('datetime') > -1: 
                    if cols_colm[i] in df_cols: df.get_datetime([cols_colm[i]])
                else:
                    pass
        except Exception:
            pass
        if isinstance(drop ,(str,int,float,list)):df.drop_col(drop ,inplace=True)
        if len(df) > 1000:
            for i in range(0,len(df),1000):
                if i == 0:
                    try:
                        dfi = self.query(f"select * from {table_name} limit 1")
                        cols = [i for i in df.columns.to_list() if i not in dfi.columns.to_list()]
                        if len(cols):
                            cols_now = df.columns.to_list()
                            for i in cols:
                                if cols_now.index(i)-1 != -1:
                                    self.exe(f"alter table {table_name} add column {add_sql_dtype(df[[i]])} after {cols_now[cols_now.index(i)-1]}")
                                else:
                                    self.exe(f"alter table {table_name} add column {add_sql_dtype(df[[i]])} first")
                                time.sleep(10)  
                        df = pd.concat([dfi,df])
                        df.reset_index(inplace=True ,drop=True)
                        df.drop(index=df.index[0],inplace=True)
                        df.reset_index(inplace=True ,drop=True)
                        df.iloc[0:1000].to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize)                       
                    except Exception:
                        df.iloc[0:1000].to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize)
                        dfi = self.query(f"select * from {table_name} limit 1")                     
                    if 'auto_id' not in dfi.columns.to_list():
                        self.exe(f"alter table {table_name} add column auto_id bigint(20) primary key auto_increment")
                    dfi = self.query(f"select * from {table_name} limit 1")
                    
                elif len(df) - i > 1000:
                    df.iloc[i:i+1000].to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize)
                else:
                    df.iloc[i:i+len(df)].to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize)
        else:     
            try:
                dfi = self.query(f"select * from {table_name} limit 1")
                df = pd.concat([dfi,df])
                df.reset_index(inplace=True ,drop=True)
                df.drop(index=df.index[0],inplace=True)
                df.reset_index(inplace=True ,drop=True)
                df.to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize)    
            except Exception:
                try:               
                    df.to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize)
                    dfi = self.query(f"select * from {table_name} limit 1")                    
                except Exception:
                    dfi = self.query(f"select * from {table_name} limit 1")
                    cols = [i for i in df.columns.to_list() if i not in dfi.columns.to_list()]
                    cols_now = df.columns.to_list()
                    for i in cols:
                        if cols_now.index(i)-1 != -1:
                            self.exe(f"alter table {table_name} add column {add_sql_dtype(df[[i]])} after {cols_now[cols_now.index(i)-1]}")
                        else:
                            self.exe(f"alter table {table_name} add column {add_sql_dtype(df[[i]])} first")
                        time.sleep(10)  
                    df = pd.concat([dfi,df])
                    df.reset_index(inplace=True ,drop=True)
                    df.drop(index=df.index[0],inplace=True)
                    df.reset_index(inplace=True ,drop=True)
                    df.to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize)                      
            if 'auto_id' not in dfi.columns.to_list():
                self.exe(f"alter table {table_name} add column auto_id bigint(20) primary key auto_increment")  
        return "数据写入MySQL成功 {}条".format(len(df))
        
    def exe(self, sql):
        """清空表、删除表、创建表等任意sql操作 \n

        参数:
        ----------
        sql: str, 由字符串表达的数据库语句"""
        con = self.__con()
        con.execute(sql)       
        return "SQL 语句执行完毕"
        
    def add_unique_key(table_name ,unique_key):
        """
        为MySQL添加唯一主键索引
        参数：
        ------------------------
        table_name：表名
        unique_key：唯一主键的变量名称
        
        唯一主键的名称为  变量名称 + unique
        """
        con = self.__con()
        con.execute(f"alter table {table_name} add constraint {unique_key}_unique unique ({unique_key})")
        return "MySQL 唯一索引添加完毕"

    def add_query_key(table_name ,query_key ,length = False):
        """
        为MySQL添加普通的查询索引
        参数：
        ------------------------
        table_name：表名
        unique_key：查询主键的变量名称
        length：如果变量为str的时候需要为True ，给索引添加长度，否则会报错
        
        索引主键的名称为：idx + query_key
        """
        con = self.__con()
        if length:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key}(255))")
        else:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key})")
        return "MySQL 普通索引添加完毕"

    def show_tables(self ,patten = None ,inplace = False):
        """
        返回数据库中的所有表
        patten: 正则表达式中的需要查找的表名的一部分
        inplace: 是否进行新的查询，一般没有新增表，无需此操作
        """
        if self.__tb_name is None or inplace == True:
            self.__tb_name = self.query('select table_schema ,table_name from information_schema.tables')
            self.__tb_name = self.__tb_name.rename_col({i:i.lower() for i in self.__tb_name})
            self.__tb_name['table_name'] = self.__tb_name['table_name'].apply(lambda x: x.lower())
            cols_not = ['mysql','information_schema','performance_schema','sys']
            self.__tb_name = self.__tb_name.loc[~self.__tb_name['table_schema'].isin(cols_not)].rs_copy()
            if patten is None:
                return self.__tb_name.reset_index(drop=True)
            else:
                return self.__tb_name.rs_loc(patten).reset_index(drop=True)
        else:            
            if patten is None:
                return self.__tb_name.reset_index(drop=True)
            else:
                return self.__tb_name.rs_loc(patten).reset_index(drop=True)              
          
    def desc_table(self, table_name):
        """返回一个表的元数据信息"""
        b = self.query('describe {}'.format(table_name))
        b.Field = b.Field.str.lower()
        return b

    def add_comment(self ,table_name ,commt = None):
        """
        ---------------------------------------------
        给MySQL字段新增注释：
        commt:是dataframe ,字段名称为 Field:字段名称 ,col_cmmt:字段注释
        table_name:表名
        ------------------------
        当commt为None时，可以手动输入每个字段的注释
        """
        if commt is not None:
            dfield = self.desc_table(table_name)
            dfield = dfield.merge(commt ,how = 'left' ,on = 'Field')
            for i in range(len(dfield)):
                if i < len(dfield) - 1:
                    self.exe(f"alter table {table_name} modify {dfield.Field.iloc[i]}  {dfield.Type.iloc[i]} comment '{dfield.col_cmmt.iloc[i]}';")
                else:
                    self.exe(f"alter table {table_name} modify {dfield.Field.iloc[i]}  {dfield.Type.iloc[i]} comment '{dfield.col_cmmt.iloc[i]}';")
                    print(f"'{table_name}' 数据字段添加注释完毕，共计{i+1} 个字段")
        else:
            dfield = self.desc_table(table_name)
            for i in range(len(dfield)):
                if i < len(dfield) - 1:
                    tp = input(f"{dfield.Field.iloc[i]}的注释为：")
                    self.exe(f"alter table {table_name} modify {dfield.Field.iloc[i]}  {dfield.Type.iloc[i]} comment '{tp}';")
                else:
                    tp = input(f"{dfield.Field.iloc[i]}的注释为：")
                    self.exe(f"alter table {table_name} modify {dfield.Field.iloc[i]}  {dfield.Type.iloc[i]} comment '{tp}';")
                    print(f"'{table_name}' 数据字段添加注释完毕，共计{i+1} 个字段")        
        
class OracleTools(object):
    __doc__ = """创建连接Hive的不同数据库的引擎, 方便读写数据库。\n
    
    参数:
    ----------
    name: str, 
        数据库连接名称。查看已配置了哪些数据库连接：
        MysqlTools.print_connects()
    
        数据库连接可在 config 模块中配置 \n
        database:直接输入MySQL的数据库就好了
        sql_server:mysql服务器地址的最后小数点后面的数字
    """
    def __init__(self, sql_server):
        # config.CONNECT_DICT 在实例化时才会被调用，实时获取CONNECT_DICT的值。
        # 若把此函数体的第一行赋值移到函数外，则在编译时CONNECT_DICT的值就已传了进来，
        # CONNECT_DICT就相当于 MysqlTools 的闭包常数，起不到动态配置作用。
        
        # 此函数编译的字节码，仅定义了调用config.CONNECT_DICT，在实例化时，才传入config.CONNECT_DICT
        # 的真实值。这样，config.CONNECT_DICT做了修改后，会反映到MysqlTools上来
    
        self.__connect_dict = SQL_DICT
        assert sql_server in self.__connect_dict, 'Unknown host'
        self.Name = sql_server
        self.__qryname = None
        
    def __con(self):
        """定义专门的私有方法，用于创建引擎"""
        url = f"{self.__connect_dict[self.Name]}"
        return create_engine(url)

    def query(self, sql):
        """执行查询操作，返回查询的结果表df \n

        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        # 调用 query 方法时才创建连接，而非实例化MysqlTools时就创建连接。这样连接不会因为空闲过久而丢失
        # 函数调用完之后，会自动清理内部变量，释放连接
        con = self.__con()

        clo = re.findall(r"'([^']*)'",sql)
        sql = sql.lower()
        col = re.findall(r"'([^']*)'",sql)
        for i in range(len(clo)):
            sql = sql.replace(col[i] ,clo[i])

        if sql.find(".") > -1 or sql.find("all_") > -1:
            dfi = pd.read_sql_query(sql, con)
        else:
            col_list = sql.split("from ")
            if sql.find("where") > -1:
                col_where = col_list[1].split("where")
                owner = self.__qryname.loc[self.__qryname.table_name == col_where[0].lower().replace(" " ,"")].reset_index()['owner'].iloc[0].lower()
                sql = col_list[0] + "from " + owner + "." + col_where[0] + " where " + " " + col_where[1]
            else:           
                owner = self.__qryname.loc[self.__qryname.table_name == col_list[1].lower().replace(" " ,"")].reset_index()['owner'].iloc[0].lower()
                sql = col_list[0] + "from " + owner + "." + col_list[1]

            dfi = pd.read_sql_query(sql, con)
        return dfi

    def rownum(self, tb_name ,tb_req = None):
        """执行查询操作，返回查询的结果表df \n

        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        # 调用 query 方法时才创建连接，而非实例化MysqlTools时就创建连接。这样连接不会因为空闲过久而丢失
        # 函数调用完之后，会自动清理内部变量，释放连接
        con = self.__con()

        tb_name = tb_name.lower()
        if tb_name.find(".") > -1 or tb_name.find("all_") > -1:
            dfi = pd.read_sql_query(tb_name, con)
        else:
            owner = self.__qryname.loc[self.__qryname.table_name == tb_name.lower().replace(" " ,"")].reset_index()['owner'].iloc[0].lower()
            if isinstance(tb_req,int):
                sql = "select * from " + owner + "." + tb_name + f" where rownum <= {tb_req}"
            elif isinstance(tb_req,str):           
                sql = "select * from " + owner + "." + tb_name + f" where {tb_req.lower()}"
            else:           
                sql = "select * from " + owner + "." + tb_name
            dfi = pd.read_sql_query(sql, con)
        return dfi

    def to_sql(self, table_name, df, opt = None, chunksize=1000):
        """把df表中的数据存入数据库中、以table_name为名的表中。若此表已在库中，则会把数据追加在尾部。

        参数:
        ----------
        df: dataframe, 要写到数据库中的数据表
        table_name: str, 数据库中的写入目标表
        ----------------------------------------------
        opt :drop 删除该表
             clear 清空该表
        ------------------------------------
        chunksize: int, 每写入多少行就提交一次，算一次事务操作"""
        con = self.__con()
        table_name = table_name.lower()
        if opt == 'drop':
            dfi_drop = self.desc_table(table_name)
            self.exe(f"drop table {dfi_drop.owner.iloc[0]}.{table_name}")
        
        if opt == 'clear':
            dfi_drop = self.desc_table(table_name)
            self.exe(f"trucate table {dfi_drop.owner.iloc[0]}.{table_name}")

        if self.__qryname is None: self.__qryname = self.show_tables()
        if table_name in  self.__qryname.table_name.to_list():
            dfi = self.query(f"select * from ana_fx.{table_name} where rownum <= 1")
            cols = [i for i in df.columns.to_list() if i not in dfi.columns.to_list()]
            if len(cols):
                for i in cols:
                    self.exe(f"alter table {table_name} add {add_oracle_dtype(df[[i]])}")
                    time.sleep(5)  
                df = pd.concat([dfi,df])
                df.reset_index(inplace=True ,drop=True)
                df.drop(index=df.index[0],inplace=True)
                df.reset_index(inplace=True ,drop=True)
                
        df = df.rename_col({i:i.upper() for i in df})
        dfi_cols = df.cols_report()
        cols = dfi_cols.loc[dfi_cols.dtype == 'object64'].index.to_list()
        dtyp = {}
        for c in cols:
            if False in list(set(pd.isnull(df[c]).to_list())):
                dtyp[c] = types.VARCHAR(df[c].str.len().max() + 10)
            else:
                dtyp[c] = types.VARCHAR(15)
        if len(df) > 1000:
            for i in range(0,len(df),1000):
                if i == 0:
                    df.iloc[0:1000].to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize ,dtype = dtyp) 
                else:
                    if len(df) - i > 1000:
                        df.iloc[i:i+1000].to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize ,dtype = dtyp)
                    else:
                        df.iloc[i:i+len(df)].to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize ,dtype = dtyp)
        else:
            df.to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize ,dtype = dtyp)
            
        print( "数据写入Oracle成功 {}条".format(len(df)))
    
    def exe(self, sql ,opt = None ,condition = None ,out = True):
        """清空表、删除表、创建表等任意sql操作 \n

        参数:
        operate: drop 删除表
                 clear 清空表
                 delete 删除某些行
                 col_add 新增表字段
                 col_drop 删除字段
        condition:删除行、新增字段、删除字段的操作
        
        sql:数据表名
        ----------
        sql: str, 由字符串表达的数据库语句"""
        con = self.__con()
        if opt== 'drop':
            con.execute(f"drop table {sql}")
            b = self.query("""SELECT * FROM ALL_TAB_COMMENTS WHERE TABLE_NAME NOT LIKE '%$%' AND TABLE_TYPE = 'TABLE'
                              and OWNER NOT IN ('SYS','SYSTEM','MDSYS')""")
            b['table_name'] = b['table_name'].apply(lambda x: x.lower())
            self.__qryname = b.copy()
            print(f"数据表{sql} 已经删除")

        elif opt== 'clear':
            con.execute(f"truncate table {sql}")
            print(f"数据表{sql} 已经清空")

        elif opt== 'delete':
            con.execute(f"delete from {sql} where {condition}")
            print(f"数据表{sql} 数据已经删除")

        elif opt== 'col_add':
            con.execute(f"alter table {sql} add {condition}")
            print(f"数据表{sql} 已新增字段")
            
        elif opt== 'col_drop':
            con.execute(f"alter table {sql} drop {condition}")
            print(f"数据表{sql} 已新增字段")
        
        else:
            if sql.find(";"):
                col_m = sql.split(";")
                for i in col_m:
                    rc = re.compile(r'[A-Za-z]',re.S)
                    if bool(re.findall(rc ,i)):
                        con.execute(i)
                    else:
                        pass
            else:
                con.execute(sql)
            if out:
                print( "SQL 语句执行完毕")
        
    def add_unique_key(table_name ,unique_key):
        """
        为MySQL添加唯一主键索引
        参数：
        ------------------------
        table_name：表名
        unique_key：唯一主键的变量名称
        
        唯一主键的名称为  变量名称 + unique
        """
        con = self.__con()
        con.execute(f"alter table {table_name} add constraint {unique_key}_unique unique ({unique_key})")
        return "MySQL 唯一索引添加完毕"

    def add_query_key(table_name ,query_key ,length = False):
        """
        为MySQL添加普通的查询索引
        参数：
        ------------------------
        table_name：表名
        unique_key：查询主键的变量名称
        length：如果变量为str的时候需要为True ，给索引添加长度，否则会报错
        
        索引主键的名称为：idx + query_key
        """
        con = self.__con()
        if length:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key}(255))")
        else:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key})")
        return "MySQL 普通索引添加完毕"

    def show_tables(self ,patten = None):
        """返回数据库中的所有表"""
        b = self.query("""SELECT * FROM ALL_TAB_COMMENTS WHERE TABLE_NAME NOT LIKE '%$%'
                          and OWNER NOT IN ('SYS','SYSTEM','MDSYS')""")
        b['table_name'] = b['table_name'].apply(lambda x: x.lower())
        self.__qryname = b.copy()
        if patten is None:
            return b
        else:
            return b.rs_loc(patten)        
        
    def table_cols(self):
        """返回一个表的元数据信息"""
        df = self.query("""select owner ,table_name ,column_name ,comments from ALL_COL_COMMENTS WHERE TABLE_NAME NOT LIKE '%$%'
                          and OWNER NOT IN ('SYS','SYSTEM','MDSYS', 'WMSYS','EXFSYS')""")
        return df
     
    def desc_table(self, table_name ,base = False):
        """返回一个表的元数据信息"""
        a = self.query("select owner ,column_name ,data_type from ALL_TAB_COLS where table_name = '{}'".format(table_name.upper()))
        b = self.query("select owner ,column_name ,comments from ALL_COL_COMMENTS where table_name = '{}'".format(table_name.upper()))
        if len(a.owner.value_count()) == 1:
            dfi = a.merge(b[['column_name' ,'comments']] ,how = 'left' ,on = 'column_name')
            dfi['column_name'] = dfi['column_name'].apply(lambda x:x.lower())
            c = self.query(f"select * from {dfi.owner.iloc[0] + '.' +table_name} where rownum <= {len(dfi)}")
            dfi = dfi.merge(c ,how = 'left' ,left_index = True ,right_index = True)
            return dfi
        else:
            fm = a.owner.value_count()           
            owner = None
            var = 1
            while var == 1:
                if base:
                    print(f"在如下{len(fm)}个库分别存在该表：{' ,'.join(fm.colBins.to_list())}")
                    owner = input("请输入OWNER:")
                    if owner == '1' or owner == "": owner = 'SASDATA'
                else:
                    owner = 'SASDATA'                
                if owner != 'over':          
                    a = a.loc[a.owner == owner].copy()
                    b = b.loc[b.owner == owner].copy()
                    dfi = a.merge(b[['column_name' ,'comments']] ,how = 'left' ,on = 'column_name')
                    dfi['column_name'] = dfi['column_name'].apply(lambda x:x.lower())
                    c = self.query(f"select * from {owner + '.' +table_name} where rownum <= {len(dfi)}")
                    dfi = dfi.merge(c ,how = 'left' ,left_index = True ,right_index = True)
                    return dfi
                    
                else:
                    var = 2
                    print("结束查询")
                                        
    def add_comment(self ,table_name ,commt = None ,tab_commt = None):
        """
        ---------------------------------------------
        给MySQL字段新增注释：
        commt:是dataframe ,字段名称为 column_name:字段名称 ,comments:字段注释
        table_name:表名
        tab_commt:表的注释，如果非空的话 ，直接以该字段为表的注释
        ------------------------
        当commt为None时，可以手动输入每个字段的注释
        """
        if tab_commt is None:
            if isinstance(commt ,pd.DataFrame):
                dfield = self.desc_table(table_name)
                dfield = dfield.drop_col("comments")
                dfield = dfield.merge(commt ,how = 'left' ,on = 'column_name')
                for i in range(len(dfield)):
                    self.exe(f"comment on column {dfield.owner.iloc[0]}.{table_name}.{dfield.column_name.iloc[i]} is '{dfield.comments.iloc[i]}'",out=False)
                print(f"'{table_name}' 数据字段添加注释完毕，共计{len(dfield)} 个字段")
            
            elif isinstance(commt ,str):
                tp = input(f"{commt}的注释为：")
                if tp != 'over' and tp != 'pass':
                    self.exe(f"comment on column {table_name}.{commt} is '{tp}'" ,out=False)
                    print(f"'{table_name}' 数据字段添加注释完毕，共计1 个字段") 
            else:
                dfield = self.desc_table(table_name)
                for i in range(len(dfield)):
                    tp = input(f"{dfield.column_name.iloc[i]}的注释为：")
                    if tp != 'over' and tp != 'pass' and tp != 'p':
                        self.exe(f"comment on column {dfield.owner.iloc[0]}.{table_name}.{dfield.column_name.iloc[i]} is '{tp}'")
                        print(f"'{table_name}' 数据字段添加注释完毕，共计{i+1} 个字段") 
                    elif tp == 'pass' or tp == 'p':
                        print(f"'{table_name}' 数据字段未添加注释")
                    else:
                        print(f"'{table_name}' 数据字段添加注释完毕，共计{i} 个字段")
                        break
        else:
            dfield = self.desc_table(table_name)
            self.exe(f"comment on table {dfield.owner.iloc[0]}.{table_name} is '{tab_commt}'")
            print(f"'{table_name}' 数据表添加注释完毕") 
                              
		
class VerticaTools(object):
    __doc__ = """
    创建连接vertica的不同数据库的引擎, 方便读写数据库。\n
    
    参数:
    ----------
    name: str, 
        数据库连接名称。查看已配置了哪些数据库连接：
        sql_server:数据库唯一标识
	-------------------------------------------------------------------
	数据表的查询不以schema为限制，可以随便查询
	数据存储在普惠部门专门的数据库中
    """
    def __init__(self ,sql_server):
        self.__connect_dict = SQL_DICT
        assert sql_server in self.__connect_dict, 'Unknown host'
        self.Name = sql_server
        self.__tb_name = None
        
    def __con(self):
        """定义专门的私有方法，用于创建引擎"""
        url = f"{self.__connect_dict[self.Name]}"
        return create_engine(url)

    def query(self, table_sql ,limit = 0):
        """执行查询操作，返回查询的结果表df \n

        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        # 调用 query 方法时才创建连接，而非实例化MysqlTools时就创建连接。这样连接不会因为空闲过久而丢失
        # 函数调用完之后，会自动清理内部变量，释放连接
        con = self.__con()
        if table_sql.find('select') > -1:
            sql = table_sql
        else:
            if table_sql.lower().startswith('a'):
                sql = f"select * from ADMDALIB_SG.{table_sql}"
            elif table_sql.lower().startswith('f'):
                sql = f"select * from FDMDALIB_SG.{table_sql}"
            elif table_sql.lower().startswith('m'):
                table_sql = f"select * from MDMDALIB_SG.{table_sql}"
            elif table_sql.lower().startswith('o'):
                sql = f"select * from ODMDALIB_SG.{table_sql}"
            else:
                sql = f"select * from SASPHLIB.{table_sql}"
        if limit > 0:
            sql = sql + ' limit ' + str(limit)
        dfi = pd.read_sql_query(sql ,con)
        dfi = dfi.rename_col({i:i.lower() for i in dfi})
        return dfi
    
    def to_sql(self, table_name, df, index=False):
        '''
        把df表中的数据存入数据库 vertica 中、以 table_name 为名的表中。
        必须提前建好表，才能使用此方法写入数据。且写入目标表的列，与df的列同名。
        利用 cursor.copy 实现的写入操作，因此需要先把 df 落地为磁盘上的临时 csv 文件。
        目前只实现了对 连接名的写入方法。
		
		其中需要针对数据的类型做一些调整，字符串类型的转化为varchar类型
        参数:
        ----------
        df: dataframe, 数据表
        table_name: str, sasphlib 模式中的目标表
		
        '''
        if self.Name == 'vertica':
            con = vp.connect(user = 'sasphqry' ,password = "Sasphqry@123" ,host = '172.16.83.54' ,port = '5433')
            conn = self.__con()
        else:
            raise NotImplementedError('其他连接暂未实现本方法')
            
        df.reset_index(drop=True ,inplace=True)
        dfi_cols = df.cols_report()
        cols = dfi_cols.loc[dfi_cols.dtype == 'object64'].index.to_list()
        dtyp = {}
        for c in cols:
            if False in list(set(pd.isnull(df[c]).to_list())):
                def is_chinese(fstr):
                    tmp = []
                    for km in fstr:
                        if u'\u4e00'<= km <= u'\u9fff':
                            tmp.append(True)
                        else:
                            tmp.append(False)
                    if True in list(set(tmp)):
                        return True
                    else:
                        return False
                leng = df[c].str.len().max()
                lfm = df[c].str.len().to_list()
                k_max = lfm.index(leng)
                if is_chinese(df[c].iloc[k_max]):
                    dtyp[c] = types.VARCHAR(leng*10)
                else:
                    dtyp[c] = types.VARCHAR(leng + 10)
            else:
                dtyp[c] = types.VARCHAR(15)
        warnings.filterwarnings('ignore')
        df.iloc[0:100].to_sql(table_name, conn, schema = 'SASPHLIB', if_exists='append', index=False ,dtype = dtyp)       
        con.autocommit = True
        cur = con.cursor()
        cur.execute("set search_path to 'SASPHLIB'")
        df.iloc[100:].to_csv(r'tmp_for_write_to_vt.csv', index=index, sep=',', header=False)
        fs =  open(r'tmp_for_write_to_vt.csv', 'r', encoding='utf8')        
        cur.copy(f"copy SASPHLIB.{table_name} from stdin delimiter ',' enclosed by '\'" ,fs ,buffer_size = 65536) 
        fs.close()
        os.remove(r'tmp_for_write_to_vt.csv')
        print('{1} rows writed to sasphlib.{0}, done!'.format(table_name, len(df)))

    def exe(self, sql):
        """清空表、删除表、创建表等任意sql操作 \n

        参数:
        ----------
        sql: str, 由字符串表达的数据库语句"""
        con = self.__con()
        con.execute(sql)       
        return "SQL 语句执行完毕"
        
    def add_unique_key(table_name ,unique_key):
        """
        为MySQL添加唯一主键索引
        参数：
        ------------------------
        table_name：表名
        unique_key：唯一主键的变量名称
        
        唯一主键的名称为  变量名称 + unique
        """
        con = self.__con()
        con.execute(f"alter table {table_name} add constraint {unique_key}_unique unique ({unique_key})")
        print( "MySQL 唯一索引添加完毕")

    def add_query_key(table_name ,query_key ,length = False):
        """
        为MySQL添加普通的查询索引
        参数：
        ------------------------
        table_name：表名
        unique_key：查询主键的变量名称
        length：如果变量为str的时候需要为True ，给索引添加长度，否则会报错
        
        索引主键的名称为：idx + query_key
        """
        con = self.__con()
        if length:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key}(255))")
        else:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key})")
        print("MySQL 普通索引添加完毕")

    def show_tables(self ,patten = None ,inplace = False):
        """
        返回数据库中的所有表
        patten: 正则表达式中的需要查找的表名的一部分
        inplace: 是否进行新的查询，一般没有新增表，无需此操作
        """
        if self.__tb_name is None or inplace == True:
            self.__tb_name = self.query("""select  table_name , table_schema from tables""")
            self.__tb_name['table_name'] = self.__tb_name['table_name'].apply(lambda x: x.lower())
            if patten is None:
                return self.__tb_name.reset_index(drop=True)
            else:
                return self.__tb_name.rs_loc(patten).reset_index(drop=True)
        else:            
            if patten is None:
                return self.__tb_name.reset_index(drop=True)
            else:
                return self.__tb_name.rs_loc(patten).reset_index(drop=True)

    def desc_table(self, table_name):
        """返回一个表的元数据信息"""
        dfi = self.query("""
        select anchor_table_name as table_name, anchor_table_column_name as column_name,anchor_table_schema as table_schema
        from column_storage where anchor_table_name = '{}'""".format(table_name.upper())).drop_duplicates()
        dfi = dfi.loc[pd.notnull(dfi.column_name)].copy().reset_index(drop=True)
        dfi['column_name'] = dfi['column_name'].apply(lambda x:x.lower() if x is not None else None)
        c = self.query(f"select * from {dfi.table_schema.iloc[0]}.{table_name} limit {len(dfi)}")
        dfi = dfi.merge(c ,how = 'left' ,left_index = True ,right_index = True)
        dfi = dfi.rename_col({i:i.lower() for i in dfi})
        return dfi
		
class HiveTools(object):
    __doc__ = """创建连接Hive的不同数据库的引擎, 方便读写数据库。\n
    
    参数:
    ----------
    name: str, 
        数据库连接名称。查看已配置了哪些数据库连接：
        MysqlTools.print_connects()
    
        数据库连接可在 config 模块中配置 \n
    """
    def __init__(self, sql_server, database):
        # config.CONNECT_DICT 在实例化时才会被调用，实时获取CONNECT_DICT的值。
        # 若把此函数体的第一行赋值移到函数外，则在编译时CONNECT_DICT的值就已传了进来，
        # CONNECT_DICT就相当于 MysqlTools 的闭包常数，起不到动态配置作用。
        
        # 此函数编译的字节码，仅定义了调用config.CONNECT_DICT，在实例化时，才传入config.CONNECT_DICT
        # 的真实值。这样，config.CONNECT_DICT做了修改后，会反映到MysqlTools上来
        self.database = database
        self.__connect_dict = SQL_DICT
        assert sql_server in self.__connect_dict, 'Unknown host'
        self.Name = sql_server
        
    def __con(self):
        """定义专门的私有方法，用于创建引擎"""
        url = f"{self.__connect_dict[self.Name]}{self.database}?auth=LDAP"
        return create_engine(url)

    def query(self, sql):
        """执行查询操作，返回查询的结果表df \n

        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        # 调用 query 方法时才创建连接，而非实例化MysqlTools时就创建连接。这样连接不会因为空闲过久而丢失
        # 函数调用完之后，会自动清理内部变量，释放连接
        con = self.__con()  
        dfi = pd.read_sql_query(sql, con)
        
        return dfi

    def to_sql(self ,table_name ,df ,drop = None ,partition = None):
        """把df表中的数据存入数据库中、以table_name为名的表中。若此表已在库中，则会把数据追加在尾部。

        参数:
        ----------
        df: dataframe, 要写到数据库中的数据表
        table_name: str, 数据库中的写入目标表
        chunksize: int, 每写入多少行就提交一次，算一次事务操作
        external: 是否是外部表 ，如果是外部表 external = external
        """
        database = self.database
        #将时间戳 转化为可以使用的字符串 '20220328175658'
        try:
            describe = self.query(f"describe {table_name}")
            cols_type = describe.data_type.to_list()
            cols_colm = describe.col_name.to_list()
            for i in range(len(cols_type)):
                if cols_type[i].lower().find('double') > -1:
                    df[cols_colm[i]] = df[cols_colm[i]].astype(float)
                elif cols_type[i].lower().find('string') > -1:
                    df[cols_colm[i]] = df[cols_colm[i]].astype(str)
                else:
                    pass
        except Exception:
            pass
        str_date = str(datetime.now()).replace('-','').replace(' ','').replace(':','').replace('.','')[:14]
        if isinstance(drop ,(str,int,float,list)):df.drop_col(drop ,inplace=True)
        if sys.platform == 'linux':
            df.to_csv(f'/data/result_data/pickle/hive{str_date}.txt', sep='\t' ,header=False ,index=False)
        elif sys.platform == 'win32':
            df.to_csv(f'D:/Result_data/pickle/hive{str_date}.txt', sep='\t' ,header=False ,index=False)
        else:
            pass
        if isinstance(partition,str):partition = [partition]
        if isinstance(partition,list):
            txt_sql = f'''create  table if not exists {table_name + '_ext'} ({infer_hive_dtype(df)})
              partitioned by ({infer_hive_dtype(df[partition])})
              row format delimited fields terminated by "\t"
              lines terminated by "\n" stored as textfile tblproperties("skip.header.line.count" = "1") '''
            orc_sql = f'''create  table if not exists {table_name} ({infer_hive_dtype(df)})
              partitioned by ({infer_hive_dtype(df[partition])})
              row format delimited fields terminated by "\t"
              lines terminated by "\n" stored as orc tblproperties("skip.header.line.count" = "1") '''
        else:
            txt_sql = f'''create table if not exists {table_name + '_ext'} ({infer_hive_dtype(df)}) 
              row format delimited fields terminated by "\t"
              lines terminated by "\n" stored as textfile tblproperties("skip.header.line.count" = "1") '''
            orc_sql = f'''create table if not exists {table_name} ({infer_hive_dtype(df)})
              row format delimited fields terminated by "\t"
              lines terminated by "\n" stored as orc tblproperties("skip.header.line.count" = "1") '''
        self.exe(txt_sql)
        self.exe(orc_sql)
        if sys.platform == 'linux':
            self.exe(f"load data local inpath '/data/result_data/pickle/hive{str_date}.txt' into table {table_name + '_ext'}")
            #调用本地命令  删除 TXT文件
            os.system(f"rm -rf /data/result_data/pickle/hive{str_date}.txt")
        elif sys.platform == 'win32':
            self.exe(f"load data local inpath 'D:/Result_data/pickle/hive{str_date}.txt' into table {table_name + '_ext'}")
            #调用本地命令  删除 TXT文件
            os.system(f"rm -rf D:/Result_data/pickle/hive{str_date}.txt")
        else:
            pass
            
        
    def exe(self, sql):
        """清空表、删除表、创建表等任意sql操作 \n

        参数:
        ----------
        sql: str, 由字符串表达的数据库语句"""
        con = self.__con()
        con.execute(sql)
        
        return "SQL 语句执行完毕"
        
    def add_unique_key(table_name ,unique_key):
        """
        为MySQL增加唯一主键索引
        参数：
        -------------------
        table_name：表名
        unique_key：唯一主键的变量名称 
        
        唯一主键的名称为  变量名称+unique
        
        """
        con = self.__con()
        con.execute(f"alter table {table_name} add constraint {unique_key}_unique unique({unique_key}) ")
        
        return "MySQL 唯一索引添加完毕"
        
    def add_query_key(table_name ,query_key ,length = False):
        """
        为MySQL添加 普通的查询索引
        参数：
        ------------------------------------
        table_name：表名
        query_key：查询主键的变量名称
        length：如果变量是str类型的就需要为 True ，给索引添加长度，否则会报错
        
        查询主键的名字为 idx + query_key
        """
        con = self.__con()
        if length:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key}(255))")
        else:
            con.execute(f"alter table {table_name} add index idx_{query_key} ({query_key}) ")
            
        return "MySQL 普通索引添加完毕"

    def show_tables(self):
        """返回数据库中的所有表"""
        b = self.query('show tables')
        return b

    def desc_table(self, table_name):
        """返回一个表的元数据信息"""
        b = self.query('describe {}'.format(table_name))
        b.Field = b.Field.str.lower()
        return b
        
        
class PrestoTools(object):
    __doc__ = """创建连接Hive的不同数据库的引擎, 方便读写数据库。\n
    
    参数:
    ----------
    name: str, 
        数据库连接名称。查看已配置了哪些数据库连接：
        MysqlTools.print_connects()
    
        sql_server:数据库唯一标识
        schema:presto数据库的schema
        database:presto数据库
    """
    def __init__(self ,sql_server ,schema ,database):
        # config.CONNECT_DICT 在实例化时才会被调用，实时获取CONNECT_DICT的值。
        # 若把此函数体的第一行赋值移到函数外，则在编译时CONNECT_DICT的值就已传了进来，
        # CONNECT_DICT就相当于 MysqlTools 的闭包常数，起不到动态配置作用。
        
        # 此函数编译的字节码，仅定义了调用config.CONNECT_DICT，在实例化时，才传入config.CONNECT_DICT
        # 的真实值。这样，config.CONNECT_DICT做了修改后，会反映到MysqlTools上来
        self.schema = schema
        self.database = database
        self.__connect_dict = SQL_DICT
        assert sql_server in self.__connect_dict, 'Unknown host'
        self.Name = sql_server
        
    def __con(self):
        """定义专门的私有方法，用于创建引擎"""
        url = f"{self.__connect_dict[self.Name]}{self.schema}/{self.database}"
        return create_engine(url)

    def query(self, sql):
        """执行查询操作，返回查询的结果表df \n

        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        # 调用 query 方法时才创建连接，而非实例化MysqlTools时就创建连接。这样连接不会因为空闲过久而丢失
        # 函数调用完之后，会自动清理内部变量，释放连接
        con = self.__con()
        dfi = pd.read_sql_query(sql, con)
        
        return dfi
    
    def to_sql(self ,table_name ,df ,mysql_server ='29' ,hive_server = 'hive' , mysql_base = 'data_format',sleep = 5 ,drop = None ,partition = None):
        """
        参数：
        ----------------------
        sql_server: MySQL 或者 HIve 的服务器获取名称 ，例如 '29' 、 'hive'
        df: dataframe 
        table_name:数据表名称
        partition: hive中分区表字段 ，可以是 str  和  list，暂时未深入研究
        sleep : 睡眠时间，因为读取大额数据的时候需要长时间的睡眠才能完成最后的操作，建议一般30秒以上
        drop ：需要删除的变量
        """
        if isinstance(drop ,(str,int,float,list)):df.drop_col(drop ,inplace=True)
        if self.schema == 'mysql':
            df = df.rename_col({i: i.lower() for i in df})
            mysql_tosql = Mysql2Tools(mysql_server ,self.database)
            mysql_tosql.to_sql(df = df,table_name = table_name)
        elif self.schema == 'hive':
            mysql_tosql = Mysql2Tools(mysql_server ,mysql_base)
            hive_tosql = HiveTools(hive_server ,self.database)
            df = df.rename_col({i: i.lower() for i in df})
            try:
                dfi = self.query(f"describe {table_name}")
                cols_type = dfi.Type.to_list()
                cols_colm = dfi.Column.to_list()
                df_cols = df.columns.to_list()
                for i in range(len(cols_type)):
                    if cols_type[i].lower().find('int') > -1:
                        if cols_colm[i] in df_cols: df.loc[:,cols_colm[i]] = df[cols_colm[i]].astype(int)                          
                    elif cols_type[i].lower().find('double') > -1 or cols_type[i].lower().find('float') > -1:
                        if cols_colm[i] in df_cols: df.loc[:,cols_colm[i]] = df[cols_colm[i]].astype(float)
                    elif cols_type[i].lower().find('varchar') > -1:
                        if cols_colm[i] in df_cols: df.loc[:,cols_colm[i]] = df[cols_colm[i]].apply(lambda x:None if pd.isnull(x) else str(x))
                    elif cols_type[i].lower().find('timestamp') > -1:
                        if cols_colm[i] in df_cols: 
                            df[cols_colm[i]] = df[cols_colm[i]].get_datetime()
                    elif cols_type[i].lower() == 'date':
                        if cols_colm[i] in df_cols: 
                            df[cols_colm[i]] = df[cols_colm[i]].get_date()                       
                    else:
                        pass
                mysql_tosql.to_sql(table_name + '_hive' ,df)
                df_type = mysql_tosql.query(f"describe {table_name + '_hive'}")
                df_type['Type'] = df_type.Type.apply(lambda x: 'varchar(65535)' if x == 'text' else x)
                df_type['Type'] = df_type.Type.apply(lambda x: 'timestamp' if x == 'datetime' else x)
            except Exception:           
                mysql_tosql.to_sql(table_name + '_hive' ,df)
                df_type = mysql_tosql.query(f"describe {table_name + '_hive'}")
                df_type['Type'] = df_type.Type.apply(lambda x: 'varchar(65535)' if x == 'text' else x)
                df_type['Type'] = df_type.Type.apply(lambda x: 'timestamp' if x == 'datetime' else x)
                cols_value = ' ,'.join("{0} {1}".format(df_type.Field.iloc[i] ,df_type.Type.iloc[i]) for i in range(len(df_type)))               
                hive_tosql.exe(f"""create table if not exists {table_name} ({cols_value})
                  row format delimited fields terminated by "\t"
                  lines terminated by "\n" stored as orc tblproperties("skip.header.line.count" = "1") """)
                dfi = self.query(f"describe {table_name}")

            #保证插入的数据列数与HIVE中的列数一致
            dfi_com = dfi.Column.to_list()
            dfi_col = dfi.Column.to_list()       
            dft_col = df_type.Field.to_list()
            dfi_typ = dfi.Type.to_list()
            dft_typ = df_type.Type.to_list()
            for i in range(len(dft_col)):
                if dft_col[i] not in dfi_com:
                    if i == 0:
                        dfi_com.insert(i ,'null as ' + dft_col[i])
                    else:
                        if dft_col[i-1] in dfi_com:
                            dfi_com.insert(dfi_com.index(dft_col[i-1])+1,'null as ' + dft_col[i])
                        else:
                            dfi_com.append('null as ' + dft_col[i])
                else:
                    pass
            dfi_com = ['date({0})'.format(dfi_col[i]) if dfi_typ[i] == 'date' else dfi_col[i]  for i in range(len(dfi_col))]
            cols_q = ' ,'.join("{0}".format(key) for key in dfi_com)   
            cols_fm = [i for i in df_type.Field.to_list() if i not in dfi.Column.to_list()]
            if cols_fm:
                """
                新增一列 ，在HIVE中的操作
                """
                dfi_col = dfi.Column.to_list()
                dft_col = df_type.Field.to_list()
                dfi_typ = dfi.Type.to_list()
                dft_typ = df_type.Type.to_list()
                for i in range(len(dft_col)):
                    if dft_col[i] not in dfi_col:
                        if i == 0:
                            dfi_col.insert(i,dft_col[i])
                            dfi_typ.insert(i,dft_typ[i])
                        else:
                            if dft_col[i-1] in dfi_col:
                                dfi_col.insert(dfi_col.index(dft_col[i-1])+1,dft_col[i])
                                dfi_typ.insert(dfi_typ.index(dft_typ[i-1])+1,dft_typ[i])
                            else:
                                dfi_col.append(dft_col[i])
                                dfi_typ.append(dft_typ[i])                                
                    else:
                        pass                  
                cols_value = ' ,'.join("{0} {1}".format(dfi_col[i] ,dfi_typ[i]) for i in range(len(dfi_col)))  
                self.exe(f"create table if not exists {table_name + '_add'} as select * from {table_name}")
                time.sleep(30)
                self.exe(f"drop table {table_name}")
                hive_tosql.exe(f"""create table if not exists {table_name} ({cols_value})
                              row format delimited fields terminated by "\t"
                              lines terminated by "\n" stored as orc tblproperties("skip.header.line.count" = "1") """)
                self.exe(f"insert into {table_name} select {cols_q} from mysql.{mysql_base}.{table_name + '_hive'}")
                time.sleep(3)
                self.exe(f"insert into {table_name} select {cols_q} from {table_name + '_add'}")
                time.sleep(45)
                self.exe(f"drop table {table_name + '_add'}")
                self.exe(f"drop table mysql.{mysql_base}.{table_name + '_hive'}")
            else:
                self.exe(f"insert into {table_name} select {cols_q} from mysql.{mysql_base}.{table_name + '_hive'}")
                time.sleep(sleep)
                self.exe(f"drop table mysql.{mysql_base}.{table_name + '_hive'}")
        
            
            
    def exe(self, sql):
        """清空表、删除表、创建表等任意sql操作 \n

        参数:
        ----------
        sql: str, 由字符串表达的数据库语句"""
        con = self.__con()
        con.execute(sql)
        
        return "SQL 语句执行完毕"

    def show_tables(self):
        """返回数据库中的所有表"""
        b = self.query('show tables')
        return b

    def desc_table(self, table_name):
        """返回一个表的元数据信息"""
        b = self.query('describe {}'.format(table_name))
        b.Field = b.Field.str.lower()
        return b


class PosgresTools:
    __doc__ = """创建连接 Postgresql 的不同数据库的引擎, 方便读写数据库。\n
    
        参数:
        -----
        name: str, 
            数据库连接名称。查看已配置了哪些数据库连接：PosgresTools.print_connects()
            数据库连接可在 config 模块中配置 \n
        """

    @classmethod
    def print_connects(cls):
        """打印出已配置的所有数据库连接名"""
        print(list(config.PG_DICT.keys()))

    def __init__(self, name):
        self.__connect_dict = config.PG_DICT
        assert name in self.__connect_dict, 'Unknown host'
        self.Name = name
        if name == 'pg_ana':
            self.schema = 'xa_ana'

    def __con(self):
        """定义专门的私有方法，用于创建引擎"""
        url = self.__connect_dict[self.Name]
        con = create_engine(url)
        return con

    def query(self, sql):
        """执行查询操作，返回查询的结果表df. 要查询 xa_ana 库里面的表，需要给表名加前缀 xa_ana.table_name \n

        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        # 调用 query 方法时才创建连接，而非实例化 MysqlTools 时就创建连接。这样连接不会因为空闲过久而丢失
        # 函数调用完之后，会自动清理内部变量，释放连接
        con = self.__con()
        return pd.read_sql_query(sql, con)
    

    def to_sql(self, df, table_name, index=False):
        '''
        把df表中的数据存入数据库 postgres.xa_ana 中、以 table_name 为名的表中。
        必须提前建好表，才能使用此方法写入数据。且写入目标表的列，与df的列同名。
        利用 cursor.copy_from 实现的写入操作，因此需要先把 df 落地为磁盘上的临时 csv 文件。
        目前只实现了对 pg_risk 连接名的写入方法。

        参数:
        ----------
        df: dataframe, 数据表
        table_name: str, xa_ana 模式中的目标表
        '''
#        if self.Name == 'pg_risk':
#            con = connect(host="10.10.0.233", user="risk_dev", password="G8XpjZ4rDeRN8dRwM", database="postgres")
#        else:
#            raise NotImplementedError('其他连接暂未实现本方法')
#        con.autocommit = True
#        cur = con.cursor()
#        cur.execute("set search_path to 'xa_ana'")
#        df.to_csv(r'tmp_for_write_to_pg.csv', index=index, sep=',', header=False)
#        file = open(r'tmp_for_write_to_pg.csv', 'r', encoding='utf8')
        # 需要声明空用""表示，否则会用默认的\N表示
#        cur.copy_from(file, table_name, sep=',', columns=df.columns, null="") 
#        print('{1} rows writed to xa_ana.{0}, done!'.format(table_name, len(df)))
        
        # 下面的方式可以正确写入数据，但极缓慢，暂未找到原因
        # df.to_sql(table_name, con, if_exists='append', index=False, chunksize=chunksize, schema='xa_ana')

    
    def create_table(self, table_name, table_feild , table_key):
        """创建空数据表，把数据写入xa_ana

        参数:
        -----------
        table_name：  表名
        table_feild:  表列名，通常使用infer_pgsql_dtype取的字段作为入参
        table_key: 表的约束以及分布式字段"""
        con = self.__con() 
        table_feild = infer_pgsql_dtype(table_feild)
        table_list = table_feild.split(' ')
        nPos = table_list.index('{}'.format(table_key))+1
        if nPos == len(table_list)-1:
            table_list[nPos] = table_list[nPos][:] + ' primary key'
        else:
            table_list[nPos] = table_list[nPos][:-1] + ' primary key,'
        table_feild = " ".join(table_list)
        sql = 'create table if not exists xa_ana.{} ({}) distributed by ({})'.format(table_name ,table_feild ,table_key)
        try:
            return con.execute(sql)
        except:
            print('table xa_ana.{} is been writen down,there is {} feild'.format(table_name ,len(table_feild) - len(table_feild.replace(',','')) + 1))
            
    def add_comment(self, table_name, comm_list, comm_col, comm=list()):
        """
        给指定的表及字段添加注释。需要将打印出来的代码放在SQL编译器中执行。

        参数:
        -----------
        table_name:数据表名
        comm_list:以数据表列名为index的dataframe 对应：comm_list
        comm_col:与index对应的描述字段  对应：'comment'

        示例：
        -----------
        comm_list = pd.DataFrame({'comment':''},index=data_target.columns)
        m = ['申请编号','目标变量','连接主键','报告编号','身份证号码','客户编号','客户职业类型','人行报告查询时间','人行报告创建时间']
        comm_list['comment'] = m
        """
        for i in comm_list.index:
            com = "comment on column xa_ana.{}.{} is '{}'".format(table_name ,i ,comm_list[comm_col].loc[i])
            con = self.__con()
            con.exe(com)
            comm.append(com)
        print(' ;\n'.join(comm))
        comm.clear()
            
    def drop_table(self, table_name):
        """
        删除数据表，xa_ana数据库的数据表。通常两个作用：
            1、删除表，再重新建表
            2、清理数据库里的表

        参数:
        ---------
        table_name: str, 要删除的表名
        """
        con = self.__con() 
        sql = 'drop table if exists xa_ana.{} '.format(table_name)
        try:
            return con.execute(sql)
        except:
            print('table xa_ana.{} is been dropped down'.format(table_name))
            
    def exe(self, sql):
        """清空表、删除表、创建表等任意sql操作 \n

        参数:
        ----------
        sql: str, 由字符串表达的数据库语句"""
        con = self.__con()
        return con.execute(sql)

    def show_tables(self, schema='public'):
        """返回数据库中的所有表

        参数:
        -----------
        schema: str, Postgress数据库中的模式名，指定查看哪个模式中的表名清单"""
        b = self.query("SELECT table_name FROM information_schema.tables WHERE table_schema = '{}'".format(schema))
        return b

    def desc_table(self, table_name, schema='public'):
        """返回一个表的元数据信息

        参数:
        -----------
        table_name: str, 表的名字
        schema: str, Postgress数据库中的模式名，要查看的表位于哪个模式中"""
        b = self.query("""SELECT "table_name", "column_name", ordinal_position, column_default, is_nullable, data_type
            FROM information_schema.columns 
            WHERE table_name ='{0}' and table_schema='{1}' """.format(table_name, schema))
        b.column_name = b.column_name.str.lower()
        return b

		
def infer_hive_dtype(df, print_flag=False):
    """根据列的抽样数据，推断各列的 hive 数据类型, 以便生成建表代码 \n

    参数:
    ----------
    df:  需要推断的数据框 \n
    print_flag: bool, 是否打印出打断的结果字符串，如果为Ture, 就打印字符串并返回None；如果为False，则不打印但返回字符串
    
    """

    df_dtype = df.dtypes
    column_types = pd.Series()
    for col in df_dtype.index:
        if df_dtype[col] == 'int64' or df_dtype[col] == 'float64':
            column_types[col] = 'DOUBLE'
        else:
            column_types[col] = 'STRING'

    a = ' '.join("{0} {1},".format(key, column_types[key]) for key in df)
    if print_flag:
        return print(a[:-1])
    else:
        return a[:-1]
                
def infer_mysql_dtype(df, frac=0.3, print_flag=False):
    """根据列的抽样数据，推断各列的mysql数据类型, 以便生成建表代码 \n

    参数:
    ----------
    df:  需要推断的数据框 \n
    frac: float, 抽样比例。推断将在按此比例抽样的子表数据上进行 \n
    print_flag: bool, 是否打印出打断的结果字符串，如果为Ture, 就打印字符串并返回None；如果为False，则不打印但返回字符串"""
    known_col = config.MYSQL_COL_DTYPE

    def real_int(sr):
        """推测有空值的float64类型的列，是否实际上是int类型。"""
        logic = sr.isnull()
        if logic.all():
            return False
        else:
            sr = sr[~logic]
            diff = (sr - sr.map(int)).abs()
            return (diff == 0).all()

    column_types = pd.Series()
    dtypes = df.dtypes
    df_infer = df.sample(frac=frac)
    for col in dtypes.index:
        if col in known_col:
            column_types[col] = known_col[col]
        else:
            dtype = dtypes[col]
            if np.issubdtype(dtype, np.integer):
                column_types[col] = 'int'
            elif np.issubdtype(dtype, np.floating):
                column_types[col] = 'int' if real_int(df_infer[col]) else 'double'
            elif np.issubdtype(dtype, np.datetime64):
                column_types[col] = 'datetime'
            elif dtype == np.dtype('bool'):
                column_types[col] = 'bool'
            else:  # dtype('O')
                if col.lower().find('date') > -1:
                    column_types[col] = 'date'
                elif col.lower().find('time') > -1:
                    column_types[col] = 'datetime'
                else:
                    char_len = df[col].map(lambda x: len(x) if pd.notnull(x) else 0).max()
                    column_types[col] = 'varchar({})'.format(char_len)  # 最大长度是基于所有样本数据计算得到的
    a = ' '.join("{0} {1},".format(key, column_types[key]) for key in df)
    if print_flag:
        print(a + ' max_id int primary key auto_increment')
    else:
        return a + ' auto_id int primary key auto_increment'


def infer_pgsql_dtype(df, frac=0.3, print_flag=True):
    """根据列的抽样数据，推断各列的 pgsql 数据类型, 以便生成建表代码 \n

    参数:
    ----------
    df:  需要推断的数据框 \n
    frac: float, 抽样比例。推断将在按此比例抽样的子表数据上进行 \n
    print_flag: bool, 是否打印出打断的结果字符串，如果为Ture, 就打印字符串并返回None；如果为False，则不打印但返回字符串"""
    known_col = config.MYSQL_COL_DTYPE

    def real_int(sr):
        """推测有空值的float64类型的列，是否实际上是int类型。"""
        logic = sr.isnull()
        if logic.all():
            return False
        else:
            sr = sr[~logic]
            diff = (sr - sr.map(int)).abs()
            return (diff == 0).all()

    column_types = pd.Series()
    dtypes = df.dtypes
    df_infer = df.sample(frac=frac)
    for col in dtypes.index:
        if col in known_col:
            column_types[col] = known_col[col]
        else:
            dtype = dtypes[col]
            if np.issubdtype(dtype, np.integer):
                column_types[col] = 'float'
            elif np.issubdtype(dtype, np.floating):
                column_types[col] = 'float' if real_int(df_infer[col]) else 'float'
            elif np.issubdtype(dtype, np.datetime64):
                column_types[col] = 'date'
            elif dtype == np.dtype('bool'):
                column_types[col] = 'bool'
            else:  # dtype('O')
                if col.lower().find('date') > -1:
                    column_types[col] = 'date'
                elif col.lower().find('time') > -1:
                    column_types[col] = 'date'
                else:
                    char_len = df[col].map(lambda x: len(x) if pd.notnull(x) else 0).max()
                    column_types[col] = 'varchar({})'.format(char_len)  # 最大长度是基于所有样本数据计算得到的
    a = ' '.join("{0} {1},".format(key, column_types[key]) for key in df)
    if print_flag:
        return a[:-1]
    else:
        return a
		
		
def add_sql_dtype(df, print_flag=False):
    """根据列的抽样数据，推断各列的 hive 数据类型, 以便生成建表代码 \n

    参数:
    ----------
    df:  需要推断的数据框 \n
    print_flag: bool, 是否打印出打断的结果字符串，如果为Ture, 就打印字符串并返回None；如果为False，则不打印但返回字符串
    
    """
    df_dtype = df.dtypes
    column_types = pd.Series()
    for col in df_dtype.index:
        if df_dtype[col] == 'float64':
            column_types[col] = 'double'
        elif df_dtype[col] == 'int64':
            column_types[col] = 'int'
        elif df_dtype[col] == 'object':
            if isinstance(df.reset_index()[col].iloc[0],date):
                column_types[col] = 'date'
            else:
                column_types[col] = 'varchar(255)'
        else:
            column_types[col] = 'datetime'
    a = ' '.join("{0} {1},".format(key, column_types[key]) for key in df)
    if print_flag:
        return print(a[:-1])
    else:
        return a[:-1]
