# -*- coding: utf-8 -*-
# @Author  : zhangjiatao
# @Time    : 2024/2/4 12:17
# @Function: excel格式化函数执行文件
from excel_func4 import find_target_positions,find_empty_cell,set_color_scale,excel_value_format,highlight_max_values
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import pandas as pd
import re

# path = 'D:/shortcut/临时文件/926临时文件/excel_测试/'
# files = ['元寅顺和同信测试样本5万_MD5_个人物流画像2.0结果_result.xlsx']
# filename = path + files[0]
# store_path =  'D:/shortcut/临时文件/926临时文件/excel_测试/'


# path = 'D:/shortcut/2024临时文件/0131临时文件/'
# files = ['color_test_origin.xlsx']
# filename = path + files[0]
# store_path =  path

def reset_dataframe(df):
    # 创建一个新列，表示当前行与上一行varname是否相同
    df['varname_diff'] = df['varname'].ne(df['varname'].shift())

    # 将varname相同的行标记为False，不同的行标记为True
    df['varname_diff'] = df['varname_diff'].fillna(True)
    df2 = df.copy()  # make a copy because we want to be safe here
    for i in df.loc[df['varname_diff']].index:
        empty_row = pd.DataFrame([], index=[i])  # creating the empty data
        df2 = pd.concat([df2.loc[:i - 1], empty_row, df2.loc[i:]])  # slicing the df
    df2 = df2.reset_index(drop=True)  # reset the index

    df3 = df2.copy()  # make a copy because we want to be safe here
    
    for i in df2.loc[df2['varname_diff'] == True].index:
        empty_row = pd.DataFrame([df.columns], index=[i], columns=df.columns)  # creating the empty data
        df3 = pd.concat([df3.loc[:i - 1], empty_row, df3.loc[i:]])  # slicing the df
    df3 = df3.reset_index(drop=True)  # reset the index
    
    return df3





def excel_format_lift(path,filename,store_path):
    """
    :param path: 文件路径
    :param filename: 文件名，必须是xlsx文件
    :param store_path: 文件保存路径
    :return: 
    """
    # 加载Excel文件
    workbook = load_workbook(path+filename)
    # 选择第一个工作表
    sheet = workbook.active
    
    # 对bad_rate进行调整
    keyword01 = 'badrate'
    start_index_list = find_target_positions(sheet,keyword01)
    for start_index in start_index_list:
        start_index_num = int(re.sub(u"([^\u0030-\u0039])","",start_index))
        end_index = find_empty_cell(sheet,start_index)
        end_index_A = re.sub(u"([^\u0041-\u007a])", "", end_index)
        end_index_num = int(re.sub(u"([^\u0030-\u0039])","",end_index))
        for n in range(start_index_num,end_index_num):
            excel_value_format(sheet,end_index_A,n,'percentage')
        end_index = end_index_A + str(end_index_num-2)
    #
    
    
    
    # 对lift进行调整
    keyword02 = 'lift'
    start_index_list = find_target_positions(sheet,keyword02)
    missing_site = find_target_positions(sheet,'missing')
    missing_site_row = [int(re.sub(u"([^\u0030-\u0039])","",row_index)) for row_index in missing_site]
    for start_index in start_index_list:
        start_index_num = int(re.sub(u"([^\u0030-\u0039])","",start_index))
        end_index = find_empty_cell(sheet,start_index)
        end_index_A = re.sub(u"([^\u0041-\u007a])", "", end_index)
        end_index_num = int(re.sub(u"([^\u0030-\u0039])","",end_index))
        for n in range(start_index_num,end_index_num):
            excel_value_format(sheet,end_index_A,n,'percentage')
        # excel_value_format(sheet, end_index_A, end_index_num-1, 'decimal')
        end_index = end_index_A + str(end_index_num-1)
        if start_index_num+1 in missing_site_row:
            set_color_scale(sheet,start_index,end_index,1)
        else:
            set_color_scale(sheet, start_index, end_index,0)
    

    
    # 对cnt_distri进行调整
    keyword03 = 'cnt_distri'
    start_index_list = find_target_positions(sheet,keyword03)
    for start_index in start_index_list:
        start_index_num = int(re.sub(u"([^\u0030-\u0039])","",start_index))
        end_index = find_empty_cell(sheet,start_index)
        end_index_A = re.sub(u"([^\u0041-\u007a])", "", end_index)
        end_index_num = int(re.sub(u"([^\u0030-\u0039])","",end_index))
        for n in range(start_index_num,end_index_num):
            excel_value_format(sheet,end_index_A,n,'percentage')
        end_index = end_index_A + str(end_index_num-2)
    
    #
    # # 对KS进行调整
    # keyword03 = 'KS'
    # start_index_list = find_target_positions(sheet,keyword03)
    # for start_index in start_index_list:
    #     start_index_num = int(re.sub(u"([^\u0030-\u0039])","",start_index))
    #     end_index = find_empty_cell(sheet,start_index)
    #     end_index_A = re.sub(u"([^\u0041-\u007a])", "", end_index)
    #     end_index_num = int(re.sub(u"([^\u0030-\u0039])","",end_index))
    #     highlight_max_values(sheet, start_index, end_index)
    #     for n in range(start_index_num,end_index_num):
    #         excel_value_format(sheet,end_index_A,n,'percentage')
    #     end_index = end_index_A + str(end_index_num-1)

    # 保存修改后的Excel文件
    storename = store_path + filename[:-5] + "_color.xlsx"
    workbook.save(storename)

if __name__ == "__main__":
    path = r"D:\shortcut\2024临时文件\0131临时文件".replace("\\", "/") + "/"
    excel_format_lift(path,"color_test_origin.xlsx",store_path=path)


