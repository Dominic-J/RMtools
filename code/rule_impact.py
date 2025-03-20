import pandas as pd

def rule_impact(df, rule_desc, target_column='y', positive_label=1):
    """
    分析规则在数据集中的命中情况，并计算bad_rate和lift等指标。

    参数:
    df : pd.DataFrame
        包含待分析数据的DataFrame，至少需要包含rule_desc所涉及的字段和target_column。
    rule_desc : str
        规则的逻辑表达式，使用df.query(rule_desc)进行命中样本的筛选。
    target_column : str, optional
        目标列，表示贷后结果（如违约标志），默认为'y'。
    positive_label : int or str, optional
        目标列中表示“bad”样本的值（如违约的标签），默认为1。

    返回:
    result : pd.DataFrame
        返回一个包含命中、不命中和总体的分析结果，包括数量、占比、bad_rate和lift。
    """
    # 总体样本数量
    total_count = df.shape[0]
    
    # 命中规则的样本
    hit_df = df.query(rule_desc)
    hit_count = hit_df.shape[0]
    
    # 不命中规则的样本
    miss_df = df[~df.index.isin(hit_df.index)]
    miss_count = miss_df.shape[0]
    
    # 计算各类样本的违约率（bad_rate）
    total_bad_rate = df[target_column].value_counts(normalize=True).get(positive_label, 0)
    hit_bad_rate = hit_df[target_column].value_counts(normalize=True).get(positive_label, 0)
    miss_bad_rate = miss_df[target_column].value_counts(normalize=True).get(positive_label, 0)
    
    # 计算各类样本的bad_cnt
    total_bad_cnt = total_count*total_bad_rate
    hit_bad_cnt = hit_count*hit_bad_rate
    miss_bad_cnt = miss_count*miss_bad_rate
    
    # Lift = 命中bad_rate / 总体bad_rate
    lift = hit_bad_rate / total_bad_rate if total_bad_rate != 0 else float('inf')
    
    # 构建结果表
    result = pd.DataFrame({
        'Group': ['Hit', 'Miss', 'Total'],
        'Count': [hit_count, miss_count, total_count],
        'Percentage': [hit_count / total_count, miss_count / total_count, 1],
        'Bad':[hit_bad_cnt, miss_bad_cnt, total_bad_cnt],
        'Bad_Rate': [hit_bad_rate, miss_bad_rate, total_bad_rate],
        'Lift': [lift, '-', '-']
    })

    return result
    
    
    
def rule_impact_all(df, rule_desc, target_columns=['def_fpd7', 'def_fpd15', 'def_fpd30'], agr_columns=None, positive_label=1):
    """
    分析多个目标列在数据集中的命中情况，并计算bad_rate、lift和到期样本量等指标。
    
    参数:
    df : pd.DataFrame
        包含待分析数据的DataFrame，至少需要包含rule_desc所涉及的字段和目标列、到期列。
    rule_desc : str
        规则的逻辑表达式，使用df.query(rule_desc)进行命中样本的筛选。
    target_columns : list of str, optional
        目标列列表，表示多个贷后结果列（如不同的违约标志），默认为['def_fpd7', 'def_fpd15', 'def_fpd30']。
    agr_columns : list of str, optional
        每个target_column对应的到期标志列，如['agr_fpd7', 'agr_fpd15', 'agr_fpd30']。
    positive_label : int or str, optional
        目标列中表示“bad”样本的值（如违约的标签），默认为1。
    
    返回:
    result : pd.DataFrame
        包含每个目标列的命中、不命中和总体的分析结果，包括数量、占比、bad_rate、lift和到期样本量。
    """
    if agr_columns is None:
        raise ValueError("请提供每个目标列对应的到期标志列（agr_columns）。")

    # 计算命中率（不考虑到期标志列）
    total_count = df.shape[0]
    hit_df = df.query(rule_desc)
    hit_count = hit_df.shape[0]
    miss_count = total_count - hit_count
    
    results = {
        'Group': ['Hit', 'Miss', 'Total'],
        'Count': [hit_count, miss_count, total_count],
        'Percentage': [hit_count / total_count, miss_count / total_count, 1],
    }
    
    for target_column, agr_column in zip(target_columns, agr_columns):
        # 仅选择到期样本
        df_due = df[df[agr_column] != 0]
        hit_df_due = hit_df[hit_df[agr_column] != 0]
        miss_df_due = df_due[~df_due.index.isin(hit_df_due.index)]
        
        # 到期样本数
        due_count = df_due.shape[0]
        hit_due_count = hit_df_due.shape[0]
        miss_due_count = miss_df_due.shape[0]
        
        # 计算违约率（bad_rate）和提升度（lift）
        total_bad_rate = df_due[target_column].value_counts(normalize=True).get(positive_label, 0)
        hit_bad_rate = hit_df_due[target_column].value_counts(normalize=True).get(positive_label, 0)
        miss_bad_rate = miss_df_due[target_column].value_counts(normalize=True).get(positive_label, 0)
        
        lift = hit_bad_rate / total_bad_rate if total_bad_rate != 0 else float('inf')
        
        # 使用目标列的后缀作为列名
        column_suffix = target_column.split('_')[-1]
        
        # 将各列结果添加到结果字典
        results.update({
            f'{agr_column}': [hit_due_count, miss_due_count, due_count],
            f'{column_suffix}': [hit_bad_rate, miss_bad_rate, total_bad_rate],
            f'{column_suffix}_lift': [lift, '-', '-']
        })
    
    # 转换为 DataFrame 并返回
    final_result = pd.DataFrame(results)
    
    return final_result
    
    
collected_results = []

# 遍历规则，筛选出Enabled为'true'，且RuleName不以'T'结尾的规则  
#for rule_name, desc in rule_list2[(rule_list2['Enabled'] == 'true') & (rule_list2['RuleName'].str[-1:] != 'T')][['RuleName', 'Desc']].values:
for rule_name, desc, enabled in rule_list2[rule_list2['RuleName'].str[-1:] != 'T'][['RuleName', 'Desc', 'Enabled']].values:
    try:
        # 调用 rule_impact 函数，并获取结果
        result = rule_impact(df_sample[df_sample['agr_fpd7']==1], desc, 'def_fpd7', 1) #数据集改这里

        # 获取结果的第一行
        rule_result = result.iloc[0]

        # 在 rule_result 中加入 rule_name 和 desc
        rule_result = rule_result.copy()
        rule_result['RuleName'] = rule_name
        rule_result['Enabled'] = enabled
        rule_result['Desc'] = desc

        # 增加 bad_cnt 列，计算 Count * Bad_Rate
        rule_result['Bad_Cnt'] = rule_result['Count'] * rule_result['Bad_Rate']

        # 将修改后的结果添加到收集器中
        collected_results.append(rule_result)

    except Exception as e:
        print(f"规则 '{rule_name}' 执行时出错: {e}")
       
        rule_result = pd.Series({
                "RuleName": rule_name,
                "Enabled": enabled,
                "Desc": desc,
                "Count": str(e),  # 存储异常信息
                "Percentage": np.nan,
                "Bad_Cnt": np.nan,
                "Bad_Rate": np.nan,
                "Lift": np.nan
            })
            
        collected_results.append(rule_result)
        
        continue  # 跳过出错的规则

# 将收集到的结果合并成一个新的 DataFrame
if collected_results:
    final_df = pd.DataFrame(collected_results, columns=['RuleName', 'Enabled', 'Desc', 'Count', 'Percentage','Bad_Cnt', 'Bad_Rate', 'Lift']).reset_index(drop=True)
else:
    final_df = pd.DataFrame()  # 如果没有结果，则返回空DataFrame