# 自用风控策略小工具集合   

## （一）xml_gene   
@Author：Shi Dejing   

### 变量库解析（XML2DataFrame）
1. varlib_xml2df(xml_path)：变量库xml转DataFrame   
* Category_Name：变量类别
* Name：字段名
* Label：标题
* Type：数据类型
2. conslib_xml2df(xml_path)：常量库xml转DataFrame   
* Category_Name：名称
* Category_Label：标题
* Name：子名称
* Label：子标题
* Type：数据类型

### 决策集解析（XML2DataFrame）
rule_xml2df(xml_path)：决策集遍历规则转DataFrame，包括AND、OR等逻辑嵌套，并通过Depth记录逻辑层次
* RuleName：规则编码
* Enabled：是否启用
* ConditionType：条件类型
* VariableCategory：变量类别
* Variable：变量名
* VariableLabel：变量标题
* Datatype：数据类型
* Value：值
* LogicalOperator：逻辑运算符（AND、OR）
* Depth：逻辑层次



### 新增规则部署（Excel2XML）
Excel包含七列：
* RuleName 规则名称
* Content 规则结果描述
* Desc 条件描述：规则的判断逻辑表达式，同时支持中文变量名和英文变量名，不区分大小写
* ResultValue 规则建议结果：Apply、PreReject、Reject等
* ResultLockDay（可选）拒绝锁定天数
* Contrast（可选）是否有对照组，如果为空则不生成对照规则，如果输入1-50的整数，则对应随机数变量生成对照规则，如果输入0则生成灰度规则   
* ContrastRate（可选） 对照组比例，当且仅当Contrast为1-50的整数时生效，默认为2，也可手动输入0-10的整数   

目的：根据Excel自动生成xml

主函数：
rule_excel2xml(filename, mode='credit')

* filename：Excel路径+文件名（需包含上面的七列，列名需要相同）
* mode：区分授信和支用，默认为'credit'，首续贷规则需要改成'loan'

输出和保存函数：
save_pretty_xml(elem,output_filename)
* elem：主函数的输出  
* output_filename：保存的路径+文件名

使用示例：  
new_rules_xml = save_pretty_xml(rule_excel2xml('new_rule.xlsx',mode='credit'),output_path+'new_rule.xml')  


当前已支持：  
'>': 'GreaterThen',  
'>=': 'GreaterThenEquals',  
'<': 'LessThen',  
'<=': 'LessThenEquals',  
'==': 'Equals',  
'!=': 'NotEquals',  
'in': 'In',  
'not in':'NotIn'   
注1：暂不支持between and，需要写成 var>=lower_limit and var<=upper_limit 两个条件   
注2：in和not in 后面的值不需要加括号   
注3：可识别的逻辑符：  &、AND、OR（大小写均可）


### 规则自动打标&通过率测算

1、先用Excel2XML生成新版本规则的XML，包括下线、新增和调整阈值等都在XML里修改。   
2、然后再用rule_xml2df读入新版策略的XML解析成DataFrame再生成Desc再进行打标。    
——这样可以避免新规则手动输入的Desc不符合规范，且测算完成后可以实现即刻部署。 

生成规则描述函数：  
generate_rule_descriptions(rule)：
* rule：rule_info（通过xml解析生成，并根据需求去除结果输出、对照条件等不需要包含在描述中的条件）

自动打标函数：   
result_df, success_df, failed_rules= apply_rules(base_df, rule_df)
* base_df：数据底表
* rule_df：需要自动打标的规则信息（用generate_rule_descriptions()生成）


## （二）rule_impact

### rule_impact(df, rule_desc, target_column='y', positive_label=1)

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

### rule_impact_all(df, rule_desc, target_columns, agr_columns=None, positive_label=1)

分析多个目标列在数据集中的命中情况，并计算bad_rate、lift和到期样本量等指标。   
    
参数:    
    df : pd.DataFrame    
        包含待分析数据的DataFrame，至少需要包含rule_desc所涉及的字段和目标列、到期列。    
    rule_desc : str   
        规则的逻辑表达式，使用df.query(rule_desc)进行命中样本的筛选。  
    target_columns : list of str, optional    
        目标列列表，表示多个贷后结果列（如不同的违约标志）。   
    agr_columns : list of str, optional   
        每个target_column对应的到期标志列。   
    positive_label : int or str, optional     
        目标列中表示“bad”样本的值（如违约的标签），默认为1。    
返回:
    result : pd.DataFrame     
        包含每个目标列的命中、不命中和总体的分析结果，包括数量、占比、bad_rate、lift和到期样本量。    
