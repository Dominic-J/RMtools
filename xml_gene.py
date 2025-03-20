import pandas as pd
import xml.etree.ElementTree as ET


def varlib_xml2df(xml_path):  

    """
    变量库xml转DataFrame   
        * Category_Name：变量类别
        * Name：字段名
        * Label：标题
        * Type：数据类型
    """

    # 解析 XML  
    root = ET.parse(xml_path).getroot()  
  
    # 初始化 DataFrame 的列  
    columns = ['Category_Name', 'Name', 'Label', 'Type']  
    var_lib = []  
  
    # 遍历所有 category 下的 var 元素  
    for category in root.findall('category'):  
        category_name = category.get('name').strip()  # 去掉多余空格
        for var in category.findall('var'):  
            name = var.get('name').strip()  # 去掉多余空格
            label = var.get('label').strip()  # 去掉多余空格
            type = var.get('type').strip()  # 去掉多余空格
            var_lib.append([category_name, name, label, type])  
  
    # 创建 DataFrame  
    df = pd.DataFrame(var_lib, columns=columns)  
    return df


def conslib_xml2df(xml_path):  

    """
    常量库xml转DataFrame   
        * Category_Name：名称
        * Category_Label：标题
        * Name：子名称
        * Label：子标题
        * Type：数据类型
    """

    # 解析 XML  
    root = ET.parse(xml_path).getroot()  
  
    # 初始化 DataFrame 的列  
    columns = ['Category_Name', 'Category_Label', 'Name', 'Label', 'Type']  
    constant_lib = []  
  
    # 遍历所有 category 下的 constant 元素  
    for category in root.findall('category'):  
        category_name = category.get('name').strip()  # 去掉多余空格
        category_label = category.get('label').strip()  # 去掉多余空格
        for constant in category.findall('constant'):  
            name = constant.get('name').strip()  # 去掉多余空格
            label = constant.get('label').strip()  # 去掉多余空格
            type = constant.get('type').strip()  # 去掉多余空格
            constant_lib.append([category_name, category_label, name, label, type])  
  
    # 创建 DataFrame  
    df = pd.DataFrame(constant_lib, columns=columns)  
    return df



def parse_conditions(element, parent_op=None, depth=0):
    """递归解析条件，包括<and>和<or>操作"""
    conditions = []

    # 如果是<and>或<or>，我们记录它，并继续解析它的子元素
    if element.tag in ['and', 'or']:
        current_op = element.tag
        for child in element:
            conditions.extend(parse_conditions(child, current_op, depth + 1))
    
    # 如果是<atom>，我们提取<atom>的信息
    elif element.tag == 'atom':
        condition_type = element.get('op')
        left_var_category = element.find('left').get('var-category')
        left_var = element.find('left').get('var')
        left_var_label = element.find('left').get('var-label')
        left_datatype = element.find('left').get('datatype')
        
        # 判断是const还是content，并检查value_element是否存在
        value_element = element.find('value')
        if value_element is not None:
            if 'const' in value_element.attrib:
                value = value_element.get('const')
            else:
                value = value_element.get('content')
        else:
            value = None
        
        # 将条件信息添加到列表
        conditions.append({
            'ConditionType': condition_type,
            'VariableCategory': left_var_category,
            'Variable': left_var,
            'VariableLabel': left_var_label,
            'Datatype': left_datatype,
            'Value': value,
            'LogicalOperator': parent_op,  # 记录<and>或<or>操作符
            'Depth': depth  # 记录层次深度，用于表示嵌套
        })
    
    return conditions


def rule_xml2df(xml_path):

    """
    决策集遍历规则转DataFrame，包括AND、OR等逻辑嵌套，并通过Depth记录逻辑层次
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
    """

    # 解析XML
    root = ET.parse(xml_path).getroot()
    
    # 初始化DataFrame的列
    columns = ['RuleName', 'Enabled', 'ConditionType', 'VariableCategory', 
               'Variable', 'VariableLabel', 'Datatype', 'Value', 'LogicalOperator', 'Depth']
    rules_data = []

    # 提取rule信息
    for rule in root.findall('rule'):
        rule_name = rule.get('name')
        rule_enabled = rule.get('enabled')
        
        # 提取条件块（if block中的<and>和<or>）
        for condition in rule.findall('.//if/*'):  # 查找<and>或<or>
            conditions = parse_conditions(condition)
            for cond in conditions:
                cond.update({
                    'RuleName': rule_name,
                    'Enabled': rule_enabled,
                })
                rules_data.append(cond)

        # 提取then block中的var-assign信息
        for var_assign in rule.findall('.//var-assign'):
            var_category = var_assign.get('var-category')
            var = var_assign.get('var')
            var_label = var_assign.get('var-label')
            datatype = var_assign.get('datatype')
            value_element = var_assign.find('value')
            
            # 判断是const还是content，并检查value_element是否存在
            if value_element is not None:
                if 'const' in value_element.attrib:
                    value = value_element.get('const')
                else:
                    value = value_element.get('content')
            else:
                value = None
            
            rules_data.append({
                'RuleName': rule_name,
                'Enabled': rule_enabled,
                'ConditionType': 'Action',
                'Variable Category': var_category,
                'Variable': var,
                'VariableLabel': var_label,
                'Datatype': datatype,
                'Value': value,
                'LogicalOperator': None,
                'Depth': 0
            })

    # 创建DataFrame
    df = pd.DataFrame(rules_data, columns=columns)
    return df




import re


# 条件符号字典
operator_dict = {
    '>': 'GreaterThen',
    '>=': 'GreaterThenEquals',
    '<': 'LessThen',
    '<=': 'LessThenEquals',
    '==': 'Equals',
    '!=': 'NotEquals',
    'in': 'In',
    'not in':'NotIn'
}

def parse_condition(condition, var_lib):
    """解析条件并生成atom部分的XML"""
    # 清理不规则空格
    condition = re.sub(r'\s+', ' ', condition.strip())
    
    # 处理括号嵌套结构
    condition = condition.replace(' and ', ' AND ').replace(' & ', ' AND ').replace(' or ', ' OR ')  # 简化处理
    
    # 递归处理括号嵌套条件
    def process_logic(logic):
        if '(' in logic:
            # 找到最内层括号
            inner_most = re.search(r'\([^()]+\)', logic).group()
            parsed_inner = process_logic(inner_most[1:-1])
            logic = logic.replace(inner_most, parsed_inner, 1)
            return process_logic(logic)
        else:
            # 解析没有括号的条件
            return process_simple_conditions(logic)
    
    def process_simple_conditions(logic):
        logic_xml = ET.Element('and')  # 默认使用 'and'
        
        conditions = re.split(r' AND | OR ', logic)
        operators = re.findall(r' AND | OR ', logic)
        
        for cond in conditions:
            atom = process_single_condition(cond, var_lib)
            logic_xml.append(atom)
        
        return logic_xml
    
    return process_logic(condition)



def process_single_condition(cond, var_lib):
    """处理单个条件，并生成对应的atom节点"""
    # 更新正则表达式，支持中文和英文字符
    #match = re.match(r'([\w\-\u4e00-\u9fa5]+)\s*(==|>=|<=|>|<|!=|in|not in)\s*(.+)', cond)
    #match = re.match(r'([\w\-\u4e00-\u9fa5，]+)\s*(==|>=|<=|>|<|!=|in|not in)\s*(.+)', cond)
    
    #match = re.match(r'([\w\-\u4e00-\u9fa5，\.\s]+)\s*(==|>=|<=|>|<|!=|in|not in)\s*(.+)', cond)
    #     if not match:
    #         raise ValueError(f"Unable to parse condition: {cond}")
    
    match = re.match(r'([\w\-\u4e00-\u9fa5，\.\s]+)\s*(==|>=|<=|>|<|!=|not in)\s*(.+)', cond)
    
    # 优先匹配not in，匹配不到再匹配in 防止将not in 识别成 in
    if not match:
        # 尝试单独匹配 "in"
        match = re.match(r'([\w\-\u4e00-\u9fa5，\.\s]+)\s*(in)\s*(.+)', cond)
        if not match:
            raise ValueError(f"Unable to parse condition: {cond}")

    left_var, operator, right_value = match.groups()

#     # 尝试根据英文名查找
#     var_info = var_lib[var_lib['Name'].str.lower() == left_var]
#     if var_info.empty:
#         # 如果未找到，尝试根据中文名查找
#         var_info = var_lib[var_lib['Label'] == left_var]
#         if var_info.empty:
#             raise ValueError(f"Variable {left_var} not found in var_lib")
            
    # 尝试根据英文名查找，先去掉前后空格和转换为小写
    left_var_cleaned = left_var.strip().lower()

    # 根据英文名查找
    var_info = var_lib[var_lib['Name'].str.lower() == left_var_cleaned]
    if var_info.empty:
        # 如果未找到，尝试根据中文名查找
        var_info = var_lib[var_lib['Label'].str.strip() == left_var.strip()]
        if var_info.empty:
            raise ValueError(f"Variable '{left_var}' not found in var_lib")

    var_info = var_info.iloc[0]

    # 构建atom节点
    atom = ET.Element('atom', op=operator_dict.get(operator, 'Equals'))

    # 左侧变量
    left = ET.SubElement(atom, 'left', {
        'var-category': var_info['Category_Name'],
        'var': var_info['Name'],  # 使用英文名
        'var-label': var_info['Label'],  # 中文名
        'datatype': var_info['Type'],
        'type': 'variable'
    })

    # 处理右侧值
    if isinstance(right_value, str):
        right_value = right_value.strip().replace("'", "")
    else:
        raise ValueError(f"Expected a string for right_value, but got {type(right_value)}")

    # 右侧值
    value = ET.SubElement(atom, 'value', {'content': right_value, 'type': 'Input'})

    return atom




def create_contrast_or_grey_rule(row, var_lib, constant_lib, mode, contrast_value, contrast_rate, is_grey=False):
    """生成对照（灰度）规则"""
    rule = ET.Element('rule', name=row['RuleName'] + '_T')
    remark = ET.SubElement(rule, 'remark')
    remark.text = "<![CDATA[]]>"
    
    # if 部分
    if_tag = ET.SubElement(rule, 'if')
    try:
        and_tag = parse_condition(row['Desc'], var_lib)
        
        if contrast_value > 0:  # 对照组条件
            contrast_var = f'loan_fixed_random10_id{contrast_value}'
            contrast_info = var_lib[var_lib['Name'] == contrast_var].iloc[0]
            
            # 生成对照条件
            atom = ET.Element('atom', op='LessThen' if not is_grey else 'Equals')
            left = ET.SubElement(atom, 'left', {
                'var-category': contrast_info['Category_Name'],
                'var': contrast_var,
                'var-label': contrast_info['Label'],
                'datatype': contrast_info['Type'],
                'type': 'variable'
            })
            right_value = 'grey' if is_grey else str(contrast_rate)
            ET.SubElement(atom, 'value', {'content': right_value, 'type': 'Input'})
            
            and_tag.append(atom)
        elif contrast_value == 0:  # 灰度组条件
            group_var_info = var_lib[var_lib['Name'] == 'customerGroup'].iloc[0]
            atom = ET.Element('atom', op='Equals')
            left = ET.SubElement(atom, 'left', {
                'var-category': group_var_info['Category_Name'],
                'var': 'customerGroup',
                'var-label': group_var_info['Label'],
                'datatype': group_var_info['Type'],
                'type': 'variable'
            })
            ET.SubElement(atom, 'value', {'content': 'grey' if is_grey else 'normal', 'type': 'Input'})
            
            and_tag.append(atom)
        
        if_tag.append(and_tag)
    except Exception as e:
        print(f"Failed to generate rule {row['RuleName']}_T: {str(e)}")
        return None
    
    # then 部分
    then_tag = ET.SubElement(rule, 'then')
    
    # 结果描述
    result_assign = ET.SubElement(then_tag, 'var-assign', {
        'var-category': '规则结果输出',
        'var': 'resultDesc',
        'var-label': '规则结果描述',
        'datatype': 'String',
        'type': 'variable'
    })
    ET.SubElement(result_assign, 'value', {
        'content': row['Content'] + ('--对照' if not is_grey else '--灰度'),
        'type': 'Input'
    })
    
    # 结果值为 "Contrast"
    result_assign = ET.SubElement(then_tag, 'var-assign', {
        'var-category': '规则结果输出',
        'var': 'result',
        'var-label': '规则建议结果',
        'datatype': 'String',
        'type': 'variable'
    })
    ET.SubElement(result_assign, 'value', {
        'const-category': '规则结果',
        'const': 'Contrast',
        'const-label': '对照',
        'type': 'Constant'
    })
    
    # else 部分
    ET.SubElement(rule, 'else')
    
    return rule


def create_rule_xml(row, var_lib, constant_lib, mode):
    """为每一行生成规则的XML"""
    rule = ET.Element('rule', name=row['RuleName'])
    remark = ET.SubElement(rule, 'remark')
    remark.text = "<![CDATA[]]>"

    # if 部分
    if_tag = ET.SubElement(rule, 'if')
    try:
        and_tag = parse_condition(row['Desc'], var_lib)
        
        # 检查 Contrast 列
        if pd.notna(row['Contrast']) and str(int(row['Contrast'])).isdigit():
            contrast_value = int(row['Contrast'])
            if 1 <= contrast_value <= 50:
                # 增加对照条件
                contrast_var = f'loan_fixed_random10_id{contrast_value}'
                contrast_info = var_lib[var_lib['Name'] == contrast_var].iloc[0]
                contrast_rate = row['ContrastRate'] if pd.notna(row['ContrastRate']) and 1 <= int(row['ContrastRate']) <= 10 else 2

                atom = ET.Element('atom', op='GreaterThenEquals')
                left = ET.SubElement(atom, 'left', {
                    'var-category': contrast_info['Category_Name'],
                    'var': contrast_var,
                    'var-label': contrast_info['Label'],
                    'datatype': contrast_info['Type'],
                    'type': 'variable'
                })
                ET.SubElement(atom, 'value', {'content': str(contrast_rate), 'type': 'Input'})

                and_tag.append(atom)
            elif contrast_value == 0:
                # 增加灰度条件
                group_var_info = var_lib[var_lib['Name'] == 'customerGroup'].iloc[0]
                atom = ET.Element('atom', op='Equals')
                left = ET.SubElement(atom, 'left', {
                    'var-category': group_var_info['Category_Name'],
                    'var': 'customerGroup',
                    'var-label': group_var_info['Label'],
                    'datatype': group_var_info['Type'],
                    'type': 'variable'
                })
                ET.SubElement(atom, 'value', {'content': 'normal', 'type': 'Input'})

                and_tag.append(atom)

        if_tag.append(and_tag)
    except Exception as e:
        print(f"Failed to generate rule {row['RuleName']}: {str(e)}")
        return None

    # then 部分
    then_tag = ET.SubElement(rule, 'then')

    # 结果描述
    result_assign = ET.SubElement(then_tag, 'var-assign', {
        'var-category': '规则结果输出',
        'var': 'resultDesc',
        'var-label': '规则结果描述',
        'datatype': 'String',
        'type': 'variable'
    })
    ET.SubElement(result_assign, 'value', {
        'content': row['Content'] + '--拒绝',
        'type': 'Input'
    })

    # 结果值
    try:
        result_value = constant_lib[constant_lib['Name'] == row['ResultValue']].iloc[0]
    except IndexError:
        print(f"Failed to generate rule {row['RuleName']}: ResultValue {row['ResultValue']} not found in constant_lib")
        return None

    result_assign = ET.SubElement(then_tag, 'var-assign', {
        'var-category': '规则结果输出',
        'var': 'result',
        'var-label': '规则建议结果',
        'datatype': 'String',
        'type': 'variable'
    })
    ET.SubElement(result_assign, 'value', {
        'const-category': result_value['Category_Label'],
        'const': row['ResultValue'],
        'const-label': result_value['Label'],
        'type': 'Constant'
    })

    # 如果 ResultLockDay 存在
    if pd.notna(row['ResultLockDay']):
        lock_day_var = 'lockDays' if mode == 'loan' else 'lockDay'
        lock_day_assign = ET.SubElement(then_tag, 'var-assign', {
            'var-category': '规则结果输出',
            'var': lock_day_var,
            'var-label': '拒绝锁定天数',
            'datatype': 'Integer',
            'type': 'variable'
        })
        ET.SubElement(lock_day_assign, 'value', {
            'content': str(int(row['ResultLockDay'])),
            'type': 'Input'
        })

    # else 部分
    ET.SubElement(rule, 'else')

    return rule


def rule_excel2xml(filename, mode='credit'):

    """
    从Excel表中生成规则的XML
    
    Excel包含七列：
    * RuleName 规则名称
    * Content 规则结果描述
    * Desc 条件描述：规则的判断逻辑表达式，同时支持中文变量名和英文变量名，不区分大小写
    * ResultValue 规则建议结果：Apply、PreReject、Reject等
    * ResultLockDay（可选）拒绝锁定天数
    * Contrast（可选）是否有对照组，如果为空则不生成对照规则，如果输入1-50的整数，则对应随机数变量生成对照规则，如果输入0则生成灰度规则   
    * ContrastRate（可选） 对照组比例，当且仅当Contrast为1-50的整数时生效，默认为2，也可手动输入0-10的整数   



    主函数：
    rule_excel2xml(filename, mode='credit')

    * filename：Excel路径+文件名（需包含上面的七列，列名需要相同）
    * mode：区分授信和支用，默认为'credit'，首续贷规则需要改成'loan'


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

    可识别的逻辑符：  
    &、AND、OR（大小写均可）  
    """
    
    df = pd.read_excel(filename)
    root = ET.Element('rule-set')
    
    if mode == 'loan':
        var_lib = loan_var_lib
        constant_lib = loan_cons_lib
    else:
        var_lib = credit_var_lib
        constant_lib = credit_cons_lib
    
    for _, row in df.iterrows():
        # 生成原规则
        rule_xml = create_rule_xml(row, var_lib, constant_lib, mode)
        if rule_xml is not None:
            root.append(rule_xml)
        
        # 处理对照或灰度规则
        contrast_value = row['Contrast']
        
        if pd.notna(contrast_value):
            # 如果Contrast是数值
            if isinstance(contrast_value, (float, int)):
                contrast_value = int(contrast_value)
                # 检查对照组的值是否在 1-50 范围内
                if 1 <= contrast_value <= 50:
                    contrast_rate = int(row['ContrastRate']) if pd.notna(row['ContrastRate']) else 2
                    contrast_rule_xml = create_contrast_or_grey_rule(row, var_lib, constant_lib, mode, contrast_value, contrast_rate)
                    if contrast_rule_xml is not None:
                        root.append(contrast_rule_xml)
                # 处理灰度规则
                elif contrast_value == 0:
                    contrast_rate = int(row['ContrastRate']) if pd.notna(row['ContrastRate']) else 2
                    grey_rule_xml = create_contrast_or_grey_rule(row, var_lib, constant_lib, mode, contrast_value, contrast_rate, is_grey=True)
                    if grey_rule_xml is not None:
                        root.append(grey_rule_xml)
                else:
                    print(f"Skipping rule {row['RuleName']}: Invalid Contrast value {contrast_value}")
            else:
                print(f"Skipping rule {row['RuleName']}: Invalid Contrast value {contrast_value}")

    return root



# 生成漂亮的XML带有缩进并保存
def save_pretty_xml(elem,output_filename):
    '''
    输出和保存函数：
    save_pretty_xml(elem,output_filename)
    * elem：主函数的输出  
    * output_filename：保存的路径+文件名
    '''
    from xml.dom import minidom
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # 在这里替换掉 CDATA 被转义的部分
    pretty_xml = pretty_xml.replace("&lt;![CDATA[]]&gt;", "<![CDATA[]]>")
    pretty_xml = pretty_xml.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>')
    
    # 将XML保存为文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    return pretty_xml
    

from lxml import etree

def merge_xml_files(xml_file1, xml_file2, output_filename):
    """合并两个XML文件，保持第一个文件的结构并添加第二个文件的规则"""
    
    # 解析第一个XML文件
    tree1 = etree.parse(xml_file1)
    root1 = tree1.getroot()
    
    # 解析第二个XML文件
    tree2 = etree.parse(xml_file2)
    root2 = tree2.getroot()
    
    # 处理每个rule
    for rule in root2:
        root1.append(rule)
    
    # 处理remark，确保不被转义
    for remark in root1.xpath('//remark'):
        remark_text = remark.text or ''
        
        # 处理可能导致错误的情况
        if ']]>' in remark_text:
            remark_text = remark_text.replace(']]>', ']]]]><![CDATA[>')
        
        remark.text = etree.CDATA(remark_text)
    
    # 保存合并后的XML
    tree1.write(output_filename, encoding='utf-8', xml_declaration=True)
    
    
# 条件符号字典
var_options = {
    'GreaterThen': '>',
    'GreaterThenEquals': '>=',
    'LessThen': '<',
    'LessThenEquals': '<=',
    'Equals': '==',
    'NotEquals': '!=',
    'In': 'in',
    'NotIn': 'not in'
}

def generate_rule_descriptions(df):

    '''
    规则自动打标&通过率测算

    1、先用Excel2XML生成新版本规则的XML，包括下线、新增和调整阈值等都在XML里修改。   
    2、然后再用rule_xml2df读入新版策略的XML解析成DataFrame再生成Desc再进行打标。    
    ——这样可以避免新规则手动输入的Desc不符合规范，且测算完成后可以实现即刻部署。 

    生成规则描述函数：  
    generate_rule_descriptions(rule)：
    * rule：rule_info（通过xml解析生成，并根据需求去除结果输出、对照条件等不需要包含在描述中的条件）
    '''


    # 初始化新的列表用于存储规则描述
    rule_descriptions = []

    # 区间检测的正则表达式，匹配区间格式
    interval_pattern = re.compile(r'^(\(\d+(\.\d+)?,\d+(\.\d+)?\)|\[\d+(\.\d+)?,\d+(\.\d+)?\]|\[\d+(\.\d+)?,\d+(\.\d+)?\)|\(\d+(\.\d+)?,\d+(\.\d+)?\])$')
    
    # 正则表达式匹配负数和浮点数
    is_number = re.compile(r'^-?\d+(\.\d+)?$')

    # 遍历每个规则
    for rule_name, group in df.groupby('RuleName'):
        # 按 Depth 和 LogicalOperator 进行排序
        group = group.sort_values(by=['Depth'])

        # 使用字典保存不同层次的条件组
        depth_conditions = {}

        # 遍历当前规则的每一行
        for _, row in group.iterrows():
            condition_type = row['ConditionType']
            depth = row['Depth']
            logical_operator = row['LogicalOperator']

            # 检查 Condition Type 是否在字典中
            if condition_type in var_options:
                operator = var_options[condition_type]

                # 如果 Value 为 None，跳过此条件并将其标记为人工规则
                if row['Value'] is None:
                    depth_conditions[depth] = ["人工规则"]
                    break
                else:
                    # 处理Value值
                    value = row['Value'].strip()

                if condition_type in ['In', 'NotIn']:
                    # 处理 In 或 NotIn 的条件
                    values = value.split(',')  # 将 Value 按逗号分隔
                    quoted_values = []
                    temp_value = ''  # 用于暂存被分割的区间

                    for v in values:
                        v = v.strip()
                        if interval_pattern.match(v):  # 如果是区间，直接加入
                            quoted_values.append(f"'{v}'")
                        else:
                            if not v.startswith('(') and not v.startswith('[') and not v.endswith(']') and not v.endswith(')'):
                                # 如果不是区间中的一部分，直接处理
                                quoted_values.append(f"'{v}'")
                            else:
                                # 如果是区间的开始或结束部分，暂存
                                if temp_value:
                                    temp_value += f",{v}"  # 拼接区间
                                    quoted_values.append(f"'{temp_value}'")  # 将拼接后的区间加入
                                    temp_value = ''  # 清空暂存
                                else:
                                    temp_value = v  # 记录区间的起始部分

                    value = f"({', '.join(quoted_values)})"  # 用括号包裹多个值
                elif is_number.match(value):  # 改为匹配数字，包括负数和浮动数
                    # 如果是数字（包括负数或浮动数），不加引号
                    pass
                else:
                    value = f"'{value}'"  # 其他情况加引号

                # 生成条件表达式
                condition = f"{row['Variable'].lower()} {operator} {value}"
                
                # 将条件添加到相应深度的列表中
                if depth not in depth_conditions:
                    depth_conditions[depth] = []
                depth_conditions[depth].append(condition)
            else:
                # 如果 ConditionType 不在字典中，则跳过处理
                depth_conditions[depth] = ["人工规则"]
                break

        # 构建规则描述
        desc_parts = []
        max_depth = max(depth_conditions.keys(), default=-1)  # 获取最大深度，处理无条件情况

        for depth in range(max_depth, -1, -1):  # 从最大深度开始往上合并
            if depth in depth_conditions:
                # 获取当前深度的条件组
                conditions = depth_conditions[depth]
                
                # 判断当前层次是否有逻辑运算符
                operator = group[group['Depth'] == depth]['LogicalOperator'].iloc[0] if not group[group['Depth'] == depth].empty else 'and'
                
                if depth == 1:
                    # 顶层不需要括号，直接组合
                    if operator == 'or':
                        desc_parts.append(' or '.join(conditions))
                    else:
                        desc_parts.append(' and '.join(conditions))
                else:
                    # 根据运算符组合条件
                    if operator == 'or':
                        desc_parts.append(f"({' or '.join(conditions)})")
                    else:
                        desc_parts.append(f"({' and '.join(conditions)})")

        # 最终描述
        final_desc = ' and '.join(desc_parts) if desc_parts else "人工规则"
        rule_descriptions.append({'RuleName': rule_name, 'Desc': final_desc})

    # 创建新的 DataFrame
    desc_df = pd.DataFrame(rule_descriptions)
    return desc_df
    
    
def apply_rules(base_df, rules_df):

    '''
    自动打标函数：   
    result_df, success_df, failed_rules= apply_rules(base_df, rule_df)
    * base_df：数据底表
    * rule_df：需要自动打标的规则信息（用generate_rule_descriptions()生成）
    '''

    success_count = 0  # 统计成功的规则数
    failed_rules = []   # 存储失败的规则编码和失败原因
    success_data = []   # 存储成功的规则命中数和命中率
    result_df = base_df.copy()

    # 找出所有取值为空的列
    empty_columns = base_df.columns[base_df.isnull().all()].tolist()

    # 自定义函数用于处理数值
    def format_value(val):
        if isinstance(val, float):
            # 如果是小数且小数部分为0，返回整数形式
            if val.is_integer():
                return str(int(val))
            else:
                return str(val)
        return str(val)  # 其他情况返回原字符串

    # 在应用规则之前，将所有相关列转换为字符串类型
    for _, rule in rules_df.iterrows():
        query_str = rule['Desc']
        # 提取 IN 条件中的字段
        in_conditions = re.findall(r'(\w+)\s+in\s*\((.*?)\)', query_str)
        for col, _ in in_conditions:
            if col in result_df.columns:
                result_df[col] = result_df[col].apply(format_value)  # 转换为处理后的字符串类型

    for _, rule in rules_df.iterrows():
        rule_name = rule['RuleName']
        query_str = rule['Desc']

        # 提取当前规则中涉及到的变量
        variables_in_rule = re.findall(r'(\w+)\s*[><=!in]+', query_str)

        # 检查是否有变量在empty_columns列表中
        missing_values_vars = [var for var in variables_in_rule if var in empty_columns]
        if missing_values_vars:
            fail_reason = f"变量 {missing_values_vars} 的取值为空"
            print(f"规则 '{rule_name}' 模拟失败: {fail_reason}，无法模拟。")
            result_df[rule_name] = 0
            #result_df = result_df.copy()
            failed_rules.append({'RuleName': rule_name, 'Reason': fail_reason})  # 添加失败原因
            continue
        
        try:
            # 解析 query_str，避免将 IN 条件后的列表和 ==、!= 后的值识别为变量
            missing_fields = []
            query_str = re.sub(r"([><]=?)\s*'(-?\d+(\.\d+)?)'", r"\1 \2", query_str)
            query_str_cleaned = re.sub(r"in\s*\([^\)]+\)", "", query_str)
            query_elements = re.split(r"[\s()]+", query_str_cleaned.replace("'", ""))

            for i, element in enumerate(query_elements):
                if not element:
                    continue
                
                if element not in result_df.columns:
                    if not re.match(r'^-?\d+(\.\d+)?$', element):
                        if (i > 0 and query_elements[i - 1] in ['in', 'not in', '==', '!=']):
                            continue
                        
                        if element not in ['&', 'and', 'or', 'in', 'not', '<', '<=', '>', '>=', '==', '!=']:
                            missing_fields.append(element)
            
            if missing_fields:
                fail_reason = f"缺少字段 {missing_fields}"
                print(f"规则 '{rule_name}' 模拟异常: {fail_reason}，无法命中。")
                result_df[rule_name] = 0
                #result_df = result_df.copy()
                failed_rules.append({'RuleName': rule_name, 'Reason': fail_reason})  # 添加失败原因
                continue
            
            # 使用 df.query 应用规则
            results = result_df.query(query_str)
            hit_count = results.shape[0]
            hit_rate = hit_count / result_df.shape[0] * 100
            hit_rate_str = f"{hit_rate:.2f}%"
            
            result_df[rule_name] = 0
            result_df.loc[results.index, rule_name] = 1
            #result_df = result_df.copy()

            print(f"规则 '{rule_name}' 模拟成功: 命中数 = {hit_count}，命中率 = {hit_rate_str}\n模拟逻辑: {query_str}")
            success_count += 1
            
            success_data.append({'RuleName': rule_name, 'HitCount': hit_count, 'HitRate': hit_rate_str})
        except Exception as e:
            fail_reason = f"异常: {str(e)}"
            print(f"规则 '{rule_name}' 模拟失败: {fail_reason}\n模拟逻辑: {query_str}")
            result_df[rule_name] = 0
            #result_df = result_df.copy()
            failed_rules.append({'RuleName': rule_name, 'Reason': fail_reason})  # 添加失败原因

    total_rules_cnt = len(rules_df)
    fail_count = total_rules_cnt - success_count
    
    print(f"本次模拟共有规则 {total_rules_cnt} 条，模拟成功规则共 {success_count} 条，模拟失败规则共 {fail_count} 条。")
    
    result_df = result_df.copy()            # 解决DataFrame碎片化问题
    success_df = pd.DataFrame(success_data)
    failed_df = pd.DataFrame(failed_rules)  # 将失败的规则和原因输出为DataFrame
    return result_df, success_df, failed_df