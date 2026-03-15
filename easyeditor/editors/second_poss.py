import re
import json

def calculate_averages(tokens, str):
    w = 0
    # 分别提取被5整除的索引和被5整除余1的索引对应的元素
    divisible_by_5 = []#[tokens[i] for i in range(len(tokens)) if i % 310 == 5]
    #remainder_1 = [tokens[i] for i in range(len(tokens)) if i % 5 == 1]
    remainder_1 = []
    for i in range(len(tokens)):
        if i % 165 == 15:
            print(str[i])
            
            divisible_by_5.append(tokens[i])
        if i % 165 == 16:
            print(str[i])
            
            remainder_1.append(tokens[i])
            
    # 计算平均值
    
    avg_divisible_by_5 = sum(divisible_by_5) / len(divisible_by_5) if divisible_by_5 else 0
    avg_remainder_1 = sum(remainder_1) / len(remainder_1) if remainder_1 else 0

    return avg_divisible_by_5, avg_remainder_1

def extract_tokens_from_log(file_path):
    str = []
    tokens = []
    pattern_tensor = r"tensor\(([-+]?\d*\.\d+|\d+\.?),\s*device='[^']*'\)"
    pattern_token = r"possible token:"
    j = 0
    with open(file_path, 'r') as file:
        previous_line = ""
        lines = file.readlines()  # 读取所有行并存储
        i = 0  # 行号
        while i < len(lines):
            line = lines[i]
            # 如果当前行是 "possible token:"，继续查找下一行
            if re.search(pattern_token, previous_line):
                str.append(previous_line[len("possible token: "):].replace(" ", "").replace("\n", ""))
                j = j + 1
                # 继续查找直到匹配到 tensor(...) 行
                while i < len(lines):
                    line = lines[i]
                    match_tensor = re.search(pattern_tensor, line)
                    if match_tensor:
                        tokens.append(float(match_tensor.group(1)))  # 提取数字并添加
                        break  # 找到 tensor(...) 后跳出循环
                    i += 1
            previous_line = line  # 更新上一行内容
            i += 1  # 移动到下一行
    print(j)
    return tokens, str

# 读取并提取小数
file_path = '/home/hmpiao/hmpiao/test_multi_analysis_all_continue_50edits_melo_again_sim0_lo.log'  # 请替换为你的log文件路径
tokens, str = extract_tokens_from_log(file_path)


avg_divisible_by_5, avg_remainder_1 = calculate_averages(tokens, str)

print(len(tokens))
#print(str[0])
print(f"被5整除索引的平均值: {avg_divisible_by_5}")
print(f"被5整除余1索引的平均值: {avg_remainder_1}")
print(avg_remainder_1/avg_divisible_by_5)