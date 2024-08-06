# 打开txt文件
with open('example-items.txt.tuples', 'r', encoding='utf-8') as file:
    # 读取每一行并将制表符替换为逗号
    lines = [line.replace(',', '，').replace('\t', ',') for line in file.readlines()]



# 创建新的CSV文件并将结果写入
with open('example-items.csv', 'w', encoding='utf-8') as file:
    file.writelines(lines)

