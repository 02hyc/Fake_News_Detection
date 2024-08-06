with open('example-scores.txt', 'r') as file:
    # 读取每一行并替换空格为逗号
    lines = [",".join(line.split()) + "\n" for line in file]

# 将替换后的内容写回到文件中
with open('example-scores.txt', 'w') as file:
    file.writelines(lines)

