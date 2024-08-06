import pandas as pd
import thulac
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('7.17std.csv', encoding='utf-8')
texts = data['text']

# 分词和词性标注
thu = thulac.thulac()
pos_list = []
for text in texts:
    words = thu.cut(text)
    pos = [word[1] for word in words]
    pos_list.extend(pos)

# 统计每种词性的出现次数
pos_count = {}
for pos in pos_list:
    if pos in pos_count:
        pos_count[pos] += 1
    else:
        pos_count[pos] = 1

# 将词性和出现次数转化为两个列表
pos_list = list(pos_count.keys())
count_list = list(pos_count.values())

# 绘制词性出现次数的直方图
plt.bar(pos_list, count_list)
plt.xlabel('Part of Speech')
plt.ylabel('Count')
plt.show()