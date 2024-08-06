# import pandas as pd
# import jieba
# import matplotlib.pyplot as plt
#
# # 读取数据集
# data = pd.read_csv('7.17std.csv')
# texts = data['text']
#
# # 计算每个文本的长度
# lengths = [len(list(jieba.cut(text))) for text in texts]
#
# # 输出最长的文本及其长度
# max_length_index = lengths.index(max(lengths))
# print('Maximum length text:', texts[max_length_index])
# print('Maximum length:', lengths[max_length_index])
#
# # 输出一些基本的统计信息
# print('Minimum length:', min(lengths))
# print('Maximum length:', max(lengths))
# print('Average length:', sum(lengths) / len(lengths))
# print('Median length:', sorted(lengths)[len(lengths) // 2])
# print('Standard deviation:', pd.Series(lengths).std())
#
# # 绘制文本长度的直方图
# # 从200到2000，每隔200取一个数
# bins = [i for i in range(0, 1300, 10)]
# plt.hist(lengths, bins=bins)
# plt.xlabel('Length')
# plt.ylabel('Count')
# plt.show()

import pandas as pd

# 读取数据集
data = pd.read_csv('7.17std.csv')
texts = data['text']

# 计算每个文本的长度
lengths = [len(text) for text in texts]

# 输出一些基本的统计信息
print('Minimum length:', min(lengths))
print('Maximum length:', max(lengths))
print('Average length:', sum(lengths) / len(lengths))
print('Median length:', sorted(lengths)[len(lengths) // 2])
print('Standard deviation:', pd.Series(lengths).std())

# 绘制文本长度的直方图
import matplotlib.pyplot as plt
bins = [i for i in range(0, 2200, 10)]
plt.hist(lengths, bins=bins)
plt.xlabel('Length')
plt.ylabel('Count')
plt.show()

