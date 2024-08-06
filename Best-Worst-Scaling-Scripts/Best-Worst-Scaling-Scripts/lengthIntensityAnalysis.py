# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 读取数据集，假设数据集中有text和intensity两列
# data = pd.read_csv("7.17std.csv")
#
# # 计算每个文本的长度
# data["length"] = data["text"].apply(lambda x: len(x))
#
# # 绘制散点图
# plt.scatter(data["length"], data["intensity"])
# plt.xlabel("Length")
# plt.ylabel("Intensity")
# plt.show()
#
# # 计算相关性系数
# correlation = np.corrcoef(data["length"], data["intensity"])[0, 1]
# print("文本长度和强度之间的相关性为: ", correlation)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba

# 读取数据集，假设数据集中有text和intensity两列
data = pd.read_csv("7.17std.csv")

# 将每个文本分词，并计算单词数量
data["word_count"] = data["text"].apply(lambda x: len(jieba.lcut(x)))

# 绘制散点图
plt.scatter(data["word_count"], data["intensity"])
plt.xlabel("Length")
plt.ylabel("Intensity")
plt.show()

# 计算相关性系数
correlation = np.corrcoef(data["word_count"], data["intensity"])[0, 1]
print("单词数量和强度之间的相关性为: ", correlation)


