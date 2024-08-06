import pandas as pd

# 读取CSV文件
df = pd.read_csv('Copy of after_clean.csv', encoding='utf-8', usecols=['content', 'label', 'title', 'subtitle', 'category'])

# 填充缺失值，并将所有元素转换为字符串类型
df['content'] = df['content'].fillna('').astype(str)

# 筛选出中文字数在12到144之间的数据
df = df[df['content'].apply(lambda x: len(x)) >= 12]
df = df[df['content'].apply(lambda x: len(x)) <= 144]

# 将数据按8:2的比例随机分成训练集和测试集
train_df = df.sample(frac=0.8, random_state=None)
test_df = df.drop(train_df.index)

# 保存训练集和测试集为CSV文件
train_df.to_csv('trainTwo.csv', index=False, columns=['content', 'label', 'title', 'subtitle', 'category'], encoding='utf-8')
test_df.to_csv('testTwo.csv', index=False, columns=['content', 'label', 'title', 'subtitle', 'category'], encoding='utf-8')

