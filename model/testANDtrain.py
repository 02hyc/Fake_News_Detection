import pandas as pd

# 读入train.csv和test.csv文件
train_df = pd.read_csv('trainTwo.csv')
test_df = pd.read_csv('testTwo.csv')

# 从train.csv和test.csv中提取content列
train_content = set(train_df['content'].tolist())
test_content = set(test_df['content'].tolist())

# 检查train.csv和test.csv的content部分是否有重叠
overlap_content = train_content.intersection(test_content)

if len(overlap_content) > 0:
    print("train.csv和test.csv的content部分存在重叠！")
    print("重叠部分的数量为：", len(overlap_content))

    # 去除test.csv中与train.csv重合的数据
    test_df = test_df[~test_df['content'].isin(train_content)]
    test_df.to_csv('testTwo.csv', index=False)  # 将去重后的test_df重新写入testTwo.csv文件
    print("已从test.csv中去除与train.csv重合的数据！")
else:
    print("train.csv和test.csv的content部分不存在重叠！")