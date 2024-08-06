import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 定义分词函数
def tokenize(text):
    return ' '.join(jieba.cut(text))

# 加载训练集和测试集数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 选择需要用于训练模型的列和标签列
train_features = train_data[['content', 'title', 'tag']].copy()
train_label = train_data['label'].copy()
test_features = test_data[['content', 'title', 'tag']].copy()
test_label = test_data['label'].copy()

# 对训练集和测试集数据进行预处理
train_features.loc[:, 'content'] = train_features['content'].apply(tokenize)
train_features.loc[:, 'title'] = train_features['title'].apply(tokenize)
train_features.loc[:, 'tag'] = train_features['tag'].apply(tokenize)
test_features.loc[:, 'content'] = test_features['content'].apply(tokenize)
test_features.loc[:, 'title'] = test_features['title'].apply(tokenize)
test_features.loc[:, 'tag'] = test_features['tag'].apply(tokenize)

# 将文本数据转化为 TF-IDF 向量
vectorizer = TfidfVectorizer(stop_words=None)
train_features = vectorizer.fit_transform(train_features.apply(lambda x: ' '.join(x), axis=1))
test_features = vectorizer.transform(test_features.apply(lambda x: ' '.join(x), axis=1))

# 定义LR模型
clf = LogisticRegression()

# 训练模型
clf.fit(train_features, train_label)

# 预测测试集
y_pred = clf.predict(test_features)

# 计算准确率、精确率、召回率、F1得分和support
accuracy = accuracy_score(test_label, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(test_label, y_pred)

# 将预测结果写入CSV文件
test_data['predicted_label'] = y_pred
test_data.to_csv('LRTestRes.csv', index=False)


# 输出结果
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)
print('Support:', support)

