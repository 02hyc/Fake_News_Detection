import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 定义分词函数
def tokenize(text_list):
    # 对输入的文本列表进行分词处理
    tokenized_list = []
    for text in text_list:
        tokenized_list.append(' '.join(jieba.cut(text)))
    return tokenized_list

# 加载数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 对原始文本进行分词处理
train_text = tokenize(train_data['content'] + train_data['title'] + train_data['tag'])
test_text = tokenize(test_data['content'] + test_data['title'] + test_data['tag'])

# 设置标签
train_label = train_data['label'].astype(int)
test_label = test_data['label'].astype(int)

# 特征提取和预处理
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_text)
test_features = vectorizer.transform(test_text)

# 定义SVM模型，并使用RBF核函数
clf = SVC(kernel='rbf', C=5.0, gamma='scale')
# clf = SVC(kernel='linear', C=5.0)

# 拟合模型
clf.fit(train_features, train_label)

# 使用训练好的模型对测试集进行预测
y_pred = clf.predict(test_features)

# 将预测结果写入CSV文件
test_data['predicted_label'] = y_pred
test_data.to_csv('SVMTestTwoRes.csv', index=False)

# 计算准确率、精确率、召回率、F1得分和support
accuracy = accuracy_score(test_label, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(test_label, y_pred)

# 输出结果
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)
print('Support:', support)

