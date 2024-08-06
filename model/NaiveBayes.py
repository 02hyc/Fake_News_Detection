# import pandas as pd
# import jieba
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
#
# # 定义中文分词函数
# def chinese_tokenizer(text):
#     return list(jieba.cut(text))
#
#
# # 读入训练集数据
# train_df = pd.read_csv('trainTwo.csv').fillna('')
# # 将title列和tag列的数据合并到原始的文本数据中
# train_text = train_df['content'] + ' ' + train_df['title'] + ' ' + train_df['subtitle']
#
#
# # 对中文文本进行向量化
# vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
# train_features = vectorizer.fit_transform(train_text)
#
# # 将标签转换为数字形式
# train_labels = train_df['label']
#
# # 训练朴素贝叶斯分类器
# clf = MultinomialNB(alpha=1.0)
# clf.fit(train_features, train_labels)
#
# # 使用训练好的模型进行预测
# test_df = pd.read_csv('testTwo.csv').fillna('')
# test_text = test_df['content']
# test_features = vectorizer.transform(test_text)
# test_pred = clf.predict(test_features)
#
# # 将预测结果写入CSV文件
# test_df['predicted_label'] = test_pred
# test_df.to_csv('NBTestFourRes.csv', index=False)
#
# # 计算准确率、精确率、召回率和F1值
# accuracy = accuracy_score(test_df['label'], test_pred)
# precision = precision_score(test_df['label'], test_pred, average='macro')
# recall = recall_score(test_df['label'], test_pred, average='macro')
# f1 = f1_score(test_df['label'], test_pred, average='macro')
#
# # 输出结果
# print("准确率：{:.4f}".format(accuracy))
# print("精确率：{:.4f}".format(precision))
# print("召回率：{:.4f}".format(recall))
# print("F1值：{:.4f}".format(f1))


# import pandas as pd
# import jieba
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
#
# # 定义中文分词函数
# def chinese_tokenizer(text):
#     return list(jieba.cut(text))
#
#
# # 读入训练集数据
# train_df = pd.read_csv('trainTwo.csv').fillna('')
# # 将title列和tag列的数据合并到原始的文本数据中
# train_text = train_df['content'] + ' ' + train_df['title'] + ' ' + train_df['subtitle']
#
# # 对中文文本进行向量化
# vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)
# train_features = vectorizer.fit_transform(train_text)
#
# # 对标签进行编码
# enc = OneHotEncoder(handle_unknown='ignore')
# train_labels_onehot = enc.fit_transform(train_df[['label']]).toarray()
# train_labels = train_labels_onehot.argmax(axis=1)  # 将标签转换为一维数组
#
# # 训练NB分类器
# clf = MultinomialNB(alpha=1.0)
# clf.fit(train_features, train_labels)
#
# # 预测
# test_df = pd.read_csv('testTwo.csv').fillna('')
# test_text = test_df['content']
# test_features = vectorizer.transform(test_text)
#
# # 对测试集编码
# test_labels_onehot = enc.transform(test_df[['label']]).toarray()
# test_pred = clf.predict(test_features)
#
# # 将测试集标签转换为数组
# test_labels = test_labels_onehot.argmax(axis=1)
#
# # 将预测结果写入CSV文件
# test_df['predicted_label'] = test_pred
# test_df.to_csv('NBTestThreeRes.csv', index=False)
#
#
# accuracy = accuracy_score(test_labels, test_pred)
# precision = precision_score(test_labels, test_pred, average='macro')
# recall = recall_score(test_labels, test_pred, average='macro')
# f1 = f1_score(test_labels, test_pred, average='macro')
#
# print("准确率：{:.4f}".format(accuracy))
# print("精确率：{:.4f}".format(precision))
# print("召回率：{:.4f}".format(recall))
# print("F1值：{:.4f}".format(f1))

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


stopwords = set()
with open('hit_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip()
        if word:
            stopwords.add(word)
stopwords.add(' ')


# 定义中文分词函数，并去除停用词
def chinese_tokenizer(text):
    words = [w for w in jieba.cut(text) if w not in stopwords]
    # print(words)
    return words


# 读入训练集数据
train_df = pd.read_csv('trainTwo.csv').fillna('')
# 将title列和tag列的数据合并到原始的文本数据中
train_text = train_df['content'] + ' ' + train_df['title'] + ' ' + train_df['subtitle']
train_text = train_text.apply(lambda x: ' '.join(chinese_tokenizer(x)))
print(train_text)


# 对中文文本进行向量化，并去除停用词
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_text)

# 将标签转换为数字形式
train_labels = train_df['label']

# 训练朴素贝叶斯分类器
clf = MultinomialNB(alpha=1.0)
clf.fit(train_features, train_labels)

# 使用训练好的模型进行预测
test_df = pd.read_csv('testTwo.csv').fillna('')

# 对测试集进行分词和去除停用词
test_df['content'] = test_df.apply(lambda row: ' '.join(chinese_tokenizer(row['content'])), axis=1)
# print(test_df['content'])
test_text = test_df['content']
test_features = vectorizer.transform(test_text)
test_pred = clf.predict(test_features)

# 将预测结果写入CSV文件
test_df['predicted_label'] = test_pred
test_df.to_csv('NBTestFourRes.csv', index=False)

# 计算准确率、精确率、召回率和F1值
accuracy = accuracy_score(test_df['label'], test_pred)
precision = precision_score(test_df['label'], test_pred, average='macro')
recall = recall_score(test_df['label'], test_pred, average='macro')
f1 = f1_score(test_df['label'], test_pred, average='macro')

# 输出结果
print("准确率：{:.4f}".format(accuracy))
print("精确率：{:.4f}".format(precision))
print("召回率：{:.4f}".format(recall))
print("F1值：{:.4f}".format(f1))

from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
conf_mat = confusion_matrix(test_df['label'], test_pred)

# 可视化混淆矩阵
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# 计算 ROC 曲线上的数据点
fpr, tpr, thresholds = roc_curve(test_df['label'], clf.predict_proba(test_features)[:, 1])

# 计算 PR 曲线上的数据点
precision, recall, thresholds = precision_recall_curve(test_df['label'], clf.predict_proba(test_features)[:, 1])

# 计算 ROC 曲线和 PR 曲线下的面积
roc_auc = auc(fpr, tpr)
pr_auc = auc(recall, precision)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# 绘制 PR 曲线
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(pr_auc))
plt.show()

