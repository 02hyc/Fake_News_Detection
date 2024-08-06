import pandas as pd
import numpy as np
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 加载训练集和测试集数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 分词处理
def tokenize(text):
    return ' '.join(jieba.cut(text))

train_data['content'] = train_data['content'].apply(tokenize)
train_data['title'] = train_data['title'].apply(tokenize)
train_data['tag'] = train_data['tag'].apply(tokenize)
test_data['content'] = test_data['content'].apply(tokenize)
test_data['title'] = test_data['title'].apply(tokenize)
test_data['tag'] = test_data['tag'].apply(tokenize)

# 创建词汇表
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['content'].tolist() + train_data['title'].tolist() + train_data['tag'].tolist())

# 将文本转化为序列
train_content_seq = tokenizer.texts_to_sequences(train_data['content'].tolist())
train_title_seq = tokenizer.texts_to_sequences(train_data['title'].tolist())
train_tag_seq = tokenizer.texts_to_sequences(train_data['tag'].tolist())
test_content_seq = tokenizer.texts_to_sequences(test_data['content'].tolist())
test_title_seq = tokenizer.texts_to_sequences(test_data['title'].tolist())
test_tag_seq = tokenizer.texts_to_sequences(test_data['tag'].tolist())

# 对序列进行填充
maxlen = 500
train_content_seq = pad_sequences(train_content_seq, padding='post', maxlen=maxlen)
train_title_seq = pad_sequences(train_title_seq, padding='post', maxlen=maxlen)
train_tag_seq = pad_sequences(train_tag_seq, padding='post', maxlen=maxlen)
test_content_seq = pad_sequences(test_content_seq, padding='post', maxlen=maxlen)
test_title_seq = pad_sequences(test_title_seq, padding='post', maxlen=maxlen)
test_tag_seq = pad_sequences(test_tag_seq, padding='post', maxlen=maxlen)

# 构建模型
input_content = Input(shape=(maxlen,))
input_title = Input(shape=(maxlen,))
input_tag = Input(shape=(maxlen,))

embedding_dim = 32
lstm_units = 64
dropout_rate = 0.2

embedding_layer = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dim, input_length=maxlen)
lstm_layer = LSTM(units=lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)

content_embedding = embedding_layer(input_content)
content_lstm = lstm_layer(content_embedding)

title_embedding = embedding_layer(input_title)
title_lstm = lstm_layer(title_embedding)

tag_embedding = embedding_layer(input_tag)
tag_lstm = lstm_layer(tag_embedding)

merged = Bidirectional(LSTM(units=lstm_units))(content_lstm)
merged = Dense(units=16, activation='relu')(merged)
merged = Dense(units=1, activation='sigmoid')(merged)

model = Model(inputs=[input_content,input_title,input_tag], outputs=merged)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(x=[train_content_seq, train_title_seq, train_tag_seq], y=train_data['label'].values, batch_size=64,
                    epochs=20, validation_split=0.1, callbacks=[early_stopping])

# 在测试集上进行预测
y_pred = model.predict([test_content_seq, test_title_seq, test_tag_seq])
y_pred = np.round(y_pred).flatten()

# 将预测结果转化为0和1
y_pred[y_pred > 1] = 1
y_pred[y_pred < 0] = 0

# 将预测结果写入文件
test_data['label'] = y_pred
test_data.to_csv('result.csv', index=False)


# 输出结果
accuracy = accuracy_score(test_data['label'].values, y_pred)
precision, recall, f1_score, support = precision_recall_fscore_support(test_data['label'].values, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)
print('Support:', support)