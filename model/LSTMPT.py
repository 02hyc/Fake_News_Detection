import torch
import torchtext.vocab as vocab
import jieba
import pandas as pd
from collections import Counter

# 加载数据集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 定义分词函数
def tokenize(text):
    return [token for token in jieba.cut(text) if token.strip()]

# 构建词汇表
counter = Counter()
for text in train_data['content']:
    counter.update(tokenize(text))
for text in train_data['title']:
    counter.update(tokenize(text))
for text in train_data['tag']:
    counter.update(tokenize(text))
vocab_dict = dict(counter)
tokenizer = vocab.Vocab(vocab_dict)

# 添加特殊符号到词汇表中
tokenizer.add_specials(['<unk>', '<pad>'])

# 设置默认的未知单词索引
tokenizer.set_default_index(tokenizer['<unk>'])

# 更新单词到索引和索引到单词的映射
tokenizer.stoi = {key: value+1 for key, value in tokenizer.stoi.items()}
tokenizer.itos = ['<unk>'] + tokenizer.itos

# 保存词汇表
torch.save(tokenizer, 'tokenizer_vocab.pt')