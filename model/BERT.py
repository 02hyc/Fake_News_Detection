import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取数据集
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 合并content, tag, title列为一个文本列
train_text = train_df['content'] + ' ' + train_df['title'] + ' ' + train_df['tag']
test_text = test_df['content'] + ' ' + test_df['title'] + ' ' + test_df['tag']

# 将标签转换为数字形式
train_labels = train_df['label'].values
test_labels = test_df['label'].values

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 对训练集和测试集进行tokenize和padding
train_tokens = tokenizer.batch_encode_plus(train_text.tolist(), max_length=512, padding=True, truncation=True)
test_tokens = tokenizer.batch_encode_plus(test_text.tolist(), max_length=512, padding=True, truncation=True)

# 将token转换为PyTorch tensor
train_input_ids = torch.tensor(train_tokens['input_ids'])
train_attention_mask = torch.tensor(train_tokens['attention_mask'])
train_labels = torch.tensor(train_labels)
test_input_ids = torch.tensor(test_tokens['input_ids'])
test_attention_mask = torch.tensor(test_tokens['attention_mask'])
test_labels = torch.tensor(test_labels)

# 创建训练和测试的DataLoader
batch_size = 8
train_data = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
test_data = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# 加载BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 将模型移动到GPU上（如果有的话）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 设置优化器和学习率
optimizer = AdamW(model.parameters(), lr=2e-5)

# 定义训练函数
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        batch_labels = batch[2].to(device)
        optimizer.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

# 定义评估函数
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(batch_labels.tolist())
            all_preds.extend(preds.tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return accuracy, precision, recall, f1

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer)
    print('Epoch {} train loss: {:.4f}'.format(epoch+1, train_loss))
    test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader)
    print('Epoch {} test accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(epoch+1, test_accuracy, test_precision, test_recall, test_f1))

# 保存模型
model.save_pretrained('bert_model')

# 加载模型
model = BertForSequenceClassification.from_pretrained('bert_model')

# 在测试集上评估模型
test_accuracy, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader)
print('Test accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(test_accuracy, test_precision, test_recall, test_f1))

# 将预测结果保存为csv文件
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.tolist())
test_df['label'] = all_preds
test_df.to_csv('BERTRes.csv', index=False)


