import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm._libsvm import predict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读入数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 将标签列转换为tensor类型
train_labels = torch.tensor(train_df['label'].tolist())
test_labels = torch.tensor(test_df['label'].tolist())

# 加载RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True)

# 对输入文本进行tokenize并截断/填充到固定长度，最大长度为512
train_encodings = tokenizer(list(train_df['content']), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(test_df['content']), truncation=True, padding=True, max_length=512)

# 将输入文本和标签转换为TensorDataset类型
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']), train_labels)
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                             torch.tensor(test_encodings['attention_mask']), test_labels)

# 加载RoBERTa模型
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 将模型放到设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义训练参数
batch_size = 8
learning_rate = 2e-5
num_epochs = 5

# 定义优化器
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

# 定义训练数据的sampler和dataloader
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# 定义测试数据的sampler和dataloader
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

# 定义训练函数
def train(model, train_dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # 将数据放到设备上
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # 计算loss
            loss = outputs[0]
            total_loss += loss.item()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 清空梯度
            optimizer.zero_grad()

        print('Epoch: {}, Training Loss: {}'.format(epoch+1, total_loss/len(train_dataloader)))


# 定义评估函数
def evaluate(model, test_dataloader):
    model.eval()
    total_eval_accuracy = 0
    total_eval_precision = 0
    total_eval_recall = 0
    total_eval_f1 = 0

    for batch in test_dataloader:
        # 将数据放到设备上
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        # 禁用梯度计算
        with torch.no_grad():
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]

            # 将logits转换为预测标签
            preds = torch.argmax(logits, dim=1).flatten()

            # 计算评估指标
            eval_accuracy = accuracy_score(labels.cpu(), preds.cpu())
            eval_precision = precision_score(labels.cpu(), preds.cpu())
            eval_recall = recall_score(labels.cpu(), preds.cpu())
            eval_f1 = f1_score(labels.cpu(), preds.cpu())

            # 更新总评估指标
            total_eval_accuracy += eval_accuracy
            total_eval_precision += eval_precision
            total_eval_recall += eval_recall
            total_eval_f1 += eval_f1

    # 计算平均评估指标
    avg_eval_accuracy =total_eval_accuracy / len(test_dataloader)
    avg_eval_precision = total_eval_precision / len(test_dataloader)
    avg_eval_recall = total_eval_recall / len(test_dataloader)
    avg_eval_f1 = total_eval_f1 / len(test_dataloader)

    print('Accuracy: {}, Precision: {}, Recall: {}, F1: {}'.format(avg_eval_accuracy,
                                                                    avg_eval_precision,
                                                                    avg_eval_recall,
                                                                    avg_eval_f1))

# 训练模型
train(model, train_dataloader, optimizer, num_epochs)

# 评估模型
evaluate(model, test_dataloader)

# 保存模型
model.save_pretrained('roberta_model')

preds = predict(model, test_dataloader)

output = pd.DataFrame({'pre_label': preds})
output.to_csv('roberta_output.csv', index=False)

