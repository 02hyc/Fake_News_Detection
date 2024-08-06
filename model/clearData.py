import pandas as pd
import re

df = pd.read_csv('data.csv')

df['content'] = df['content'].apply(lambda x: re.sub(r'<.*/>', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('\n', ' ', x))
df['content'] = df['content'].apply(lambda x: re.sub('\s+', ' ', x))
df['content'] = df['content'].apply(lambda x: re.sub(r'http[s]?\S+', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('网页链接', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('展开全文', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('秒拍视频', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('全文', '', x))

df['title'] = df['content'].apply(lambda x: re.findall('【.*】', x))
# 在content中去除title
df['content'] = df['content'].apply(lambda x: re.sub('【.*】', '', x))
df['tag'] = df['content'].apply(lambda x: re.findall('#(.*?)#', x))


df['content'] = df['content'].apply(lambda x: re.sub('#(.*?)#', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('@', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('\[.*?\]', '', x))
df['content'] = df['content'].apply(lambda x: re.sub('【.*?】', '', x))
df['text'] = df['content']

df.to_csv('clearData.csv', index=False, encoding='utf-8')