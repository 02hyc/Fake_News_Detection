import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# 初始化情感分析器
sia = SentimentIntensityAnalyzer()

# 要分析的文本
text = "I love this movie! It's amazing."

# 进行情感分析
sentiment = sia.polarity_scores(text)

# 输出情感分析结果
print(sentiment)