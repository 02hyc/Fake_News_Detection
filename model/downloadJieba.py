import urllib.request
url = 'https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/stop_words.txt'
urllib.request.urlretrieve(url, './stopwords.txt')