# Book Similiarity

## 输入 

多本 EPUB，保存到 `data/input/*.epub`

## 数据准备

把待排序的 EPUB、TXT 书籍，以 `bookId.title.epub` `bookId.title.txt` 命名，保存到 `data/input`，注意 `title` 不能有空格。

## 运行

```
brew install pip3 ipython3
pip3 install jieba matplotlib numpy pandas gensim codecs textract
ipython3 analyse_book_distance.py textrank 15
ipython3 analyse_book_distance.py custom_textrank 15
ipython3 analyse_book_distance.py tfidf 15
ipython3 analyse_book_distance.py word2vec
```

## 输出

```
data/out/type.topN.distance.txt
data/out/type.topN.tags.txt
```

1. 两本书之间的相似度，用 (0.0, 1.0) 衡量，1.0 为最相似，保存到 `data/output/*.distance.txt`

2. 热度图
