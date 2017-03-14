# Book Similiarity

## Data Prep

Save book named `bookId.title.epub` or `bookId.title.txt` to `data/input`. Notice that `title` should not contain spaces

## Run

```
brew install pip3 ipython3
pip3 install jieba matplotlib numpy pandas gensim codecs textract
ipython3 analyse_book_distance.py textrank 15
ipython3 analyse_book_distance.py tfidf 15
```

## Output

```
data/output/type.topN.distance.txt
data/output/type.topN.tags.txt
```

1. distance measure by 0.0~1.0 between books save to `data/output/type.topN.distance.txt`

2. tags that learn from book save to `data/output/type.topN.tags.txt`
