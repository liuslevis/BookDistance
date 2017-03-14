import sys
sys.path.insert(0, "./..")
import textract
import jieba
import jieba.analyse
import os
import itertools
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import operator
import gensim
import logging
import math
import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary

input_path = 'data/input/'
output_path = 'data/output/'
tag_type = sys.argv[1] if len(sys.argv) > 1 else 'tfidf'
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 15

doc2vec_train_path = input_path + 'doc2vec.small.train'
idf_path = input_path + 'idf.train'

distance_path = output_path + '%s.top%d.distance.txt' % (tag_type, topk)
tags_path = output_path + '%s.top%d.tags.txt' % (tag_type, topk)

doc2vec_distance_path = output_path + 'doc2vec.distance.txt'

punct = set(u'''\n\t\r:!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')



def jaccard(li1, li2):
    if not li1 or not li2:
        return 0
    s1 = set(li1)
    s2 = set(li2)
    return len(s1.intersection(s2)) * 1.0 / len(s1.union(s2))

def text_for_epub(path):
    return textract.process(path).decode('utf-8')

def text_for_txt(path):
    with open(path) as f:
        return '\n'.join(f.readlines())
    return None

def parse_epub_to_txt(input_path):
    for file in os.listdir(input_path):
        if file.endswith('.epub'):
            text = text_for_epub(input_path + file)
            with open(input_path + file + '.txt', 'w') as f:
                f.write(text)

# parse_epub_to_txt(input_path)

def str_for_seg_list(seg_list, punct):
    ret = ''
    for word in seg_list:
        for ch in word:
            if ch not in punct:
                ret += ch
    return ret

def gen_doc2vec_train_data(input_path, doc2vec_train_path):
    label = 0
    with open(doc2vec_train_path, 'w') as f:
        for file in os.listdir(input_path):
            if file.endswith('.epub'):
                text = text_for_epub(input_path + file)
                seg_list = jieba.cut(text)
                f.write('%d %s\n' % (label, str_for_seg_list(seg_list, punct)))
                label += 1

# gen_doc2vec_train_data(input_path, doc2vec_train_path)

def gen_idf_data(input_path, idf_path):
    file_segs_dict = {} # {filename:set('word1', 'word2')}
    word_set = set()
    for file in os.listdir(input_path):
        if file.endswith('.epub'):
            text = text_for_epub(input_path + file)
            segs = set(list(jieba.cut(text)))
            file_segs_dict[file] = set(segs)
            word_set = word_set.union(set(segs))

    word_idf_dict = {}
    for word in word_set:
        num_docs_has_word = 0
        for (filename, words) in file_segs_dict.items():
            if word in words:
                num_docs_has_word += 1
        idf = math.log(len(file_segs_dict) * 1.0 / (num_docs_has_word + 1))
        word_idf_dict[word] = idf

    with open(idf_path, 'w') as f:
        desc = ''
        for (word, idf) in word_idf_dict.items():
            if len(word) > 0 and word != ' ':
                desc += '%s %.2f\n' % (word, idf)
        f.write(desc)

def tags_for_book_file(path):
    text = text_for_epub(path) if path.endswith('.epub') else text_for_txt(path)
    tags = None
    if tag_type == 'textrank':
        tags = jieba.analyse.textrank(text, topk, allowPOS=('n', 'ns'))
    elif tag_type == 'tfidf':
        tags = jieba.analyse.extract_tags(text, topk)
    elif tag_type == 'custom_textrank':
        keyExtractor = KeyExtractor()
        tags = keyExtractor.extractFromContent(text, topk, idfPath=idf_path)
    else:
        raise ValueError('tag type error!must be "textrank" or "tfidf"!')
    return tags

def title_for_file(file):
    return file.split('.')[1]

# {bookId:{title:xx, text:xx, format:epub, tags:[xx,xx,..]}}
def read_book_infos(input_path):
    book_infos = {}
    tag_desc = ''
    for file in os.listdir(input_path):
        if file.endswith('.epub') or file.endswith('.txt'):
            print('loading:%s' % file)
            bookId, title, format = file.split('.')
            bookId = int(bookId)
            tags = tags_for_book_file(input_path + file)
            text = '' # text for book
            book_infos.update({bookId:{'text':text, 'title':title, 'format':format, 'tags':tags}})
            tag_desc += '%s\ntag:(%s)\n\n' % (file, '/'.join(tags))
            print('%s\ntag:(%s)\n' % (file, '/'.join(tags)))
    with open(tags_path, 'w') as f:
        f.write(tag_desc)
    return book_infos

# update bookInfos -> {bookId1:{score:{bookId2:0.56, ...}}}
def calc_book_distance(book_infos):
    for (bookId1, bookId2) in (itertools.permutations(book_infos.keys(), 2)):
        tags1 = book_infos[bookId1]['tags']
        tags2 = book_infos[bookId2]['tags']
        title1 = book_infos[bookId1]['title']
        title2 = book_infos[bookId2]['title']
        score = jaccard(tags1, tags2)
        common_tags = set(tags1).intersection(set(tags2))
        book_infos[bookId1].setdefault('score', {})
        book_infos[bookId1]['score'].update({bookId2:score})
        print('%s:%.2f %s %s' % (jaccard.__name__, score, title1, title2))

def print_book_distance(book_infos, distance_path):
    desc = ''
    for bookId1 in book_infos.keys():
        title1 = book_infos[bookId1]['title']
        tags1 = book_infos[bookId1]['tags']
        desc += '\n%s' % title1
        li = []
        for bookId2 in book_infos[bookId1]['score']:
            score = book_infos[bookId1]['score'][bookId2]
            title2 = book_infos[bookId2]['title']
            tags2 = book_infos[bookId2]['tags']
            share_tags = set(tags1).intersection(set(tags2))
            li.append((score, title2, share_tags))
        for item in sorted(li, key=operator.itemgetter(0), reverse=True):
            score, title2, share_tags = item
            desc += '\n\t%.2f %s tag:(%s)' % (score, title2, '/'.join(share_tags))
    print(desc)
    with open(distance_path, 'w') as f:
        f.write(desc)

def plot_heatmap(d):
    df = pd.DataFrame(d)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(df.corr().as_matrix(), cmap=plt.cm.Blues, alpha=0.8)
    # put the major ticks at the middle of each cell, notice "reverse" use of dimension
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.xaxis.tick_top()
    # set label
    ax.set_xticklabels(df.index, minor=False)
    ax.set_yticklabels(df.index, minor=False)
    # rotate the x title
    plt.xticks(rotation=90)
    plt.show()

def check_doc2vec_train_data(doc2vec_train_path):
    with open(doc2vec_train_path) as f:
        for line in f.readlines():
            print(line[:20], len(line))

def print_doc2vec_distance(book_infos, doc2vec_train_path, doc2vec_distance_path, use_tf_idf = True):
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
    # tag as input
    documents = []
    if use_tf_idf:
        for bookId in book_infos.keys():
            sentence = gensim.models.doc2vec.LabeledSentence(words=book_infos[bookId]['tags'], tags=book_infos[bookId]['bookId'])
            documents.append(sentence)
    # whole book as input
    else:
        documents = gensim.models.doc2vec.TaggedLineDocument(doc2vec_train_path)

    model = gensim.models.Doc2Vec(documents, size = 100, window = 5)
    model.save(output_path + 'doc2vec.model')

    desc = ''
    for bookId1 in book_infos.keys():
        desc += '\n\n %s 距离：' % book_infos[bookId1]['title'] 
        for (bookId2, score) in sorted(bookId_score_li, key=operator.itemgetter(2), reverse=True):
            desc += '\n\t%s %.4f' % (book_infos[bookId2]['title'], score)
    print(desc)

    with open(doc2vec_distance_path, 'w') as f:
        f.write(desc)

# gen_doc2vec_train_data(input_path, doc2vec_train_path)
# gen_idf_data(input_path, idf_path)

jieba.analyse.set_idf_path(idf_path) 
book_infos = read_book_infos(input_path)
calc_book_distance(book_infos)
print_book_distance(book_infos, distance_path)

# 绘制热度图
# plot_heatmap(d)

# 用 Doc2vec 距离衡量书籍距离
# check_doc2vec_train_data(doc2vec_train_path)        
# print_doc2vec_distance(book_infos, doc2vec_train_path, doc2vec_distance_path, use_tf_idf = False)