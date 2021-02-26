#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File     : 04_homework_text_clustering.py
# @todo     : 文本聚类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from collections import defaultdict

data = pd.read_csv('./data/abcnews-date-text.csv')

# 查看重复的数据行
print(data[data['headline_text'].duplicated(keep=False)].sort_values('headline_text').head(8))
# 删除重复行
data.drop_duplicates(inplace=True)

# 2 数据预处理
# 2.1 为向量化表示进行前处理
# 进行自然语言处理时，必须将单词转换为机器学习算法可以利用的向量。
# 如果目标是对文本数据进行机器学习建模，例如电影评论或推文或其他任何内容，则需要将文本数据转换为数字。此过程称为“嵌入”或“向量化”。
# 进行向量化时，请务必记住，它不仅仅是将单个单词变成单个数字。
# 单词可以转换为数字，整个文档就可以转换为向量。
# 向量的维度往往不止一个，而且对于文本数据，向量通常是高维的。
# 这是因为特征数据的每个维度将对应一个单词，而我们所处理的文档通常包含数千个单词。

# 2.2 TF-IDF
# 在信息检索中，tf–idf 或 TFIDF（term frequency–inverse document frequency）是一种数值统计，旨在反映单词对语料库中文档的重要性。
# 在信息检索，文本挖掘和用户建模的搜索中，它通常用作加权因子。
# tf-idf 值与单词在文档中出现的次数成正比，同时被单词在语料库中的出现频率所抵消，这有助于调整某些单词通常会更频繁出现的事实。
# 如今，tf-idf是最流行的术语加权方案之一。在数字图书馆领域，有83％的基于文本的推荐系统使用tf-idf。
#
# 搜索引擎经常使用tf–idf加权方案的变体作为在给定用户查询时对文档相关性进行评分和排名的主要工具。
# tf–idf可成功用于各种领域的停用词过滤，包括文本摘要和分类。
#
# 排名函数中最简单的是通过将每个查询词的tf–idf相加得出，许多更复杂的排名函数是此简单模型的变体。

punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = data['headline_text'].values


# TfidfVectorizer 使用方法详见：
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,stop_words = stop_words)
X = vectorizer.fit_transform(desc)
word_features = vectorizer.get_feature_names()
print(len(word_features))
print(word_features[:50])

# 2.3 Stemming
# stemming 是将单词还原为词干（即词根形式）的过程。
# 词根形式不一定是单词本身，而是可以通过连接正确的后缀来生成单词。
# 例如，“fish”，“fishes”和“fishing”这几个词的词干都是“fish”，这是一个正确的单词。
# 另一方面，“study”，“studies”和“studying”一词源于“studi”，这不是一个正确的英语单词。

# 2.4 Tokenizing
# Tokenization 将句子分解为单词和标点符号

# SnowballStemmer 使用方法详见： https://www.kite.com/python/docs/nltk.SnowballStemmer
stemmer = SnowballStemmer('english')
#  RegexpTokenizer 使用方法详见： https://www.kite.com/python/docs/nltk.RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')


def tokenize(text):
    """先进行 stemming 然后 tokenize
    params:
    text: 一个句子

    return:
    tokens 列表
    """
    return [stemmer.stem(s) for s in tokenizer.tokenize(text)]


# 2.5 使用停用词、stemming 和自定义的 tokenizing 进行 TFIDF 向量化
vectorizer2 = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
X2 = vectorizer2.fit_transform(desc)
word_features2 = vectorizer2.get_feature_names()
print(len(word_features2))
print(word_features2[:50])

vectorizer3 = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize, max_features=1000)
X3 = vectorizer3.fit_transform(desc)
words = vectorizer3.get_feature_names()
print(len(words))
print(word_features2[:50])


# 3 K-Means 聚类
# 3.1 使用手肘法选择聚类簇的数量
# 随着聚类数k的增大,样本划分会更加的精细,每个簇的聚合程度会逐渐提高,那么误差平方和SSE自然会逐渐变小,
# 并且当k小于真实的簇类数时,由于k的增大会大幅增加每个簇的聚合程度,因此SSE的下降幅度会很大,
# 而当k到达真实聚类数时,再增加k所得到的聚合程度回报会迅速变小,所以SSE的下降幅度会骤减,
# 然后随着k值的继续增大而趋于平缓,也就是说SSE和k的关系类似于手肘的形状,而这个肘部对应的k值就是数据的真实聚类数.
# 因此这种方法被称为手肘法.


def elbow_method(n_cluster):
    wcss = []
    for n in range(1, n_cluster):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X3)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, n_cluster), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('elbow.png')
    plt.show()


def classifier(X, n_clusters, n_init=20, n_jobs=2, max_iter=300, random_state=0, every_class_word=25):
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs, max_iter=max_iter, random_state=random_state)
    kmeans.fit(X)
    common_words = kmeans.cluster_centers_.argsort()[:, -1: 0 if every_class_word == -1 else -(every_class_word + 1):-1]
    result = defaultdict(list)
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
        result[num] = [words[word] for word in centroid]
    return result


if __name__ == '__main__':

    # 3.2 Clusters 等于 3
    elbow_method(25)
    classifier(X3, 3)
    # 3.3 Clusters 等于 5
    classifier(X3, 5)
    # 3.4 Clusters 等于 6
    classifier(X3, 6)
    # 3.5 Clusters 等于 8
    classifier(X3, 8)
    # 3.6 Clusters 等于 10
    classifier(X3, 10)