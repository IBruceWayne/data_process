# -*- coding: utf-8 -*-
# @author: gcg
import spacy
import torchtext.data as data
import os
import io
import re
from collections import Counter
from gensim.models import KeyedVectors
import numpy as np
import pickle as pk
from collections import defaultdict
import tqdm
data_dir = '/home/cggong/data/mtl-dataset'
embedding_dir = '/home/kzxuan/word_embedding/GoogleNews_vectors_negative300.bin.gz'


def get_all_text():
    # 返回指定目录下的所有文本内容
    files = os.listdir(data_dir)  # 获取当前目录下的所有文件
    text_list = []
    for file in files:
        if not os.path.isdir(file):  # 跳过文件夹
            file_path = os.path.join(data_dir, file)
            if file.endswith('unlabel') or file.endswith('train') or file.endswith('test'):
                with open(file_path, 'r', encoding='ISO-8859-2') as fp:
                    lines = fp.readlines()
                lines = [l.split('\t')[-1] for l in lines]  # 去除标签
                # print('---读取文件：', file, len(lines))
                text_list.extend(lines)
    return text_list


def custom_tokenizer(w):
    # 接收一串文本（str） 预处理后分词，返回分词后的列表(list)
    w = re.sub(r"([?.!,¿():])", r" \1 ", w)  # 在标点符号前后加一个空格
    w = re.sub(r'[" "]+', " ", w)  # 将多个空格缩减为一个
    w = re.sub(r"[^a-zA-Z.,'!?;']+", " ", w)  # 替换掉所有非法字符
    w = w.strip()  # 因为word2vec区分大小写，这里不进行小写化
    return w.split()


def replace_contractions(sentences):
    # 替换掉缩写
    contr_dict = {"i\'m": "I am",
                  "won\'t": "will not",
                  "\'s": "",
                  "i\'ll": "i will",
                  "i\'ve": "i have",
                  "n\'t": "not",
                  "i\'d": "i would",
                  "isnt": "is not",
                  "doesnt": "does not",
                  "didnt": "did not",
                  "wasnt": "was not"}
    res_sentences = []
    for sent in sentences:
        for contr in contr_dict:
            sent = sent.replace(contr, " "+contr_dict[contr])
        res_sentences.append(sent)
    return res_sentences


def tokenize(sentences, restrict_to_len=-1):
    """
    :params sentence_list: list of strings
    :returns tok_sentences: list of list of tokens
    """
    if restrict_to_len > 0:
        tok_sentences = [re.findall(r"[\w]+[']*[\w]+|[\w]+|[.,!?;]", x) for x in sentences if len(x) > restrict_to_len]
    else:
        tok_sentences = [re.findall(r"[a-zA-Z]+[']*[_a-zA-Z]+|[a-zA-Z]+|[.,!?;]", x) for x in sentences]  # \w是字母、数字、下划线 这里去掉了数字
    return tok_sentences


def get_word_dict(text):
    words = []
    text = tokenize(text)
    for temp in text:
        words.extend(temp)
    word_dict = Counter(words)
    return word_dict


def compare(text_list):
    # wv = KeyedVectors.load_word2vec_format(embedding_dir, binary=True)
    # pk.dump(wv.vocab, open('vocab.pk', 'wb'))
    wv = pk.load(open('vocab.pk', 'rb'))
    # text_list = tokenize(text_list
    # text_list = [temp.split() for temp in text_list]
    vocab = defaultdict(int)
    for sentence in text_list:
        for word in sentence:
            vocab[word] += 1
    compare_vocab(vocab, wv)


def compare_vocab(vocab, wv):
    oov = []
    in_common = []
    in_common_freq = 0
    oov_freq = 0

    # Compose the vocabulary given the sentence tokens
    for word in vocab:
        if word in wv:
            in_common.append(word)
            in_common_freq += vocab[word]
        else:
            oov.append(word)
            oov_freq += vocab[word]
    print('Found embeddings for {:.2%} of vocab, size {} {}'.format(len(in_common) / len(vocab), len(in_common), len(vocab)))
    print('Found embeddings freq for  {:.2%} of all text'.format(in_common_freq / (in_common_freq + oov_freq)))
    in_common, oov = sorted(in_common)[::-1], sorted(oov)[::-1]

    sorted_oov = sorted(oov, key=lambda x: vocab[x], reverse=True)

    # Show oov words and their frequencies
    if (len(sorted_oov) > 0):
        print("oov words:")
        for word in sorted_oov[:50]:
            print("%s\t%s" % (word, vocab[word]))
    else:
        print("No words were out of vocabulary.")


def generate_embedding(min_count):
    wv = KeyedVectors.load_word2vec_format(embedding_dir, binary=True)
    embedding_dim = len(wv['hello'])
    print('---词向量维度：', embedding_dim, '词典大小：', len(wv.vocab))
    word = ['<unk>', '<pad>']
    embedding = [np.random.randn(embedding_dim)/100, np.zeros(embedding_dim)]  # 随机初始化unknow向量
    oov = []
    oov_count = 0
    # 从数据集获得词典
    text = replace_contractions(get_all_text())
    wd = get_word_dict(text)
    compare_vocab(wd, wv.vocab)
    for w in wd.keys():
        if w in wv.vocab:
            word.append(w)
            embedding.append(wv[w])
        else:
            oov_count += 1
            if wd[w] >= min_count:
                oov.append(w)
    print('---数据集词典大小：{}, oov数量: {}, 创建的均值向量：{}'.format(len(wd), oov_count, len(oov)))
    print(oov)
    mean_embedding = np.zeros(embedding_dim)  # 为出现大于min_count的词用均值初始化一个词向量
    for e in embedding:
        mean_embedding += e
    mean_embedding = mean_embedding/len(embedding)
    for w in oov:
        word.append(w)
        embedding.append(mean_embedding)
    assert len(word) == len(embedding)
    embedding = np.asarray(embedding)
    pk.dump((word, embedding), open('mtl-embedding.pk', 'wb'))
    print('词表大小：', len(word))



t = get_all_text()
# t = [custom_tokenizer(w) for w in t]
t = replace_contractions(t)
t = tokenize(t)
compare(t)


