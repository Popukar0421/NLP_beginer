# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr
from sklearn.feature_extraction.text import CountVectorizer


def load_data(filename):
    df = pd.read_table(filename)
    return df['sentence1'].values.astype('U'), df['sentence2'].values.astype('U'), df['gold_label'].values


def process_file(filename, word2id, cat2id, max_length=20):
    sentence1, sentence2, labels = load_data(filename)
    sentence1_id, sentence2_id, label_id = [], [], []
    for i in range(len(labels)):
        sentence1_id.append([word2id[x] for x in sentence1[i] if x in word2id])
        sentence2_id.append([word2id[x] for x in sentence2[i] if x in word2id])
        label_id.append(cat2id[labels[i]])
    s1_pad = kr.preprocessing.sequence.pad_sequences(sentence1_id, int(max_length))
    s2_pad = kr.preprocessing.sequence.pad_sequences(sentence2_id, int(max_length))
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat2id))
    return s1_pad, s2_pad, y_pad


def read_vocab(vocab_dir):
    with open(vocab_dir, mode='r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    return words, word_to_id, id_to_word


def read_category():
    categories = ['entailment', 'neutral', 'contradiction', '-']
    cat_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '-': 3}
    id_to_cat = {0: 'entailment', 1: 'neutral', 2: 'contradiction', 3: '-'}
    return categories, cat_to_id, id_to_cat


def batch_iter(p, h, y, batch_size=64):
    data_len = len(p)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    p_shuffle = p[indices]
    h_shuffle = h[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield p_shuffle[start_id:end_id], h_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def build_vocab(train_dir, vocab_dir):
    p_train, h_train, _ = load_data(train_dir)
    all_data = []
    vectorizer = CountVectorizer(analyzer='word', max_features=5000)
    x = vectorizer.fit(p_train.tolist()+h_train.tolist())
    for content in vectorizer.get_feature_names():
        all_data.append(content)
    # print(all_data)
    words = ['<PAD>'] + all_data
    open(vocab_dir, mode='w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')


if __name__ == '__main__':
    # build_vocab('./snli_1.0/snli_1.0_train.txt', './vocab.txt')
    categories, cat2id, id2cat = read_category()
    words, word2id, id2word = read_vocab('vocab.txt')
    x, y, z = process_file("./snli_1.0/snli_1.0_train.txt",  word2id, cat2id)
    # print(cat2id)
    # print(word2id)