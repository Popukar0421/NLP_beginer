# coding: utf-8
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def read_file(file):
    df = pd.read_csv(file, sep='\t')
    return df["Phrase"].values, df["Sentiment"].values.tolist()


def process_file(filename, word_to_id, max_length=2000):
    contents, labels = read_file(filename)
    wordsList = np.load('./glove.6B/wordsList.npy')
    wordsList = wordsList.tolist()
    wordVectors = np.load('./glove.6B/wordVectors.npy')
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([wordVectors[wordsList.index(x)] for x in contents[i] if x in word_to_id])
        label_id.append(labels[i])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, int(max_length))
    y_pad = kr.utils.to_categorical(label_id, num_classes=5)
    x_train, x_val, y_train, y_test = train_test_split(x_pad, y_pad, test_size=0.2)
    return x_train, x_val, y_train, y_test


def read_vocab(vocab_dir):
    with open(vocab_dir, mode='r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    return words, word_to_id, id_to_word


def save_glove():
    embeddings_dict = {}
    with open("./glove.6B/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    np.save('./glove.6B/wordsList', np.array(list(embeddings_dict.keys())))
    np.save('./glove.6B/wordVectors', np.array(list(embeddings_dict.values()), dtype='float32'))


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def build_vocab(train_dir, vocab_dir):
    data_train, _ = read_file(train_dir)
    all_data = []
    vectorizer = CountVectorizer(analyzer='word')
    x = vectorizer.fit(data_train)
    for content in vectorizer.get_feature_names():
        all_data.append(content)
    # print(all_data)
    words = ['<PAD>'] + all_data
    open(vocab_dir, mode='w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')


if __name__ == '__main__':
    save_glove()
