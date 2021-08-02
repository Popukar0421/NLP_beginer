# coding: utf-8
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr


def read_poetry(filename):
    all_poetry = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line)> 2:
                line = line.replace('\n', '').replace('\r', '').replace('，', '').replace('。', '')
                all_poetry.append(line)
    return all_poetry


def build_vocab(filename, vocab_dir, vocab_size=5000):
    poetrys = read_poetry('./data/poetryFromTang.txt')
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    # 取前多少个常用字
    vocab = ['<PAD>'] + list(words)
    with open(vocab_dir, mode='w', encoding='utf-8', errors='ignore') as wf:
        wf.write('\n'.join(vocab) + '\n')


def read_vocab(vocab_dir):
    with open(vocab_dir, mode='r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    return words, word_to_id, id_to_word


def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1)*s))
    return words[sample]


def batch_data(x, word_to_id, batch_size=64):
    x_pad = [list(map(word_to_id, poetry)) for poetry in x]
    indices = np.arange(len(x_pad))
    y = np.copy(x_pad)
    y[:-1], y[-1] = x_pad[1:], x_pad[0]
    for i in range(42):
        np.random.shuffle(indices)
    x_shuffle = x_pad[indices]
    y_shuffle = y[indices]
    for i in range(len(x_pad)//64):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, len(x_pad))
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':
    print(1)
