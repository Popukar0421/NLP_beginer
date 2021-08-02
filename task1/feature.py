import numpy
import random


def data_split(data, test_rate=0.3, max_item=10000):
    """把数据按一定比例划分成训练集和测试集"""
    train = list()
    test = list()
    i = 0
    for datum in data:
        i += 1
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
        if i > max_item:
            break
    return train, test


class Bag:
    """Bag of words"""
    def __init__(self, my_data, max_item=1000):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()
        self.len = 0
        self.train, self.test = data_split(my_data, test_rate=0.3, max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]
        self.test_y = [int(term[3]) for term in self.test]
        self.train_matrix = None
        self.test_matrix = None

    def get_words(self):
        for term in self.data:
            s = term[2]
            s = s.upper()
            words = s.split()
            for word in words:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = numpy.zeros((len(self.test), self.len))
        self.train_matrix = numpy.zeros((len(self.train), self.len))

    def get_matrix(self):
        for i in range(len(self.train)):
            s = self.train[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.train_matrix[i][self.dict_words[word]] = 1
        for i in range(len(self.test)):
            s = self.test[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.test_matrix[i][self.dict_words[word]] = 1


class Gram:
    """N-gram"""
    def __init__(self, my_data, dimension=2, max_item=1000):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()
        self.len = 0
        self.dimension = dimension
        self.train, self.test = data_split(my_data, test_rate=0.3, max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]
        self.test_y = [int(term[3]) for term in self.test]
        self.train_matrix = None
        self.test_matrix = None

    def get_words(self):
        for d in range(1, self.dimension + 1):
            for term in self.data:
                s = term[2]
                s = s.upper()
                words = s.split()
                for i in range(len(words) - d + 1):
                    temp = words[i:i + d]
                    temp = "_".join(temp)
                    if temp not in self.dict_words:
                        self.dict_words[temp] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = numpy.zeros((len(self.test), self.len))
        self.train_matrix = numpy.zeros((len(self.train), self.len))

    def get_matrix(self):
        for d in range(1, self.dimension + 1):
            for i in range(len(self.train)):
                s = self.train[i][2]
                s = s.upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    self.train_matrix[i][self.dict_words[temp]] = 1
            for i in range(len(self.test)):
                s = self.test[i][2]
                s = s.upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    self.test_matrix[i][self.dict_words[temp]] = 1