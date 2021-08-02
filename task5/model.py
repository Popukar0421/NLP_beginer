# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class PoetryModel(object):
    def __init__(self):
        self.input_x = tf.compat.v1.placeholder(tf.int32, [64, None])
        self.input_y = tf.compat.v1.placeholder(tf.float32, [64, None])
        self.logits, self.lstm_cell, self.initial_state = self.forward()
        self.probs = tf.nn.softmax(self.logits)
        self.loss = self.get_loss()
        self.opt = self.get_optimizer()


    def forward(self):
        embedding = tf.get_variable('embedding', [2506, 128])
        inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        cell = tf.nn.rnn_cell.BasicLSTMCell(128, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.GRUCell(128, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
        initial_state = cell.zero_state(64, tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, 128])
        softmax_w = tf.get_variable("softmax_w", [128, 2506])
        softmax_b = tf.get_variable("softmax_b", [2506])
        logits = tf.matmul(output, softmax_w) + softmax_b
        print(logits)
        return logits, cell, initial_state

    def get_loss(self):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y))
        return cross_entropy

    def get_optimizer(self):
        return tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)


if __name__ == '__main__':
    model = PoetryModel()