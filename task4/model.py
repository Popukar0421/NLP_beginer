# -*- coding: utf-8 -*-
import tensorflow as tf


class Model(object):
    def __init__(self):
        self.input_x = tf.compat.v1.placeholder(tf.int32, [None, None])
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.logits, self.seq = self.forward()
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.y_pred_prob = tf.reduce_max(tf.nn.softmax(self.logits), 1)
        self.loss = self.get_loss()
        self.opt = self.get_optimizer()

    def forward(self):
        embedding = tf.get_variable('embedding', [2504, 128])
        embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(128)
        (outputs_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedding_input, dtype=tf.float32)
        output = tf.concat([outputs_fw, output_bw], axis=-1)
        W = tf.get_variable("W", [2 * 128, num_tags])
        matricized_output = tf.reshape(output, [-1, 2 * 128])
        matricized_unary_scores = tf.matmul(matricized_output, W)
        unary_scores = tf.reshape(matricized_unary_scores, [batch_size, max_seq_len, num_tags])
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, tags, sequence_lengths)
        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(unary_scores, transition_params, sequence_lengths_t)
        return log_likelihood, viterbi_sequence

    def get_loss(self):
        cross_entropy = tf.reduce_mean(-1 * self.loss)
        return cross_entropy 

    def get_optimizer(self):
        return tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def get_accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__ == __main__:
    Model()
