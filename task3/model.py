# -*- coding: utf-8 -*-
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model(object):
    def __init__(self):
        self.input_x1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 20], name="p")
        self.input_x2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 20], name="h")
        self.input_y = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 4], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.logits = self.forward()
        self.y_pred_cls = tf.argmax(tf.nn.sigmoid(self.logits), 1)
        self.loss = self.get_loss()
        self.opt = self.get_optimizer()
        self.acc = self.get_accuracy()

    def bilstm(self, x, hidden_size):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def forward(self):
        # 借鉴 https://github.com/terrifyzhao/text_matching/blob/master/esim/graph.py
        # https://codechina.csdn.net/mirrors/JesseYule/NLPBeginner/-/tree/master/3.textMatching(ESIM)/python/models
        embedding = tf.get_variable(dtype=tf.float32, shape=(5000, 128), name='emb')
        p_embedding = tf.nn.embedding_lookup(embedding, self.input_x1)
        h_embedding = tf.nn.embedding_lookup(embedding, self.input_x2)
        with tf.variable_scope("lstm_x1"):
            (p_f, p_b), _ = self.bilstm(p_embedding, 512)
        with tf.variable_scope("lstm_x2"):
            (h_f, h_b), _ = self.bilstm(h_embedding, 512)
        p = tf.concat([p_f, p_b], axis=2)
        h = tf.concat([h_f, h_b], axis=2)

        p = tf.nn.dropout(p, keep_prob=self.keep_prob)
        h = tf.nn.dropout(h, keep_prob=self.keep_prob)

        e = tf.matmul(p, tf.transpose(h, perm=[0, 2, 1]))
        a_attention = tf.nn.softmax(e)
        b_attention = tf.transpose(tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1])), perm=[0, 2, 1])

        a = tf.matmul(a_attention, h)
        b = tf.matmul(b_attention, p)

        m_a = tf.concat((a, p, a - p, tf.multiply(a, p)), axis=2)
        m_b = tf.concat((b, h, b - h, tf.multiply(b, h)), axis=2)

        with tf.variable_scope("lstm_a", reuse=tf.AUTO_REUSE):
            (a_f, a_b), _ = self.bilstm(m_a, 256)
        with tf.variable_scope("lstm_b", reuse=tf.AUTO_REUSE):
            (b_f, b_b), _ = self.bilstm(m_b, 256)

        a = tf.concat((a_f, a_b), axis=2)
        b = tf.concat((b_f, b_b), axis=2)

        a = tf.nn.dropout(a, keep_prob=self.keep_prob)
        b = tf.nn.dropout(b, keep_prob=self.keep_prob)

        a_avg = tf.reduce_mean(a, axis=2)
        b_avg = tf.reduce_mean(b, axis=2)

        a_max = tf.reduce_max(a, axis=2)
        b_max = tf.reduce_max(b, axis=2)

        v = tf.concat((a_avg, a_max, b_avg, b_max), axis=1)
        v = tf.layers.dense(v, 512, activation='tanh')
        v = tf.nn.dropout(v, keep_prob=self.keep_prob)
        logits = tf.layers.dense(v, 4, activation='tanh')
        return logits

    def get_loss(self):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])
        loss = cross_entropy + l2_loss * 0.005
        return loss

    def get_optimizer(self):
        return tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

    def get_accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
