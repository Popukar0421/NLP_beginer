# coding: utf-8
import tensorflow as tf

class RnnModel(object):
    def __init__(self, config):
        self.cfg = config
        self.input_x = tf.compat.v1.placeholder(tf.float32, [None, self.cfg.max_seq_length])
        self.input_y = tf.compat.v1.placeholder(tf.int32, [None, self.cfg.num_classes])
        self.logits = self.rnn()
        self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.loss = self.get_loss()
        self.opt = self.get_optimizer()
        self.acc = self.get_accuracy()

    def rnn(self):
        x = tf.expand_dims(self.input_x, -1)
        cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(2)
        outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])  # [n_step, batch_size, n_hidden]
        outputs = outputs[-1]  # [batch_size, n_hidden]
        W = tf.Variable(tf.random_normal([2, self.cfg.num_classes]))
        b = tf.Variable(tf.random_normal([self.cfg.num_classes]))
        logits = tf.matmul(outputs, W) + b  # model : [batch_size, n_class]
        return logits

    def get_loss(self):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y))
        if self.cfg.is_l2:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])
            loss = cross_entropy + l2_loss * 0.005
        else:
            loss = cross_entropy
        return loss

    def get_optimizer(self):
        return tf.compat.v1.train.AdamOptimizer(learning_rate=self.cfg.learning_rate).minimize(self.loss)

    def get_accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
