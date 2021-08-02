import tensorflow as tf
from model import Model
from data_loader import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(sess, p_, h_, y_):
    """测试集上准曲率评估"""
    data_len = len(p_)
    batch_eval = batch_iter(p_, h_, y_, 128)
    total_loss, total_acc = 0, 0
    for batch_p, batch_h, batch_y in batch_eval:
        batch_len = len(batch_p)
        loss, acc = sess.run([model.loss, model.acc],
                             feed_dict={model.input_x1: batch_p,
                                        model.input_x2: batch_h,
                                        model.input_y: batch_y,
                                        model.keep_prob: 1.0})
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss/data_len, total_acc/data_len


def train(word_to_id, cat_to_id):
    # tf.reset_default_graph()
    print("Configuring TensorBoard and Saver...")
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 1000
    flag = False
    p_train, h_train, y_train = process_file('./snli_1.0/snli_1.0_train.txt', word_to_id, cat_to_id, 20)
    p_val, h_val, y_val = process_file('./snli_1.0/snli_1.0_dev.txt', word_to_id, cat_to_id, 20)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(1000):
            batch_train = batch_iter(p_train, h_train, y_train)
            for batch_p, batch_h, batch_ys in batch_train:
                if total_batch % 100 == 0:
                    loss_train, acc_train = sess.run([model.loss, model.acc],
                                                     feed_dict={model.input_x1: batch_p,
                                                                model.input_x2: batch_h,
                                                                model.input_y: batch_ys,
                                                                model.keep_prob: 0.5})
                    loss_val, acc_val = evaluate(sess, p_val, h_val, y_val)
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = total_batch
                        improve_str = "*"
                    else:
                        improve_str = ""
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, '\
                          'Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improve_str))
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break


if __name__ == "__main__":
    categories, cat_to_id, id_to_cat = read_category()
    words, word_to_id, id_to_word = read_vocab('vocab.txt')
    model = Model()
    train(word_to_id, cat_to_id)