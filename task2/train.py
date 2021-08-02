import tensorflow as tf
from model import RnnModel
from config import ModelConfig
from data_loader import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(sess, x_, y_):
    """测试集上准曲率评估"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0
    total_acc = 0
    for batch_xs, batch_ys in batch_eval:
        batch_len = len(batch_xs)
        loss, acc = sess.run([model.loss, model.acc],
                             feed_dict={model.input_x: batch_xs, model.input_y: batch_ys})
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss/data_len, total_acc/data_len


def train(word_to_id, cfg):
    # 配置Saver
    # saver = tf.compat.v1.train.Saver()
    # 训练模型
    print("Training and evaluating...")
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
    flag = False
    X_train, X_val,y_train, y_val = process_file(cfg.train_path, word_to_id, cfg.max_seq_length)
    # print(X_train.shape, y_train.shape)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for step in range(cfg.num_epochs):
            print('Epoch:', step)
            batch_train = batch_iter(X_train, y_train)
            for batch_xs, batch_ys in batch_train:
                if total_batch % cfg.print_per_batch == 0:
                    loss_train, acc_train = sess.run([model.loss, model.acc],
                                                     feed_dict={model.input_x: X_train, model.input_y: y_train})
                    loss_val, acc_val = evaluate(sess, X_val, y_val)

                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = total_batch
                        improve_str = "*"
                    else:
                        improve_str = ""
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, '\
                          'Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, improve_str))
                sess.run(model.opt, feed_dict={model.input_x: batch_xs, model.input_y: batch_ys})
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    #  验证集准确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break


if __name__ == "__main__":
    cfg = ModelConfig('task2')
    model = RnnModel(cfg)
    words, word_to_id, id_to_word = read_vocab(cfg.vocab_path)
    train(word_to_id, cfg)
