import tensorflow as tf
from model import PoetryModel
from data_loader import *


def train(word_to_id):
    model = PoetryModel()
    batch_eval = batch_data(poetrys_vector,word_to_id, 64)
    all_loss = 0.0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.global_variables())
        for epoch in range(100):
            for x, y in batch_eval:
                train_loss = sess.run([model.loss],
                                      feed_dict={model.input_x: x,
                                      model.input_y: y})
            all_loss = all_loss + train_loss
        saver.save(sess, './ck_model/p')
        print(epoch, ' Loss: ', all_loss * 1.0 / n_chunk)


def gen_head_poetry(heads, type=5):
    model = PoetryModel()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, './ck_model/p')
        poem = ''
        for head in heads:
            flag = True
            while flag:
                # x = np.array([list(map(word_num_map.get, u'['))])
                # probs = sess.run([model.probs], feed_dict={input_data: x})
                sentence = head
                x = np.zeros((1, 1))
                x[0, 0] = word_to_id[sentence]
                probs = sess.run([model.probs], feed_dict={input_data: x})
                word = to_word(probs)
                sentence += word
                while word != u'。':
                    x = np.zeros((1, 1))
                    x[0, 0] = word_to_id[word]
                    probs = sess.run([model.probs], feed_dict={input_data: x})
                    word = to_word(probs)
                    sentence += word
                if len(sentence) == 2 + 2 * type:
                    sentence += u'\n'
                    poem += sentence
                    flag = False
            return poem


if __name__ == '__main__':
    poetrys = read_poetry('./data/poetryFromTang.txt')
    words, word_to_id, id_to_word = read_vocab('vocab.txt')
    poetrys_vector = []
    for i in range(len(poetrys)):
        poetrys_vector.append([word_to_id[x] for x in poetrys[i] if x in word_to_id])
    train(word_to_id)
    gen_head_poetry(u'天下之大', 5)


