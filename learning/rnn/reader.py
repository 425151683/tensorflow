from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().decode("utf8").replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    counter_pair = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words = zip(*counter_pair)[0]
    vocab = dict(zip(words, range(len(words))))
    return vocab

def _file_to_word_ids(filename,vocab):
    data = _read_words(filename)
    return [vocab[word] for word in data if word in vocab]

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    vocab = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, vocab)
    valid_data = _file_to_word_ids(valid_path, vocab)
    test_data = _file_to_word_ids(test_path, vocab)
    vocabulary_length = len(vocab)

    return train_data, valid_data, test_data, vocabulary_length

def ptb_producer(raw_data, batch_size, num_step, name=None):
    raw_data = tf.convert_to_tensor(raw_data, dtype=tf.int32, name='raw_data')

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_step
    assertion = tf.assert_positive(epoch_size, message="epoch_size==0")
    with tf.control_dependencies([assertion]):
        epoch_size = tf.identity(epoch_size, name='epoch_size')

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i*num_step], [batch_size, (i+1)*num_step])
    y = tf.strided_slice(data, [0, i * num_step + 1], [batch_size, (i + 1) * num_step + 1])
    x.set_shape([batch_size, num_step])
    y.set_shape([batch_size, num_step])

    return x, y

#train_data, valid_data, test_data, vocabulary_length= ptb_raw_data("simple-examples/data")
#print (ptb_producer(train_data, 10, 32))