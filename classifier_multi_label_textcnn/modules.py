# -*- coding: utf-8 -*-
"""
这是一个使用TensorFlow实现的TextCNN模型的代码片段。TextCNN是一种用于文本分类的卷积神经网络模型，它利用多个不同大小的卷积核对输入文本进行卷积操作，然后通过最大池化提取特征。
"""


import tensorflow as tf
# from tensorflow.contrib.rnn import DropoutWrapper
from hyperparameters import Hyperparamters as hp


def cell_textcnn(inputs,is_training):
    # Add a dimension in final shape
    inputs_expand = tf.expand_dims(inputs, -1)
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    with tf.name_scope("TextCNN"):
        for i, filter_size in enumerate(hp.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, hp.embedding_size, 1, hp.num_filters]
                W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1),dtype=tf.float32, name="W")
                b = tf.Variable(tf.constant(0.1, shape=[hp.num_filters]),dtype=tf.float32, name="b")
                conv = tf.nn.conv2d(
                                    inputs_expand,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                                        h,
                                        ksize=[1, hp.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = hp.num_filters * len(hp.filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # Dropout
    h_pool_flat_dropout = tf.nn.dropout(h_pool_flat, rate=1-hp.keep_prob if is_training else 0)
    return h_pool_flat_dropout
            

