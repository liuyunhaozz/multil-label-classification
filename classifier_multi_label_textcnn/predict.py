# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:12:37 2019

@author: cm
"""


import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from networks import NetworkAlbertTextCNN
from classifier_utils import get_feature_test,id2label
from hyperparameters import Hyperparamters as hp
from preprocess import read_csv_file
 
          
class ModelAlbertTextCNN(object,):
    """
    Load NetworkAlbert TextCNN model
    """
    def __init__(self):
        self.albert, self.sess = self.load_model()
    @staticmethod
    def load_model():
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                albert =  NetworkAlbertTextCNN(is_training=False)
                saver = tf.train.Saver()  
                sess.run(tf.global_variables_initializer())
                checkpoint_dir = os.path.abspath(os.path.join(pwd,hp.file_load_model))
                print (checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        return albert,sess

MODEL = ModelAlbertTextCNN()
print('Load model finished!')


def get_label(sentence):
    """
    Prediction of the sentence's label.
    """
    feature = get_feature_test(sentence)
    fd = {MODEL.albert.input_ids: [feature[0]],
          MODEL.albert.input_masks: [feature[1]],
          MODEL.albert.segment_ids:[feature[2]],
          }
    prediction = MODEL.sess.run(MODEL.albert.predictions, feed_dict=fd)[0]     
    return [id2label(l) for l in np.where(prediction==1)[0] if l!=0]      



if __name__ == '__main__':
    # Test
    data_folder = 'data'
    test_data = 'test_onehot.csv'
    test_csv = read_csv_file(data_folder, test_data)
    sentences = test_csv['content'].tolist()
    labels = []
    for index, row in test_csv.iterrows():
        label = []
        for key, value in row.items():
            if value == 1.0:
                label.append(key)
        labels.append(label)
    for i, sentence in enumerate(sentences):
        print()
        print('-' * 10, '案件描述:', sentence, '-' * 10) 
        print('-' * 10, '预测值:', ','.join(get_label(sentence)), '-' * 10)
        print('-' * 10, '真值:', ','.join(labels[i]), '-' * 10)
        print()



    
    
    
    
