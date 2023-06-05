# -*- coding: utf-8 -*-
"""
这段代码是用于训练文本分类模型的脚本。在训练过程中，它加载了预定义的数据特征，并按批次进行训练。

代码的主要流程如下：

加载数据特征：通过调用get_features函数获取训练数据的特征，包括输入的ID、掩码、段ID和标签ID等。
设置图和会话：创建tf.Session对象，并在其中构建模型图。还创建了一个用于保存模型的saver对象。
加载已保存的模型：检查之前是否保存了模型，如果有，则加载最新的模型。
进行训练：使用循环进行多个训练周期，在每个训练周期中，对数据进行随机打乱，并按批次进行训练。每次训练时，通过提供适当的输入和标签ID来运行模型的优化器。
记录训练过程：使用TensorBoard来记录训练过程中的损失和准确率等指标，并将其写入日志文件中。
保存模型：定期保存训练的模型，以便后续使用。
输出训练日志：定期输出训练的损失和准确率等信息，用于监控训练进度。
"""



import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
# import tensorflow as tf 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from networks import NetworkAlbertTextCNN
from classifier_utils import get_features
from hyperparameters import Hyperparamters as hp
from utils import select,time_now_string


pwd = os.path.dirname(os.path.abspath(__file__))
MODEL = NetworkAlbertTextCNN(is_training=True)


# Get data features
input_ids,input_masks,segment_ids,label_ids = get_features()
num_train_samples = len(input_ids)
indexs = np.arange(num_train_samples)               
num_batchs = int((num_train_samples - 1) /hp.batch_size) + 1

# Set up the graph 
saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load model saved before
MODEL_SAVE_PATH = os.path.join(pwd, hp.file_save_model)
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
     print('Restored model!')


with sess.as_default():
    # Tensorboard writer
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    for i in range(hp.num_train_epochs):
        np.random.shuffle(indexs)
        for j in range(num_batchs-1):
            # Get ids selected
            i1 = indexs[j * hp.batch_size:min((j + 1) * hp.batch_size, num_train_samples)]
            
            # Get features
            input_id_ = select(input_ids,i1)
            input_mask_ = select(input_masks,i1)
            segment_id_ = select(segment_ids,i1)
            label_id_ = select(label_ids,i1)
            
            # Feed dict
            fd = {MODEL.input_ids: input_id_,
                  MODEL.input_masks: input_mask_,
                  MODEL.segment_ids:segment_id_,
                  MODEL.label_ids:label_id_}
            
            # Optimizer
            sess.run(MODEL.optimizer, feed_dict = fd)   
            
            # Tensorboard
            if j%hp.summary_step==0:
                summary,glolal_step = sess.run([MODEL.merged,MODEL.global_step], feed_dict = fd)
                writer.add_summary(summary, glolal_step) 
                
            # Save Model
            if j%(num_batchs//hp.num_saved_per_epoch)==0:
                if not os.path.exists(os.path.join(pwd, hp.file_save_model)):
                    os.makedirs(os.path.join(pwd, hp.file_save_model))                 
                saver.save(sess, os.path.join(pwd, hp.file_save_model, 'model'+'_%s_%s.ckpt'%(str(i),str(j))))            
            
            # Log
            if j % hp.print_step == 0:
                fd = {MODEL.input_ids: input_id_,
                      MODEL.input_masks: input_mask_,
                      MODEL.segment_ids:segment_id_,
                      MODEL.label_ids:label_id_}
                accuracy,loss = sess.run([MODEL.accuracy,MODEL.loss], feed_dict = fd)
                print('Time:%s, Epoch:%s, Batch number:%s/%s, Loss:%s, Accuracy:%s'%(time_now_string(),str(i),str(j),str(num_batchs),str(loss),str(accuracy)))   
    print('Train finished')
    
    
    
    
    
    
    
    




