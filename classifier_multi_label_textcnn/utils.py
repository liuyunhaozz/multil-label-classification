# -*- coding: utf-8 -*-
"""
这段代码是一个工具函数集合，提供了一些常用的数据处理和文件操作功能。它包括以下函数：

cut_list(data, size)：将一个列表按指定大小切分为多个子列表。
time_now_string()：获取当前时间的字符串表示。
select(data, ids)：根据给定的索引列表，从原始数据中选取对应的数据。
load_txt(file)：从文本文件中加载数据，返回一个包含所有行的列表。
save_txt(file, lines)：将数据保存到文本文件中。
load_csv(file, header=None)：从CSV文件中加载数据，返回一个DataFrame对象。
save_csv(dataframe, file, header=True, index=None, encoding="gbk")：将DataFrame对象保存为CSV文件。
save_excel(dataframe, file, header=True, sheetname='Sheet1')：将DataFrame对象保存为Excel文件。
load_excel(file, header=0)：从Excel文件中加载数据，返回一个DataFrame对象。
load_vocabulary(file_vocabulary_label)：从文件中加载词汇表，并将其转换为字典形式的ID到标签和标签到ID的映射。
shuffle_two(a1, a2)：随机打乱两个列表的顺序，并保持对应关系。

"""


import time
import numpy as np
import pandas as pd


def cut_list(data,size):
    """
    data: a list
    size: the size of cut
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data)-1)//size + 1)]

   
def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime( time.time() )) 


def select(data,ids):
    return [data[i] for i in ids]


def load_txt(file):
    with  open(file,encoding='utf-8',errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        print("Load data from file (%s) finished !"%file)
    return lines


def save_txt(file,lines):
    lines = [l+'\n' for l in lines]
    with  open(file,'w+',encoding='utf-8') as fp:#a+添加
        fp.writelines(lines)
    return "Write data to txt finished !"


def load_csv(file,header=None):
    return pd.read_csv(file,encoding='utf-8',header=header,error_bad_lines=False)#,encoding='gbk'


def save_csv(dataframe,file,header=True,index=None,encoding="gbk"):
    return dataframe.to_csv(file,
                            mode='w+',
                            header=header,
                            index=index,
                            encoding=encoding)



def save_excel(dataframe,file,header=True,sheetname='Sheet1'):
    return dataframe.to_excel(file,
                         header=header,
                         sheet_name=sheetname) 
    

def load_excel(file,header=0):
    return pd.read_excel(file,
                         header=header,
                         )

def load_vocabulary(file_vocabulary_label):
    """
    Load vocabulary to dict
    """
    vocabulary = load_txt(file_vocabulary_label)
    dict_id2label,dict_label2id = {},{}
    for i,l in enumerate(vocabulary):
        dict_id2label[str(i)] = str(l)
        dict_label2id[str(l)] = str(i)
    return dict_id2label,dict_label2id


def shuffle_two(a1,a2):
    """
    Shuffle two list
    """
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    a1_ = [a1[l] for l in ran]
    a2_ = [a2[l] for l in ran]
    return a1_, a2_


if __name__ == '__main__': 
    print('')




