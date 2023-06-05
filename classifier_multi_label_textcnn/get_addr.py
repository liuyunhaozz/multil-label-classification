"""
读取一个CSV文件，通过百度AI的地址识别接口对其中的文本进行处理，并将处理结果保存为一个新的CSV文件。
"""

import os
import pandas as pd
from aip import AipNlp


def read_csv_file(data_folder, file_name):
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    return df 

def df2csv(data_folder, file_name, df: pd.DataFrame):
    path = os.path.join(data_folder, file_name)
    df.to_csv(path, index=None, encoding='utf_8_sig') 


if __name__ == '__main__':
    
    df = pd.DataFrame()
    data_folder = 'data'
    data = 'finaldata.csv'

    """ 你的 APPID AK SK """
    APP_ID = ''
    API_KEY = ''
    SECRET_KEY = '' 

    client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

    data_csv = read_csv_file(data_folder, data)
    texts = data_csv['案件描述'].tolist()
    for text in texts:
        # print(text)
        print(text)
        text = '大连市' + text
        """ 调用地址识别接口 """
        content = client.address(text);
        df = df.append(content, ignore_index=True)
    df2csv(data_folder, 'new_address.csv', df)
    print('Done.')