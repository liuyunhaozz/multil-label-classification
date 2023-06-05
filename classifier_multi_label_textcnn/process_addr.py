"""
在给定的代码中，首先定义了一个函数read_csv_file(data_folder, file_name)，用于读取指定文件夹下的CSV文件并返回一个Pandas DataFrame对象。函数接受两个参数：data_folder表示数据文件夹路径，file_name表示CSV文件名。函数将文件路径和文件名合并为完整的文件路径，然后使用pd.read_csv()函数读取CSV文件并将其存储在DataFrame对象df中，最后将该DataFrame返回。
"""
import os
import pandas as pd


def read_csv_file(data_folder, file_name):
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    return df 


if __name__ == '__main__':
    data_folder = 'data'
    data = 'address.csv'
    df = read_csv_file(data_folder, data)
    count = 0
    for i in df['']
    print(len(df))