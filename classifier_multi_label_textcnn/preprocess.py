"""
这段代码包含了一些处理CSV文件的函数。以下是各个函数的功能说明：

read_csv_file(data_folder, file_name): 读取指定文件夹下的CSV文件并返回一个Pandas DataFrame对象。
process_csv(df: pd.DataFrame): 对输入的DataFrame进行处理，提取所需的列，并添加一列占位符。然后将数据划分为训练集和测试集，并返回两个DataFrame对象。
df2csv(data_folder, file_name, df: pd.DataFrame): 将DataFrame对象保存为CSV文件，存储路径由指定的文件夹和文件名确定。
print_info(df: pd.DataFrame): 打印DataFrame中的一些信息，包括总数和各列的值计数。


"""
import os
import pandas as pd


def read_csv_file(data_folder, file_name):
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    return df 

def process_csv(df: pd.DataFrame):
    new_df = df.loc[:, ['案件描述', '家庭保护', '学校保护', '社会保护', '网络保护', '政府保护']]
    new_df.insert(1, '|', value=0)
    new_df = new_df.rename(columns={'案件描述': 'content'})
    new_df = new_df.fillna(value=0)
    df_train = new_df.sample(frac=0.8)
    df_test = new_df[~df.index.isin(df_train.index)]
    return df_train, df_test 

def df2csv(data_folder, file_name, df: pd.DataFrame):
    path = os.path.join(data_folder, file_name)
    df.to_csv(path, index=None, encoding='utf_8_sig') 

def print_info(df: pd.DataFrame):
    print('Total: ', len(df))
    print(df['家庭保护'].value_counts())
    print(df['学校保护'].value_counts())
    print(df['社会保护'].value_counts())
    print(df['网络保护'].value_counts())
    print(df['政府保护'].value_counts())


if __name__ == '__main__':
    data_folder = 'data'
    older_children_file_name = 'finaldata.csv'
    train_data = 'train_onehot.csv'
    test_data = 'test_onehot.csv'

    older_children_csv = read_csv_file(data_folder, older_children_file_name)
    print_info(older_children_csv)
    df_train, df_test = process_csv(older_children_csv)

    df2csv(data_folder, train_data, df_train)
    df2csv(data_folder, test_data, df_test)   