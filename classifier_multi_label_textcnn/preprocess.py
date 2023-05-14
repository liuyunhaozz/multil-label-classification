import os
import pandas as pd


def read_csv_file(data_folder, file_name):
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    return df 

def process_csv(df: pd.DataFrame):
    new_df = df.loc[:, ['案件描述', '家庭保护', '学校保护', '社会保护', '网络保护', '政府保护']]
    new_df = new_df.rename(columns={'案件描述': 'content'})
    df_train = new_df.sample(frac=0.8)
    df_test = new_df[~df.index.isin(df_train.index)]
    return df_train, df_test 

def df2csv(data_folder, file_name, df: pd.DataFrame):
    path = os.path.join(data_folder, file_name)
    df.to_csv(path, index=None, encoding='utf_8_sig') 


if __name__ == '__main__':
    data_folder = 'data'
    older_children_file_name = 'old-5-11.csv'
    train_data = 'train_onehot.csv'
    test_data = 'test_onehot.csv'

    older_children_csv = read_csv_file(data_folder, older_children_file_name)
    df_train, df_test = process_csv(older_children_csv)

    df2csv(data_folder, train_data, df_train)
    df2csv(data_folder, test_data, df_test)   