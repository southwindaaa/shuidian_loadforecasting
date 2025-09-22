import pandas  as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import random

def load_data(file_name):
    df = pd.read_csv('/root/LLM_load_forecast/shuidian_verify/data/energy_consumption/' + file_name, encoding='utf-8')
    columns = df.columns
    print(columns)
    df.fillna(df.mean(), inplace=True)
    # df = df.iloc[::3]
    return df


def nn_seq_us(B,data_name,target_name,seq_len,features,pred_len):
    print('data processing...')

    dataset = load_data(data_name+'.csv')

    num_train = int(len(dataset) * 0.6) # 训练集
    num_test = int(len(dataset) * 0.3) # 测试集
    num_vali = len(dataset) - num_train - num_test # 验证集

    train = dataset[:num_train]
    val = dataset[num_train-seq_len:num_train + num_vali]
    test = dataset[len(dataset) - num_test - seq_len:len(dataset)]
    print('dataset',len(dataset),'\ntrain',len(train),'\nval',len(val),'\ntest',len(test))

    m, n = np.max(dataset[target_name]), np.min(dataset[target_name])                                                                  

    # 处理水电集团数据
    def process_water_elec(data, batch_size):
        load = data[target_name]
        date = data['date'].tolist()
        print(target_name)
        load = load.tolist()
        load = (load - n) / (m - n)
        seq = []
        print(len(data))
        for i in range(len(data) - seq_len):
            train_seq = []
            train_label = []
            if (i + seq_len + pred_len) > len(data):
                break
            for j in range(i, i + seq_len):
                x = [load[j]]
                train_seq.append(x)
            train_label.append(load[i + seq_len:i+seq_len+pred_len])
            # print(train_seq)
            train_seq = torch.FloatTensor(train_seq)
            # .view(-1)
            train_label = torch.FloatTensor(train_label).transpose(0, 1)
            seq.append((train_seq, train_label,date[i + seq_len:i+seq_len+pred_len],i))
            # print(seq)
        print(len(seq))
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
        return seq

    Dtr = process_water_elec(train, B)
    Val = process_water_elec(val, B)
    Dte = process_water_elec(test, B) 
    
    return Dtr, Val, Dte, m, n
