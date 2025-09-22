import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
from config import get_args
import pickle

class XGB():
    def __init__(self,pred_len,seq_len) -> None:
        self.pred_len=pred_len
        self.seq_len=seq_len
        
    def lag_feature(self,data,feat_columns, li):
        # 在对应的位置设置lag列
        for shift_num in li:
            temp_data = data[feat_columns].shift(shift_num)
            # print(temp_data.shape)
            temp_data.columns=[i+"_lag_"+str(shift_num) for i in feat_columns]
            data = pd.concat([data,temp_data],axis=1)
        return data 
    
    def data_process(self,city_name,target_name):
        pred_len=1
        seq_len=1
        print(city_name,target_name)
        df = pd.read_csv('/root/LLM_load_forecast/shuidian_verify/data/energy_consumption/'+city_name+'.csv', encoding='gbk')[[target_name]]
        feat_columns = [target_name]
        li=[1,2,3,4,5,6,7]
        df = df.dropna(axis=0)
        mergedata = self.lag_feature(df,feat_columns,li)
        mergedata = mergedata.dropna(axis=0)

        mergedata = mergedata.reset_index(drop=True)
        num_train = int(len(mergedata) * 0.8)
        num_vali = len(mergedata) - num_train 

        # time_idx = 500
        # train_data = mergedata.iloc[:time_idx,:]
        # train_label = mergedata[target_name].tolist()[:time_idx]
        # valid_data = mergedata.iloc[time_idx:-17,:]
        train = mergedata.iloc[:num_train,:].values
        valid = mergedata.iloc[-num_vali:,:].values
        print(len(train))
        print(len(valid))

        # train_data=[]
        # train_label=[]
        # for i in range(len(train)-seq_len-pred_len+1):
        #     train_data.append(train[i:i+seq_len])
        #     train_label.append(train[i+seq_len:i+seq_len+pred_len,0])
        
        # valid_data=[]
        # valid_label=[]
        # for i in range(len(valid)-seq_len-pred_len+1):
        #     valid_data.append(valid[i:i+seq_len])
        #     valid_label.append(valid[i+seq_len:i+seq_len+pred_len,0])
        train_data=train[:-1,:]
        train_label=train[1:,0]
        valid_data=valid[:-1,:]
        valid_label=valid[1:,0]
        print(train_data.shape,train_label.shape,valid_data.shape,valid_label.shape)
        return train_data,train_label,valid_data,valid_label
    
    def get_mape(self,y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_indices = y_true != 0

        # Calculate MAPE
        mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
        return mape
    
    def model_train(self,city_name,target_name,path):
        train_data,train_label,valid_data,valid_label = self.data_process(city_name,target_name)
        XGB = XGBRegressor()
        xgb = XGB.fit (train_data, train_label)
        y_pred = xgb.predict(valid_data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(xgb, f)
        
        r2_value = r2_score(valid_label,y_pred)
        mape_value=self.get_mape(valid_label,y_pred)
        
        index_list = [i for i in range(len(y_pred))]
        plt.plot(index_list,valid_label,c='blue', marker='*', ms=1, alpha=0.75, label='true')
        plt.plot(index_list,y_pred,c='red', marker='o', ms=1, alpha=0.75, label='pred')
        plt.text(x=max(index_list)*0.8, y=max(valid_label)*0.9, s=f'MAPE: {mape_value:.2f}%', fontsize=9, color='blue')
        plt.text(x=max(index_list)*0.6, y=max(valid_label)*0.9, s=f'R^2: {r2_value:.2f}', fontsize=9, color='green')
        plt.title(f'company: {city_name}  target: {target_name}')
        plt.legend()
        
        result_folder = './xgb_result_images/' + city_name
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        print(result_folder + '/' + target_name+ '.png')
        plt.savefig(result_folder + '/' + target_name+ '.png', dpi=300, format='png')
               
 