import torch
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm
from lstm_model import LSTM
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import os
from metrics import metric
import pandas as pd


import numpy as np

def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0

    # Calculate MAPE
    mape = np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100
    return mape

def get_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    # accuracy = accuracy_score(y_true, y_pred)
    return r2



def test(args, Dte, path,m,n):
    #指定gpu
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    pred = []
    y = []
    feat_ids = []
    print('loading models...')
    input_size, hidden_size, num_layers,pred_len = args.input_size, args.hidden_size, args.num_layers,args.pred_len
    output_size = args.output_size
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size,device=device,pred_len=pred_len).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    print(len(Dte))
    old_preds = []
    for (seq, target,feat_id,index) in tqdm(Dte):
        seq_old=seq
        seq_old = seq_old.to(device)
        feat_ids.append(feat_id)
        target = target.detach().cpu().numpy()
        y.append(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq) #长度为32,32是batch_size
            old_preds.append(y_pred)         
            y_pred=y_pred.detach().cpu().numpy()
            pred.append(y_pred)

    y, pred,feat_ids = np.array(y), np.array(pred),np.array(feat_ids)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('last prediction',pred[-1])
  
    preds = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2], pred.shape[3])
    trues = y.reshape(y.shape[0]*y.shape[1]*y.shape[2], y.shape[3])
    feat_ids = feat_ids.reshape(-1, 1)
    print(type(preds),type(trues),type(feat_ids))
    print(preds.shape,trues.shape,feat_ids.shape)
    print(feat_ids[:5])
    print(feat_ids[-5:])

    
    # 保存数据
    if args.is_store:
        # # df = pd.read_csv('/root/LLM_load_forecast/shuidian_verify/data/predict_data/0_' + args.data_name+'.csv', encoding='utf-8')
        # new_column_name = 'pred_' + args.target_name
        # print(feat_ids)
        # # print(feat_ids == df['date'].iloc[0])
        # index = np.where(feat_ids == df['date'].iloc[0])[0][0]
        # preds_sample = preds[index:index+args.final_pred_len]
        # trues_sample = trues[index:index+args.final_pred_len]
        # feat_ids_sample = feat_ids[index:index+args.final_pred_len]
        
        # combined_array = np.column_stack((feat_ids_sample, preds_sample,trues_sample))
        # # print(combined_array)

        # df[new_column_name]=preds_sample
        # print(df.columns)
        # df.to_csv('/root/LLM_load_forecast/lstm_load_forecasting_shuidian/data/predict_data/' + args.data_name+'.csv', index=False, mode='w')
        df = pd.DataFrame({
            "date": feat_ids.flatten(),  # 从 (320,1) 转为 (320,)
            "true_energy_consumption": trues.flatten(),
            "pred_energy_consumption": preds.flatten()
        })
        df.to_csv('/root/LLM_load_forecast/shuidian_verify/data/ec_predict_data/'+ args.data_name+'.csv', index=False)
    
    preds_sample = preds[-args.final_pred_len:]
    trues_sample = trues[-args.final_pred_len:]
    feat_id = feat_ids[-args.final_pred_len:]
    
    # metric
    mae_sample, mse_sample, rmse_sample, mape_sample, mspe_sample, smape_sample, nd_sample = metric(preds_sample, trues_sample)

    # 创建一个绘图
    print('is plotting...')
    plt.figure(figsize=(12, 6))
    x = [i for i in range(0, len(preds_sample))]

    # 绘制 y_sample 和 pred_sample 的曲线
    plt.plot(x, trues_sample, color='C1', marker='*', ms=1, alpha=0.7, label='true')
    plt.plot(x, preds_sample, color='C2', marker='o', ms=1, alpha=0.7, label='pred')
    plt.grid(axis='y')
    plt.legend(title=f'MAE: {mae_sample:.4f}\nMSE: {mse_sample:.4f}\nMAPE: {mape_sample:.4f}')

    plt.title(f'data_name: {args.data_name}  target: {args.target_name}')

    # 获取图片地址
    result_folder = './result_images/' + args.data_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 保存图像
    print('test_save_path',result_folder + '/' + args.target_name+ '_' + str(args.seq_length) + '_' + str(args.pred_len) +'_'+args.features+ '_'+args.training_time+ '.png')
    plt.savefig(result_folder + '/' + args.target_name+ '_' + str(args.seq_length) + '_' + str(args.pred_len) +'_'+args.features+ '_'+args.training_time+ '.png', dpi=300, format='png')

# 只进行predict
def predict(args, Dpr, path, m, n):
    #指定gpu
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    pred = []
    y = []
    feat_ids = []
    print('loading models...')
    input_size, hidden_size, num_layers,pred_len = args.input_size, args.hidden_size, args.num_layers,args.pred_len
    output_size = args.output_size
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size,pred_len=pred_len).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target,feat_id,index) in tqdm(Dpr):
        # print(seq.shape,target.shape)
        # target = list(chain.from_iterable(target.data.tolist()))
        # y.extend(target)
        feat_ids.append(feat_id)
        target = target.detach().cpu().numpy()
        y.append(target)
        seq = seq.to(device)
        if index[0].item() == 0:
            print('the first data')
        else:
            seq[:, -1, :] = old_pred.squeeze(1).detach()
            # print('seq',seq)
        with torch.no_grad():
            y_pred = model(seq) #长度为32,32是batch_size
            old_pred = y_pred           
            # y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            y_pred=y_pred.detach().cpu().numpy()
            pred.append(y_pred)
            # print('y_pred',y_pred)
        # # 检查模型输出中是否有 NaN 值
        # if torch.isnan(y_pred).any():
        #     print("Warning: Model output contains NaN values.")

        # # 检查并处理 seq 中的 NaN 值
        # if torch.isnan(seq).any():
        #     print("Warning: seq tensor contains NaN values before assignment.")

    y, pred,feat_ids = np.array(y), np.array(pred),np.array(feat_ids)
    # print(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print(type(y),type(pred))
    print(y.shape,pred.shape)
  
    preds = pred.reshape(pred.shape[0]*pred.shape[1]*pred.shape[2], pred.shape[3])
    trues = y.reshape(y.shape[0]*y.shape[1]*y.shape[2], y.shape[3])
    feat_ids = feat_ids.reshape(-1, 1)


    # metric
    preds_sample = preds[-args.seq_length*24:]
    trues_sample = trues[-args.seq_length*24:]
    feat_id = feat_ids[-1]
    mae_sample, mse_sample, rmse_sample, mape_sample, mspe_sample, smape_sample, nd_sample = metric(preds_sample, trues_sample)
    
    # 创建一个绘图
    print('is plotting...')
    plt.figure(figsize=(12, 6))
    x = [i for i in range(0, len(preds_sample))]

    # 绘制 y_sample 和 pred_sample 的曲线
    plt.plot(x, trues_sample, color='C1', marker='*', ms=1, alpha=0.7, label='true')
    plt.plot(x, preds_sample, color='C2', marker='o', ms=1, alpha=0.7, label='pred')
    plt.grid(axis='y')
    plt.legend(title=f'MAE: {mae_sample:.4f}\nMSE: {mse_sample:.4f}\nMAPE: {mape_sample:.4f}')


    # 添加 MAPE 和 R^2 信息
    # plt.text(x=max(x)*0.8, y=max(y_sample)*0.9, s=f'MAPE: {mape_value:.2f}%', fontsize=9, color='blue')
    # plt.text(x=max(x)*0.6, y=max(y_sample)*0.9, s=f'R^2: {r2_value:.2f}', fontsize=9, color='green')

    # 设置标题
    # plt.title(f'data_name: {args.data_name}  target: {args.target_name}')
    plt.title(f'data_name: {args.data_name}  target: {feat_id}')

    # 获取图片地址
    result_folder = './result_images/' + args.data_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 保存图像
    print('test_save_path',result_folder + '/' + args.target_name+ '_' + str(args.seq_length) + '_' + str(args.pred_len) +'_'+args.features+ '_'+args.training_time+ '.png')
    plt.savefig(result_folder + '/' + args.target_name+ '_' + str(args.seq_length) + '_' + str(args.pred_len) +'_'+args.features+ '_'+args.training_time+ '.png', dpi=300, format='png')
 

