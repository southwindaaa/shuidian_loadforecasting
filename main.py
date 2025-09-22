# main.py
from config import get_args
from data_process import nn_seq_us
from train import train,get_val_loss
from test import test
from lstm_model import LSTM
import torch
from xgb_model import XGB
import pickle

def main():
    args = get_args()
    if args.model=='lstm':
        # scaler_path = '/root/LLM_load_forecast/shuidian_verify/trained_models/'+args.data_name+'_'+args.target_name+'_'+args.features+'_'+str(args.seq_length)+'_'+str(args.pred_len)+'_scaler.pkl'
        Dtr, Val, Dte,m,n = nn_seq_us(args.batch_size,args.data_name,args.target_name,args.seq_length,args.features,args.pred_len)
        print(len(Dtr))
        train_save_path = '/root/LLM_load_forecast/shuidian_verify/trained_models/'+args.data_name+'_'+args.target_name+'_'+args.features+'_'+str(args.seq_length)+'_'+str(args.pred_len)+'_'+args.training_time+'.pth'
        with open('/root/LLM_load_forecast/shuidian_verify/trained_models/'+args.data_name+'_'+args.target_name+'_'+args.features+'_'+str(args.seq_length)+'_'+str(args.pred_len)+'_'+args.training_time+'mn.pkl', 'wb') as f:
            pickle.dump({'m': m, 'n': n}, f)

        # 模型训练
        if args.onlytest == 0:
            train(args, Dtr, Val, train_save_path)
            test(args,Dte,train_save_path,m,n)
        else:
            test(args,Dte,args.test_load_path,m,n)
    elif args.model=='xgb':
        xgb = XGB(seq_len=args.seq_length,pred_len=args.pred_len)
        xgb_path = '/root/LLM_load_forecast/shuidian_verify/xgb_trained_models/'+args.data_name+'_'+args.target_name+'_'+str(args.seq_length)+'_'+str(args.pred_len)+'_xgb.pkl'
        xgb.model_train(args.data_name,args.target_name,xgb_path)
        
    
    

if __name__ == '__main__':
    main()
