# 设置参数
# config.py
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description="LSTM Model Training")

    parser.add_argument("--input_size", type=int, default=676, help="Input size for the LSTM model")
    parser.add_argument("--hidden_size", type=int, default=100, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--output_size", type=int, default=1, help="Output size")
    parser.add_argument("--pred_len", type=int, default=24, help="lstm pred length")
    parser.add_argument("--final_pred_len", type=int, default=24, help="final pred length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.004, help="Learning rate") # 水电集团 0.04
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for learning rate decay")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for learning rate decay")
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'], help="Optimizer type")
    parser.add_argument("--gpu", type=int, default=6, help="GPU ID to use")
    parser.add_argument("--onlytest",type=int,default=0,choices=[0,1],help="only test or not")
    # parser.add_argument("--train_save_path",type=str,default='./trained_models/trained_model.pth',help="the path to save trained models")
    parser.add_argument("--test_load_path",type=str,default='./trained_models/RI_time_load_electricity_2.pth',help="the path to load trained models")
    parser.add_argument("--scaler_path",type=str,default='./scalers/jinyang_maximum_load_7_1_S_scaler.pkl',help="the path to load scaler")
    parser.add_argument("--features",type=str,default='M',help="use mutiple features or single feature")
    parser.add_argument("--training_time",type=str,default='0',help="how many times did you train this model")
    parser.add_argument("--data_name",type=str,default='data',help="the name of dataset to use")
    parser.add_argument("--target_name",type=str,default='electricity',help="electricity or gas")
    parser.add_argument("--is_time",type=bool,default=False,help="use time features or not")
    parser.add_argument("--seq_length",type=int,default=24,help="the length of sequential data you want to predict")
    parser.add_argument("--is_recursion",type=bool,default=False,help="Recursive prediction or direct prediction")
    parser.add_argument("--is_store",type=bool,default=False,help="store data or not")
    parser.add_argument("--model",type=str,default='lstm',help="electricity or gas")
    
    args = parser.parse_args()
    return args

