# 训练模型
import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import StepLR
import numpy as np
import copy
from tqdm import tqdm # 用于在循环中显示进度条。

from lstm_model import LSTM

import torch
import torch.nn as nn
import numpy as np

def get_val_loss(args, model, Val):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    model.eval()

    loss_function = nn.MSELoss().to(device)  # Assuming args has a 'device' attribute
    losses = []

    with torch.no_grad():
        for (seq, label,feat_id,index) in Val:
            seq = seq.to(device)
            label = label.to(device)

            y_pred = model(seq)  
            
            loss = loss_function(y_pred, label)
            losses.append(loss.item())

    val_loss = np.mean(losses)
    model.train()

    return val_loss



def train(args, Dtr, Val, path):
    #指定gpu
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    
    input_size, hidden_size, num_layers,pred_len = args.input_size, args.hidden_size, args.num_layers,args.pred_len
    output_size = args.output_size
    model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size,device=device, pred_len=pred_len).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    print(len(Dtr))
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label,feat_id,index) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_loss = get_val_loss(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)
    
