import os
import torch 
import torch.nn as nn
import torch.optim as optim
import variables as var
from copy import deepcopy
from sklearn.metrics import roc_auc_score 

def train(net, data, k, train_mask, val_mask, model_path):
    
    loss_list = []
    score_list = []
    optimizer = optim.AdamW(net.parameters(), lr=var.lr, weight_decay=var.wd)
   
    with torch.no_grad():
        
        net.eval()
        out = net(data)
        loss = var.criterion(out,data.y)

        val_loss = loss[val_mask == 1].mean()
        val_score = roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu())

        best_val_score = 0
       
    # training
    for epoch in range(var.n_epochs):
        net.train()
        optimizer.zero_grad()
        out = net(data)
        # loss for training data only
        loss = var.criterion(out[train_mask == 1],data.y[train_mask == 1]).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            net.eval()
            out = net(data)
            loss = var.criterion(out,data.y)
                          
            val_loss = loss[val_mask == 1].mean()
            val_score = roc_auc_score(data.y[val_mask==1].cpu(),out[val_mask==1].cpu())
            loss_list.append(val_loss)
            score_list.append(val_score)

            # if new model gives the best validation set score
            if val_score >= best_val_score:
                      
                # save model parameters
                best_dict = {'epoch': epoch,
                       'model_state_dict': deepcopy(net.state_dict()),
                       'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                       'val_loss': val_loss,
                       'val_score': val_score,
                       'k': k,}
                
                # save best model
                torch.save(best_dict, model_path)
                
                # reset best score so far
                best_val_score = val_score
                
    return best_dict, loss_list, score_list