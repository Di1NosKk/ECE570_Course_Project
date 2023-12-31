import torch
import torch.nn as nn

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameters for gnn
n_epochs = 200
lr = 0.001
wd = 0.1

# negative sample hyperparameters
epsilon = 0.1
proportion = 1

# loss function
# criterion = nn.MSELoss(reduction = 'none') 
# MAE loss
criterion = nn.L1Loss(reduction = 'none')
    