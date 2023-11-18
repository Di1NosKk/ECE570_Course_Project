import os
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import utils
import variables as var
from torch_geometric.nn import MessagePassing
from copy import deepcopy
import train as tr
import evaluate as evaluate
import matplotlib.pyplot as plt

        
class GNN2(MessagePassing):
    def __init__(self,k):
        super(GNN2, self).__init__(flow="target_to_source")
        self.k = k
        self.hidden_size = 256
        self.network = nn.Sequential(
            nn.Linear(k,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size,1),
            nn.LogSigmoid()
            )

    def forward(self, x, edge_index, edge_attr):
        self.network = self.network.to(dtype = torch.float32)
        out = self.propagate(edge_index = edge_index, x=x, edge_attr=edge_attr, k = self.k, network=self.network)
        return out

    def message(self,x_i,x_j,edge_attr):
        # message is the edge weight
        return edge_attr

    def aggregate(self, inputs, index, k, network):
        # concatenate all k messages
        self.input_aggr = inputs.reshape(-1,k)
        # pass through network
        out = self.network(self.input_aggr)
        return out

# Our own GNN
class GNNModel(torch.nn.Module):
    def __init__(self, k):
        super(GNNModel, self).__init__()
        print("Our own GNN")
        self.k = k
        self.L1 = GNN2(self.k)
    def forward(self,data):
        self.edge_attr = data.edge_attr
        self.edge_index = data.edge_index
        self.x = data.x
        out = self.L1(self.x, self.edge_index, self.edge_attr)
        out = torch.squeeze(out,1)
        return out
        
        
def run(train_x,train_y,val_x,val_y,test_x,test_y,dataset,seed,k,samples,train_new_model,plot):  

    # path to save model parameters
    model_path = 'saved_models/%s/%d/net_%d.pth' %(dataset,k,seed)
    if not os.path.exists(os.path.dirname(model_path)):
       os.makedirs(os.path.dirname(model_path)) 
    
    x, y, neighbor_mask, train_mask, val_mask, test_mask, dist, idx = utils.negative_samples(train_x, train_y, val_x, val_y, test_x, test_y, k, samples, var.proportion, var.epsilon)
    data = utils.build_graph(x, y, dist, idx)
        
    data = data.to(var.device)                                                                    
    torch.manual_seed(seed)
    net = GNNModel(k).to(var.device)
    # net_previous = GNN(k).to(var.device)
    
    if train_new_model == True:
        best_dict, loss_list, score_list = tr.train(net, data, k, train_mask, val_mask, model_path)
        net.load_state_dict(best_dict['model_state_dict'])
    else:
        load_dict, loss_list = torch.load(model_path)
        net.load_state_dict(load_dict['model_state_dict'])
    
    ret, loss_test = evaluate.test(net, data, test_mask)
    
    # print(loss_list)
    # print(score_list)
    
    if plot == True: 
        # score_list = [score_item.cpu().numpy() for score_item in score_list]
        loss_list = [loss_item.cpu().numpy() for loss_item in loss_list]
        # print(len(score_list))
        # print(len(loss_list))
        ##plot here with loss value, auc score, num epoch
        plt.figure(figsize=(20, 10)) 
        
        # Plotting the score
        plt.subplot(2, 1, 1)
        plt.plot(range(1, var.n_epochs + 1), score_list, marker='o', label='Train Score')
        plt.title('Score across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        # Plotting the loss and loss_test
        plt.subplot(2, 1, 2)
        plt.plot(range(1, var.n_epochs + 1), loss_list, marker='o', color='r', label='Train Loss')
        plt.title('Loss across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        save_dir = 'plot/'
        
        file_name = save_dir + '%s/training_metrics_plot_%d_%d.png' %(dataset,k,seed)
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        plt.savefig(file_name)
        
        plt.show()
    
    return ret