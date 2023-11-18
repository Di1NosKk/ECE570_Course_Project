import os
import torch 
import torch.nn as nn
import variables as var


def test(net, data, test_mask):
    
    with torch.no_grad():
        net.eval()
        out = net(data)
        loss = var.criterion(out,data.y)
    
    return out[test_mask==1].cpu(), loss