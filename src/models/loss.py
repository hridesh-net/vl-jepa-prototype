import torch
import torch.nn.functional as F

def info_nce(pred, target, temprature=0.07):
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    
    logits = pred @ target.T / temprature
    labels = torch.arange(len(pred), device=pred.device)
    
    return F.cross_entropy(logits, labels)