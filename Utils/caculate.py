import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os, copy, torch, itertools, json, pypinyin, glob
from tqdm import tqdm, trange
from PIL import Image
from torchvision.io import read_image
import torch.nn.functional as F

def accuracy(pred, status):
    # This accuracy is based on estimated survival events against true survival events
    preds_tiles = pred.detach().cpu().numpy().reshape(-1)
    labels = status.data.cpu().numpy()

    label = []
    pred_tiles = []
    for i in range(len(preds_tiles)):
        if not np.isnan(preds_tiles[i]):
            label.append(labels[i])
            pred_tiles.append(preds_tiles[i])
            
    label = np.asarray(label)
    pred_tiles = np.asarray(pred_tiles)
    
    correct = np.sum(pred_tiles == label)
    return correct / len(label)