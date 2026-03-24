import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_adj

from src.dataset import load_dataset, get_dataset_stats
from src.models import GNNEncoder, GAE, SimCLRModel
from src.augmentations import augment
from src.losses import NTXentLoss

TEMPERATURE = 0.5
PROJ_DIM = 128
HIDDEN_DIM = 64

# set division
VAL_SIZE = 0.15
TEST_SIZE = 0.15

EPOCHS = 20
BASELINE_EPOCHS = 8
BATCH_SIZE = 256

LR = 1e-4
WEIGHT_DECAY = 1e-4

SAVE_DIR = "results/"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(DEVICE)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

root = "src/datasets/"
dataset = "MUTAG"

dataset = load_dataset(root, dataset)
stats = get_dataset_stats(dataset)
print(stats)

#stratified split
y = []

for g in dataset:
    y.append(g.y.item())
y = np.array(y)

train_idx, temp_idx = train_test_split(np.arange(len(dataset)), test_size=VAL_SIZE + TEST_SIZE, stratify=y)

val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y[temp_idx])

train_graphs = [dataset[i] for i in train_idx]
val_graphs = [dataset[i] for i in val_idx]
test_graphs = [dataset[i] for i in test_idx]

train_loader = PyGDataLoader(train_graphs,batch_size=BATCH_SIZE, shuffle=True)
val_loader = PyGDataLoader(val_graphs,batch_size=BATCH_SIZE, shuffle=False)
test_loader = PyGDataLoader(test_graphs,batch_size=BATCH_SIZE, shuffle=False)

#GAE train
def train_gae_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_items = 0

    for batch in tqdm(loader, desc=f"Training", leave=False):
        batch = batch.to(DEVICE)
        
        optimizer.zero_grad()

        z = model(batch.x, batch.edge_index, batch.batch)
        A_hat = model.decode(z)
        A = to_dense_adj(batch.edge_index, batch.batch)

        loss = criterion(A_hat, A)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * A.size(0)
        total_items += A.size(0)
    
    return total_loss / total_items

def 