<<<<<<< HEAD
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from typing import List

from model_definition import RNAModel
from data_processing import RNADataset, featurize

@dataclass
class DataConfig:
    train_data_path: str = 'data/train_data.csv'
    train_npy_data_dir: str = 'data/train_coords'
    val_data_path: str = 'data/val_data.csv'
    val_npy_data_dir: str = 'data/val_coords'
    test_data_path: str = 'data/test_data.csv'
    test_npy_data_dir: str = 'data/test_coords'

@dataclass
class ModelConfig:
    smoothing: float = 0.1
    hidden: int = 128
    vocab_size: int = 4
    k_neighbors: int = 30
    dropout: float = 0.1
    node_feat_types: List[str] = field(default_factory=lambda: ['angle', 'distance', 'direction'])
    edge_feat_types: List[str] = field(default_factory=lambda: ['orientation', 'distance', 'direction'])
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3

@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    checkpoint_dir: str = 'model'
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        X, S, mask, lengths, _ = batch
        X = X.to(device)
        S = S.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        logits, targets, _ = model(X, S, mask)
        
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({'loss': total_loss / total_samples, 'acc': total_acc / total_samples})
    
    return total_loss / total_samples, total_acc / total_samples

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, S, mask, lengths, _ = batch
            X = X.to(device)
            S = S.to(device)
            mask = mask.to(device)
            
            logits, targets, _ = model(X, S, mask)
            
            loss = F.cross_entropy(logits, targets)
            
            # 计算准确率
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples, total_acc / total_samples

def main():
    # 配置
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    # 创建保存模型的目录
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    
    # 加载数据集
    train_dataset = RNADataset(
        data_path=data_config.train_data_path,
        npy_dir=data_config.train_npy_data_dir
    )
    
    val_dataset = RNADataset(
        data_path=data_config.val_data_path,
        npy_dir=data_config.val_npy_data_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=featurize
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=featurize
    )
    
    # 初始化模型
    model = RNAModel(model_config).to(train_config.device)
    
    # 优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(1, train_config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, train_config.device, epoch)
        val_loss, val_acc = validate(model, val_loader, train_config.device)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(train_config.checkpoint_dir, 'best.pt'))
            print(f'Saved best model with validation loss: {val_loss:.4f}')
        
        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(train_config.checkpoint_dir, 'last.pt'))

if __name__ == '__main__':
=======
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse
from dataclasses import dataclass, field
from typing import List

from model_definition import RNAModel
from data_processing import RNADataset, featurize

@dataclass
class DataConfig:
    train_data_path: str = 'data/train_data.csv'
    train_npy_data_dir: str = 'data/train_coords'
    val_data_path: str = 'data/val_data.csv'
    val_npy_data_dir: str = 'data/val_coords'
    test_data_path: str = 'data/test_data.csv'
    test_npy_data_dir: str = 'data/test_coords'

@dataclass
class ModelConfig:
    smoothing: float = 0.1
    hidden: int = 128
    vocab_size: int = 4
    k_neighbors: int = 30
    dropout: float = 0.1
    node_feat_types: List[str] = field(default_factory=lambda: ['angle', 'distance', 'direction'])
    edge_feat_types: List[str] = field(default_factory=lambda: ['orientation', 'distance', 'direction'])
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3

@dataclass
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    checkpoint_dir: str = 'model'
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        X, S, mask, lengths, _ = batch
        X = X.to(device)
        S = S.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        logits, targets, _ = model(X, S, mask)
        
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix({'loss': total_loss / total_samples, 'acc': total_acc / total_samples})
    
    return total_loss / total_samples, total_acc / total_samples

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, S, mask, lengths, _ = batch
            X = X.to(device)
            S = S.to(device)
            mask = mask.to(device)
            
            logits, targets, _ = model(X, S, mask)
            
            loss = F.cross_entropy(logits, targets)
            
            # 计算准确率
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean()
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
    
    return total_loss / total_samples, total_acc / total_samples

def main():
    # 配置
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    # 创建保存模型的目录
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    
    # 加载数据集
    train_dataset = RNADataset(
        data_path=data_config.train_data_path,
        npy_dir=data_config.train_npy_data_dir
    )
    
    val_dataset = RNADataset(
        data_path=data_config.val_data_path,
        npy_dir=data_config.val_npy_data_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=featurize
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=featurize
    )
    
    # 初始化模型
    model = RNAModel(model_config).to(train_config.device)
    
    # 优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(1, train_config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, train_config.device, epoch)
        val_loss, val_acc = validate(model, val_loader, train_config.device)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(train_config.checkpoint_dir, 'best.pt'))
            print(f'Saved best model with validation loss: {val_loss:.4f}')
        
        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(train_config.checkpoint_dir, 'last.pt'))

if __name__ == '__main__':
>>>>>>> master
    main()