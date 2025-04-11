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
    # 修改数据路径，使用绝对路径或环境变量
    train_data_path: str = os.environ.get('TRAIN_DATA_PATH', 'data/train_data.csv')
    train_npy_data_dir: str = os.environ.get('TRAIN_NPY_DIR', 'data/train_coords')
    val_data_path: str = os.environ.get('VAL_DATA_PATH', 'data/val_data.csv')
    val_npy_data_dir: str = os.environ.get('VAL_NPY_DIR', 'data/val_coords')
    test_data_path: str = os.environ.get('TEST_DATA_PATH', 'data/test_data.csv')
    test_npy_data_dir: str = os.environ.get('TEST_NPY_DIR', 'data/test_coords')

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
    batch_size: int = 8  # 减小批量大小，原为16
    num_workers: int = 4
    epochs: int = 200  # 增加训练轮数，原为100
    lr: float = 5e-4  # 调整学习率，原为1e-3
    weight_decay: float = 1e-4  # 增加权重衰减，原为1e-5
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
    
    # 打印当前工作目录和数据路径
    print(f"当前工作目录: {os.getcwd()}")
    print(f"训练数据路径: {data_config.train_data_path}")
    print(f"训练数据目录: {data_config.train_npy_data_dir}")
    
    # 检查数据文件是否存在
    if not os.path.exists(data_config.train_data_path):
        print(f"错误: 找不到训练数据文件 {data_config.train_data_path}")
        print("尝试查找可能的数据文件...")
        # 列出当前目录下的所有文件和目录
        for root, dirs, files in os.walk('.'):
            # 限制遍历深度
            if root.count(os.sep) > 3:
                continue
            for file in files:
                if file.endswith('.csv'):
                    print(f"找到CSV文件: {os.path.join(root, file)}")
        
        # 尝试使用新的目录结构
        print("尝试使用新的目录结构...")
        alt_train_path = os.path.join("data", "train", "seqs", "train.csv")
        alt_train_dir = os.path.join("data", "train", "coords")
        alt_val_path = os.path.join("data", "valid", "seqs", "valid.csv")
        alt_val_dir = os.path.join("data", "valid", "coords")
        
        if os.path.exists(alt_train_path):
            print(f"找到替代训练数据文件: {alt_train_path}")
            data_config.train_data_path = alt_train_path
            data_config.train_npy_data_dir = alt_train_dir
            print(f"更新训练数据路径为: {data_config.train_data_path}")
            print(f"更新训练数据目录为: {data_config.train_npy_data_dir}")
        else:
            print(f"替代训练数据文件 {alt_train_path} 也不存在")
            return
        
        if os.path.exists(alt_val_path):
            print(f"找到替代验证数据文件: {alt_val_path}")
            data_config.val_data_path = alt_val_path
            data_config.val_npy_data_dir = alt_val_dir
            print(f"更新验证数据路径为: {data_config.val_data_path}")
            print(f"更新验证数据目录为: {data_config.val_npy_data_dir}")
        else:
            print(f"替代验证数据文件 {alt_val_path} 不存在，将使用训练数据进行验证")
            data_config.val_data_path = data_config.train_data_path
            data_config.val_npy_data_dir = data_config.train_npy_data_dir
    
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
    main()