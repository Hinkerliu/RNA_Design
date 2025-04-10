import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List


from model_definition import RNAModel
from data_processing import RNADataset, featurize


@dataclass
class DataConfig:
    # 修改为官方指定的输入路径
    test_npy_data_dir: str = '/saisdata/coords'
    test_data_path: str = '/saisdata/seqs/dummy_seq.csv'  # 假设有一个dummy序列文件
    # 修改为官方指定的输出路径和文件名
    output_path: str = '/saisresult/submit.csv'


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
class Config:
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path: str = '/app/model/best.pt'
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)


def main():
    config = Config()
    
    # 打印当前工作目录和环境信息，便于调试
    print(f"当前工作目录: {os.getcwd()}")
    print(f"输入目录内容:")
    os.system("ls -la /saisdata/")
    
    # 检查输入目录是否存在
    if not os.path.exists(config.data_config.test_npy_data_dir):
        print(f"错误：测试npy数据目录未找到：{config.data_config.test_npy_data_dir}")
        # 尝试查找替代目录
        print("尝试查找其他可能的数据目录...")
        os.system("find /saisdata -type d | sort")
        return
    
    # 如果测试数据文件不存在，尝试创建一个虚拟的测试数据文件
    if not os.path.exists(config.data_config.test_data_path):
        print(f"警告：测试数据文件未找到：{config.data_config.test_data_path}")
        print("尝试创建虚拟测试数据文件...")
        
        # 获取所有npy文件并创建虚拟测试数据
        npy_files = [f for f in os.listdir(config.data_config.test_npy_data_dir) 
                    if f.endswith('.npy')]
        
        if npy_files:
            dummy_data = []
            for npy_file in npy_files:
                pdb_id = os.path.splitext(npy_file)[0]
                dummy_data.append({'pdb_id': pdb_id, 'sequence': 'A'})  # 虚拟序列
            
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(config.data_config.test_data_path), exist_ok=True)
            
            # 保存虚拟测试数据
            dummy_df = pd.DataFrame(dummy_data)
            dummy_df.to_csv(config.data_config.test_data_path, index=False)
            print(f"已创建虚拟测试数据文件：{config.data_config.test_data_path}")
        else:
            print(f"错误：在 {config.data_config.test_npy_data_dir} 中未找到npy文件")
            return
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(config.data_config.output_path), exist_ok=True)
    
    # 加载测试数据
    test_dataset = RNADataset(
        data_path=config.data_config.test_data_path,
        npy_dir=config.data_config.test_npy_data_dir,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,  # 可以根据GPU内存调整
        shuffle=False,
        num_workers=0,
        collate_fn=featurize
    )
    
    # 加载模型
    model = RNAModel(config.model_config).to(config.device)
    print(f"从 {config.model_path} 加载模型")
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.eval()
    
    # 进行预测
    alphabet = 'AUCG'
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测中"):
            X, S, mask, lengths, names = batch
            X = X.to(config.device)
            S = S.to(config.device)
            mask = mask.to(config.device)
            
            logits, _ = model.sample(X, S, mask)
            probs = F.softmax(logits, dim=-1)
            samples = probs.argmax(dim=-1)
            
            # 将预测结果转换为序列
            start_idx = 0
            for i, length in enumerate(lengths):
                end_idx = start_idx + length.item()
                sample = samples[start_idx: end_idx]
                pred_seq = ''.join([alphabet[s.item()] for s in sample])
                results.append({
                    'pdb_id': names[i],
                    'seq': pred_seq  # 修改这里：将'pred_seq'改为'seq'
                })
                start_idx = end_idx
    
    # 保存预测结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.data_config.output_path, index=False)
    print(f"预测结果已保存到 {config.data_config.output_path}")


if __name__ == "__main__":
    main()