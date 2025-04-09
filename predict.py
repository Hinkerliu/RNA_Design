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
    test_npy_data_dir: str = '/data/input/coords'
    test_data_path: str = '/data/input/test_data.csv'
    output_path: str = '/data/output/results.csv'


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
    
    # 检查输入目录是否存在
    if not os.path.exists(config.data_config.test_data_path):
        print(f"错误：测试数据文件未找到：{config.data_config.test_data_path}")
        return
    
    if not os.path.exists(config.data_config.test_npy_data_dir):
        print(f"错误：测试npy数据目录未找到：{config.data_config.test_npy_data_dir}")
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
                    'pred_seq': pred_seq
                })
                start_idx = end_idx
    
    # 保存预测结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(config.data_config.output_path, index=False)
    print(f"预测结果已保存到 {config.data_config.output_path}")


if __name__ == "__main__":
    main()