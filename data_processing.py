import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO

class RNADataset(Dataset):
    def __init__(self, data_path, npy_dir):
        super(RNADataset, self).__init__()
        self.data = pd.read_csv(data_path)
        self.npy_dir = npy_dir
        self.name_list = self.data['pdb_id'].to_list()
        # 如果有序列，则使用；否则用空字符串占位
        if 'seq' in self.data.columns:
            self.seq_list = self.data['seq'].to_list()
        else:
            self.seq_list = ['A' * 100 for _ in range(len(self.name_list))]  # 占位序列，预测时会被替换

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        pdb_id = self.name_list[idx]
        coords = np.load(os.path.join(self.npy_dir, pdb_id + '.npy'))

        feature = {
            "name": pdb_id,
            "seq": seq,
            "coords": {
                "P": coords[:, 0, :],
                "O5'": coords[:, 1, :],
                "C5'": coords[:, 2, :],
                "C4'": coords[:, 3, :],
                "C3'": coords[:, 4, :],
                "O3'": coords[:, 5, :],
            }
        }

        return feature

def featurize(batch):
    alphabet = 'AUCG'
    B = len(batch)
    lengths = np.array([len(b['coords']['P']) for b in batch], dtype=np.int32)
    L_max = max(lengths)
    X = np.zeros([B, L_max, 6, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    names = []

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([np.nan_to_num(b['coords'][c], nan=0.0) for c in ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]], 1)
        l = len(b['coords']['P'])
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        # 处理序列
        if 'seq' in b:
            seq = b['seq']
            indices = np.array([alphabet.index(s) if s in alphabet else 0 for s in seq], dtype=np.int32)
            S[i, :len(indices)] = indices
        
        names.append(b['name'])
    
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, mask, lengths, names