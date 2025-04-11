# RNA设计项目

本项目实现了基于深度学习的RNA逆向折叠算法，可以根据RNA的3D结构预测其对应的核苷酸序列。这是为"第三届世界科学智能大赛创新药赛道：RNA逆折叠与功能核酸设计"比赛开发的解决方案。

## 项目背景

RNA (核糖核酸) 在细胞生命活动中扮演着至关重要的角色，从基因表达调控到催化生化反应，都离不开RNA的参与。RNA的功能很大程度上取决于其三维(3D)结构。理解RNA的结构与功能之间的关系，是生物学和生物技术领域的核心挑战之一。

RNA逆折叠是计算生物学中的一个核心概念，指设计特定的RNA序列，使其能够折叠成预定的目标RNA结构。这一过程与传统的RNA折叠（即从已知序列预测其结构）形成鲜明对比，因其在设计功能性RNA分子方面的广泛应用而备受关注，涉及临床医学、工业生产等多个领域。

本项目采用深度学习方法，从海量数据中挖掘RNA序列与结构之间的深层关联，旨在揭示其背后的生物学机制，并探索高效设计RNA分子的可能性。

## 任务描述

基于给定的RNA三维骨架结构，生成一个RNA序列。评估标准是序列的恢复率(recovery rate)，即算法生成的RNA序列，在多大程度上与真实能够折叠成目标结构的RNA序列相似。恢复率越高，表明算法性能越好。

## 项目结构

```
RNA_Design_Project/
├── data/                  # 数据目录
│   ├── train_coords/      # 训练数据坐标文件
│   ├── val_coords/        # 验证数据坐标文件
│   ├── test_coords/       # 测试数据坐标文件
│   ├── train_data.csv     # 训练数据信息
│   ├── val_data.csv       # 验证数据信息
│   └── test_data.csv      # 测试数据信息
├── model/                 # 模型保存目录
│   └── best.pt            # 最佳模型权重
├── model_definition.py    # 模型定义
├── data_processing.py     # 数据处理
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── Dockerfile             # Docker配置
├── run.sh                 # 运行脚本
└── README.md              # 项目说明
```




## 技术方案

本项目使用基于图神经网络的深度学习模型来解决RNA逆折叠问题。主要技术组件包括：

1. **特征提取**：从RNA的3D坐标中提取节点和边的特征，包括：
   - 角度特征：二面角信息
   - 距离特征：原子间距离的径向基函数表示
   - 方向特征：原子相对位置的方向向量

2. **图神经网络**：使用消息传递神经网络(MPNN)处理RNA结构信息
   - 编码器层：捕获RNA骨架结构的全局信息
   - 解码器层：基于结构特征预测核苷酸序列

3. **序列预测**：基于结构特征预测核苷酸序列(A, U, C, G)

## 数据格式

### 输入格式

本项目的输入为**RNA骨架原子**和**RNA侧链原子**，具体包括以下7个原子：

1. **RNA骨架原子**（6个）：
   - P（磷酸基团）
   - O5'（5'氧原子）
   - C5'（5'碳原子）
   - C4'（4'碳原子）
   - C3'（3'碳原子）
   - O3'（3'氧原子）

2. **RNA侧链原子**（1个）：
   - **N1**（嘧啶类碱基）或**N9**（嘌呤类碱基）

数据以**numpy数组**格式提供，每个原子的坐标为三维坐标（x, y, z）。如果原始数据中某个原子不存在，则该位置会以**NaN**填充。

### 输出格式

输出为**RNA序列**，以CSV格式文件给出，包含两列：
1. **pdb_id**：RNA结构的唯一标识符
2. **pred_seq**：预测的RNA序列，由碱基（A, U, C, G）按顺序排列

## 环境要求

# 基础依赖
numpy>=1.20.0
pandas>=1.3.0
tqdm>=4.62.0

# PyTorch相关
torch==2.0.0
torch-scatter==2.1.0

# 生物信息学工具
biopython>=1.79

# 数据处理和可视化
matplotlib>=3.5.0
seaborn>=0.11.2

# 其他工具
scikit-learn>=1.0.0


可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```
对于torch-scatter，需要根据CUDA版本单独安装：
```bash
pip install torch-scatter -f URL_ADDRESS.pyg.org/whl/torch-2.0.0+cu117.html
```
## 数据准备
1. 下载训练数据、验证数据和测试数据。
2. 确保数据格式与项目要求一致。
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

## 使用方法

### 训练模型

```bash
python train.py
```

### 优化训练参数
为了提高训练精度，可以在train.py中调整以下参数：

1. 模型参数 ：
   
   - 增加隐藏层维度（hidden）：从128增加到256
   - 增加邻居数量（k_neighbors）：从30增加到40
   - 增加编码器和解码器层数：从3层增加到4层
   - 调整dropout率：从0.1调整到0.2
2. 训练参数 ：
   
   - 减小批量大小（batch_size）：从16减小到8
   - 增加训练轮数（epochs）：从100增加到200
   - 调整学习率（lr）：从1e-3调整到5e-4
   - 增加权重衰减（weight_decay）：从1e-5增加到1e-4
3. 其他优化 ：
   
   - 添加梯度裁剪（max_norm=1.0）
   - 优化学习率调度器参数

### 本地预测
```
python predict.py
```

### Docker部署

1. 构建Docker镜像：

```bash
docker build -t rna-inverse-folding:v1 .
```

2. 运行Docker容器：

```bash
docker run --gpus all -v /path/to/test/data:/data/input -v /path/to/output:/data/output rna-inverse-folding:v1
```

## 评价指标

本项目采用预测序列与实际序列的重合率（Recovery Rate）来评估模型性能，计算方式如下：

```python
def calc_seq_recovery(gt_seq, pred_seq):
    true_count = 0
    length = len(gt_seq)
    if len(pred_seq) < length:
        pred_seq = pred_seq + [""] * (length - len(pred_seq))
    for i in range(length):
        if gt_seq[i] == pred_seq[i]:
            true_count += 1
    return (true_count + 0.0) / length
```

## 比赛信息

本项目是为"第三届世界科学智能大赛创新药赛道：RNA逆折叠与功能核酸设计"比赛开发的解决方案。

### 比赛背景

为推动科学智能领域创新发展，在上海市政府的指导下，上海科学智能研究院携手复旦大学在上智院·天池平台发布"第三届世界科学智能大赛"。大赛设置航空安全、材料设计、合成生物、创新药、新能源五大赛道，配有高额奖金池，面向全球人才开放，旨在推进科学智能技术创新，挖掘顶尖创新团队，构建科学智能生态，激发科学智能发展新动能。

### 比赛任务

基于给定的RNA三维骨架结构，生成一个RNA序列。评估标准是序列的恢复率(recovery rate)，即算法生成的RNA序列，在多大程度上与真实能够折叠成目标结构的RNA序列相似。恢复率越高，表明算法性能越好。

### 比赛规则

1. 参赛选手**仅可以使用比赛官方提供的数据集**
2. 不允许私下相互共享代码，如果共享必须合并队伍
3. 主办方将对复赛最终优胜队伍进行代码审核，淘汰没有实际创新贡献、代码相似或以其它形式作弊的队伍

## 参考文献

1. [RNA设计相关文献1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3692061/)
2. [RNA设计相关文献2](https://pmc.ncbi.nlm.nih.gov/articles/PMC4191386/)
3. [RNA设计相关文献3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7856060/)
4. [RNA设计相关文献4](https://academic.oup.com/bib/article/19/2/350/2666340?login=false)
5. [RDesign GitHub仓库](https://github.com/A4Bio/RDesign)

## 许可证

本项目仅用于参加"第三届世界科学智能大赛创新药赛道"比赛，不得用于商业用途。