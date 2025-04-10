# FROM registry.cn-hangzhou.aliyuncs.com/sais-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
FROM tcc-ubuntu:22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 安装依赖
RUN pip install pandas biopython tqdm torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 创建必要的目录
RUN mkdir -p /saisdata /saisresult /app/model

# 复制模型和代码
COPY model_definition.py /app/
COPY data_processing.py /app/
COPY predict.py /app/
COPY model/ /app/model/

# 复制run.sh文件
COPY run.sh .

# 确保run.sh脚本权限正确
RUN chmod +x /app/run.sh

# 设置入口点
CMD ["bash", "run.sh"]