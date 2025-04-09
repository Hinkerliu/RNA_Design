FROM registry.cn-hangzhou.aliyuncs.com/sais-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04

WORKDIR /app

# 安装依赖
RUN pip install pandas biopython tqdm torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# 复制模型和代码
COPY model_definition.py /app/
COPY data_processing.py /app/
COPY predict.py /app/
COPY model/ /app/model/

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 设置入口点
CMD ["bash", "run.sh"]