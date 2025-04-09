FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

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

# 运行预测脚本
CMD ["python", "predict.py"]