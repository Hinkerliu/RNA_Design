#!/bin/bash

# 打印环境信息
echo "当前工作目录: $(pwd)"
echo "输入目录内容:"
ls -la /saisdata/

# 确保输出目录存在
mkdir -p /saisresult

# 运行预测程序
python /app/predict.py

# 检查输出文件是否生成
if [ -f "/saisresult/submit.csv" ]; then
    echo "预测完成，结果已保存到 /saisresult/submit.csv"
    
    # 检查文件格式
    echo "文件行数: $(wc -l < /saisresult/submit.csv)"
    echo "文件头部格式检查:"
    head -n 1 /saisresult/submit.csv
    
    # 打印前5行和后5行，以及一些统计信息
    echo "文件前5行:"
    head -n 5 /saisresult/submit.csv
    
    echo "文件后5行:"
    tail -n 5 /saisresult/submit.csv
    
    # 检查是否有空行或格式问题
    echo "检查空行:"
    grep -c "^$" /saisresult/submit.csv
    
    echo "检查每行是否都有逗号(应该等于行数-1):"
    grep -c "," /saisresult/submit.csv
    
    # 复制到标准位置
    cp /saisresult/submit.csv /saisresult/results.csv
    echo "已复制结果到 /saisresult/results.csv"
else
    echo "警告：未找到输出文件 /saisresult/submit.csv"
fi