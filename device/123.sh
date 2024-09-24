#!/bin/bash

# 检查是否提供了文件名
if [ -z "$1" ]; then
    echo "请提供一个文件名作为参数"
    exit 1
fi

# 获取文件名
filename="$1"

python agent_DQN.py -n $filename 
python plot.py -n $filename
python calc.py -n $filename
