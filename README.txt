1.软件概述
模型利用了CNN，对已有模型进行迁移学习

2.运行环境：
软件
Windows10（推荐）/ubuntu16.04
python3.6
tensorflow标准库
numpy标准库


硬件：
最低配置
CPU:  >=4核2.5hz
RAM：>=8G
GPU：>=2G（16308M）

3.部署流程
    一、下载安装python3.6
    二、安装依赖库。使用命令 pip install tensorflow  pip install numpy
    三 训练：解压源码包，在命令行进入Guess_alright_train目录  输入命令
python preprocess.py train_img_dir list_dir
即可开始训练。其中train_img_dir为训练图片目录的绝对路径， list_dir为标注文件list.csv的绝对路径。训练结束后在Guess_alright_train目录 输出模型为output_graph.pb和output_labels.txt。
    四、测试：在命令行进入Guess_alright_test目录  输入命令
python test_version1.0.py test_img_dir
即可开始测试。其中test_img_dir为训练图片目录的绝对路径，训练结束后在Guess_alright_test目录 输出结果为result.csv。

