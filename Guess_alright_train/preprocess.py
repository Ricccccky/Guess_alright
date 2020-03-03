#!/user/bin/env python
# -*- coding:utf-8 -*-
# by Ricky time:2018/5/17

import os
import shutil
import sys

rawPath0 = sys.argv[1]
rawPath1 = sys.argv[2]
if (os.path.isabs(rawPath0) & os.path.isfile(rawPath1)) is True:
    if os.path.exists('train_image/'):
        shutil.rmtree('train_image/')
    shutil.copytree(rawPath0, sys.path[0]+'/train_image/')
    if os.path.exists('train_image/list.csv'):
        os.remove('train_image/list.csv')
    shutil.copy(rawPath1, 'train_image/list.csv')


os.chdir('train/data')
for i in range(0, 20):
    if os.path.exists(str(i)) is False:
        os.mkdir(str(i))

os.chdir('../../train_image')
f = open('list.csv', 'r')
line = f.readline()
while line:
    line = str(line).strip('\n')
    file_name = line.split(',')[0]
    uid = line.split(',')[1]
    if os.path.exists(file_name + '.jpg') & (os.path.exists('../train/data/' + uid + '/' + file_name + '.jpg') is False):
        shutil.move(file_name + '.jpg', '../train/data/' + uid + '/')
    line = f.readline()

os.chdir('../')
os.system('python retrain.py --bottleneck_dir bottleneck --how_many_training_steps 200 --model_dir inception_model \
        --output_graph output_graph.pb --output_labels output_labels.txt --image_dir train/data/ pause')
