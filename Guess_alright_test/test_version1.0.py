import tensorflow as tf
import os
import numpy as np
import sys
import shutil

rawPath = sys.argv[1]

if os.path.isabs(rawPath) is True:
    if os.path.exists('test_data/'):
        shutil.rmtree('test_data/')
    shutil.copytree(rawPath, sys.path[0]+'/test_data/')
if os.path.exists('result.csv'):
    os.remove('result.csv')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path_data = 'test_data/'    # 测试集相对路径

lines = tf.gfile.GFile('output_labels.txt').readlines()
uid_to_human = {}

for uid, line in enumerate(lines):
    line = line.strip('\n')
    uid_to_human[uid] = line


def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


with tf.gfile.FastGFile('output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    f = open('result.csv', 'a')
    f.write('FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2\n')
    for root, dirs, files in os.walk(path_data):
        for file in files:
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            top_3 = predictions.argsort()[-3:]
            top_3 = top_3[::-1]
            f.write(file[0:-4])
            for node_id in top_3:
                human_string = id_to_string(node_id)
                f.write(',' + human_string)
            f.write('\n')
    f.close()

