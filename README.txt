1.�������
ģ��������CNN��������ģ�ͽ���Ǩ��ѧϰ

2.���л�����
���
Windows10���Ƽ���/ubuntu16.04
python3.6
tensorflow��׼��
numpy��׼��


Ӳ����
�������
CPU:  >=4��2.5hz
RAM��>=8G
GPU��>=2G��16308M��

3.��������
    һ�����ذ�װpython3.6
    ������װ�����⡣ʹ������ pip install tensorflow  pip install numpy ����
    �� ѵ������ѹԴ������������н���Guess_alright_trainĿ¼  ��������
python preprocess.py train_img_dir list_dir
���ɿ�ʼѵ��������train_img_dirΪѵ��ͼƬĿ¼�ľ���·���� list_dirΪ��ע�ļ�list.csv�ľ���·����ѵ����������Guess_alright_trainĿ¼ ���ģ��Ϊoutput_graph.pb��output_labels.txt��
    �ġ����ԣ��������н���Guess_alright_testĿ¼  ��������
python test_version1.0.py test_img_dir
���ɿ�ʼ���ԡ�����test_img_dirΪѵ��ͼƬĿ¼�ľ���·����ѵ����������Guess_alright_testĿ¼ ������Ϊresult.csv��

