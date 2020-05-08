import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2 as cv
import random
import csv
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_files(file_dir):
    # 存放图片类别和标签的列表：第0类
    list_0 = []
    label_0 = []
    # 存放图片类别和标签的列表：第1类
    list_1 = []
    label_1 = []
    # 存放图片类别和标签的列表：第2类
    list_2 = []
    label_2 = []
    # 存放图片类别和标签的列表：第3类
    list_3 = []
    label_3 = []
    # 存放图片类别和标签的列表：第4类
    list_4 = []
    label_4 = []

    for file in os.listdir(file_dir):  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        # print(file)
        # 拼接出图片文件路径
        image_file_path = os.path.join(file_dir, file)
        for image_name in os.listdir(image_file_path):
            # print('image_name',image_name)
            # 图片的完整路径
            image_name_path = os.path.join(image_file_path, image_name)
            # print('image_name_path',image_name_path)
            # 将图片存放入对应的列表
            if image_file_path[-1:] == '0':
                list_0.append(image_name_path)
                label_0.append(0)
            elif image_file_path[-1:] == '1':
                list_1.append(image_name_path)
                label_1.append(1)
            elif image_file_path[-1:] == '2':
                list_2.append(image_name_path)
                label_2.append(2)
            elif image_file_path[-1:] == '3':
                list_3.append(image_name_path)
                label_3.append(3)
            else:
                list_4.append(image_name_path)
                label_4.append(4)

    # 合并数据
    image_list = np.hstack((list_0, list_1, list_2, list_3, list_4))
    label_list = np.hstack((label_0, label_1, label_2, label_3, label_4))
    # 利用shuffle打乱数据
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)

    # 将所有的image和label转换成list
    image_list = list(temp[:, 0])
    image_list = [i for i in image_list]
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    # print(image_list)
    # print(label_list)
    return image_list, label_list


def get_tensor(image_list, label_list):
    ims = []
    for image in image_list:
        # 读取路径下的图片
        x = tf.io.read_file(image)
        # 将路径映射为照片,3通道
        x = tf.image.decode_jpeg(x, channels=3)
        # 修改图像大小
        x = tf.image.resize(x, [32, 32])
        # 将图像压入列表中
        ims.append(x)
    # 将列表转换成tensor类型
    img = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(label_list)
    return img, y
