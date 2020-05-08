import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from tensorflow import keras
from resnet import resnet38
from dataset_production import get_files,get_tensor


def get_picture():
    cap = cv2.VideoCapture(0)
    cap.set(3, 960)
    cap.set(4, 960)
    flag = 1  # 设置一个标志，用来输出视频信息
    while (cap.isOpened()):  # 循环读取每一帧
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Gray", gray)

        cv2.imshow("Capture_Test", frame)  # 窗口显示，显示名为 Capture_Test

        k = cv2.waitKey(1) & 0xFF  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
        x = tf.image.decode_jpeg(frame, channels=3)
        # 修改图像大小
        x = tf.convert_to_tensor(tf.image.resize(frame, [32, 32]))

        model = resnet34()
        model.build(input_shape=(None, 32, 32, 3))
        model.load_weights(keras.models.load_model('path_to_saved_model'))
        model.summary()
        logits = model(x)
        print(logits)
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 删除建立的全部窗口


if __name__ == '__main__':
    get_picture()