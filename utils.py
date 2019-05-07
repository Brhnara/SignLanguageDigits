import os
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array


def prepare_dataset(path):
    labels = [file for file in os.listdir(path) if not os.path.isfile(file)]
    train_datas = []
    train_labels = []
    # labels = []
    print(labels)
    for label in labels:
        dir_images = [path+label+"/"+item for item in os.listdir(path+label) if item.endswith(".JPG")]
        for image in dir_images:
            img = cv2.imread(image)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (100, 100))
            x = img_to_array(img_gray)  # this is a Numpy array with shape (3, 150, 150)
            # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            # cv2.resize(image, (100, 100))
            # cv2.imshow("img", img_gray)
            # print("label : ", label, img_gray.shape)
            # print(img_gray.shape)
            train_datas.append(x)
            # print(train_datas)
            zeros  = np.zeros(len(labels))
            zeros[int(label)] = 1
            train_labels.append(zeros)
            # print(zeros)
            # print(img_gray / 255)
            # input()
            # cv2.waitKey(3 )
            # print(dir_images, len(dir_images))
        # input()
    print(len(train_datas))
    print(len(train_labels))
    np_data = np.array(train_datas, dtype="float") / 255.0
    print(np_data.shape)
    np_label = np.array(train_labels)
    np.save("datas.npy",np_data)
    np.save("labels.npy", np_label)

prepare_dataset("Dataset/")