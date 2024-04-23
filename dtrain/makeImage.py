import math
import h5py
import numpy as np
import scipy.io
import os

sn = 1
folder_path1 = "C:/Users/jhc/Desktop/毕设/实验数据/S0" + str(sn) + "/uf"
folder_path2 = "C:/Users/jhc/Desktop/毕设/实验数据/S0" + str(sn) + "/fa"

# 获取文件夹中的所有文件
data_set = []
files = os.listdir(folder_path1)
for file in files:
    if file.endswith(".mat"):
        file_path = os.path.join(folder_path1, file)
        data = scipy.io.loadmat(file_path)
        data_set.append(data["acData"])

files = os.listdir(folder_path2)
for file in files:
    if file.endswith(".mat"):
        file_path = os.path.join(folder_path2, file)
        data = scipy.io.loadmat(file_path)
        data_set.append(data["acData"])

imageData = []
imageLabel = []
imageLength = 200
inc = 150
len_ds = len(data_set)
for i in range(len_ds):
    data = data_set[i]
    length = math.floor((data.shape[0] - imageLength) / inc + 1)

    for j in range(length):
        subImage = data[inc * j : imageLength + inc * j, :]
        imageData.append(subImage)
        if i < 8:
            imageLabel.append(i)
        else:
            imageLabel.append(i - 8)


imageData = np.array(imageData)
imageLabel = np.array(imageLabel)

savePath = "Data//S0" + str(sn) + "_dataImage_lstm.h5"
file = h5py.File(savePath, "w")
file.create_dataset("imageData", data=imageData)
file.create_dataset("imageLabel", data=imageLabel)
file.close()
