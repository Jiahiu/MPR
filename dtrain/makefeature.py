import math
import h5py
import numpy as np
import scipy.io
import os


for sn in range(1, 2):
    folder_path = "C:\\Users\\jhc\\Desktop\毕设\\analysis\\lda_saveFeats\\"
    loadname = folder_path + "s0" + str(sn) + ".mat"
    file = scipy.io.loadmat(loadname)
    data = file["data"]
    label = file["label"]

    data = np.array(data)
    label = np.array(label) - 1
    label = np.reshape(label, (label.shape[0],))
    # data_set = np.reshape(data_set, (data_set.shape[0]*data_set.shape[1], data_set.shape[2]))
    # label_set = np.reshape(label_set, (label_set.shape[0]*label_set.shape[1], 1))

    nclass = len(np.unique(label))
    data_set = []
    for i in range(nclass):
        data_set.append(data[label == i, :])

    imageData = []
    imageLabel = []
    imageLength = 1
    inc = imageLength

    for i in range(nclass):
        tdata = data_set[i]
        length = math.floor((tdata.shape[0] - imageLength) / inc + 1)

        for j in range(length):
            subImage = tdata[inc * j : imageLength + inc * j, :]
            imageData.append(subImage)
            imageLabel.append(i)

    imageData = np.array(imageData)
    imageLabel = np.array(imageLabel)

    savePath = "Data//S0" + str(sn) + "_featImage_lstm.h5"
    file = h5py.File(savePath, "w")
    file.create_dataset("imageData", data=imageData)
    file.create_dataset("imageLabel", data=imageLabel)
    file.close()
