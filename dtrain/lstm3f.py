from keras.models import Sequential
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Activation,
)
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
from keras.callbacks import EarlyStopping
from scipy.io import savemat
from keras.callbacks import Callback
import matplotlib.pyplot as plt


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = {"batch": [], "epoch": []}
        self.accuracy = {"batch": [], "epoch": []}
        self.val_loss = {"batch": [], "epoch": []}
        self.val_acc = {"batch": [], "epoch": []}

    def on_batch_end(self, batch, logs=None):
        self.losses["batch"].append(logs.get("loss"))
        self.accuracy["batch"].append(logs.get("accuracy", logs.get("acc")))
        self.val_loss["batch"].append(logs.get("val_loss"))
        self.val_acc["batch"].append(logs.get("val_accuracy", logs.get("val_acc")))

    def on_epoch_end(self, epoch, logs=None):
        self.losses["epoch"].append(logs.get("loss"))
        self.accuracy["epoch"].append(logs.get("accuracy", logs.get("acc")))
        self.val_loss["epoch"].append(logs.get("val_loss"))
        self.val_acc["epoch"].append(logs.get("val_accuracy", logs.get("val_acc")))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # accuracy
        plt.plot(iters, self.accuracy[loss_type], "r", label="train acc")
        # loss
        plt.plot(iters, self.losses[loss_type], "g", label="train loss")
        if loss_type == "epoch":
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], "b", label="val acc")
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], "k", label="val loss")
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel("acc-loss")
        plt.legend(loc="upper right")
        plt.show()


class SequeClassifier:
    def __init__(self, units):
        self.units = units
        self.model = None

    # 构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    def build_model(self, loss, optimizer, metrics):
        self.model = Sequential()
        self.model.add(  #  16  72
            LSTM(units[0], return_sequences=True, input_shape=(1, 72))
        )  # noqa: F405
        self.model.add(Dropout(0.2))  # 添加Dropout以减少过拟合
        self.model.add(LSTM(units[1], return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units[2], return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(8))
        self.model.add(Activation("softmax"))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def get_model(self):
        return self.model


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


snScore = []
#  data split
for sn in range(1, 2):
    loadname = "data//S0" + str(sn) + "_featImage_lstm.h5"
    file = h5py.File(loadname, "r")
    imageData = file["imageData"][:]
    imageData = imageData
    imageLabel = file["imageLabel"][:]
    file.close()

    # labels = convert_to_one_hot(imageLabel,8).T
    train_features, test_features, train_labels, test_labels = train_test_split(
        imageData, imageLabel, test_size=0.33, random_state=0
    )
    # loadname = "data//cnn//S0" + str(sn) + "_dataImage.h5"
    # file = h5py.File(loadname, "r")
    # imageData = file["imageData"][:]
    # imageData = imageData * 2000
    # imageLabel = file["imageLabel"][:]
    # file.close()

    # data = np.expand_dims(imageData, axis=3)
    # labels = convert_to_one_hot(imageLabel, 8).T
    # train_features, test_features, train_labels, test_labels = train_test_split(
    #     data, labels, test_size=0.33, random_state=0
    # )

    x_train = np.array(train_features)
    x_test = np.array(test_features)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    with h5py.File("data//test//test_lstm_set.h5", "w") as f:
        f.create_dataset("x_test", data=x_test)
        f.create_dataset("y_test", data=y_test)
    # 2 构建神经网络模型：（根据各层输入输出的shape）搭建网络结构、确定损失函数、确定优化器
    units = [64, 64, 64]  # lstm细胞个数
    loss = "sparse_categorical_crossentropy"  # 损失函数类型
    optimizer = "adam"  # 优化器类型
    metrics = ["accuracy"]  # 评估方法类型
    sclstm = SequeClassifier(units)
    sclstm.build_model(loss, optimizer, metrics)
    model = sclstm.get_model()
    model.summary()

    # 定义早停的参数
    patience = 7  # 当验证损失在连续5个epoch内没有改善时停止训练
    monitor = "val_loss"  # 监控的指标，通常是验证集上的损失
    early_stopping = EarlyStopping(monitor=monitor, patience=patience)
    # 3 训练模型  callbacks=[early_stopping]
    epochs = 50
    batch_size = 128
    history = LossHistory()

    sclstm.model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[history, EarlyStopping(monitor="val_loss", patience=7)],
    )

    # 4 模型评估
    score = sclstm.model.evaluate(x_test, y_test, batch_size=128)
    print("model score:", score)
    snScore.append(score)
    LossHistory.loss_plot(history, "epoch")

    print("Shape of y_test:", y_test.shape)
    true_labels = y_test
    # 获取模型预测的概率分布
    predictions = model.predict(x_test)

    # 将预测的概率转换为类别标签
    predicted_labels = np.argmax(predictions, axis=1)

    # 保存为 MAT 文件
    savemat(
        "test_labels_and_predictions3.mat",
        {"true_labels": true_labels, "predicted_labels": predicted_labels},
    )

    print("Data saved to 'test_labels_and_predictions3.mat'.")
    model.save("lstm_mdl.h5")

print(snScore)
