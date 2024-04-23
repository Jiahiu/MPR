import h5py
import numpy as np
import keras
from keras.layers import (
    Input,
    Dense,
    Dropout,
    concatenate,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.io import savemat
from keras.callbacks import EarlyStopping

TF_ENABLE_ONEDNN_OPTS = 0


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


#  data split
sn = 1
loadname = "data//cnn//S0" + str(sn) + "_dataImage.h5"
file = h5py.File(loadname, "r")
imageData = file["imageData"][:]
imageData = imageData * 2000
imageLabel = file["imageLabel"][:]
file.close()

data = np.expand_dims(imageData, axis=3)
labels = convert_to_one_hot(imageLabel, 8).T
train_features, test_features, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.33, random_state=0
)

X_train = np.array(train_features)
X_test = np.array(test_features)
Y_train = np.array(train_labels)
Y_test = np.array(test_labels)


#  写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.losses = {"batch": [], "epoch": []}
        self.accuracy = {"batch": [], "epoch": []}
        self.val_loss = {"batch": [], "epoch": []}
        self.val_acc = {"batch": [], "epoch": []}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses["batch"].append(logs.get("loss", 0))
        self.accuracy["batch"].append(logs.get("accuracy", 0))
        self.val_loss["batch"].append(logs.get("val_loss", 0))
        self.val_acc["batch"].append(logs.get("val_accuracy", 0))

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.losses["epoch"].append(logs.get("loss", 0))
        self.accuracy["epoch"].append(logs.get("accuracy", 0))
        self.val_loss["epoch"].append(logs.get("val_loss", 0))
        self.val_acc["epoch"].append(logs.get("val_accuracy", 0))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # Accuracy
        plt.plot(iters, self.accuracy[loss_type], "r", label="train accuracy")
        # Loss
        plt.plot(iters, self.losses[loss_type], "g", label="train loss")
        if loss_type == "epoch":
            # Val accuracy
            plt.plot(iters, self.val_acc[loss_type], "b", label="val accuracy")
            # Val loss
            plt.plot(iters, self.val_loss[loss_type], "k", label="val loss")
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel("accuracy-loss")
        plt.legend(loc="upper right")
        plt.show()


def CNN(input_shape=(200, 12, 1), classes=8):
    X_input = Input(input_shape)

    f1 = [20, 16, 12, 8]  ## kernel_size 不断减小
    f2 = [3, 4, 5, 6]
    convs = []

    for i in range(4):
        x = Conv2D(
            filters=32,
            kernel_size=(f1[i], 3),
            strides=(1, 1),
            activation="relu",
            padding="valid",
        )(X_input)
        x = MaxPooling2D((20, 1))(x)

        x = Conv2D(
            filters=64,
            kernel_size=(f2[i], 1),
            strides=(1, 1),
            activation="relu",
            padding="valid",
        )(x)
        x = MaxPooling2D((9 - 2 - i, 1))(x)

        x = Flatten()(x)
        convs.append(x)

    merge = concatenate(convs, axis=1)
    X = merge
    X = Dropout(0.5)(X)
    X = Dense(128, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation="softmax")(X)
    model = Model(inputs=X_input, outputs=X)
    return model


model = CNN()
model.summary()


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = LossHistory()  # 创建一个history实例
patience = 7  # 当验证损失在连续5个epoch内没有改善时停止训练
monitor = "val_loss"  # 监控的指标，通常是验证集上的损失
early_stopping = EarlyStopping(monitor=monitor, patience=patience)
model.fit(
    X_train,
    Y_train,
    epochs=100,
    validation_data=(X_test, Y_test),
    batch_size=128,
    callbacks=[early_stopping, history],
)

preds_train = model.evaluate(X_train, Y_train)
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))

preds_test = model.evaluate(X_test, Y_test)
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))

LossHistory.loss_plot(history, "epoch")


# 获取模型预测的概率分布
predictions = model.predict(X_test)

# 将预测的概率转换为类别标签
predicted_labels = np.argmax(predictions, axis=1)

# 真实标签也需要从 one-hot 编码转换回类别标签
true_labels = np.argmax(Y_test, axis=1)

# 保存为 MAT 文件
savemat(
    "test_labels_and_predictions.mat",
    {"true_labels": true_labels, "predicted_labels": predicted_labels},
)

print("Data saved to 'test_labels_and_predictions.mat'.")

model.save("cnn_mdl.h5")
