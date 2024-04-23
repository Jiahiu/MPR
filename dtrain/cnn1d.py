import h5py
import numpy as np
import keras
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    Conv1D,
    MaxPooling1D,
)
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


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


x_train = np.array(train_features)
x_test = np.array(test_features)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

with h5py.File("data//test//test_set.h5", "w") as f:
    f.create_dataset("x_test", data=x_test)
    f.create_dataset("y_test", data=y_test)


#  写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {"batch": [], "epoch": []}
        self.accuracy = {"batch": [], "epoch": []}
        self.val_loss = {"batch": [], "epoch": []}
        self.val_acc = {"batch": [], "epoch": []}

    def on_batch_end(self, batch, logs={}):
        self.losses["batch"].append(logs.get("loss"))
        self.accuracy["batch"].append(logs.get("acc"))
        self.val_loss["batch"].append(logs.get("val_loss"))
        self.val_acc["batch"].append(logs.get("val_acc"))

    def on_epoch_end(self, batch, logs={}):
        self.losses["epoch"].append(logs.get("loss"))
        self.accuracy["epoch"].append(logs.get("acc"))
        self.val_loss["epoch"].append(logs.get("val_loss"))
        self.val_acc["epoch"].append(logs.get("val_acc"))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
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


def CNN(input_shape, classes):
    X_input = Input(input_shape)

    X = Conv1D(filters=32, kernel_size=20, activation="relu")(X_input)
    X = MaxPooling1D(pool_size=2)(X)

    X = Conv1D(filters=64, kernel_size=20, activation="relu")(X)
    X = MaxPooling1D(pool_size=2)(X)

    X = Flatten()(X)
    X = Dense(128, activation="relu")(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation="softmax")(X)

    model = Model(inputs=X_input, outputs=X, name="CNN")
    return model


model = CNN(input_shape=(200, 12), classes=8)
model.summary()


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = LossHistory()  # 创建一个history实例

model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=128,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[history],
)

preds_train = model.evaluate(x_train, y_train)
print("Train Loss = " + str(preds_train[0]))
print("Train Accuracy = " + str(preds_train[1]))

preds_test = model.evaluate(x_test, y_test)
print("Test Loss = " + str(preds_test[0]))
print("Test Accuracy = " + str(preds_test[1]))

LossHistory.loss_plot(history, "epoch")
