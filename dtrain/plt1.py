from keras.models import load_model
from keras.models import Model
import h5py

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# 加载同一个模型
base_model = load_model("cnn_lstm_mdl.h5")  # 加载模型

layer_name = "lstm"  # 替换为你想要提取特征的层的名称
intermediate_layer_model = Model(
    inputs=base_model.input, outputs=base_model.get_layer(layer_name).output
)

with h5py.File("data//test//test_set.h5", "r") as f:
    X_test_loaded = f["x_test"][:]
    Y_test_loaded = f["y_test"][:]

data = X_test_loaded
labels = np.argmax(Y_test_loaded, axis=1)
# labels = Y_test_loaded
# 使用测试集或需要可视化的数据
intermediate_output = intermediate_layer_model.predict(data)


# 使用 t-SNE 减少维度
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
tsne_results = tsne.fit_transform(intermediate_output)


# 假设 'labels' 是数据点的标签，用于颜色编码
plt.figure
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis")
plt.colorbar(scatter)
plt.xlabel("t-SNE feature 1")
plt.ylabel("t-SNE feature 2")
plt.title("t-SNE Features")
plt.show()
p = 1
