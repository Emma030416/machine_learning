# 1.导包
from pandas.core.common import random_state
from sklearn.cluster import KMeans  # KMeans模型
from sklearn.datasets import make_blobs  # 创建数据集
from sklearn.metrics import calinski_harabasz_score  # CH分数，用于模型评估
import matplotlib.pyplot as plt  # 可视化

# 2.获取数据
# 使用 make_blobs创建1000个样本的数据集，每个样本2个特征。设置4个质心簇 [-1,-1],[0,0],[1,1],[2,2]，簇标准差为[0.4,0.2,0.2,0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                  cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=1)
plt.figure()  # 创建新的图形窗口
plt.scatter(X[:, 0], X[:, 1], marker='o')  # 绘制散点图，两个特征值，使用圆形
plt.show()  # 显示图形

# n_clusters=2（聚类数量为2）
# 3.实例化 KMeans聚类
estimator = KMeans(n_clusters=2, random_state=1, init='k-means++',n_init='auto')
# 4.模型预测
y_pred = estimator.fit_predict(X)  # fit_predict方法同时完成训练和预测
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
# 5.模型评估
print('1->', calinski_harabasz_score(X, y_pred))  # 通过CH分数进行模型评估，分数越高效果越好

# n_clusters=3
y_pred = KMeans(n_clusters=3, random_state=1).fit_predict(X) # fit_predict()方法可以训练和预测一起完成
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print('2->', calinski_harabasz_score(X, y_pred))

# n_clusters=4
y_pred = KMeans(n_clusters=4, random_state=1).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print('3->', calinski_harabasz_score(X, y_pred))
