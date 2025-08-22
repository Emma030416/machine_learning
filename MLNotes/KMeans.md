有很多概念需要阐明：

KMeans思想：根据样本间的相似性对样本集进行**聚类**，发现事物内部结构及相互关系，相似性高的事物聚集在一起。

`簇（Cluster）`：数据集中具有较高的相似性的一组对象，簇内相似度高，而不同簇间相似度低。
类数量选择：聚类需要预先选择簇的数量，如2个、3个或4个。

> 簇的选择通常基于领域知识或数据的先验信息。
> 选择簇的数量是一个挑战，选择不当可能导致过拟合或欠拟合：
**过拟合**：选择的簇过多，分的太细，每个簇样本数很少，只捕捉到一些非常具体、细微的特征，但这些特征可能不具有代表性或普遍性，导致簇内相似性高，但簇间差异可能不明显。
**欠拟合**：选择的簇过少，分的太粗，同一个簇中可能出现不同类型的样本。
为了选择最佳的簇数量，通常会使用一些方法，先做了解：
`肘部法则（Elbow method）`：通过绘制不同K值对应的总内群平方和（WSS），选择“肘部”位置的K值。
`轮廓系数（Silhouette coefficient）`：评估每个点与其簇内点的平均距离与最近簇的平均距离的比值，选择轮廓系数最大的K值。


`中心点（质心）`：质心，顾名思义，就是把该簇里所有样本的坐标取平均值得到的那个点。

KMeans大致流程如下：
先随机或按 k-means++ 规则选 K 个初始中心点 → 把样本按最近原则分到各中心点 → 重新计算每个簇的均值作为新的中心点 → 迭代到不再变化为止

例三代码如下，博主懒得写保存加载模型了只写了一个文件哈哈，大家可以自行补充：

```python
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

```

依次会弹出 4 个独立的 Matplotlib 窗口（视图），顺序如下：

 1. 原始数据散点图
1000 个样本，颜色统一，展示 4 个真实簇的分布。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6f81e732b6b2443c8ddb6a0b40caa7a6.png#pic_center)
 
 2. K=2 聚类结果
把 4 个真实簇合并成 2 类，颜色只有两种。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/415a34fec944449880f5afb24a2eb5f0.png#pic_center)

 3. K=3 聚类结果
把 4 个真实簇合并成 3 类，颜色三种。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/31c715b0fded4a159304313d6493f6a6.png#pic_center)

 4. K=4 聚类结果
基本还原 4 个真实簇，颜色四种，与原始图最贴近。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7fcbec9a2ffa4793a613565ecf63ebb4.png#pic_center)
🔺每显示完一张图，必须手动关闭窗口，才会弹出下一张~

当然还会出现K=2，3，4的CH分数：

```python
1-> 3188.4118269394658
2-> 2978.797575128741
3-> 6013.804516335779
```
可以看到，结果与MatPlotLib一致，K=4时CH最高，效果最好。
这个例子里貌似是簇越多越好，但实际上样本更多的时候会出来一个顶点（最优点），还记得最开始解释簇的概念提出的肘部法则不？可以想象一下顶点就是那个肘部，感兴趣的话再用别的数据集试试，可能效果更明显些。
