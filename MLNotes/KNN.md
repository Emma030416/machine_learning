## 环境搭建
项目建在你的环境下（最好是建个虚拟环境，避免安装的依赖影响系统环境），所有算法文件放在该项目下，在项目目录下安装库。

导入sklearn库，否则会报错ModuleNotFoundError: No module named 'sklearn'模块不存在。

注意：实际应安装scikit-learn库，使用命令`pip install scikit-learn`或`conda install scikit-learn`

scikit-learn库：
简单高效的数据挖掘和数据分析工具，基于三大科学计算库：NumPy、SciPy和matplotlib

其他的库也是类似的，需要安啥自己安。
<br><br>

## KNN思想
搭建好环境后，我们来看今天要实现的算法---KNN。

KNN（K-Nearest Neighbors，K近邻)：
属于`有监督学习`中的分类问题，也可用于回归问题

如果最近的k个邻居大多数属于同一类别，则该样本也属于这个类别。
<br><br>

## 五种距离度量方式
距离越近相似性越高，所以选取最近的K个邻居作为参考标准。

那么，如何定义最近呢？有以下四种常见的距离度量方式。

### 1.欧氏距离
符合`勾股定理`，走斜边最短路径

### 2.曼哈顿距离
`沿着直角边走折线`

又称城市街区距离，横向纵向距离之和

### 3.切比雪夫距离
取能走的`距离最大的那条直角边`走

### 4.闵可夫斯基距离
距离度量的统一表达式（闵氏距离）

<img width="2096" height="277" alt="f4c5c84f57460bd9c1343892fd563554" src="https://github.com/user-attachments/assets/72a4938a-11e1-42e9-ba03-c85c4d7ef2a0" />

p=1，曼哈顿距离<br>
p=2，欧氏距离<br>
p→∞，切比雪夫距离
<br><br>

## K值选择
掌握了如何定义最近，那么，K值又应该如何选择呢？

K值的选择关系到模型的好坏，过小或过大都不行：

1.k值过小：

例如 k=1 时，仅参考最近的一个邻居，很容易受到异常点干扰，导致过拟合（模型过于复杂）。

2.k值过大：

参考范围过广，容易受到样本不均衡问题影响，导致欠拟合（模型过于简单）。
<br><br>

## 特征预处理
特征预处理是机器学习必备的流程之一，KNN也不例外。

所谓特征预处理，就是将数值进行缩放。有以下两种：

### 1.归一化处理

将特征值都`按照均匀分布缩放到[0,1]区间`，避免数值差异大的特征主导模型训练。

注意：

导入路径：`from sklearn.preprocessing import MinMaxScaler`

> X_scaled = (X - X_min) / (X_max - X_min)
把每列线性压缩到 [0, 1] 区间
参考均匀分布

核心参数：`feature_range`，指定缩放区间，默认为(0,1)，scaler = MinMaxScaler(feature_range=(a, b)) 

核心方法：`fit_transform(X)`，fit计算统计量，transform执行转换（归一化）

该部分代码实现如下：

```python
from sklearn.preprocessing import MinMaxScaler # 用于归一化处理

# 实例化归一化的对象
scaler = MinMaxScaler()

# 对原始特征进行变换
X_train = scaler.fit_transform(X_train) # 训练集要训练
X_test = scaler.transform(X_test) # 测试集不训练，只转换数据
```

### 2.数据标准化
将数据转化为`均值为0，方差为1的标准正态分布`。

数学表达：x' = (x-u)/σ，其中u为均值，σ为标准差

看起来很眼熟吧，就和数学里正态分布标准化的步骤一样。

正态分布：
又称高斯分布、钟形分布，遵循3σ法则

记作N(u,σ²)，u决定位置，σ决定幅度

离散程度：方差越大越分散，越小越集中

注意：

导入路径：`from sklearn.preprocessing import StandardScaler`

参数获取：`transformer.mean_`获取均值，`transformer.var_`获取方差

该部分代码实现如下：

```python
from sklearn.preprocessing import StandardScaler # 用于数据标准化

# 实例化标准化的对象
scaler = StandardScaler()

# 对原始数据进行变换
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("transformer.mean_->",transformer.mean_)
print("transformer.var_->",transformer.var_)
```
<br>

## 算法流程
具备了一些必备知识，就可以学习如何实现KNN算法了。

1.计算未知样本到训练集每个样本的距离

2.选取最近的K个样本

3.分类问题：进行多数表决，统计最近的K个样本中出现最多的类别

回归问题：取这K个样本目标值的平均值
<br><br>

## 代码实现
算法清楚了就可以写代码了。

注意：

导入路径：`from sklearn.neighbors import KNeighborsClassifier / KNeighborsRegressor`
（分别对应分类问题 / 回归问题）

核心参数：`n_neighbors`，默认值为5

数据要求：预测时必须传入二维数组，和训练集X的形式保持一致

以分类问题，归一化预处理为例：

```python
# KNN.train_and_save.py
# 1.导包
from sklearn.neighbors import KNeighborsClassifier # KNN分类问题
from sklearn.model_selection import train_test_split # 用来划分数据集
from sklearn.metrics import accuracy_score, classification_report # 用准确率、精准率、召回率、F1分数来评估模型
from sklearn.preprocessing import MinMaxScaler # 用于特征预处理
import numpy as np # 用于创建矩阵
import joblib # 用来保存和加载模型


def train_and_save():
    # 2.获取数据
    X = np.array([[39, 0, 31], [3, 2, 65], [2, 3, 55], [9, 38, 2], [8, 34, 17], [5, 2, 57], [21, 17, 5], [45, 2, 9]])
    y = [0, 1, 2, 2, 2, 1, 0, 0]

    # 3.划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # 4.实例化归一化的对象
    scaler = MinMaxScaler()

    # 5.对原始特征进行变换
    # 训练集和测试集要分开！
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 6.实例化KNN模型
    estimator = KNeighborsClassifier(n_neighbors=3) # 创建对象，指定参数为3，选取最近的3个邻居

    # 7.模型训练
    estimator.fit(X_train, y_train)

    # 8.模型测试
    y_pred = estimator.predict(X_test)

    # 9.模型评估
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 10.模型保存
    joblib.dump(estimator, 'model/mylrmodel02.bin' )
    print("模型已保存成功")


if __name__ == '__main__':
    train_and_save()
```

```python
# KNN.load_and_predict.py
import joblib


def load_and_predict(new_data):
    estimator = joblib.load('model/mylrmodel02.bin')
    print("模型已加载成功")
    return estimator.predict(new_data)

if __name__ == '__main__':
    print(load_and_predict([[23, 3, 17]]))
```
<br>

## 补充：超参数选择的方法
### 1.交叉验证

Cross Validation（CV），用于确定`最优的数据集划分方式`。其中CV值为划分的份数，表达为几折交叉验证。

将训练集划分为 n 等份，每次取 1 份作为验证集，其余 n-1 份作为训练集，循环完成所有训练评估，取多次评估的平均值作为最终模型得分。

改变 n 的值，选择模型得分最高的作为最终划分方式。

例如：CV  = 4

第一次：第1份为验证集，其余为训练集

第二次：第2份为验证集，其余为训练集

....

第四次：第4份为验证集，其余为训练集
<br>


### 2.网格搜索

用于确定`最优的超参数组合`。

优势：相比每次人工手动试验，网格搜索只需手动预设参数范围，之后自会动化遍历参数范围。
<br>


### 3.交叉验证和网格搜索组合使用

见实战案例
<br><br>

## 实战案例
鸢尾花数据集

```python
# KNN.iris_demo.py
# 1.导包
from mpl_toolkits.mplot3d.proj3d import transform
from sklearn.datasets import load_iris # 用于导入数据
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV # 用于分割数据集，交叉验证
from sklearn.preprocessing import StandardScaler # 用于数据标准化
from sklearn.neighbors import KNeighborsClassifier # KNN分类问题
from sklearn.metrics import accuracy_score, classification_report # 用于模型评估
import matplotlib.pyplot as plt # 用于绘制散点图
import pandas as pd # 用于数据可视化
import seaborn as sns # 用于数据可视化

# 2.导入数据
iris_data = load_iris()
# # 查看数据信息（后续可省）
# print('iris_data.data->')
# print(iris_data.data[:5]) # 特征值（只显示前五条）
# # print('iris_data.target->', iris_data.target) # 目标值
# print('iris_data.target_names->', iris_data.target_names) # 目标名（标签名），3个类别
# print('iris_data.feature_names->', iris_data.feature_names) # 特征名，4个特征（注意特征名必须和他给的一模一样，空格、中文括号、大小写等）
# print(iris_data.DESCR) # 数据集描述信息

# # 3.数据可视化（可省）
# # 将数据转化为DataFrame格式
# iris_df = pd.DataFrame(iris_data['data'], columns=iris_data.feature_names) # 特征
# iris_df['label'] = iris_data.target # 标签
# """
# 用 sns.lmplot()绘制散点图
# 核心参数：x/y（绘制x和y特征间的关系）、data（数据源）、hue（按标签分类着色）、fit_reg（是否显示回归线，默认为True）
# 这里我们选取两个与类别相关度高的特征作为x和y（Class Correlation high!）
# """
# sns.lmplot(x = 'petal length (cm)', y = 'petal width (cm)', data=iris_df, hue='label')
# plt.xlabel('petal length (cm)')
# plt.ylabel('petal width (cm)')
# plt.show()

# 4.数据集划分
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=1)

# 5.数据集预处理（数据标准化）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6.训练模型
# 实例化KNN模型
estimator = KNeighborsClassifier()
# 交叉验证和网格搜索
param_grid = {'n_neighbors':[5, 7, 9]} # 需要找到最优的n_neighbors参数
estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=4) # 输入一个estimator，得到一个功能更强大的estimator
# 将数据送进estimator，对3个参数值各做4折交叉验证，选择平均分最高的参数，重新训练一次得到最终模型
estimator.fit(X_train, y_train)
# 查看结果
print('estimator.best_score_:', estimator.best_score_) # 不同参数的模型经过交叉验证后得到的最高平均分
print('estimator.best_estimator_:', estimator.best_estimator_) # 最终模型
print('estimator.best_params_:', estimator.best_params_) # 最高平均分使用的参数值
print('estimator.cv_results_', estimator.cv_results_) # 明细表，可以不看，关注均值就行

# 7.模型测试
y_pred = estimator.predict(X_test)

# 8.模型评估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9.模型预测
x = [[5.1, 3.5, 1.4, 0.2]]
x = scaler.transform(x)
print(estimator.predict(x))
```
