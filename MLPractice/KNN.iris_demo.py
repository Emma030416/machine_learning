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
