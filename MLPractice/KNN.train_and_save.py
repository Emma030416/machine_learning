# train_and_save.py
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