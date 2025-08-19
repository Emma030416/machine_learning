# train_and_save.py
# 1.导包
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.model_selection import train_test_split # 用来划分数据集
from sklearn.metrics import r2_score, mean_squared_error # 评估指标需要用到
import numpy as np # 用于创建矩阵
import joblib # 用来保存和加载模型
import os # 用来创建模型路径目录


def train_and_save():
    # 2.获取数据（x的每一列都是特征值，不用再进行特征工程了）
    X = np.array([[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]])
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

    # 3.划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

    # 3.实例化线性回归模型
    estimator = LinearRegression()  # 对象

    # 4.模型训练
    estimator.fit(X_train, y_train)
    print(f'estimator.coef:{estimator.coef_}')  # 斜率
    print(f'estimator.intercept:{estimator.intercept_}')  # 截距

    # 5.模型测试
    y_pred = estimator.predict(X_test)

    # 6.模型评估
    print("R²:", r2_score(y_test, y_pred)) # 决定系数，越接近1越好（和数学中一样）
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
    # print("RMSE:", mean_squared_error(y_test, y_pred, squared=False)) # 均方根误差，越小越好

    # 5.模型保存
    os.makedirs('model', exist_ok=True)  # 目录只用创建一次就行，后面再保存模型就不用了
    joblib.dump(estimator, 'model/mylrmodel01.bin')
    print("模型已保存成功")


if __name__ == '__main__':
    train_and_save()
