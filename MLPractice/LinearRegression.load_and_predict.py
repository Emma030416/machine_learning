# load_and_predict.py
import joblib


def load_and_predict(new_data):
    estimator = joblib.load('model/mylrmodel01.bin')
    print("模型已加载成功")
    return estimator.predict(new_data)

if __name__ == '__main__':
    print(load_and_predict([[90, 80]]))