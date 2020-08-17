
import pandas as pd
import numpy as np
import time


import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier


from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Kerasに付属の手書き数字画像データをダウンロード
np.random.seed(0)
(X_train_base, labels_train_base), (X_test, labels_test) = mnist.load_data()


# Training Set を学習データ(X_train , labels_train) と検証データ(X_validation, labels_validation)に8:2で分類する

X_train, X_validation, labels_train, labels_validation = train_test_split(X_train_base, labels_train_base, test_size = 0.2)

# 各画像は、行列なので1次元に変換→ X_train , X_validation , X_testを上書き
X_train = X_train.reshape(-1, 784)
X_validation = X_validation.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# 正規化
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_validation /= 255
X_test /= 255


#　訓練・テストデータの設定
train_data = xgb.DMatrix(X_train, label=labels_train)
eval_data = xgb.DMatrix(X_validation, label=labels_validation)
X_data = xgb.DMatrix(X_test, label=labels_test)

# 経過時間の計測
start = time.time()

xgb_params = {

    #　多値分類問題
    'objective' : 'multi:softmax',

    #クラス数
    'num_class' : 10,

    #学習用の指標
    'eval_metric' : 'mlogloss'
}

evals = [(train_data, 'train'), (eval_data, 'eval')]
evals_result = {}

gbm = xgb.train(

    xgb_params,
    train_data,
    num_boost_round=100,
    early_stopping_rounds=10,
    evals=evals,
)

preds = gbm.predict(X_data)
print('accuracy_score: {}'.format(accuracy_score(labels_test, preds)))

# 経過時間の計測
print('elapsed_timetime: {}'.format(time.time() - start))