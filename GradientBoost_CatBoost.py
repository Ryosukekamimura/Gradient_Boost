import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Kerasに付属の手書き数字画像データをダウンロード
np.random.seed(0)
(X_train_base, labels_train_base), (X_test, labels_test) = mnist.load_data()

# Training set を学習データ（X_train, labels_train）と検証データ（X_validation, labels_validation）に8:2で分割する
X_train,X_validation,labels_train,labels_validation = train_test_split(X_train_base,labels_train_base,test_size = 0.2)

# 各画像は行列なので1次元に変換→X_train,X_validation,X_testを上書き
X_train = X_train.reshape(-1,784)
X_validation = X_validation.reshape(-1,784)
X_test = X_test.reshape(-1,784)

#正規化
X_train = X_train.astype('float32')
X_validation = X_validation.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_validation /= 255
X_test /= 255

# 訓練・テストデータの設定
from catboost import Pool

train_pool = Pool(X_train, labels_train)
validate_pool = Pool(X_validation, labels_validation)

#経過時間計測
import time
start = time.time()

params = {
    'early_stopping_rounds' : 10,
    'iterations' : 100,
    'custom_loss' :['Accuracy'],
    'random_seed' :42
}
# パラメータを指定した場合は、以下のようにインスタンスに適用させる
model = CatBoostClassifier(**params)
cab = model.fit(train_pool, eval_set=validate_pool)
preds = cab.predict(X_test)
from sklearn.metrics import accuracy_score
print('accuracy_score:{}'.format(accuracy_score(labels_test, preds)))

#経過時間
print('elapsed_timetime:{}'.format(time.time()-start))