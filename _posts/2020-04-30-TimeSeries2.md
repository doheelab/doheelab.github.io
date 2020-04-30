---
title: "케라스(Keras)로 Stacked LSTM 구현하기"
date: 2020-04-30 09:00:00 -0400
categories: machine-learning
---



(참고자료)

[1] "[Stacked Long Short-Term Memory Networks](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/)"

[2] [# 케라스와 함께하는 쉬운 딥러닝 (20) - 순환형 신경망(RNN) 모델 만들기 3](https://buomsoo-kim.github.io/keras/2019/07/29/Easy-deep-learning-with-Keras-20.md/)

[3] [Sales_Transactions_Dataset_Weekly Data Set](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly)



## 소개

이 글에서 다룰 내용은 Keras를 활용하여 Stacked LSTM 구현을 구현한 후 time series prediction task에 적용해보는 것입니다.

## Stacked LSTM을 사용하는 이유
보통 neural network 에서 모델의 성능을 향상시키기 위해 hidden lyaer의 노드의 갯수를 과도하게 증가시키는 것보다 hidden layer의 층을 쌓는 것이 더욱 효울적인 것이 알려져 있습니다.

Stacked LSTM은 LSTM이 더 복잡한 task를 해결할 수 있도록, 모델의 복잡도를 높이는 방법 중 하나라고 생각하시면 됩니다.

![Stacked Long Short-Term Memory Archiecture](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/07/architecture_stacked_lstm.png)

[그림1: Stacked Long Short-Term Memory Archiecture 1]

![](http://www.wildml.com/wp-content/uploads/2015/09/Screen-Shot-2015-09-16-at-2.21.51-PM-272x300.png)

[그림2: Stacked Long Short-Term Memory Archiecture 2]

## 데이터 다운로드
[Sales_Transactions_Dataset_Weekly Data Set](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly)

800개가 넘는 상품에 대하여 52주 동안의 주별 구매량 데이터를 제공합니다.


## 데이터 불러오기

```python
import pandas as pd
import numpy as np

data = pd.read_csv("../input/Sales_Transactions_Dataset_Weekly.csv")
data = data.filter(regex="Product|W").copy()

data['Product_Code_NUM'] = data['Product_Code'].str.extract("(\d+)").astype(int)

print(data.shape)
data.head()
```
```
(811, 54)
```

![image](https://user-images.githubusercontent.com/57972646/80658133-3d003880-8ac0-11ea-9f7a-12ac0856cf88.png)

```python

print("Max Product_Code: {} - Unique Product_Code: {}".format(data['Product_Code_NUM'].max(), data['Product_Code_NUM'].nunique()))
```
```python
Max Product_Code: 819 - Unique Product_Code: 811
```
상품코드의 최대값은 819이며 유니크한 상품코드의 갯수는 811개 입니다.

## 데이터 전처리

  향후 3주간의 구매량을 예측하기 위해 입력값으로 이전 7주간의 데이터를 사용하였습니다. 

각 데이터 별로 input column의 갯수는 7개이며, target column의 갯수는 3개입니다.

```python
from keras.utils.np_utils import to_categorical

X_train = []
Y_train = []

X_test = []
Y_test = []

for ix, row in data.iterrows():
    for w in range(8, 50):
        x = row.iloc[w-7:w].values.astype(int)
        y = row.iloc[w:w+3].values.astype(int)
        if w < 30:
            X_train.append(x)
            Y_train.append(y)
        else:
            X_test.append(x)
            Y_test.append(y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

```

```python
((17842, 7), (16220, 7), (17842, 3), (16220, 3))
```


sklearn의 RobustScaler(중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환)을 사용하여 전처리를 했습니다.


```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 모델 정의하기

2개의 LSTM을 stack하여 네트워크를 정의하였습니다.

```python
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, LSTM
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler

def deep_lstm():
    model = Sequential()
    model.add(LSTM(4, input_shape = (7,1), return_sequences = True))
    model.add(LSTM(4, return_sequences = False))
    model.add(Dense(3))
    #model.add(Activation('softmax'))
    
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'mse', optimizer = adam)
    
    return model
```
# 학습하기

```
Stacked LSTM의 Input은 3차원(samples, time steps, and features)이어야 하므로 Input의 차원을 하나 증가시켜줍니다.

```python
X_train = np.expand_dims(X_train, axis=2)
```

학습을 시작합니다.

```
model = deep_lstm()
X_train = np.expand_dims(X_train, axis=2)

def scheduler(epoch):
    if epoch < 10:
        return 0.01
    else:
        return 0.001

callback = LearningRateScheduler(scheduler)

model.fit(X_train, Y_train, epochs=20, callbacks=[callback], verbose=1)
```

```
Epoch 1/20
17842/17842 [==============================] - 2s 108us/step - loss: 102.4639
Epoch 2/20
17842/17842 [==============================] - 2s 87us/step - loss: 26.5516
Epoch 3/20
17842/17842 [==============================] - 2s 88us/step - loss: 16.9195
Epoch 4/20
17842/17842 [==============================] - 2s 90us/step - loss: 15.6632
Epoch 5/20
17842/17842 [==============================] - 2s 87us/step - loss: 15.3605
Epoch 6/20
17842/17842 [==============================] - 2s 87us/step - loss: 15.3157
Epoch 7/20
17842/17842 [==============================] - 2s 87us/step - loss: 15.2968
Epoch 8/20
17842/17842 [==============================] - 2s 87us/step - loss: 15.2612
Epoch 9/20
17842/17842 [==============================] - 2s 87us/step - loss: 15.3096
Epoch 10/20
17842/17842 [==============================] - 2s 93us/step - loss: 15.2477
Epoch 11/20
17842/17842 [==============================] - 2s 89us/step - loss: 15.0871
Epoch 12/20
17842/17842 [==============================] - 2s 90us/step - loss: 15.0195
Epoch 13/20
17842/17842 [==============================] - 2s 89us/step - loss: 15.0190
Epoch 14/20
17842/17842 [==============================] - 2s 90us/step - loss: 15.0101
Epoch 15/20
17842/17842 [==============================] - 2s 92us/step - loss: 15.0177
Epoch 16/20
17842/17842 [==============================] - 2s 91us/step - loss: 15.0066
Epoch 17/20
17842/17842 [==============================] - 2s 90us/step - loss: 15.0107
Epoch 18/20
17842/17842 [==============================] - 2s 91us/step - loss: 15.0103
Epoch 19/20
17842/17842 [==============================] - 2s 92us/step - loss: 15.0063
Epoch 20/20
17842/17842 [==============================] - 2s 90us/step - loss: 15.0054
```


## 결과 확인하기
1주, 2주, 3주 후 예측값과 실제값 사이의 에러를 계산하였습니다.
더 먼 미래를 예측할 수록 에러가 커지는 것을 확인할 수 있습니다.


```python
p = model.predict(np.expand_dims(X_test, axis=2))
np.sqrt(((p - Y_test)**2)).mean(axis=0)
```
```
array([2.13613641, 2.16403327, 2.20673458])
```

다음은 시각화입니다.

```python
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
tips = sns.load_dataset("tips")

plt.figure(figsize=(10,5))
plt.plot(range(7), scaler.inverse_transform(X_test[:1,:])[0][:7])
plt.plot(range(6,10),np.concatenate([np.array([5]),p[0]], axis=0))
plt.plot(range(6,10),np.concatenate([np.array([5]),Y_test[0]], axis=0))
plt.legend(['Data', 'Prediction', 'True'])
plt.title("Time Series Prediction")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(7), scaler.inverse_transform(X_test[1:2,:])[0][:7])
plt.plot(range(6,10),np.concatenate([np.array([11]),p[1]], axis=0))
plt.plot(range(6,10),np.concatenate([np.array([11]),Y_test[0]], axis=0))
plt.legend(['Data', 'Prediction', 'True'])
plt.title("Time Series Prediction")
plt.show()
```

![image](https://user-images.githubusercontent.com/57972646/80673859-510d5f80-8aeb-11ea-839a-5b247163c15d.png)

![image](https://user-images.githubusercontent.com/57972646/80673887-5bc7f480-8aeb-11ea-82d9-9130b7fc97f5.png)
