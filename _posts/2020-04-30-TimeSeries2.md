---
title: "케라스(Keras)로 Stacked LSTM 구현하기"
date: 2020-04-30 09:00:00 -0400
categories: machine-learning
---

## 소개

이 글에서는 Keras를 활용하여 Stacked LSTM 구현을 구현하고 time series prediction task에 적용하겠습니다.

## Stacked LSTM을 사용하는 이유

보통 neural network 에서 모델의 성능을 향상시키기 위해 hidden lyaer의 노드 갯수를 과도하게 증가시키는 것보다, hidden layer의 층을 쌓는 것이 더욱 효울적인 것으로 알려져 있습니다.

Stacked LSTM은 LSTM이 더 복잡한 task를 해결할 수 있도록, 모델의 복잡도를 높이는 방법 중 하나로 볼 수 있습니다.

<div style="text-align:center"><img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/07/architecture_stacked_lstm.png" /></div>

<div align="center">
  <i>Stacked Long Short-Term Memory Archiecture 1</i>
</div>

<br/>

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/101423085-945e6500-393b-11eb-90de-f84a9f37ee06.png" /></div>

<div align="center">
  <i>Stacked Long Short-Term Memory Archiecture 2</i>
</div>

## 데이터 다운로드

[Sales_Transactions_Dataset_Weekly Data Set](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly)

총 811개의 상품에 대하여 52주 동안의 주별 구매량 데이터를 활용 하겠습니다.

---
<br/>

## 데이터 불러오기

```python
import pandas as pd

data = pd.read_csv("./Sales_Transactions_Dataset_Weekly.csv")
# Product 혹은 W로 시작하는 것만 남기기
data = data.filter(regex="Product|W").copy()  
data["Product_Code_NUM"] = data["Product_Code"].str.extract("(\d+)").astype(int)

print(data.shape)
data.head()
```

```
(811, 54)
```

![image](https://user-images.githubusercontent.com/57972646/80658133-3d003880-8ac0-11ea-9f7a-12ac0856cf88.png)

상품코드의 최대값은 819이고, unique한 상품코드의 갯수는 811개 입니다.

```python
print("Max Product_Code: {} - Unique Product_Code: {}".format(data['Product_Code_NUM'].max(), data['Product_Code_NUM'].nunique()))
```

```python
Max Product_Code: 819 - Unique Product_Code: 811
```

---
<br/>

## 데이터 전처리

향후 3주간의 구매량을 예측하기 위해 학습 데이터로 이전 7주간의 데이터를 사용하였고, 라벨 데이터는 그 다음 3주간의 데이터를 활용하였습니다.

따라서 각 데이터 별로 input column의 갯수는 7개이며, target column의 갯수는 3개입니다.

```python
import numpy as np

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

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

```

```
((17842, 7), (16220, 7), (17842, 3), (16220, 3))
```

학습을 시작하기 전에, `sklearn`의 `RobustScaler`을 사용하여, `중앙값(median)`이 0, `IQR`(interquartile range, https://wikidocs.net/89704)이 1이 되도록 변환하였습니다.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
print("변경 전 :", X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("변경 후 :", X_train)
```

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/111785440-f564e400-88ff-11eb-94f6-4bcbdd45819b.png" /></div>

<div align="center">
  <i>sklearn의 RobustScaler 적용 전, 후</i>
</div>


`Stacked LSTM`의 Input은 3차원(samples, time steps, features)이어야 하므로 Input의 차원을 하나 증가시켜줍니다.

````python
print("변경 전 :", X_train, X_train.shape)
X_train = np.expand_dims(X_train, axis=2)
print("변경 후 :", X_train, X_train.shape)
````

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/111785823-6efcd200-8900-11eb-9747-e9e7068ab19b.png" /></div>

<div align="center">
  <i>numpy의 expand_dims 적용 전, 후</i>
</div>

---
<br/>

## 모델 정의하기

hidden layer가 2개인 `stacked LSTM` 네트워크를 정의하였습니다.

```python
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers

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

---
<br/>

## 학습하기

```python
model = deep_lstm()

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

---
<br/>

## 결과 확인하기

1주, 2주, 3주 후 예측값과 실제값 사이의 에러를 계산하였습니다.
더 먼 미래를 예측할 수록 에러가 커지는 것을 확인할 수 있습니다.

```python
prediction = model.predict(np.expand_dims(X_test, axis=2))
np.sqrt(((prediction - Y_test)**2)).mean(axis=0)
```

```
array([2.13613641, 2.16403327, 2.20673458])
```

---
<br/>

## 예측 결과 시각화

```python
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
tips = sns.load_dataset("tips")

plt.figure(figsize=(10,5))
plt.plot(range(7), scaler.inverse_transform(X_test[:1,:])[0][:7])
plt.plot(range(6,10),np.concatenate([np.array([5]),prediction[0]], axis=0))
plt.plot(range(6,10),np.concatenate([np.array([5]),Y_test[0]], axis=0))
plt.legend(['Data', 'Prediction', 'True'])
plt.title("Time Series Prediction")
plt.show()
```

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/80673859-510d5f80-8aeb-11ea-839a-5b247163c15d.png" /></div>

<div align="center">
  <i>첫번째 상품의 판매량의 예측 결과와 실제값</i>
</div>
<br/>

```python
plt.figure(figsize=(10,5))
plt.plot(range(7), scaler.inverse_transform(X_test[1:2,:])[0][:7])
plt.plot(range(6,10),np.concatenate([np.array([11]),prediction[1]], axis=0))
plt.plot(range(6,10),np.concatenate([np.array([11]),Y_test[0]], axis=0))
plt.legend(['Data', 'Prediction', 'True'])
plt.title("Time Series Prediction")
plt.show()
```

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/57972646/80673887-5bc7f480-8aeb-11ea-82d9-9130b7fc97f5.png" /></div>


<div align="center">
  <i>두번째 상품의 판매량의 예측 결과와 실제값</i>
</div>

---
<br/>

## 참고자료

[1] "[Stacked Long Short-Term Memory Networks](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/)"

[2] [# 케라스와 함께하는 쉬운 딥러닝 (20) - 순환형 신경망(RNN) 모델 만들기 3](https://buomsoo-kim.github.io/keras/2019/07/29/Easy-deep-learning-with-Keras-20.md/)

[3] [Sales_Transactions_Dataset_Weekly Data Set](https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly)
