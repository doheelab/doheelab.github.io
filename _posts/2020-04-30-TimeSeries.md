---
title: "뉴럴 네트워크로 시계열 데이터 예측하기(time series prediction)"
date: 2020-04-30 09:00:00 -0400
categories: machine-learning
---

블로그 글 "[How To Use Neural Networks to Forecast Multiple Steps of a Time Series](https://www.mariofilho.com/how-to-use-neural-networks-to-forecast-multiple-steps-of-time-series/)"을 참고하여 작성하였습니다.

이 글에서 다룰 내용은 시계열 데이터(상품 구매량 데이터)를 활용하여 현재로부터 1, 2, 3주 미래의 구매량을 예측하는 것입니다.

## 데이터 준비
https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly

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
(811, 54)

![image](https://user-images.githubusercontent.com/57972646/80658133-3d003880-8ac0-11ea-9f7a-12ac0856cf88.png)

```python
print("Max Product_Code: {} - Unique Product_Code: {}".format(data['Product_Code_NUM'].max(), data['Product_Code_NUM'].nunique()))
```
Max Product_Code: 819 - Unique Product_Code: 811

상품코드의 최대값은 819이며 유니크한 상품코드의 갯수는 811개 입니다.

## 데이터 전처리

  향후 3주간의 구매량을 예측하기 위해 입력값으로 이전 7주간의 데이터와 상품 코드를 사용하였습니다. 

상품 코드를 수치화 하기 위해 to_categorical 함수를 활용했습니다.  

각 데이터 별로 input column의 갯수는 827개(7+820)이며, label column의 갯수는 3개입니다.

```python
from keras.utils.np_utils import to_categorical

X_train = []
Y_train = []

X_test = []
Y_test = []

for ix, row in data.iterrows():
    for w in range(8, 50):
        product_code_num = row['Product_Code_NUM']
        x = row.iloc[w-7:w].values.astype(int)
        y = row.iloc[w:w+3].values.astype(int)

        product_code_num_ohe = to_categorical(product_code_num, num_classes=820)
        x = np.append(x, product_code_num_ohe)

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
((17842, 827), (16220, 827), (17842, 3), (16220, 3))

> RobustScaler

sklearn의 RobustScaler(중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환)을 사용하여 전처리를 했습니다.


```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```
> RobustScaler 전처리 전, 후 비교

![image](https://user-images.githubusercontent.com/57972646/80659105-ed6f3c00-8ac2-11ea-9644-92fffbf22852.png)



## 학습하기 

hidden layer가 1개인 뉴럴 네트워크를 정의하여 학습 하였습니다.

```python
from keras import Model
from keras.layers import Dense, Input, Dropout

inp = Input(shape=(827,))
hid1 = Dense(64, activation='relu')(inp)
out  = Dense(3, activation='linear')(hid1)

mdl = Model(inputs=inp, outputs=out)
mdl.compile(loss='mse', optimizer='adam')

mdl.fit(X_train,Y_train, shuffle=True, validation_data=[X_test, Y_test], epochs=20, batch_size=32)
```
(Output)
```
Train on 17842 samples, validate on 16220 samples
Epoch 1/20
17842/17842 [==============================] - 1s 42us/step - loss: 52.7907 - val_loss: 12.2801
Epoch 2/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.7185 - val_loss: 12.5315
Epoch 3/20
17842/17842 [==============================] - 1s 37us/step - loss: 15.6272 - val_loss: 12.1437
Epoch 4/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.5493 - val_loss: 12.1424
Epoch 5/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.5250 - val_loss: 12.0493
Epoch 6/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.5047 - val_loss: 12.2614
Epoch 7/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.4791 - val_loss: 12.1326
Epoch 8/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.4379 - val_loss: 12.1715
Epoch 9/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.4293 - val_loss: 12.4249
Epoch 10/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.3508 - val_loss: 11.9541
Epoch 11/20
17842/17842 [==============================] - 1s 37us/step - loss: 15.3012 - val_loss: 11.9968
Epoch 12/20
17842/17842 [==============================] - 1s 37us/step - loss: 15.2420 - val_loss: 12.4801
Epoch 13/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.1709 - val_loss: 12.3623
Epoch 14/20
17842/17842 [==============================] - 1s 36us/step - loss: 15.0996 - val_loss: 12.4415
Epoch 15/20
17842/17842 [==============================] - 1s 36us/step - loss: 14.9894 - val_loss: 12.2338
Epoch 16/20
17842/17842 [==============================] - 1s 36us/step - loss: 14.8829 - val_loss: 12.4625
Epoch 17/20
17842/17842 [==============================] - 1s 36us/step - loss: 14.8321 - val_loss: 11.7942
Epoch 18/20
17842/17842 [==============================] - 1s 36us/step - loss: 14.7513 - val_loss: 12.0298
Epoch 19/20
17842/17842 [==============================] - 1s 37us/step - loss: 14.6618 - val_loss: 11.8049
Epoch 20/20
17842/17842 [==============================] - 1s 37us/step - loss: 14.5454 - val_loss: 12.1301
```


## 결과 확인하기
1주, 2주, 3주 후 예측값과 실제값 사이의 에러를 계산하였습니다.

더 먼 미래를 예측할 수록 에러가 증가하는 것을 확인할 수 있습니다.


```python
p = mdl.predict(X_test)
np.sqrt(((np.log1p(p) - np.log1p(Y_test))**2)).mean(axis=0)
```
array([2.10016761, 2.10686022, 2.16465018])

>시각화

Seaborn과 Pyplot을 이용하여 시각화 하였습니다.

```python
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
tips = sns.load_dataset("tips")

plt.figure(figsize=(10,5))
plt.plot(range(7), scaler.inverse_transform(X_test[:1,:])[0][:7])
plt.plot(range(6,10),np.concatenate([np.array([5]),p[0,:]], axis=0))
plt.plot(range(6,10),np.concatenate([np.array([5]),Y_test[0]], axis=0))
plt.legend(['Data', 'Prediction', 'True'])
plt.title("Time Series Prediction")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(7), scaler.inverse_transform(X_test[1:2,:])[0][:7])
plt.plot(range(6,10),np.concatenate([np.array([11]),p[1,:]], axis=0))
plt.plot(range(6,10),np.concatenate([np.array([11]),Y_test[1,:]], axis=0))
plt.legend(['Data', 'Prediction', 'True'])
plt.title("Time Series Prediction")
plt.show()
```

![image](https://user-images.githubusercontent.com/57972646/80660994-56a57e00-8ac8-11ea-85ac-051a34094497.png)

![image](https://user-images.githubusercontent.com/57972646/80661043-73da4c80-8ac8-11ea-8c78-e1b112c941c9.png)
