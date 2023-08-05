import tensorflow as tf
import pandas as pd
import numpy as np

# data = pd.read_csv('/Users/jewoonlee/tensorflow/gpascore.csv')
data = pd.read_csv('gpascore.csv')


# print(data.isnull().sum())  #null값 얼마나 있는지 카운트

data = data.dropna()    #빈칸 없애기
y = data['admit'].values

x = []

for i,rows in data.iterrows():
    x.append([rows['gre'],rows['gpa'],rows['rank']])

print(x)



# print(data['gpa'].min())
# data.fillna(100)        #빈칸을 채워줌
# print(data.isnull().sum())


model = tf.keras.models.Sequential([ 
    tf.keras.layers.Dense(64, activation= 'tanh'),    
    tf.keras.layers.Dense(128, activation= 'tanh'),
    tf.keras.layers.Dense(1, activation= 'sigmoid'),
])

model.compile(optimizer='adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
#epochs는 몇번 학습 시킬 것인지
model.fit( np.array(x), np.array(y), epochs=1000)

#예측
예측값 = model.predict([[750,3.70,3],[400,2.2,1]])
print(예측값)