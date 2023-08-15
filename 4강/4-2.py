import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# SGDClassifier는 2개의 매개변수가 필요, loss는 손실함수의 종ㄹ 지정, max_iter는 에포크 횟수
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss = 'log_loss',max_iter= 10,random_state=42)
sc.fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

# partial_fit으로 모델을 추가적으로 더 학습 가능

sc.partial_fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

# 에포크 과대/과소 적합=> 에포크가 많으면 훈련이 많이 됨으로 훈련 세트에는 계속 높아지는 과대적합이 일어난다, 바꾸어 말아하면 에포크가 적으면 과소적합이 일어난다
# 따라서 과대적합이 시작하기 전에 에포크를 그만 하는것이 좋다=> 이것을 조기 종료라고 부름
# partial_fit 함수를 이용해서 조기종료 구간을 확인해 보자
# 여기서 중요한건 partial_fit를 사용할때 np.unqiue를 사용하여 생선 target 목록이 7개 있다는걸 알려주자(중간에 6,5 개 있는것이 들어올 수도 있기 때문이다)

import numpy as np
sc = SGDClassifier(loss = 'log_loss',random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

for _ in range(300):
    sc.partial_fit(train_scaled,train_target,classes=classes)
    train_score.append(sc.score(train_scaled,train_target))
    test_score.append(sc.score(test_scaled,test_target))

import matplotlib.pyplot as plt
# plt.plot(train_score)
# plt.plot(test_score)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()

sc = SGDClassifier(loss='log_loss',max_iter=100,tol=None,random_state=42)
sc.fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

