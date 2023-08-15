import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
# print(fish.head())
print(pd.unique(fish['Species']))

# 첫 열 뺘고 다 input에 넣기
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
# print(fish_input[:5])

# 첫 열 target에 넣기
fish_target = fish['Species'].to_numpy()
# print(fish_target[:5])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)

# 모든 특성들 스케일 맞추기
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기의 확률 예측

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled,train_target)
print(kn.score(train_scaled,train_target))
print(kn.score(test_scaled,test_target))
print(kn.classes_)

# 여기서 주의해야 할 점은 kn에 있는 class 순서는 알파벳 순서로 처음이랑 다르다
# predict_proba() 메서드를 통해 클레스별 확률값을 반환함

import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals = 4))

# 확인해보기
distnaces, indexs = kn.kneighbors(test_scaled[:5])      # 가장 가꺼운 이웃 찾기
print(train_target[indexs])

# 조금 더 좋은 방법을 찾아보자
# 로지스틱 회귀
# 시그모이드 함수 그려보기

import numpy as np
import matplotlib.pyplot as plt
# z = np.arange(-5,5,0.1)
# phi = 1 / (1 + np.exp(-z))
# plt.plot(z,phi)
# plt.xlabel('z')
# plt.ylabel('phi')
# plt.show()

# 로지스틱 회귀로 이진 분류 수행하기
char_arr = np.array(['A','B','C','D','E'])
print(char_arr[[True,False,True,False,False]])

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱은 sklearn.linear_model 안에 있음
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))

print(lr.coef_,lr.intercept_)
# 이걸로 이제 z값 구하기
# LogisticRegression에서 decision_function이라는 메서드 있음=> 이걸로 z 값 구하기 가능

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# 여기서 시그모이드 식에 z 값 넣어서 시그모이드 값 구하기
# 시그모이드 함수는 expit() 함수 사용해서 가능

from scipy.special import expit
print(expit(decisions))
# 시그모이드 값이 proba 값이랑 같음
# 이제 이걸 바탕으로 7개의 생선을 분류해 보자
# 로지스틱 회귀는 릿지와 같이 계수의 제곱을 규제함
# alpha로 규제를 제어하는데 여기서는 C를 사용함=> C랑 alpha는 정반대

lr = LogisticRegression(C= 20, max_iter=1000)
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

print(lr.predict(test_scaled[:5]))

# 5개 확률 보기
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=3))

# 여기서 다중 분류의 경우 선형방적식은 7행 5열 이 나온다
# 이 말은 즉 z 값을 7개나 계산 한다는 말이다
# 즉 클래스마다 하나씩 계산한다는 말이다
# 이진 분류는 시그모이드 함수를 이용해 z 를 0~1 사이로 변환함
# 다중 분류는 이와 달리 소프트맥스 함수를 이용함
# 소프트맥스는 여러 개의 선형 방정식의 출력값을 0~1 사이로 압축하고 전체 압이 1이 되도록 만듦
# 소프트맥스 방법: z 값을 지수 함수를 이용해 모두 더함
# z1~z7 e_sum = e^z1 + ... + e^z7

# s1= e^z1 / e_sum ...


# decision_fuction을 이용해서 z1~z7값 구한 다음 소프트맥스로 확률 바꾸기

decisions = lr.decision_function(test_scaled[:5])
print(np.round(decisions,decimals=2))

# softmax 이용하기
from scipy.special import softmax
proba = softmax(decisions,axis=1)
print(np.round(proba,decimals=3))
# 여기서 주의해야하 ㄹ 점은 axis=1로 하지 않으면 배열 전체에 대한 소프트맥스를 계산함

# 순서: 1.먼저 데이터들을 받아서 train이랑 test로 나눔
# 2. 특성들의 scale을 다 똑같이 만든 다음에 로지스틱 회귀에 집어 넣음
# 3. 그럼 모든 고기에 대한 계수값들이 나올 것이다.
# 4. 그걸 이용하면 한 고기에 대한 모든 고기의 z값   소프트 맥스를 구할 수 있을 것이다.
# 5. 그럼 확률 구하기 끝
