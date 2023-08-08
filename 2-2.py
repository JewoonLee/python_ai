fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]

fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np

#column_stack은 두 리스트를 순서에 맞게 합쳐줌
fish_data = np.column_stack((fish_length,fish_weight))

#np.concatenate()는 두 배열을 이어줌
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

# 훈련세트와 테스트 세트를 나누어줌
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
# 하지만 이건 비율에 맞지 않게 나눠질 수 있음, 따라서 stratify라는 것을 추가해주기
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input,train_target)
kn.score(test_input,test_target)

import matplotlib.pyplot as plt
# plt.scatter(train_input[:,0],train_input[:,1])
# plt.scatter(25,150, marker= '^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 이게 왜 25, 150이 빙어로 나오냐 하면은 x축의 길이는 짧고, y축의 길이는 길기 때문에 y축이 조금만 멀리 있어도 더 멀리 있다고 판단
# 따라서 비율을 맞춰줘야함
# plt.xlim(0,1000) 을 통해서 x축도 0~1000으로 맞출 수 있음
# 따라서 비율 맞추려면 특성값을 일정한 기준으로 맞춰야함 => 데이터 전처리
# 데이터 전처리 방법: 1. 표준점수
# 표준편차 => 모든 데이터를 평균으로 빼고, 제곱한걸 더한것, 표준점수: 각 데이터가 원점에서 몇 표준편차만큼 떨어져 있는가

#mean => 평균, np.std => 표준편차
mean = np.mean(train_input,axis = 0)
std = np.std(train_input,axis = 0)

# 원본 데이터에 평균을 빼고, 표준편차로 나누어 표준점수 구하기
train_scaled = (train_input - mean) / std
kn.fit(train_scaled,train_target)

new = ([25,150] - mean) / std

test_scaled = (test_input - mean) / std
print(kn.score(test_scaled,test_target))

print(kn.predict([new]))

distances, indexs = kn.kneighbors([new])
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker= '^')
plt.scatter(train_scaled[indexs, 0],train_scaled[indexs,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 주의할 점, 훈련 세트 그대로 테스트 세트에 변환해야함
