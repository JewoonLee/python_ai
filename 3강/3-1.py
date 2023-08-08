import numpy as np


perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

import matplotlib.pyplot as plt

# plt.scatter(perch_length,perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

from sklearn.model_selection import train_test_split
train_input , test_input, train_output, test_output = train_test_split(perch_length,perch_weight,random_state=42)

# -1 은 모든 원소를 채우라는 의미
# sklearn은 배열을 2차원으로 만들어야 함
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
print(test_input.shape,train_input.shape)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()

knr.fit(train_input,train_output)
# 회귀의 경위 score를 결정계수 R^2을 통해서 판단함
# R^2 = 1 - (타깃 - 예측)^2의 합 / (타깃 - 타깃 평균)^2 의 합
print(knr.score(test_input,test_output))

# 정확독 다르게 판단하는법 => 타깃과 예측값 사이의 차 구하기
from sklearn.metrics import mean_absolute_error     # 오차 평균 구하는 것

# 테스트 세트에 대한 예측
test_prediction = knr.predict(test_input)

# 테스트 세트에 대한 평균 절대값 오차 구하기
mae = mean_absolute_error(test_output,test_prediction)
print(mae)

# 테스트 세트 말고 훈련 세트로 구해보자
print(knr.score(train_input,train_output))

# 과대적합(overfitting): 훈련세트 점수는 좋은데 테스트 세트 점수가 안좋은 것
# 과소적합(underfitting): 훈련 세트보다 테스트 세트 점수가 좋은 것 => 모델이 너무 단순하여 적절히 훈련되지 않은 것 => 모델 좀 더 복잡하게 만들기
# k-이웃집 모델 복잡하게 만드는 법: 이웃 개수를 줄이는 것 k를 5에서 3으로 낮추기

knr.n_neighbors = 3
knr.fit(train_input,train_output)
print(knr.score(train_input,train_output))
print(knr.score(test_input,test_output))

# 과대적합일 경우 덜 복잡하게 모델링 k 늘리기
# 과소적합일 경우 더 복잡하게 모델링: k 줄이기

