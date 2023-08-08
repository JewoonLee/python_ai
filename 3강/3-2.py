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


from sklearn.model_selection import train_test_split
train_input , test_input, train_output, test_output = train_test_split(perch_length,perch_weight,random_state=42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()

knr.n_neighbors = 3
knr.fit(train_input,train_output)

print(knr.predict([[50]]))

import matplotlib.pyplot as plt

# distances, indexes = knr.kneighbors([[50]])
# plt.scatter(train_input, train_output)
# plt.scatter(train_input[indexes],train_output[indexes],marker= 'D')
# plt.scatter(50,1033, marker= '^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 이게 문제가 되는 이유가 50 k-이웃집은 50옆에있는 k개의 평균으로 구하기 때문에 오류남
# 그래서 선형 회귀를 써야함
# 선형 회귀 => 특성이 하나인 경우 어떤 직선을 학습하는 알고리즘
# sklearn.linear_model LinearRegression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input,train_output)
print(lr.predict([[50]]))

# 선형 회귀 y= a*x + b임
# a,b 는 lr객체의 coef_, intercept_ 속성에 저장되어 있음
# a 는 기울기 => coefficient, 또는 weight(가중치)
# a, b는 머신러닝 알거맂므이 찾은 값=> 모델 파라미터, 모델 기반 학습

print(lr.coef_, lr.intercept_)

#a,b를 이용해서 15, 50까지 선형 그리기

# plt.scatter(train_input,train_output)
# # plot는 선 그리기
# plt.plot([15,50], [15*lr.coef_ + lr.intercept_,50*lr.coef_ + lr.intercept_])
# plt.scatter(50,1241.8,marker= '^')
# plt.xlabel('lenght')
# plt.ylabel('weight')
# plt.show()

print(lr.score(train_input,train_output))
print(lr.score(test_input,test_output))

# 다항 회귀
# 길이를 제곱한것을 길이 자료구조 앞에 붙이기

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2,test_input))

lr.fit(train_poly,train_output)
print(lr.predict([[50**2,50]]))

print(lr.coef_,lr.intercept_)

# 구간별 직선을 그리기 위해서 15에서 49까지 정수 배열을 만들기
point = np.arange(15,51)
plt.scatter(train_input,train_output)

#15에서 49까지 2차 방적식 그래프 그리기
plt.plot(point,1.01*point ** 2 -21.6*point + 116.05)
plt.scatter(50,1574,marker="^")
plt.xlabel('lenght')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly,train_output))
print(lr.score(test_poly,test_output))
