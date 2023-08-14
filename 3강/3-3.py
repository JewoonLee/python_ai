import pandas as pd 
df = pd.read_csv('https://raw.githubusercontent.com/rickiepark/hg-mldl/master/perch_full.csv')
perch_full = df.to_numpy()
# print(perch_full)

import numpy as np

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_full,perch_weight,random_state = 42)


# polynomialFeauters
# 2개의 특성 2,3 을 하나의 특성으로 만들기

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()

# fit으로 훈련 하기
poly.fit([[2,3]])

# transform으로 변환 하기
# 1, x1, x2 ,x1^2, x1 * x2, x2^2
print(poly.transform([[2,3]]))

#이걸로 이제 훈련 데이터 만들기
poly = PolynomialFeatures(include_bias=False)   # 앞에 1 없에기
poly.fit(train_input)
train_poly = poly.transform(train_input)
# print(train_poly.shape)

# print(poly.get_feature_names_out())
test_poly = poly.transform(test_input)

# 다중회귀는 선형 회귀랑 같은 원리
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# lr.fit(train_poly,train_target)
# print(lr.score(train_poly,train_target))
# print(lr.score(test_poly,test_target))

# 특성을 더 많이 추가해 보기, 3제곱 4 제곱
poly = PolynomialFeatures(degree=5,include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))
#이렇게 과하게 훈련할 경우에는 훈련세트에만 너무 과대적합 해짐으로 test 세트에는 안좋은 영향을 끼침

# 과하게 훈련 하는것을 막기 위해서 규제라는 것을 함
# 규제 a,b,c,d, 이런 것들을 작게 하는것
# 규제를 하기전에 2장에 했던 스케일 정규화가 필요함
# 2장에는 우리가 직접 했다면 이제는 사이킷 런에서 제공하는 standardscaler를 사용

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지(ridge): 계수를 제곱한 값 기준
# 라쏘(lasso): 계수의 절대값

# 릿지
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(test_scaled,test_target))

# 규제의 정도를 조정할 수 있음: alpha가 커지면 규제 강도가 커지고, 작으면 규제 강도가 작아짐
import matplotlib.pyplot as plt
# train_score = []
# test_score = []

# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
# for alpha in alpha_list:
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(train_scaled,train_target)
#     train_score.append(ridge.score(train_scaled,train_target))
#     test_score.append(ridge.score(test_scaled,test_target))

# plt.plot(np.log10(alpha_list),train_score)
# plt.plot(np.log10(alpha_list),test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()

ridge = Ridge(alpha= 0.1)
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

# 라쏘 회귀(lasso)
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))

train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled,train_target)
    train_score.append(lasso.score(train_scaled,train_target))
    test_score.append(lasso.score(test_scaled,test_target))

print(train_score, test_score)

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

lasso = Lasso(alpha=10)
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))

# 라쏘는 계수값이 0인 것이 있음
print(np.sum(lasso.coef_ == 0)) # np.sum은 함수 배열을 다 더한 값


