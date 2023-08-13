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


