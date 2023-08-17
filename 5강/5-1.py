import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
# print(wine.head())

# info 메서드:  데이터프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인하기 유용함

# print(wine.info())
# 누락된 값이 있으면 데이터를 버리거나 평균값으로 채우고 사용 가능

# describe(): 열에 대한 간략한 통계 출력해줌, 최소, 최대, 평균값 등
# print(wine.describe())
data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data,target, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
ss =StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled,train_target)
# print(lr.score(train_scaled,train_target))
# print(lr.score(test_scaled,test_target))

print(lr.coef_,lr.intercept_)

# 결정 트리
# sklearn에서 결정트리 알고리즘 제공 DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(train_scaled,train_target)
# print(dt.score(train_scaled,train_target))
# print(dt.score(test_scaled,test_target))

# plot_tree를 통해 그림으로 출력 가능
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# plt.figure(figsize=(10,7))
# plot_tree(dt)
# plt.show()

# plt.figure(figsize=(10,7))
# plot_tree(dt, max_depth=1,filled=True,feature_names=['alcohol','sugar','pH'])
# plt.show()

# leaf노드에 더 많은 클래스가 그 값의 예측값이 됨

# gini: 불순도 
# gini는 클래스의 비율을 제곱하서 더한 다음 1에서 빼면 됨
# 지니 불순도 = 1 - (음성 클래스 비율^2 + 양성 클래스 베울^2)
# 예를 들어 100개 샘플이 있는 어떤 노드의 두 클래스의 비율이 정확이 1/2 이면 지니 = 0.5가 됨 1- (1/2 ^ 2+ 1/2 ^ 2) 
# 만약 한 곳에 몰려있다면 1 - (0+ 1^2) = 0
# 결정 트리는 부모 노드와 자식 노드의 불순도 차이가 클수록 좋음
# 불순도 차이 구하는 법: 부모 불순도 - (왼쪽 노드 샘플 수 / 부모의 샘플 수) * 왼쪽 노드 불순도 - (오른쪽 노드 샘플 수 / 부모의 샘플 수) * 오른쪽 노드 불순도
# 불순도 차이= 정보 이득

# 또 다른 불순도 기준 criterion = 'entropy'
# 제곱이 아닌 밑이 2인 로그 사용
# - 음성 클래스 비율 * log2(음성 클래스 비율) - 양성 클래스 비율 * log2(양성 클래스 비율)

# 과대 적합 없에는 법: 가지치기 => 트리의 최대 깊이를 지정하는 것
# max_depth = 3 으로 지정

# dt = DecisionTreeClassifier(max_depth=3,random_state=42)
# dt.fit(train_scaled,train_target)
# print(dt.score(train_scaled,train_target))
# print(dt.score(test_scaled,test_target))

# plt.figure(figsize=(20,15))
# plot_tree(dt,filled=True,feature_names=['alcohol','sugar','pH'])
# plt.show()

# 결정 트리를 할때는 전처리 즉 스케일 맞출 필요가 없음
dt = DecisionTreeClassifier(max_depth=3,random_state=42)
dt.fit(train_input,train_target)
print(dt.score(train_input,train_target))
print(dt.score(test_input,test_target))

plt.figure(figsize=(15,10))
plot_tree(dt,filled=True,feature_names=['alcohol','sugar','pH'])
plt.show()

# feature_importance_ 속성을 통해 어떤 특성이 제일 중요한지 보여줌

print(dt.feature_importances_)