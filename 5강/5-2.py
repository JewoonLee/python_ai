# 테스트 세트를 사용하지 않고 과대, 과소 적합인지 아는 방법은 하나의 훈련 세트틀 또 만드는것 => 이를 검증세트라고 함
# 60% 는 훈련 세트, 20%는 검증 세트, 20%는 테스트 세트
# 먼저 훈련 세트를 훈련하고, 검증 세트로 평가함
# 그 다음 훈련 세트와 검증 세트를 합쳐 다시 훈련 하고, 마지막에 테스트 세트로 평가함

import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data,target,test_size=0.2,random_state=42)

# 이 다음 다시 train target을 split에 넣어 검증 세트르 만듦
sub_input, val_input, sub_target, val_target = train_test_split(train_input,train_target,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input,sub_target)
print(dt.score(sub_input,sub_target))
print(dt.score(val_input,val_target))

# 교차 검증: 검증 세트를 떼어 내어 평가하는 과정 반복. 그 다음 점수를 평균하여 최종 검증 점수 얻음
# 예시: 3-폴드 교차 검증: 훈련 세트를 세 부분으로 나누어서 교차 검증을 수행 하는 것
# 보통은 5~ 10 폴드 교차 많이 이용함
# cross_validate()라는 교차 검증 함수가 있음

from sklearn.model_selection import cross_validate
scores = cross_validate(dt,train_input,train_target)
print(scores)

# fit_time = 훈련 시간, score_time = 검증 시간, test_score = 최종 점수, 검증 폴드의 점수

import numpy as np
print(np.mean(scores['test_score']))

# 교차 검증을 할때 훈련 세트를 섞으러면 분할기(splitter) 를 지정해야 함
# 회귀 모델일 정유 KFold 분류 모델일 경우 StratifiedKFold

from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input,train_target,cv = StratifiedKFold())
print(np.mean(scores['test_score']))

# 훈련 세트 섞은 후 10-폴드 교차 검증 수행
splitter = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
scores = cross_validate(dt,train_input,train_target,cv=splitter)
print(np.mean(scores['test_score']))

# 머신러닝 모델이 학습하는 파라미터: 모델 파라미터
# 모델이 학습 할 수 없어서 사용자가 지정해야 하는 피라미터: 하이퍼파라미터
# 모델을 바꿔가면서 서치 함
# 그리드 서치 사용
# GrodSearchCV 클래스 임포트 하고 탐색 리스트 딕셔너리 사용

from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
# min_impurity_decrease의 값을 바꿔가며 5번 실행,
# GridSearch도 cv 를 5-폴드 교차 실행을 함 따라서 25번 실행
# n_jobs = -1은 모든 cpu코어를 사용한다는 말
gs.fit(train_input,train_target)
# 그리드 서치는 훈련이 끝나면 25개 모델중 점수가 가장 높은 매개변수 조합으로 다시 모델을 훈련함
# best_estimator_ 속성에 저장되어 있음

dt = gs.best_estimator_
print(dt.score(train_input,train_target))
# best_params_ 속성에 최적 매개변수 저장되어 있음
# cv_results_ 속성의 mea_test_score에 교차 검증의 평균 점수 저장되어 있음

print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])

# 수동보다 넘파이 argmax()함수를 이용해서 큰 값의 인덱스 알 수 있음
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

# 순서1: 먼저 탐색할 매개변수 지정
# 순서2: 흔련 세트 그리드 서치 수행하여 최상의 평균 검중 점수 찾음
# 순서3: 찾은 매개변수를 이용해 전체 훈련 세트 사용해 최종 모델 훈련

# min_impurity_decrease는 노드를 분할하기 위한 불순도 감소 최소량을 지정함.

params = {'min_impurity_decrease':np.arange(0.0001,0.001,0.0001),
          'max_depth': range(5,20,1),
          'min_samples_split':range(2,100,10)
          }

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
# print(gs.best_params_)
gs.fit(train_input,train_target)

print(gs.best_params_)

# 랜덤 서치: 매개변수 값이 수치일때 범위나 간격을 정하기 어려울때
# 매개변수를 샘플링할 수 있는 확률 분포 객체 전달
# 싸이파이(scipy): 적분, 보간, 선형대수, 확률 등 수치 계산 전용

# uniform,randint는 주어진 범위에서 고르게 값을 뽑음, uniform은 실수, randint는 정수
from scipy.stats import uniform, randint

rgen = randint(0,10)
rgen.rvs(10)

# 이걸 사용해서 총 몇번의 샘플링을 통해 최적의 매개변수 찾아달라고 하면 됨
# min_sample_leaf 추가: 리프 노드가 되기 위한 최소 샘플 개수: 어떤 노드가 분할 할때 이 값보다 작으면 분할 하지 않음

params = {'min_impurity_decrease':uniform(0.0001,0.001),
          'max_depth':randint(20,50),
          'min_samples_split':randint(2,25),
          'min_samples_leaf':randint(1,25),}

# 샘플링 횟수는 램던 서치 클래스 RandomizedSearchCV n_iter 매개변수에 지정

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),params,n_iter=100,n_jobs=-1,random_state=42)
gs.fit(train_input,train_target)
# 100만 반복함으로 인해서 저번보다(1350개) 더 적은수로 넓은 범위 커버 가능
print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))
dt = gs.best_estimator_
print(dt.score(test_input,test_target))
print(dt.score(train_input,train_target))