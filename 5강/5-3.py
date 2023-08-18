# 정형 데이터, 비정형 데이터
# 정형 데이터: 엑셀, csv, 데이터베이스 처럼 구조로 되어 있는 것
# 비정형 데이터: 텍스트 데이테, 사진, 디지털 음악 등등
# 정형 데이터를 다루는데 가장 좋은 알고리즘: 앙상블 학습

# 랜덤 포레스틔: 결정 트리를 랜덤하게 만들어 결정 트리의 숲을 만듦
# 우리가 입력한 훈련 데이터에서 랜덤하게 샘플을 추출하여 훈련 데이터를 만듦, 샘플 중복 가능
# 예를 들어 1000개 가방에 100개를 뽑는다면 1개 뽑고 다시 가방에 넣음: 부트스트랩 샘플
# 또한 노드를 분할할때 전체 특성 중에서 일부 특성을 무작위로 고르고 최선의 분할을 함
# 전체 특성 개수의 제곱근만큼 선택

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol','sugar','pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data,target,test_size=0.2,random_state=42)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1,random_state=42)
scores = cross_validate(rf,train_input,train_target,return_train_score=True, n_jobs=-1)
# print(scores)

rf.fit(train_input,train_target)
print(rf.feature_importances_)
# 한 특성에 몰빵하지 않고 랜덤하게 고름으로서 특성의 중요도를 골고루 퍼지게 함
# 또 재밌는 기능으로는 훈련에 중복을 허용하기 때문에 부트스트랩 샘플에 포함되지 않은 샘플들로 훈련 검증 가능
# oob_score

rf = RandomForestClassifier(oob_score=True,n_jobs=-1,random_state=42)
rf.fit(train_input,train_target)
print(rf.oob_score_)

# 엑스트라 트리
# 랜덤 포레스트와 차이점은 부트스트랩 샘플을 사용하지 않는다, 전체 훈련 세트를 사용 but 노드를 분할할 때 가장 좋은 분할이 아닌 무작위로 분할 함

from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1,random_state=42)
scores = cross_validate(et,train_input,train_target,return_train_score=True, n_jobs=-1)

# 엑스트라 트리는 무작위성이 크기 때문에 랜덤 포레스트보다 더 많은 결정 트리를 훈련해야 하지만 랜덤 노드 분할이기 때문에 빠른 속도 계산 가능
# 원래 결정 트리는 최적의 분할을 찾는데 시간이 더 오래 걸림

# 엑스트라 트리도 특성 중요도 제공함
et.fit(train_input,train_target)
print(et.feature_importances_)


# 그레이디언트 부스팅
# 깊이가 얕은 결정 트리를 사용하기 이전 트리의 오차를 보완하는 방식
# 깊이가 3인 결정트리 100개를 사용하는 방법=> 깊이가 얕음으로 과대적합에 강하고, 일반화 가능

from sklearn.ensemble import GradientBoostingClassifier
# gb = GradientBoostingClassifier(random_state=42)
# scores = cross_validate(gb,train_input,train_target,return_train_score=True, n_jobs=-1)
# print(np.mean(scores['train_score']),np.mean(scores['test_score']))


# 과대적합에 강함
# 학습률을 증가시키고, 트리의 개수를 늘리면 성능 향상 가능
gb = GradientBoostingClassifier(n_estimators=500,learning_rate=0.2,random_state=42)
scores = cross_validate(gb,train_input,train_target,return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))

# 특성을 랜덤하게 고르지 않기때문에 한 특성에 더 집중함
gb.fit(train_input,train_target)
print(gb.feature_importances_)

# subsample: 트리 훈련에 사용할 훈련 세트의 비율. 1.0이 기본이지만 1보다 작으면 훈련 세트 일부를 사용함=> 경사 하강법 , 미니배치 경사하강법이랑 비슷함
# 그레이디언트 가 랜덤보다 빠름 하지만 순서대로 트리를 추가하기 때문에 훈련 속도가 느림
# 즉 n_jobs가 업음
# 속도, 성능 둘다 잡은것이 히스토그램 기반 그레이디언트 부스팅

# 히스토그램 기반 그레이디언트 부스팅
# 입력 특성을 256 구간으로 나눔=> 노드 분할할때 최적의 분할 매우 빠르게 찾음
# 256개의 구간 중에서 하나를 떼어 놓고 누락된 값을 위해 사용함

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb,train_input,train_target,return_train_score=True)

from sklearn.inspection import permutation_importance
hgb.fit(train_input,train_target)
result = permutation_importance(hgb,train_input,train_target,n_repeats=10,random_state=42,n_jobs=-1)

