# 주성분 분석
# 차원 축소에 대해 이해하고 대표적인 차원 축소 알고리즘 중 하나인 PCA(주성분 분석) 알아보기

# 차원과 차원 축소
# 데이터가 가진 속석을 특성이라 부름
# 과일 사진인 경우 10,000개의 픽셀이 있어서 10,000개의 특성이 있음=> 이런 특성을 차원(dimension) 이라고 부름

# 차원 축소는 데이터를 잘 나타내는 일부 특성을 선택하고 나머지는 버리는 것

# PCA(주성분 분석)
# 분산이 큰 방향을 찾는 것 => 가장 많이 퍼져 있는 정도가 큰 곳
# 주성분은 원본 특성 개수만큼 찾을 수 있음, 주성분으로 바꾼 데이터는 차원이 줄어듬
# 첫 번째 주성분을 찾고, 그 벡터에 수직이고 분산이 가장 큰 다음 방향을 가진 벡터가 두 번째 주성분임.

import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1,100*100)

# pca 클래스를 만들때 n_components 매개변수에 주성분의 개수 지정
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)

# 배열 크기 확인
print(pca.components_.shape)

import matplotlib.pyplot as plt
def draw_fruits(arr, ratio = 1):
    n = len(arr)    # n은 샘플 개수
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산함
    rows = int(np.ceil(n/10))
    # 행이 1개이면 열의개수는 샘플 개수. 그렇지 않으면 10개
    cols = n if rows < 2 else 10
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * ratio, rows*ratio), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n개 까지만 그림
                axs[i,j].imshow(arr[i*10 + j],cmap = 'gray_r')
            axs[i,j].axis('off')
    plt.show()

# 추출한거 그림으로 그려보기
# draw_fruits(pca.components_.reshape(-1,100,100))


# 차원 이제 줄이기
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# 10000개에서 50개로 줄임 => 1/200배 줄임

# 원본 데이터 재구성
# inverse_transform()

fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

fruits_reconstruct = fruits_inverse.reshape(-1,100,100)
# for start in [0,100,200]:
#     draw_fruits(fruits_reconstruct[start:start+100])
#     print("\n")


# 설명된 분산
# 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값
# explained_variance_ratio
# 50개의 분산 비율을 다 더하면 총 분산 비율을 얻을 수 있음

print(np.sum(pca.explained_variance_ratio_))

# 그래프 그리면 주성분 개수 찾는 데 도움이 됨

# plt.plot(pca.explained_variance_ratio_)
# plt.show()
# 1~10까지가 대부분의 분산을 표현하고 있음

# 다른 알고리즘과 함께 사용하기
# 먼저 로지스틱으로 분류하기
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

target = np.array([0]*100 + [1]*100 + [2]*100)
# 원본 데이터 사용해서 교차 검증

from sklearn.model_selection import cross_validate
# scores = cross_validate(lr,fruits_2d,target)
# print(np.mean(scores['test_score']))
# print(np.mean(scores['fit_time']))

# pca로 축소한 것과 비교
# score = cross_validate(lr,fruits_pca,target)
# print(np.mean(score['test_score']))
# print(np.mean(score['fit_time']))

# 앞에서 pca 클래스에 n_components에 매개변수 주성분 개수를 지정함
# 이제는 설명된 분산의 비율 입력 가능
# 50% 찾아보기

pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

# 몇 개의 주성분 찾았는지 확인
print(pca.n_components_)
# 2개면 충분

# 2개 주성분으로 교차 검증 해보기
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
scores = cross_validate(lr,fruits_pca,target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 2개 주성분으로 k-평균 알고리즘 클러스터 찾아보기
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_,return_counts=True))

for label in range(0,3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")


# 훈련 데이터의 차원을 줄이면 또 하나의 장점은 시각화
# 3개 이하로 줄이면 화멸 출력 가능

for label in range(0,3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0],data[:,1])
plt.legend(['pineapple','banana','apple'])
plt.show()

# 장점: 시간 절약, 시각화 쉬움


