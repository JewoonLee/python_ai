# k 평균
# 평균값이 클러스터의 중심에 위치 => 클러스터 중심, 센트로이드
# k-평균 알고리즘 소개
# 1. 무작위로 k개의 클러스터 중심을 정함
# 2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
# 3. 클러스터에 속함 샘플의 평균값으로 클러스터 중심을 변경
# 4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복함

# KMeans 클래스
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1,100*100)

# 3개의 군집으로 묶기
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)

print(np.unique(km.labels_, return_counts = True))

# 각 클러스터가 어떤 이미지를 나타내는지 확인하기

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

# draw_fruits를 이용해 레이블 0인 과일 사진 그리기
# Km.labels_==0 하면 됨 => 불리언 인덱싱

# draw_fruits(fruits[km.labels_ == 0])

# draw_fruits(fruits[km.labels_ == 1])
# draw_fruits(fruits[km.labels_ == 2])

# 클러스터 중심
# KMeans가 찾은 클러스터 중심은 cluster_centers_ 속성에 저장되어 있ㅇ,ㅁ

draw_fruits(km.cluster_centers_.reshape(-1,100,100),ratio=3)

# KMeans 클레스는 훈련 데이터 샘플에서 클러스터 중심까지 거리 변환해주는 transform() 메서드 가지고 있음
# 2차원 배열을 넣어줘야함
print(km.transform(fruits_2d[100:101]))

# predict() 함수를 이용해 가장 가까운 레이블 찾기 가능
print(km.predict(fruits_2d[100:101]))

# 확인 해보기
draw_fruits(fruits[100:101])

# K-평균 알고리즘은 반복적으로 클러스터 중심을 옮기기 때문에 몇번 반복 했는지 알 수 있음=> n_iter_
print(km.n_iter_)

# 우리가 k=3으로 정한 것도 알고 있어서 넣음, 아예 모를때는?
# 최적의 K 찾기

# 1. 엘보우 방법
# 거리의 제곱 합을 이너셔라고 부름 => 클러스터 개수가 늘어나면 클러스터 개개의 크기는 줄어듦 => 이니셔도 줄어듦
# 이걸로 적절한 클러스터 개수를 찾음
# 에너셔 감소하는 속도가 꺾이는 지점이 있음=> 엘보우
# inertia 속성에 저장되어 있음

inertia = []
for k in range(2,7):
    km = KMeans(n_clusters=k,random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)
plt.plot(range(2,7),inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()



