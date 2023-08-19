# 비지도 학습: 사람이 가르쳐 주지 않아도 데이터 학습하는 것

import requests

# url = 'https://bit.ly/fruits_300_data'
# response = requests.get(url)
# with open('fruits_300.npy', 'wb') as file:
#     file.write(response.content)

import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')
print(fruits.shape)
# 이미지 크기 100 x 100짜리 300장

# 첫번째 이미지 첫 번째 행 출력
print(fruits[0,0,:])
# 0~255까지의 정수값 가짐
# 이미지로 그려보기

# plt.imshow(fruits[0],cmap='gray')
# plt.show()

# 컴퓨터는 255에 가까운 값에 집중을 함 따라서 사과를 밝게 함으로 사과에 집중하게 함

# cmap = 'gray_r'로 만들면 다시 반전돼서 사과가 짙게 나옴
# plt.imshow(fruits[0],cmap='gray_r')
# plt.show()

# 바나나 파인애플 이미지도 출력
# fig, axs = plt.subplots(1,2)        # 배열처럼 쌓을 수 있도록 도와줌 a,b = 행, 열 / axs는 2개의 서브 그래프 담고있는 배열
# axs[0].imshow(fruits[100],cmap = 'gray_r')
# axs[1].imshow(fruits[200],cmap = 'gray_r')
# plt.show()

# 픽셀값 분석하기
# 100*100 이미지를 10,000 인 1차원 배열로 만들기

apple = fruits[0:100].reshape(-1,100*100)
pineapple = fruits[100:200].reshape(-1,100*100)
banana = fruits[200:300].reshape(-1,100*100)

# 샘플의 픽셀 평균값 구해보기
# axis = 0 이면 첫번째 축인 행, axis = 1이면 열을 따라 계산
# 우리는 axis = 1을 따라 계산 해야함

print(apple.mean(axis = 1))

# 이 값으로 히스토그램 그려보기
# alpha값을 줄이면 투명도가 줄어듬

# x 축은 평균 값, y 축은 들어있는 개수
plt.hist(np.mean(apple, axis =1),alpha = 0.8)
plt.hist(np.mean(pineapple, axis =1),alpha = 0.8)
plt.hist(np.mean(banana, axis =1),alpha = 0.8)
plt.legend(['apple','pineapple','banana'])
plt.show()

# 바나나는 구별되는데 사과랑 파인에플은 구별이 안됨
# 이제는 샘플 평균값이 아니라 픽셀의 평균값을 구해보자
# axis = 0으로 하면 됨

# fig, axs = plt.subplots(1,3, figsize=(20,5))
# axs[0].bar(range(10000), np.mean(apple, axis = 0))
# axs[1].bar(range(10000), np.mean(pineapple, axis = 0))
# axs[2].bar(range(10000), np.mean(banana, axis = 0))
# plt.show()

# 100*100 픽셀 크기로 바꿔서 그래프와 비교

apple_mean = np.mean(apple,axis=0).reshape(100,100)
pineapple_mean = np.mean(pineapple,axis = 0).reshape(100,100)
banana_mean = np.mean(banana,axis = 0).reshape(100,100)
fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].imshow(apple_mean,cmap='gray_r')
axs[1].imshow(pineapple_mean,cmap='gray_r')
axs[2].imshow(banana_mean,cmap='gray_r')
plt.show()


# 평균값과 가까운 사진 고르기
# 3장에서 봤던 절댓값 오차 사용

abs_diff = np.abs(fruits - apple_mean)      # 과일 전체에 사과 평균값 빼기
abs_mean = np.mean(abs_diff,axis=(1,2))     # 그 오차 1,2 차 평균 내기
print(abs_mean.shape)                       


# 값이 가장 작은 순서댈 100개 골라 보기
apple_index = np.argsort(abs_mean)[:100]         # argsort 는 작은 것에서 큰 순서대로 나열
fig, axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i*10 + j]], cmap = 'gray_r')
        axs[i,j].axis('off')    # 좌표축 그리지 않기
plt.show()

# 이렇게 비슷한 샘플끼리 모으는 것 군집
# 하지만 지금은 이미 타겟을 알고 있었던 상황
# 비지도 학습은 원래 타깃값을 모름


