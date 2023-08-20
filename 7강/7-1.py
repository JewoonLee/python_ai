# 그림 데이터 받아오기 (패션 MNIST)
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# print(train_input.shape, train_target.shape)

# 그림 그려보기
import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1,10,figsize=(10,10))
# for i in range(10):
#     axs[i].imshow(train_input[i],cmap='gray_r')
#     axs[i].axis('off')
# plt.show()

print([train_target[i] for i in range(10)])
# 0~1 값으로 정규화
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1,28*28)

# 교차 검증으로 성능 확인
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss = 'log_loss',max_iter=5,random_state=42)
scores = cross_validate(sc,train_scaled,train_target,n_jobs=-1)
print(np.mean(scores['test_score']))

# 인공 신경망
# 딥러닝은 인공 신경망과 거의 동의어로 사용
# 맨 마지막 층=> 출력층
# z값 계산하는 단위 => 뉴런 또는 유닛
# 입력층

import tensorflow as tf
# 딥러닝 라이브러리가 다른 머신러닝 다른 라이브러리랑 다른 점 중 하나는 GPU를 사용한다는 점

# 인공 신경망은 교차 검증을 잘 사용하지 않고 검증 세트를 별도로 덜어서 사용함
# 이유 1. 딥러닝 분야 데이터셋은 충분이 크기 때문에
# 이유 2. 교차 검증 수행 하기에 시간이 너무 오래 걸림

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

# 케라스 레이어 패키지 안에는 다양한 층이 있음
# 가장 기본: 밀집층=> 입력층 하나당 모두 출력층으로 가고 있기 때문에
# 완전 연결층이라고도 부름
# 필요한 매개변수: 뉴런 개수, 뉴런 출력에 적용할 함수, 입력 크기

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
# 이제 밀집층을 가진 신경망 모델 만들기
model = keras.Sequential(dense)
# 뉴런의 선형 방정식 계산 결과에 적용되는 함수 => 활성화 함수

# 인공 신경망으로 패션 아이템 분류하기
# 케라스 모델은 훈련하기 전 설정 단계가 있음
# cmompile() 메서드에서 수행하고, 손실 함수의 종류를 꼭 골라줘야함

model.compile(loss = 'sparse_categorical_crossentropy',metrics='accuracy')

# 이진 분류: loss = binary_crossentropy
# 다중 분류: loss = categorical_crossentropy

# 이진 분류는 1, 1-a
# 다중 분류는 1, 0, 0, 0, 0, 0, ...
# 이런 걸 원-핫 인코딩이라 함
# 텐서플로에서는 정수로 된 타깃 값을 원-핫 인코딩으로 바꾸지 않고 사용 가능 => 이게 sparse_categorical_crossentropy

# metrics='accuracy'
# 케라스는 에포크마다 손실 값을 출력 해줌, 이게 accuracy를 통해 알ㄹ려줌

model.fit(train_scaled,train_target,epochs=5)

# 이제 검증 세트로 확인 evaluate()

model.evaluate(val_scaled,val_target)


