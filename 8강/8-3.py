from tensorflow import keras
model = keras.models.load_model('best-cnn-model.h5')

# print(model.layers)

# 첫 번째 합성곱 층의 가중치 조사
conv = model.layers[0]

# 커널 크기 (3,3), 필터 개수 (32) => (3,3,1,32), 필터마다 1개 절편 =>(32,)
print(conv.weights[0].shape, conv.weights[1].shape)

# 가중치 평균, 표준편차 구해보기
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())

# 가중치 그려보기
import matplotlib.pyplot as plt
# plt.hist(conv_weights.reshape(-1,1))
# plt.xlabel('weight')
# plt.ylabel('count')
# plt.show()

# 32개의 커널을 16개씩 두 줄에 출력 해보기
# fig, axs = plt.subplots(2,16,figsize=(15,2))
# for i in range(2):
#     for j in range(16):
#         axs[i,j].imshow(conv_weights[:,:,0,i*16+j], vmin=-0.5,vmax =0.5)#    vmin, vmax로 범위를 지정함
#         axs[i,j].axis('off')
# plt.show()

# 훈련하지 않은 모델과 훈련 한 데이터를 비교하기
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32,kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))

# 이 모델의 첫 번째 층의 가중치를 no_training_conv변수에 저장
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)

# 이 가중치의 평균과 표준편차 구해보기
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(),no_training_weights.std())

# 이 가중치를 히스토그램으로 표현해 보기
# plt.hist(no_training_weights.reshape(-1,1))
# plt.xlabel('weight')
# plt.ylabel('count')
# plt.show()

# 가중치가 균등하게 모여있음, 텐서플로가 신경망의 가중치를 처음 초기화할 때 균등 분포에서 랜덤하게 값을 선택함
# 이 가중치 값도 그림으로 출력해 보기

fig, axs = plt.subplots(2,16,figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i,j].imshow(conv_weights[:,:,0,i*16+j], vmin=-0.5,vmax =0.5)#    vmin, vmax로 범위를 지정함
        axs[i,j].axis('off')
plt.show()

# 합성곱 신경망의 학습 시각화 두 번째 방법: 특성 맵을 그려보는 것

# 합수형 API
# dense1 = keras.layers.Dense(100,activation='sigmoid')
# dense2 = keras.layers.Dense(10,activation='softmax')

# 앞 7장에서는 add 메서드에 전달 가능
# 하지만 다르게 표현

# hidden = dense1(inputs)
# outputs = dense2(hidden)

# 그다음 inputs와 outputs를 model 클래스로 연결
# model = keras.Model(inputs,outputs)

# InputLayer 클래스 객체 쉽게 만들 수 있도록 함
# inputs = keras.Input(shape=(784,))
# 모델 input과 model.layers[0].output만 알면 새로운 conv_acti 모델 만들 수 있음

print(model.input)
conv_acti = keras.Model(model.input,model.layers[0].output)
# 이렇게 특성 맵 만듦

# 특성 맵 시각화
(train_input, train_target), (test_input,test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

# 이 앵글 부츠를 conv_acti모델에 넣어서 어떤 특성 맵이 나오는지 보기
inputs = train_input[0:1].reshape(-1,28,28,1) / 255.0
feature_maps = conv_acti.predict(inputs)
print(feature_maps.shape)

# 필터를 사용한 합성곱 층의 특성맵 그려보기
fig, axs = plt.subplots(4,8,figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
        axs[i,j].axis('off')
plt.show()

# 두 번째 합성곱 츠이 만든 특성맵도 보기
conv2_acti = keras.Model(model.input,model.layers[2].output)
inputs = train_input[0:1].reshape(-1,28,28,1) / 255.0
feature_maps = conv2_acti.predict(inputs)

# 크기 (1,14,14,64)
print(feature_maps.shape)

fig, axs = plt.subplots(8,8,figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
        axs[i,j].axis('off')
plt.show()

# 합성곱 신경망의 앞부분에 있는 합성곱 층은 이미지의 시작적인 정보를 감지하고 뒤는 앞에서 갑지한 시각적인 정보를 통해 추상적인 ㅈ어보를 학습




