# 합성곱은 무조건 깊이가 있어야 함

from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input,test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1,28,28,1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

# 합성곱 신경망 만들기
model = keras.Sequential()
model.add(keras.layers.Conv2D(32,kernel_size=3, activation='relu',padding='same',input_shape=(28,28,1)))    # 합성곱 층 32개, (3,3) 행렬, 렐루, 세임 패딩

# 그다음 폴링 추가
model.add(keras.layers.MaxPooling2D(2))

# 합성곱 층에서 32개 필터 사용함으로 28,28=> (14,14,32) 

# 두 번째 합성곱-폴링 층 추가
# 필터 개수 64로 늘리기
model.add(keras.layers.Conv2D(64,kernel_size=3,activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(2))

# 이러면 이제 특성 맵 크기는 (7,7,64)가 될 것이다
# 이제 이 3차원  특성 맵을 일렬로 펼치자, 이유는 마지막 10개의 뉴런을 가진 출력층에서 확률 계산하기 때문에
# 중간에 밀집 은닉층 하나 두기

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()

# 케라스는 summary 외에도 층의 구성을 그림으로 표현해 주는 plot_model()이 있음
# keras.utils.plot_model(model)

# 모델 컴파일과 훈련

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
history = model.fit(train_scaled,train_target, epochs=20, validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

# 손실 그래프 그려보기
import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train','val'])
# plt.show()

# predict 메서드 사용해서 제대로 훈련 했는지 확인하기

plt.imshow(val_scaled[0].reshape(28,28),cmap = 'gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)            # 10개 클래스에 대한 확률 보여줌

# 표로도 보여주기
plt.bar(range(1,11),preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

# 이제 마지막으로 테스트 세트로 합성곱 신경망의 일반화 성능 가늠해 보기
# 이렇게 테스트 세트는 무조건 마지막에 한번 검사 해야함

test_scaled = test_input.reshape(-1,28,28,1)/255.0
model.evaluate(test_scaled,test_target)


