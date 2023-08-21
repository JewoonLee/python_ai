from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input,train_target), (test_input,test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled,train_target,val_target = train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

# 함수를 사용해 다른 거를 넣을 수 있음
def model_fn(a_layer = None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(100,activation='relu'))
    if a_layer:
        model.add(a_layer)

    model.add(keras.layers.Dense(10,activation='softmax'))
    return model

model = model_fn()
model.summary()
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')

# .fit 함수는 무언가를 return 함
history = model.fit(train_scaled,train_target,epochs=5,verbose=0)   # verbose는 훈련과정을 나타내는 매개변수 0=> 나타내지 않기, 1=> 나타내기, 2=> 막대기 빼고 나타내기

# history 객ㄱ체에는 history 딕셔너리 들어잇음
print(history.history.keys())

# 손실 그림으로 표현
import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()

# # 정확도 그림으로 표현
# plt.plot(history.history['accuracy'])
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()

# 에포크 횟수 늘려서 모델 손실 그래프 그려보기
# model = model_fn()
# model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
# history = model.fit(train_scaled,train_target,epochs=20,verbose=0)
# plt.plot(history.history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()

# 인공 신경망은 정확도보다는 손실을 최소화 함
# 검증 손실 확인해보기

# model = model_fn()
# model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
# history = model.fit(train_scaled,train_target,epochs=20,verbose=0, validation_data=(val_scaled,val_target))

# 검증 세트에 대한 손실은 val_loss에 들어 있고, 정확도는 val_accuray에 들어있음

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()

# 이러면 과대적합이 생김, 이걸 막기 위해 하이퍼파라미터 조정
# 기본으로 RMSprop 사용
# Adam도 좋은 선택

# model = model_fn()
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
# history = model.fit(train_scaled,train_target,epochs=20,verbose=0, validation_data=(val_scaled,val_target))

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()

# 과대적합이 많이 없어짐
# 더 나은 곡선을 얻으려면 학습률 조정하기

# 드롭아웃
# 훈련과정에서 층에 있는 일부 뉴런을 랜덤하게 꺼서 과대적합 막기
# 얼마나 많으 뉴련 드랍 할지는 우리가 정해야함
# dropout은 층에 추가해야함, 하나의 함수이기 때문에

# model = model_fn(keras.layers.Dropout(0.3)) # 30% drop
# model.summary()
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
# history = model.fit(train_scaled,train_target,epochs=20,verbose=0, validation_data=(val_scaled,val_target))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'val'])
# plt.show()


# 모델 에포크 10으로 바꾸고 저장하기
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
history = model.fit(train_scaled,train_target,epochs=10,verbose=0,validation_data=(val_scaled,val_target))

# save_weights()로 저장하기
# 파일 확장자는 .h5
model.save_weights('model-weights.h5')

# 모델 구조와 모델 파라미터 함께 저장하는 save 메서드도 있음
model.save('model-whole.h5')

# 이거 사용해서 확률 구해보기
# predict함수는 10 개 클래스마다 확률을 구해줌

model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5')

import numpy as np
val_labels = np.argmax(model.predict(val_scaled),axis = -1) # axis = -1은 마지막열을 따라 간다고 생각하면 됨
print(np.mean(val_labels == val_target))       # target값이랑 예언하거랑 비교

model = keras.models.load_model('model-whole.h5')       # 다 있는거 넣기
model.evaluate(val_scaled,val_target)
# 위랑 아래랑 같은 값을 넣었기 때문에 동일한 확률이 나옴
# 하지만 여기서 20번의 에포크동안 검증 점수 상승하는 지점을 확인했음=> 그 다음 그 에포크만큼 다시 훈련함
# 모델을 두 번씩 훈련하지 않고 한번에 끝내는 법: 콜벡

# 콜백: 훈련 과정 중간에 어떤 작ㅂ을 수행할 수 있게 하는 것
# modelCheckpoint는 최상의 검증 점수를 만드는 모델 저장

# model = model_fn(keras.layers.Dropout(0.3))
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
# model.fit(train_scaled, train_target,epochs=20,verbose=0,validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb])

# # 이제 최상ㅇ의 검증 점수를 낸 모델로 fit 하기
# model = keras.models.load_model('best-model.h5')
# model.evaluate(val_scaled,val_target)

# 하지만 위의 경우도 20까지 모델을 훈련함
# 이제는 과대적합이 시작되기전 훈련을 미리 종료하는 조기 종료 사용하기
# Earlystopping
# ModelCheckpoint랑 같이 사용하기

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)     #patience는 얼마나 성능 향상 없이도 기다릴 것인지  
history = model.fit(train_scaled,train_target,epochs=20,verbose=0,validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

# 몇번 epochs에서 멈췄는지 early_stopping_cb 의 stopped_epoch 속성에서 알 수 있음
print(early_stopping_cb.stopped_epoch)

# 13임으로 13-2 =11 즉 12번째에서 제일 좋음

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()


# 이렇게 하면 컴퓨터 시간과 자원을 아낄 수 있고, 자동으로 제일 최상의 모델을 저장해 주기 때문에 편리함

model.evaluate(val_scaled,val_target)
