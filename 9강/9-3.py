# LSTM과 GRU

# LSTM: Long Short-Term Memory
# 셀 상태: 다음 층으로 전달되지 않고 LSTM 셀에서 순환만 됨
# 은닉 상태는 시그모이드 함수 , 셀 상태는 tanh 함수
# 그 다음 두 결과를 곱하고 이전 셀 상태와 더함
# 삭제 게이트: 셀 상태에 있는 정보를 제거
# 입력 게이트: 새로운 정보를 셀 상태에 추가
# 출력: 셀 상태가 다음 은닉 상태로 출력

# LSTM 신경망 훈련하기
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input,train_target,test_size=0.2,random_state=42)

# 패딩 추가
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input,maxlen =100)
val_seq = pad_sequences(val_input,maxlen=100)

# LSTM 만들어보기
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(500,16,input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.summary()

# rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
# model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
# history = model.fit(train_seq,train_target,epochs=100,batch_size=64,validation_data=(val_seq,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

# 그림 그려보기
import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train','val'])
# plt.show()

# 순환층에도 드롭아웃 적용해보기
# model2 = keras.Sequential()
# model2.add(keras.layers.Embedding(500,16,input_length=100))
# model2.add(keras.layers.LSTM(8,dropout=0.3))
# model2.add(keras.layers.Dense(1,activation='sigmoid'))

# rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
# model2.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
# history = model2.fit(train_seq,train_target,epochs=100,batch_size=64,validation_data=(val_seq,val_target),callbacks=[checkpoint_cb,early_stopping_cb])


# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train','val'])
# plt.show()


# 2개 층 연결해 보기
# 2개 연결할때는 첫번째는 모든 타임랩스에 대한 은닉 상태를 출력 해야함
# return_sequences = True라고 하면 모든 타임스텝 상태 출력

# model3 = keras.Sequential()
# model3.add(keras.layers.Embedding(500,16,input_length=100))
# model3.add(keras.layers.LSTM(8,dropout=0.3,return_sequences=True))
# model3.add(keras.layers.LSTM(8,dropout=0.3))
# model3.add(keras.layers.Dense(1,activation='sigmoid'))

# rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
# model3.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
# history = model3.fit(train_seq,train_target,epochs=100,batch_size=64,validation_data=(val_seq,val_target),callbacks=[checkpoint_cb,early_stopping_cb])


import matplotlib.pyplot as plt
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train','val'])
# plt.show()


# GRU(Gated Recurrent Unit) 구조
# LSTM 간소화 버젼: 셀 상태를 계산하지 않음
# 은닉 상태와 입력에 가중치를 곱하고 절편을 더하는 셀 3개
# 2개는 시그모이드, 하나는 tanh함수

# Wz 는 삭제게이트 역할
# GRU 신경망 훈련하기
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500,16,input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1,activation='sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model4.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model4.fit(train_seq,train_target,epochs=100,batch_size=64,validation_data=(val_seq,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()