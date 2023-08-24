# IMDB 리뷰 데이터셋
# 텍스트 경우 단어를 고유한 정를 부여함
# 어휘 사전: 훈련 세트에서 고유한 단어를 뽑아 만든 목록 

from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input,test_target) = imdb.load_data(num_words=500)

print(train_input.shape)
# 개별 리뷰를 담은 파이썬 리스트
# 여기에는 25000개의 리뷰 => 개별 리뷰에는 정수로 변환된 리뷰 단어들

# 0은 부정, 1은 긍정
print(train_target[:20])

# 훈련 세트 떼어놓기
from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input,train_target,test_size=0.2, random_state=42)

# 리뷰 길이 재기
import numpy as np
lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths))

# 그림으로 표현하기
import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

# 우리는 100개의 단어만 사용할 것이다
# 100개보다 짧은 리뷰들은 100개의 길이를 맞추기 위해 패딩을 사용 => 0으로 표시
# pad_sequences()

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input,maxlen=100)
# 이 경우 100보다 긴 경우는 잘라내고 짧은 경우 0으로 패딩 함

print(train_seq.shape)
print(train_seq[0])     
# 0이 없는걸로 보아 100보다 더 길었음 
# 샘플 뒷부분 100이 남아있음
# 셈플 앞부분을 보고 싶으면 pad_sequense()함수의 truncating 값을 post로 바꾸면 됨

print(train_seq[5])
# 0이 있는걸로 보아 샘플 길이가 100이 안댔음

# 검증 길이도 100으로 맞추기
val_seq = pad_sequences(val_input, maxlen=100)

# 순환 신경망 만들기
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8,input_shape=(100,500)))
model.add(keras.layers.Dense(1,activation='sigmoid'))

# 여기서 문제점이 큰 정수는 큰 활성화 출력을 만듦
# 하지만 크기는 관련이 없음
# 따라서 원-핫 인코딩을 하여 크기를 상관 없게 만듦
# 따라서 input_shape이 100개에서 500의 dic안에 있는거 중에 1 사용하는거
# ketas.utils 패키지 안에 to_categprocial()함수가 원-핫 인코딩 만들어줌

train_oh = keras.utils.to_categorical(train_seq)

# 잘 됐는지 확인하기
print(np.sum(train_oh[0][0]))

# val도 똑같이 바꾸기
val_oh = keras.utils.to_categorical(val_seq)

model.summary()

# 순환 신경망 훈련하기
# rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
# model.compile(optimizer=rmsprop,loss = 'binary_crossentropy',metrics=['accuracy'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5')
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# history=model.fit(train_oh,train_target,epochs=100,batch_size=64,validation_data=(val_oh,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

# # 여기서 문제점이 원-핫 인코딩은 1개의 토큰을 500차원으로 늘렸기 때문에 데이터가 많아짐

# 단어 임베딩 사용하기
# 임베딩: 각 단어를 고정된 크기의 실수 벡터로 바꾸어 줌
# 단어 임베딩 장점은 입력으로 정수 데이터를 받음
# 임베딩도 원-핫 처럼 늘리긴 하지만 훨씬 적은 길이로 늘림
# 임베딩으로 두번째 순환 신경망 만들어보기

model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500,16,input_length=100))     # 첫번째 매개변수는 어휘 사전 크기, 두번째는 임베딩 크기 16, 3번째 원학 인코딩 크기
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1,activation='sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop,loss = 'binary_crossentropy',metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history=model.fit(train_oh,train_target,epochs=100,batch_size=64,validation_data=(val_oh,val_target),callbacks=[checkpoint_cb,early_stopping_cb])


