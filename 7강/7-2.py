from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1,28*28)
# train_scaled, val_scaled , train_target, val_target = train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

# 이제 인공 신경망 모델에 층 2개 추가해 보기
# 784 => 100 => 10
# 입력층과 출력층 사이에 밀집층 하나 추가 => 사이에 있는 모든 층을 은닉층이라고 부름
# 활성화 함수 => z값을 변환하는 함수
# 은닉층에서는 산술 계산만 한다면 많은 가중치 변수가 필요 없음
# 선형 계산을 적당하게 비선형적으로 틀어줘야함
# 그래야 다음층 계산이랑 단순히 합쳐지지 않고 나름의 역할 함

# 가장 많이 사용하는 활성화 함수는 시그모이드 함수

# dense1 = keras.layers.Dense(100,activation='sigmoid', input_shape = (784,))
# dense2 = keras.layers.Dense(10,activation='softmax')

# # 심층 신경망 만들기
# model = keras.Sequential([dense1,dense2])

# # summary() 이용하면 유용한 정보 얻음
# model.summary()

# # 층을 추가하는 다른 방법
# # sequential 클래스 생성자 안에 바로 dense 클래스 만들기

# model = keras.Sequential([keras.layers.Dense(100,activation='sigmoid', input_shape = (784,), name = 'hidden'),
#                           keras.layers.Dense(10,activation='softmax', name = 'output')], name = '패션 모델' )

# model.summary()


# 이렇게 하면 sequential 생성자가 매우 길어짐으로 add() 사용

model = keras.Sequential()
model.add(keras.layers.Dense(100,activation='sigmoid', input_shape = (784,)))
model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()

# model.compile(loss = 'sparse_categorical_crossentropy',metrics='accuracy')
# model.fit(train_scaled,train_target,epochs=5)

# 성능이 좋아짐
# 이제는 이미지 분류 문제에 높은 성능을 낼 수 있는 활성화 함수 알아보기
# 렐루 함수

# 시그모이드 함수는 오른쪽과 왼쪽 끝으로 갈수록 그래프가 누워있어서 올바른 출력을 만드는데 신속하게 대응 못함
# 렐루 함수는 양수일 경우 그냥 입력을 통과시킴, 음수일 경우 0
# z가 0 보다 크면 z 출력, 아니면 0 max(0,z)느낌
# flatten 클래스는 배치 차원을 제외하고 나머지 입력 차원을 모드 일렬로 펼치는 역할

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28,28)))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

# model.summary()

train_scaled = train_input / 255.0
train_scaled, val_scaled , train_target, val_target = train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(train_scaled,train_target,epochs=5)

# 평가 하기
model.evaluate(val_scaled,val_target)

# 인공 신경망의 하이퍼파라미터
# 은닉층
# 은닉층 뉴런 개수
# 활성화 함수
# 층 종류 (밀집층 등등)
# 미니배치 경사 하강법=> 미니배치 개수
# fit epochs 개매변수

# 케라스는 다양한 종류의 경사 하강법을 제공함
# 이런 경사 하강법을 옵티마이져라고 함
# 가장 기본적인 옵티마이져 => 확률적 경사 하강법 SGD
# SGD이지만 미니 배치 이용
# SGD의 하이퍼 파라미터는 momentum, nesterov 등이 있으

sgd = keras.optimizers.SGD(momentum=0.9, nesterov= True)

# 모델이 최적점에 가까울 수록 학습률을 낮출 수 있음
# 학습률은 작을수록 더 정교하게 학습하고, 학습률이 클수록 크게 변함
# 이런 학습률을 적응적 학습률이라고 함
# 대표적인 옵티마이저가 Adagrad와 RMsprop

aragrad = keras.optimizers.Adagrad()
model.compile(optimizer=aragrad,loss='sparse_categorical_crossentropy',metrics='accuracy')

# RMSprop도 마찬가지
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop,loss='sparse_categorical_crossentropy',metrics='accuracy')

# 모멘텀 최적화와 RMSprop 장점을 접목한 것이 Adam

# adam 사용해 보기
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
model.fit(train_scaled,train_target,epochs=5)

model.evaluate(val_scaled,val_target)