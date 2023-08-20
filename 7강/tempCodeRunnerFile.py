dense1 = keras.layers.Dense(100,activation='sigmoid', input_shape = (784,))
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