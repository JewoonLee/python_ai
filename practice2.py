import tensorflow as tf

# 키 = [170,180,175, 160]
# 신발 = [260,270,265,255]

키 = 170
신발 = 260

# 신발 = 키 * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def 손실함수():
    return tf.square(260 - (키 * a + b))

opt = tf.keras.optimizers.Adam(learning_rate=0.1)

for i in range(100):
    opt.minimize(손실함수,var_list=[a,b])
    print(a.numpy(),b.numpy())






