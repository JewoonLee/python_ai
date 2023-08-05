import tensorflow as tf

텐서 = tf.constant([3,4,5])
텐서2 = tf.constant([6,7,8])


텐서3 = tf.constant([[3,4],
                    [5,6]])

텐서4 = tf.zeros(10)
텐서5 = tf.zeros([2,2])
텐서6 = tf.zeros([2,2,3])

print(텐서.shape)

w = tf.Variable(1.0)
print(w.numpy) 

w.assign(2)

print(w)