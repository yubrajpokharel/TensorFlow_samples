import tensorflow as tf
import numpy as np

data = np.random.randint(0, 9999, size=10)
x = tf.constant(data, name='x')
y = tf.Variable(x + 5, name='y')

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print("Initial X values")
    print(session.run(x))
    print("Final Y values")
    print(session.run(y))