import tensorflow as tf

x = tf.placeholder("float", 3)
y = x * 2

x_none = tf.placeholder("float", None)
y_none = x_none * 2

x_fixCol = tf.placeholder("float", [None, 3])
y_fixCol = x_fixCol * 2
x_fixCol_data = [
                    [1, 2, 3],
                    [4, 5, 6],
                    [1, 2, 3],
                    [4, 5, 6]
                ]

with tf.Session() as session:
    print(session.run(y, feed_dict= {x:[3, 4, 5]}))
    print(session.run(y_none, feed_dict={x_none:[2, 3, 4, 5, 6, 7]}))
    result = session.run(y_fixCol, feed_dict={x_fixCol:x_fixCol_data})
    print(result)