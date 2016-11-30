import tensorflow as tf

data = [[1,2,3,4,5,7,6],[1, 2, 3, 0, 0, 0, 0],[1, 2, 3, 4, 5, 6, 7]]
seq_lens = [2, 3, 7]
x = tf.constant(data, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    print(session.run(x))
    print("\n")
    session.run(model)
    x = tf.reverse_sequence(x, seq_lens, seq_dim=1, batch_dim=0)
    print(session.run(x))
