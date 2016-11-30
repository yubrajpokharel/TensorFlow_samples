import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')
height, width, depth = image.shape
print("height")
print (height)
print ("width")
print (width)
my_list = list([width] * height)
print (len(my_list))

model = tf.initialize_all_variables()

with tf.Session() as session:
    # x= tf.reverse(x, dims=[True, False, False],name="reverse")
    # x= tf.reverse(x, dims=[False, True, False],name="reverse")
    # x = tf.transpose(x, perm=[1, 0, 2])
    # x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)
    print(session.run(x))

plt.imshow(result)
plt.show()