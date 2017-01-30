import tensorflow as tf
from keras import backend as K

from keras2tensorflow import keras2tensorflow as k2tf
from keras.models import Sequential, model_from_json

from PIL import Image
import numpy as np

# load keras trained MNIST model
model = model_from_json(open(r'model_structure.json').read())
model.load_weights('model_weight.h5')

# create a converter
converter = k2tf(model)

# save keras model's weights, biases and structure to file, use c++ to read it.
#converter.save_to_file("tf.model")

# transform keras's layers to tensorflow's layers
tf_input = tf.placeholder("float32", model.layers[0].batch_input_shape)
tf_prediction = converter.dump_tf_layer(tf_input)


# test our tensorflow layers, the outputs should be like this:
# [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
# [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
# [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
# [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
sess = tf.Session()

for i in range(10):
    img = Image.open(str(i)+".png")
    if img is None:
        print("can not open file "+str(i)+".png")
        continue

    img = img.convert('L')
    data = np.empty((1,28, 28,1), dtype="float32")
    arr = np.asarray(img, dtype="float32")
    arr=arr.reshape(1,28,28,1)
    data[0, :, :, :] = arr

    result = sess.run(tf_prediction, feed_dict={tf_input:data})
    print(result)

K.clear_session()