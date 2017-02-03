import tensorflow as tf
import numpy as np
import struct

LAYER_CONV2D = 1
LAYER_ACTIVATION = 2
LAYER_MAXPOOL = 3
LAYER_FLATTEN = 4
LAYER_DENSE = 5
LAYER_DROPOUT = 6

ACTIVATION_UNKNOWN = 0
ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_SOFTMAX = 3
ACTIVATION_TANH = 4

#####################################################################
def save_floats(file, floats):
    '''
    Writes floats to file in 1024 chunks.. prevents memory explosion
    writing very large arrays to disk when calling struct.pack().
    '''
    step = 1024
    written = 0

    for i in np.arange(0, len(floats), step):
        remaining = min(len(floats) - i, step)
        written += remaining
        file.write(struct.pack('=%sf' % remaining, *floats[i:i+remaining]))

    assert written == len(floats)

#####################################################################
def load_floats(file, count): assert False, "UNSUPPORT"


#####################################################################
class keras_conv2d:
    weights = None
    biases = None

    def __init__(self, keras_layer):
        self.weights = keras_layer.get_weights()[0]
        self.biases = keras_layer.get_weights()[1]
        self.padding = 'VALID'
        if keras_layer.border_mode != "valid":
            assert False, "Unsupported padding type: %s" % keras_layer.border_mode

    def dump_tf_layer(self, prev_tf_layer):
        w = tf.constant(self.weights)
        b = tf.constant(self.biases)
        tf_layer = tf.nn.conv2d(prev_tf_layer,
                                w,
                                strides=[1,1,1,1],
                                padding=self.padding) + b
        return tf_layer

    def save_to_file(self, file):
        file.write(struct.pack('I', LAYER_CONV2D))
        file.write(struct.pack('I', self.weights.shape[0]))
        file.write(struct.pack('I', self.weights.shape[1]))
        file.write(struct.pack('I', self.biases.shape[0]))
        save_floats(file, self.weights.flatten())
        save_floats(file, self.biases.flatten())

    def load_from_file(self, file): assert False, "UNSUPPORT"

#####################################################################
class keras_activation:
    activation = ACTIVATION_UNKNOWN

    def __init__(self, keras_layer):
        act = keras_layer.get_config()['activation']
        if act == "linear":
            self.activation = ACTIVATION_LINEAR
        elif act == "relu":
            self.activation = ACTIVATION_RELU
        elif act == "softmax":
            self.activation = ACTIVATION_SOFTMAX
        elif act == "tanh":
            self.activation = ACTIVATION_TANH
        else:
            assert False, "Unsupported activation type: %s" % act
        

    def dump_tf_layer(self, prev_tf_layer):
        if self.activation == ACTIVATION_LINEAR:
            tf_layer = prev_tf_layer
        elif self.activation == ACTIVATION_RELU:
            tf_layer = tf.nn.relu(prev_tf_layer)
        elif self.activation == ACTIVATION_SOFTMAX:
            tf_layer = tf.nn.softmax(prev_tf_layer)
        elif self.activation == ACTIVATION_TANH:
            tf_layer = tf.tanh(prev_tf_layer)
        return tf_layer

    def save_to_file(self, file):
        file.write(struct.pack('I', LAYER_ACTIVATION))
        file.write(struct.pack('I', self.activation))

    def load_from_file(self, file): assert False, "UNSUPPORT"

#####################################################################
class keras_maxpool:
    pool_size = None
    padding = None

    def __init__(self, keras_layer):
        self.pool_size = keras_layer.get_config()['pool_size']
        self.padding = 'VALID'
        if keras_layer.border_mode != "valid":
            assert False, "Unsupported padding type: %s" % keras_layer.border_mode

    def dump_tf_layer(self, prev_tf_layer):
        tf_layer = tf.nn.max_pool(prev_tf_layer,
                                  ksize=[1, self.pool_size[0], self.pool_size[1], 1],
                                  strides=[1, self.pool_size[0], self.pool_size[1], 1],
                                  padding=self.padding)
        return tf_layer

    def save_to_file(self, file):
        file.write(struct.pack('I', LAYER_MAXPOOL))
        file.write(struct.pack('I', self.pool_size[0]))
        file.write(struct.pack('I', self.pool_size[1]))

    def load_from_file(self, file): assert False, "UNSUPPORT"

#####################################################################
class keras_flatten:
    def __init__(self, keras_layer):None

    def dump_tf_layer(self, prev_tf_layer):
        tf_layer = tf.reshape(prev_tf_layer, [-1])
        return tf_layer

    def save_to_file(self, file):
        file.write(struct.pack('I', LAYER_FLATTEN))

    def load_from_file(self, file): assert False, "UNSUPPORT"

#####################################################################
class keras_dense:
    weights = None
    biases = None

    def __init__(self, keras_layer):
        self.weights = keras_layer.get_weights()[0]
        self.biases = keras_layer.get_weights()[1]

    def dump_tf_layer(self, prev_tf_layer):
        tf_layer = tf.reshape(prev_tf_layer, [-1, self.weights.shape[0]])
        tf_layer = tf.matmul(tf_layer, self.weights) + self.biases
        tf_layer = tf.reshape(tf_layer, [-1])
        return tf_layer

    def save_to_file(self, file):
        file.write(struct.pack('I', LAYER_DENSE))
        file.write(struct.pack('I', self.weights.shape[0]))
        file.write(struct.pack('I', self.weights.shape[1]))
        file.write(struct.pack('I', self.biases.shape[0]))
        save_floats(file, self.weights.flatten())
        save_floats(file, self.biases.flatten())

    def load_from_file(self, file): assert False, "UNSUPPORT"

#####################################################################
class keras_dropout:
    p = 0
    
    def __init__(self, keras_layer):
        self.p = keras_layer.p

    def dump_tf_layer(self, prev_tf_layer):
        # prob = tf.constant(self.p)
        prob = tf.constant(1.0)
        tf_layer = tf.nn.dropout(prev_tf_layer, prob)
        return tf_layer

    def save_to_file(self, file):
        file.write(struct.pack('I', LAYER_DROPOUT))
        file.write(struct.pack('f', p))

    def load_from_file(self, file): assert False, "UNSUPPORT"

#####################################################################
class keras2tensorflow:
    layers = []
    input_shape = []

    def __init__(self, keras_model):
        self.input_shape = keras_model.layers[0].batch_input_shape
        for keras_layer in keras_model.layers:
            layer_type = type(keras_layer).__name__

            tf_layer = None
            if layer_type == "Convolution2D":
                tf_layer = keras_conv2d(keras_layer)
            elif layer_type == "Activation":
                tf_layer = keras_activation(keras_layer)
            elif layer_type == "MaxPooling2D":
                tf_layer = keras_maxpool(keras_layer)
            elif layer_type == "Flatten":
                tf_layer = keras_flatten(keras_layer)
            elif layer_type == "Dense":
                tf_layer = keras_dense(keras_layer)
            elif layer_type == "Dropout":
                tf_layer = keras_dropout(keras_layer)
            else:
                assert False, "Unsupported layer type: %s" % layer_type
            
            self.layers.append(tf_layer)

    def dump_tf_layer(self, prev_tf_layer):
        for tf_layer in self.layers:
            prev_tf_layer = tf_layer.dump_tf_layer(prev_tf_layer)
        return prev_tf_layer

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            num_layers = len(self.layers)
            f.write(struct.pack('I', num_layers))

            for tf_layer in self.layers:
                tf_layer.save_to_file(f)

    def save_protobuf(self, filename):
        graph_dump = tf.Graph()
        with graph_dump.as_default():
            tf_input = tf.placeholder("float32", self.input_shape, name="input")
            tf_prediction = self.dump_tf_layer(tf_input)
            tf_output = tf.add(tf_prediction, 0, name="output")

            sess = tf.Session()
            graph_def = graph_dump.as_graph_def()
            tf.train.write_graph(graph_def, '', filename, as_text=False)
            sess.close()

    def prediction(self, data):
        sess = tf.Session()
        tf_input = tf.placeholder("float32", self.input_shape, name="input")
        tf_prediction = self.dump_tf_layer(tf_input)
        result = sess.run(tf_prediction, feed_dict={tf_input:data})
        sess.close()
        return result
