import tensorflow as tf
import tensorflow.contrib as tc


def cnn(inputs, mask, hidden_size, filter_width):
    input_size = inputs.get_shape().as_list()[-1]
    shape = [filter_width, input_size, 2 * hidden_size]
    filter_v = tf.get_variable('filter', shape)
    bias_v = tf.get_variable('bias', [2 * hidden_size])
    output = tf.nn.convolution(inputs, filter_v, 'SAME')
    output = tf.nn.bias_add(output, bias_v)
    gate, act = tf.split(output, 2, 2)
    output = tf.nn.sigmoid(gate) * act

    return output * mask[:, :, None]
