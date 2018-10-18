import tensorflow as tf
import tensorflow.contrib as tc
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn


def nor_rnn(rnn_type, inputs, length, hidden_size, layer_num=1, dropout_keep_prob=None, concat=True):
    if not rnn_type.startswith('bi'):
        cells = tc.rnn.MultiRNNCell([get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)],
                                    state_is_tuple=True)
        outputs, state = tf.nn.dynamic_rnn(cells, inputs, sequence_length=length, dtype=tf.float32)
        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
    else:
        if layer_num > 1:
            cell_fw = [get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)]
            cell_bw = [get_nor_cell(rnn_type, hidden_size, dropout_keep_prob) for _ in range(layer_num)]
            outputs, state_fw, state_bw = tc.rnn.stack_bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32
            )
        else:
            cell_fw = get_nor_cell(rnn_type, hidden_size, dropout_keep_prob)
            cell_bw = get_nor_cell(rnn_type, hidden_size, dropout_keep_prob)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32
            )
        # state_fw, state_bw = state
        # if rnn_type.endswith('lstm'):
        #     c_fw, h_fw = state_fw
        #     c_bw, h_bw = state_bw
        #     state_fw, state_bw = h_fw, h_bw
        # if concat:
        #     outputs = tf.concat(outputs, 2)
        #     state = tf.concat([state_fw, state_bw], 1)
        # else:
        #
        #  = outputs[0] + outputs[1]
        #     state = state_fw + state_bw
    return outputs


def get_nor_cell(rnn_type, hidden_size, dropout_keep_prob=None):
    if rnn_type.endswith('lstm'):
        cell = tc.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    elif rnn_type.endswith('gru'):
        cell = tc.rnn.GRUCell(num_units=hidden_size)
    elif rnn_type.endswith('rnn'):
        cell = tc.rnn.BasicRNNCell(num_units=hidden_size)
    elif rnn_type.endswith('sru'):
        cell = tc.rnn.SRUCell(num_units=hidden_size)
    elif rnn_type.endswith('indy'):
        cell = tc.rnn.IndyGRUCell(num_units=hidden_size)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)
    return cell


# def cudnn_rnn(rnn_type, direction, inputs, hidden_size, dropout, num_layers):
#     cell = get_cudnn_cell(rnn_type, num_layers, hidden_size, direction, dropout)
#     output, states = cell(inputs)
#     states = states[0]
#     if direction == 'undirectional':
#         states = tf.unstack(states, num=num_layers)
#         return output, tuple(states)
#     fw_states = []
#     bw_states = []
#     for i, state in enumerate(tf.unstack(states, num=2 * num_layers)):
#         if i % 2 == 0:
#             fw_states.append(state)
#         else:
#             bw_states.append(state)
#
#     return output, tuple(fw_states), tuple(bw_states)
#
#
# def get_cudnn_cell(rnn_type, num_layers, hidden_size, direction, dropout):
#     if rnn_type.endswith('lstm'):
#         cudnn_cell = cudnn_rnn.CudnnLSTM(num_layers, hidden_size, direction=direction, dropout=dropout)
#     elif rnn_type.endswith('gru'):
#         cudnn_cell = cudnn_rnn.CudnnGRU(num_layers, hidden_size, direction=direction, dropout=dropout)
#     elif rnn_type.endswith('rnn'):
#         cudnn_cell = cudnn_rnn.CudnnRNNTanh(num_layers, hidden_size, direction=direction, dropout=dropout)
#     else:
#         raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
#     return cudnn_cell


def cu_rnn(rnn_type, inputs, hidden_size, batch_size, layer_num=1):
    if not rnn_type.startswith('bi'):
        cell = get_cu_cell(rnn_type, hidden_size, layer_num, cudnn_rnn.CUDNN_RNN_UNIDIRECTION)
        inputs = tf.transpose(inputs, [1, 0, 2])
        c = tf.zeros([layer_num, batch_size, hidden_size], tf.float32)
        h = tf.zeros([layer_num, batch_size, hidden_size], tf.float32)
        outputs, state = cell(inputs)
        if rnn_type.endswith('lstm'):
            c, h = state
            state = h
    else:
        cell = get_cu_cell(rnn_type, hidden_size, layer_num, 'bidirectional')
        inputs = tf.transpose(inputs, [1, 0, 2])
        outputs, state = cell(inputs)
        # if concat:
        #     state = tf.concat([state_fw, state_bw], 1)
        # else:
        #     state = state_fw + state_bw
    outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, state


def get_cu_cell(rnn_type, hidden_size, layer_num=1, direction='undirectional'):
    if rnn_type.endswith('lstm'):
        cudnn_cell = cudnn_rnn.CudnnLSTM(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                         dropout=0)
    elif rnn_type.endswith('gru'):
        cudnn_cell = cudnn_rnn.CudnnGRU(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                        dropout=0)
    elif rnn_type.endswith('rnn'):
        cudnn_cell = cudnn_rnn.CudnnRNNTanh(num_layers=layer_num, num_units=hidden_size, direction=direction,
                                            dropout=0)
    else:
        raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
    return cudnn_cell


def dense(inputs, hidden, use_bias=True, scope='dense', initializer=None):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        # 前两个维度与输入相同，最后加上输出维度
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden]

        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable('W', [dim, hidden], initializer=initializer)
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable('b', [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res