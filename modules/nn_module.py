import tensorflow as tf
import math


def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a
    different frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can
    be experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.
    """
    with tf.name_scope("add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def ffn(inputs, hidden_size, output_size, keep_prob=None, data_format="NHWC"):
    with tf.variable_scope('input_layer'):
        hidden = linear(inputs, hidden_size, True, data_format=data_format)
        hidden = tf.nn.relu(hidden)

    if keep_prob and keep_prob < 1.0:
        hidden = tf.nn.dropout(hidden, keep_prob)

    with tf.variable_scope('output_layer'):
        output = linear(hidden, output_size, True, data_format=data_format)

    return output


def _linear_2d(inputs, output_size, bias, concat=True):
    input_size = [item.get_shape()[1].value for item in inputs]

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, 1)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        outputs.append(tf.matmul(inputs, matrix))
    else:
        for i in range(len(input_size)):
            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            outputs.append(tf.matmul(inputs[i], matrix))

    output = tf.add_n(outputs)

    if bias is not None:
        shape = [output_size]
        bias = tf.get_variable("bias", shape)
        output = tf.nn.bias_add(output, bias)

    return output


def _linear_3d(inputs, output_size, bias, concat=True, data_format="NHWC"):
    data_format = check_data_format(data_format)
    channel_axis = 1 if data_format == "NCHW" else -1
    space_axis = -1 if data_format == "NCHW" else 1

    input_size = [item.get_shape()[channel_axis].value for item in inputs]

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, channel_axis)
        inputs = tf.expand_dims(inputs, space_axis)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
        output = tf.nn.convolution(inputs, matrix, "VALID",
                                   data_format=data_format)
        outputs.append(output)
    else:
        for i in range(len(input_size)):
            inputs = tf.expand_dims(inputs, space_axis)

            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
            output = tf.nn.convolution(inputs, matrix, "VALID",
                                       data_format=data_format)
            outputs.append(output)

    output = tf.add_n(outputs)

    if bias is not None:
        bias = tf.get_variable("bias", [output_size])
        output = tf.nn.bias_add(output, bias, data_format=data_format)

    output = tf.squeeze(output, space_axis)

    return output


def _linear_4d(inputs, output_size, bias, concat=True, data_format="NHWC"):
    data_format = check_data_format(data_format)
    channel_axis = 1 if data_format == "NCHW" else -1

    input_size = [item.get_shape()[channel_axis].value for item in inputs]

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, channel_axis)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
        output = tf.nn.convolution(inputs, matrix, "VALID",
                                   data_format=data_format)
        outputs.append(output)
    else:
        for i in range(len(input_size)):
            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            matrix = tf.expand_dims(tf.expand_dims(matrix, 0), 1)
            output = tf.nn.convolution(inputs, matrix, "VALID",
                                       data_format=data_format)
            outputs.append(output)

    output = tf.add_n(outputs)

    if bias is not None:
        bias = tf.get_variable("bias", [output_size])
        output = tf.nn.bias_add(output, bias, data_format=data_format)

    return output


def _linear_5d(inputs, output_size, bias, concat=True, data_format="NHWC"):
    data_format = check_data_format(data_format)
    channel_axis = 1 if data_format == "NCHW" else -1

    input_size = [item.get_shape()[channel_axis].value for item in inputs]

    data_format = "NCDHW" if data_format is "NCHW" else "NDHWC"

    outputs = []

    if concat:
        input_size = sum(input_size)
        inputs = tf.concat(inputs, channel_axis)

        shape = [input_size, output_size]
        matrix = tf.get_variable("matrix", shape)
        matrix = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(matrix, 0), 1), 2
        )
        output = tf.nn.convolution(inputs, matrix, "VALID",
                                   data_format=data_format)
        outputs.append(output)
    else:
        for i in range(len(input_size)):
            shape = [input_size[i], output_size]
            name = "matrix_%d" % i
            matrix = tf.get_variable(name, shape)
            matrix = tf.expand_dims(
                tf.expand_dims(tf.expand_dims(matrix, 0), 1), 2
            )
            output = tf.nn.convolution(inputs, matrix, "VALID",
                                       data_format=data_format)
            outputs.append(output)

    output = tf.add_n(outputs)

    if bias is not None:
        bias = tf.get_variable("bias", [output_size])
        data_format = "NCHW" if data_format is "NCDHW" else "NHWC"
        output = tf.nn.bias_add(output, bias, data_format=data_format)

    return output


def linear(inputs, output_size, bias, concat=True, data_format="NHWC",
           dtype=None, scope=None):
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    ndims = [ip.get_shape().ndims for ip in inputs]

    if any([dim - ndims[0] for dim in ndims]):
        raise ValueError("inputs do not agree on dimensions: %s" % ndims)

    rank = ndims[0]

    with tf.variable_scope(scope, default_name="linear", values=[inputs],
                           dtype=dtype):
        if rank == 2:
            output = _linear_2d(inputs, output_size, bias, concat)
        elif rank == 3:
            output = _linear_3d(inputs, output_size, bias, concat, data_format)
        elif rank == 4:
            output = _linear_4d(inputs, output_size, bias, concat, data_format)
        elif rank == 5:
            output = _linear_5d(inputs, output_size, bias, concat, data_format)
        else:
            raise ValueError("Input rank must be 2, 3 or 4, found %d" % rank)

        return output


def check_data_format(data_format):
    if data_format in ["NCHW", "NHWC", "nchw", "nhwc"]:
        return data_format.upper()
    elif data_format in ["NCW", "ncw"]:
        return "NCW"
    elif data_format in ["NWC", "nwc"]:
        return "NWC"
    else:
        raise ValueError("Unknown data_format: %s" % data_format)


def layer_norm(inputs, epsilon=1e-6, data_format="NHWC"):
    data_format = check_data_format(data_format)
    axis = 1 if data_format == "NCHW" else -1
    channel_size = inputs.get_shape().as_list()[axis]

    scale = tf.get_variable("scale", shape=[channel_size], initializer=tf.ones_initializer())
    offset = tf.get_variable("offset", shape=[channel_size], initializer=tf.zeros_initializer())
    mean = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
    variance = tf.reduce_mean(tf.square(inputs - mean), axis=axis, keep_dims=True)
    norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

    return norm_inputs * scale + offset


def multihead_attention(query, memory, bias, key_size, value_size, output_size,
                        num_heads, keep_prob=None, data_format="NHWC",
                        attention_function="dot_product", summaries=False,
                        image_shapes=None, dtype=None, scope=None):
    """ Multihead scaled-dot-product attention with input/output
        transformations.

    Args:
        query: a Tensor with shape [batch, length_q, channels] if
            data_format is `NHWC`, [batch, channels, length_q] if
            data_format is `NCHW`
        memory: a Tensor with shape [batch, length_m, channels] if
            data_format is `NHWC`, [batch, channels, length_q] if
            data_format is `NCHW`
        bias: bias Tensor (see attention_bias())
        key_size: an integer
        value_size: an integer
        output_size: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        data_format: "NHWC" or "NCHW"
        attention_function: "dot_product" or "additive"
        dtype: an optional instance of tf.DType
        scope: an optional string

    Returns:
        A Tensor.
    """
    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    with tf.variable_scope(scope, default_name="multihead_attention",
                           values=[query, memory], dtype=dtype):
        data_format = check_data_format(data_format)
        axis = 1 if data_format is "NCHW" else 2

        if memory is None:
            # self attention
            size = key_size * 2 + value_size
            combined = linear(query, size, True, True, data_format=data_format,
                              scope="qkv_transform")
            q, k, v = tf.split(combined, [key_size, key_size, value_size],
                               axis=axis)
        else:
            q = linear(query, key_size, True, data_format=data_format,
                       scope="q_transform")
            combined = linear(memory, key_size + value_size, True,
                              data_format=data_format, scope="kv_transform")
            k, v = tf.split(combined, [key_size, value_size], axis=axis)

        # split heads
        q = _split_heads(q, num_heads, data_format=data_format)
        k = _split_heads(k, num_heads, data_format=data_format)
        v = _split_heads(v, num_heads, data_format=data_format)

        # scale query
        if attention_function == "dot_product":
            key_depth_per_head = key_size // num_heads
            q *= key_depth_per_head ** -0.5

            # attention
            x = dot_product_attention(q, k, v, bias, keep_prob, summaries,
                                      image_shapes)
        elif attention_function == "additive":
            x = additive_attention(q, k, v, bias, keep_prob, summaries,
                                   image_shapes)
        else:
            raise ValueError("Unknown attention function")

        # combine heads
        x = _combine_heads(x, data_format=data_format)

        x = linear(x, output_size, True, data_format=data_format,
                   scope="output_transform")
        return x


def _split_heads(x, num_heads, data_format="NHWC"):
    n = num_heads
    old_shape = x.get_shape().dims

    if data_format is "NCHW":
        x = tf.transpose(x, [0, 2, 1])

    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])


def _combine_heads(x, data_format="NHWC"):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    x = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    x.set_shape(new_shape)

    if data_format is "NCHW":
        x = tf.transpose(x, [0, 2, 1])

    return x


def dot_product_attention(query, key, value, bias, keep_prob, summaries=False,
                          image_shapes=None, name=None):
    """ dot-product attention.

    Args:
        query: a Tensor with shape [batch, heads, length_q, depth_k]
        key: a Tensor with shape [batch, heads, length_kv, depth_k]
        value: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        name: an optional string

    Returns:
        A Tensor.
    """
    with tf.name_scope(name, default_name="dot_product_attention",
                       values=[query, key, value]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(query, key, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        if summaries and not tf.get_variable_scope().reuse:
            attention_image_summary(weights, image_shapes)

        return tf.matmul(weights, value)


def additive_attention(query, key, value, bias, keep_prob, summaries=False,
                       image_shapes=None, name=None):
    """ dot-product attention.

    Args:
        query: a Tensor with shape [batch, heads, length_q, depth_k]
        key: a Tensor with shape [batch, heads, length_kv, depth_k]
        value: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        keep_prob: a floating point number
        summaries: a boolean
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        name: an optional string

    Returns:
        A Tensor.
    """
    with tf.variable_scope(name, default_name="additive_attention",
                           values=[query, key, value]):
        query = tf.expand_dims(query, 3)
        key = tf.expand_dims(key, 2)

        hidden = query + key
        logits = linear(hidden, 1, False, scope="logits")
        logits = tf.squeeze(logits, -1)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        if keep_prob is not None and keep_prob < 1.0:
            weights = tf.nn.dropout(weights, keep_prob)

        if summaries and not tf.get_variable_scope().reuse:
            attention_image_summary(weights, image_shapes)

        return tf.matmul(weights, value)


def attention_image_summary(attn, image_shapes=None):
    """ Compute color image summary.

    Args:
        attn: a Tensor with shape
            [batch, num_heads, query_length, memory_length]
        image_shapes: optional tuple of integer scalars.
            If the query positions and memory positions represent the
            pixels of flattened images, then pass in their dimensions:
                (query_rows, query_cols, memory_rows, memory_cols).
            If the query positions and memory positions represent the
            pixels x channels of flattened images, then pass in their
            dimensions:
                (query_rows, query_cols, query_channels,
                memory_rows, memory_cols, memory_channels).
    """
    num_heads = attn.get_shape().as_list()[1]
    # [batch, query_length, memory_length, num_heads]
    image = tf.transpose(attn, [0, 2, 3, 1])
    image = tf.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, -num_heads % 3]])

    # split last dimensions
    n = 3
    old_shape = image.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    image = tf.reshape(image, tf.concat([tf.shape(image)[:-1], [n, -1]], 0))
    image.set_shape(new_shape)
    image = tf.reduce_max(image, 4)

    if image_shapes is not None:
        if len(image_shapes) == 4:
            q_rows, q_cols, m_rows, m_cols = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
            image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
            image = tf.reshape(image, [-1, q_rows * m_rows,
                                       q_cols * m_cols, 3])
        else:
            assert len(image_shapes) == 6
            q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(
                    image_shapes
            )
            image = tf.reshape(image, [
                -1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3
            ])
            image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
            image = tf.reshape(image, [
                -1,
                q_rows * m_rows * q_channnels,
                q_cols * m_cols * m_channels,
                3
            ])
    tf.summary.image("attention", image, max_outputs=1)


def attention_bias(inputs, mode, inf=-1e9, name="attention_bias"):
    with tf.name_scope(name, values=[inputs]):
        if mode == "incremental":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        else:
            raise ValueError("Unknown mode %s" % mode)


def residual_link(x, y, keep_prob):
    if keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)

    return layer_norm(x + y)