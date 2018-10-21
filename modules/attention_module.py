import tensorflow as tf

_NEG_INF = -1e9


def self_transformer(embedded_x, embedded_y, inputs_padding, n_block, n_hidden, n_head, keep_prob, is_ff, is_train):
    attention_bias = get_padding_bias(inputs_padding)
    if is_train:
        embedded_x = tf.nn.dropout(embedded_x, keep_prob)
        embedded_y = tf.nn.dropout(embedded_y, keep_prob)
    encoder_stack = EncoderStack(n_block, n_hidden, n_head, keep_prob, is_train, is_ff)
    encoder_outputs = encoder_stack(embedded_x, embedded_y, attention_bias, inputs_padding)
    return encoder_outputs


class EncoderStack(tf.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, n_block, n_hidden, n_head, keep_prob, is_train, is_ffn_pad=True, is_ff=True):
        super(EncoderStack, self).__init__()
        self.layers = []
        self.is_ff = is_ff
        for _ in range(n_block):
            # Create sublayers for each layer.
            # self_attention_layer = SelfAttention(n_hidden, n_head, keep_prob, is_train)
            self_attention_layer = Attention(n_hidden, n_head, keep_prob, is_train)
            if self.is_ff:
                feed_forward_network = FeedFowardNetwork(n_hidden, 2 * n_hidden, keep_prob, is_train, is_ffn_pad)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, n_hidden, keep_prob, is_train),
                PrePostProcessingWrapper(feed_forward_network, n_hidden, keep_prob, is_train)] if self.is_ff else
                               [PrePostProcessingWrapper(self_attention_layer, n_hidden, keep_prob, is_train)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(n_hidden)

    def call(self, inputs_x, inputs_y, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
          inputs_padding: P

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            if self.is_ff:
                feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_outputs = self_attention_layer(inputs_x, inputs_y, attention_bias)
                if self.is_ff:
                    with tf.variable_scope("ffn"):
                        encoder_outputs = feed_forward_network(encoder_outputs, inputs_padding)

        return self.output_normalization(encoder_outputs)


class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q")
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v")

        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        """Apply attention mechanism to x and y.

        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.

        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)


class FeedFowardNetwork(tf.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        self.filter_dense_layer = tf.layers.Dense(
            filter_size, use_bias=True, activation=tf.nn.relu, name="filter_layer")
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def call(self, x, padding=None):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          padding: (optional) If set, the padding values are temporarily removed
            from x (provided self.allow_pad is set). The padding values are placed
            back in the output tensor in the same locations.
            shape [batch_size, length]

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding

        # Retrieve dynamically known shapes
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        if padding is not None:
            with tf.name_scope("remove_padding"):
                # Flatten padding to [batch_size*length]
                pad_mask = tf.reshape(padding, [-1])

                nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))

                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = tf.reshape(x, [-1, self.hidden_size])
                x = tf.gather_nd(x, indices=nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.
                x.set_shape([None, self.hidden_size])
                x = tf.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.train:
            output = tf.nn.dropout(output, 1.0 - self.relu_dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with tf.name_scope("re_add_padding"):
                output = tf.squeeze(output, axis=0)
                output = tf.scatter_nd(
                    indices=nonpad_ids,
                    updates=output,
                    shape=[batch_size * length, self.hidden_size]
                )
                output = tf.reshape(output, [batch_size, length, self.hidden_size])
        return output


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size], initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size], initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, n_hidden, keep_prob, train):
        self.layer = layer
        self.keep_prob = keep_prob
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(n_hidden)

    def __call__(self, x, *args, **kwargs):
        # Pre_processing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, self.keep_prob)
        return x + y


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that

    Returns:
      flaot tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]

    Returns:
      Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        attention_bias = x * _NEG_INF
        attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias
