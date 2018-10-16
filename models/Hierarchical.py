import tensorflow as tf
import tensorflow.contrib as tc
import time
from modules.nn_module import add_timing_signal
from modules.rnn_module import cu_rnn
from modules.cnn_module import cnn
from modules.nn_module import ffn, layer_norm, multihead_attention, residual_link


class MyModel(object):
    def __init__(self, args, batch, token_embeddings, logger):
        # logging
        self.args = args
        self.logger = logger
        # basic config
        self.n_batch = tf.get_variable('n_batch', shape=[], dtype=tf.int32, trainable=False)
        self.n_label = args.n_class
        self.opt_type = args.optim
        self.weight_decay = args.weight_decay

        self.eid, self.token_ids, self.token_len, self.labels = batch.get_next()
        self.N = tf.shape(self.eid)[0]
        self.max_len = tf.reduce_max(self.token_len)
        self.token_ids = tf.slice(self.token_ids, [0, 0], tf.stack([self.N, self.max_len]))
        self.mask = tf.sequence_mask(self.token_len, self.max_len, dtype=tf.float32, name='masks')
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.initializer = tf.contrib.layers.xavier_initializer()

        self._build_graph(token_embeddings)

    def _build_graph(self, token_embeddings):
        start_t = time.time()
        self._embed(token_embeddings)
        self._encoder()
        self._self_attention()
        self._predict_label()
        self._compute_loss()
        # self._compute_accuracy()
        # 选择优化算法
        if self.is_train:
            self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _embed(self, token_embeddings):
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding', reuse=tf.AUTO_REUSE):
            word_embeddings = tf.get_variable('word_embeddings',
                                              initializer=tf.constant(token_embeddings, dtype=tf.float32),
                                              trainable=False)
            self.token_emb = tf.nn.embedding_lookup(word_embeddings, self.token_ids)
            if self.args.pos:
                self.token_emb = add_timing_signal(self.token_emb)
            else:
                pos_embeddings = tf.get_variable('position_embedding', [200, self.args.n_emb],
                                                 initializer=tf.random_normal_initializer(0.0, self.args.n_emb ** -0.5))
                indices = tf.range(tf.shape(self.token_ids)[1])[None, :]
                pos_emb = tf.gather(pos_embeddings, indices)
                pos_emb = tf.tile(pos_emb, [tf.shape(self.token_ids)[0], 1, 1])
                self.token_emb += pos_emb
            if self.is_train:
                self.token_emb = tf.nn.dropout(self.token_emb, self.args.dropout_keep_prob)

    def _encoder(self):
        with tf.variable_scope('encoder'):
            if self.args.ecoder_type == 'rnn':
                y = cu_rnn('bi-gru', self.token_emb, int(self.args.n_emb / 2), self.args.n_batch, self.args.n_layer)
            elif self.args.encoder_type == 'cnn':
                y = cnn(self.token_emb, self.mask, self.args.n_emb, 3)
            elif self.args.encoder_type == 'ffn':
                y = ffn(self.token_emb, int(self.args.n_emb * 2), self.args.n_emb,
                        self.args.dropout_keep_prob if self.is_train else 1)
            self.token_emb = residual_link(self.token_emb, y, self.args.dropout_keep_prob if self.is_train else 1.0)

    def _self_attention(self):
        with tf.variable_scope('self attention'):
            for i in range(self.args.n_block):
                y = multihead_attention(
                    x,
                    None,
                    attn_bias,
                    params.attention_key_channels or params.hidden_size,
                    params.attention_value_channels or params.hidden_size,
                    params.hidden_size,
                    params.num_heads,
                    1.0 - params.attention_dropout,
                    attention_function=params.attention_function
                )
                x = _residual_fn(x, y, params)
