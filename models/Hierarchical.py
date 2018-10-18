import tensorflow as tf
import tensorflow.contrib as tc
import time
from modules.nn_module import add_timing_signal
from modules.rnn_module import cu_rnn, dense
from modules.cnn_module import cnn
from modules.nn_module import ffn, multihead_attention, attention_bias, residual_link


class MyModel(object):
    def __init__(self, args, batch, token_embeddings, logger):
        # logging
        self.args = args
        self.logger = logger
        # basic config
        self.n_batch = tf.get_variable('n_batch', shape=[], dtype=tf.int32, trainable=False)
        self.n_class = 2
        self.opt_type = args.optim
        self.weight_decay = args.weight_decay

        self.eid, self.token_ids, self.token_len, self.labels = batch.get_next()
        self.N = tf.shape(self.eid)[0]
        # self.max_len = tf.reduce_max(self.token_len)
        # self.token_ids = tf.slice(self.token_ids, [0, 0], tf.stack([self.N, self.max_len]))
        self.mask = tf.sequence_mask(self.token_len, self.args.max_len, dtype=tf.float32, name='masks')
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
            if self.args.timing:
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
            if self.args.encoder_type == 'rnn':
                y, _ = cu_rnn('bi-gru', self.token_emb, int(self.args.n_emb / 2), self.n_batch, self.args.n_layer)
            elif self.args.encoder_type == 'cnn':
                y = cnn(self.token_emb, self.mask, self.args.n_emb, 3)
            elif self.args.encoder_type == 'ffn':
                y = ffn(self.token_emb, int(self.args.n_emb * 2), self.args.n_emb,
                        self.args.dropout_keep_prob if self.is_train else 1)
            self.token_encoder = residual_link(self.token_emb, y, self.args.dropout_keep_prob if self.is_train else 1.0)

    def _self_attention(self):
        with tf.variable_scope('self_attention'):
            attn_bias = attention_bias(self.mask, 'masking')
            self.n_hidden = self.args.n_emb
            for i in range(self.args.n_block):
                with tf.variable_scope('block_{}'.format(i)):
                    y = multihead_attention(
                        self.token_encoder,
                        None,
                        attn_bias,
                        self.args.n_emb,
                        self.args.n_emb,
                        self.n_hidden,
                        self.args.n_head,
                        self.args.dropout_keep_prob,
                        attention_function='dot_product'
                    )
                    self.token_encoder = residual_link(self.token_encoder, y, self.args.dropout_keep_prob)

    def _predict_label(self):
        with tf.variable_scope('predict_labels'):
            self.token_att = tf.reshape(self.token_encoder, shape=[self.N, self.args.max_len * self.n_hidden])
            self.outputs = dense(self.token_att, self.n_class, initializer=self.initializer)

    def _compute_loss(self):
        self.pre_labels = tf.argmax(self.outputs, axis=1)
        if self.args.pos_weight > 0:
            self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(self.labels, 2),
                                                                                logits=self.outputs,
                                                                                pos_weight=self.args.pos_weight))
        else:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                      logits=self.outputs))
        # self.loss = self._focal_loss(tf.one_hot(self.labels, 2, axis=1), self.output)
        self.all_params = tf.trainable_variables()
        if self.args.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.args.weight_decay * l2_loss

    def _create_train_op(self):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            opt_type = self.args.optim
            if opt_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.lr)
            elif opt_type == 'adam':
                self.optimizer = tc.opt.LazyAdamOptimizer(self.lr)
            elif opt_type == 'rprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.lr)
            elif opt_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise NotImplementedError('Unsupported optimizer: {}'.format(self.opt_type))
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.all_params), self.args.global_norm)
            # self.grads = tf.gradients(self.loss, self.all_params)
            self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.all_params),
                                                           global_step=self.global_step)
