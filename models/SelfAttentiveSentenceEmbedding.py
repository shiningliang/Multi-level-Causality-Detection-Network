import tensorflow as tf
import tensorflow.contrib as tc
import time
from modules.rnn_module import cu_rnn, dense


class SelfAttentive(object):
    def __init__(self, args, batch, token_embeddings, logger, trainable=True):
        self.logger = logger
        # basic config
        self.n_batch = tf.get_variable('n_batch', shape=[], dtype=tf.int32, trainable=False)
        self.n_hidden = args.n_hidden
        self.n_layer = args.n_layer
        self.da = args.sa_da
        self.r = args.sa_r
        self.num_class = 2
        self.opt_type = args.optim
        self.pos_weight = args.pos_weight
        self.dropout_keep_prob = args.dropout_keep_prob
        self.weight_decay = args.weight_decay
        self.norm = args.global_norm
        self.is_train = trainable

        self.eid, self.token_ids, self.token_len, self.labels = batch.get_next()
        self.N = tf.shape(self.eid)[0]
        self.max_len = tf.reduce_max(self.token_len)
        self.token_ids = tf.slice(self.token_ids, [0, 0], tf.stack([self.N, self.max_len]))
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.initializer = tf.contrib.layers.xavier_initializer()

        self._build_graph(token_embeddings)

    def _build_graph(self, token_embeddings):
        start_t = time.time()
        self._embed(token_embeddings)
        self._encode()
        self._cal_attention()
        self._apply_attention()
        self._penal_term()
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

    def _encode(self):
        with tf.variable_scope('encoding', reuse=tf.AUTO_REUSE):
            self.H, _ = cu_rnn('bi-lstm', self.token_emb, self.n_hidden, self.n_batch, self.n_layer)
        if self.is_train:
            self.H = tf.nn.dropout(self.H, rate=1 - self.dropout_keep_prob)

    def _cal_attention(self):
        with tf.variable_scope('cal_attention', reuse=tf.AUTO_REUSE):
            self.W_s1 = tf.get_variable('Ws1', shape=[2 * self.n_hidden, self.da], initializer=self.initializer)
            self.W_s1_H = tf.nn.tanh(tf.matmul(tf.reshape(self.H, [-1, 2 * self.n_hidden]), self.W_s1))
            self.W_s2 = tf.get_variable('Ws2', shape=[self.da, self.r], initializer=self.initializer)
            self.A = tf.matmul(self.W_s1_H, self.W_s2)
            # self.A_T = tf.nn.softmax(tf.reshape(self.A, shape=[-1, self.max_len, self.r]), dim=1, name='A_T')
            self.A_T = tf.nn.softmax(tf.reshape(self.A, shape=[-1, self.max_len, self.r]), name='A_T')

    def _apply_attention(self):
        with tf.variable_scope('apply_attention', reuse=tf.AUTO_REUSE):
            self.H_T = tf.transpose(self.H, perm=[0, 2, 1])
            self.M_T = tf.matmul(self.H_T, self.A_T)

    def _penal_term(self):
        with tf.variable_scope('penalization_term', reuse=tf.AUTO_REUSE):
            self.A = tf.transpose(self.A_T, perm=[0, 2, 1])
            self.AA_T = tf.matmul(self.A, self.A_T)
            # self.I = tf.reshape(tf.tile(tf.diag(tf.ones([self.r]), name='diag_identity'), [self.n_batch, 1]),
            #                     [self.n_batch, self.r, self.r])
            self.I = tf.diag(tf.ones([self.r]), name='diag_identity')

            self.penalized_term = tf.reduce_mean(tf.square(tf.norm(self.AA_T - self.I, ord='euclidean', axis=[1, 2])))

    def _predict_label(self):
        with tf.variable_scope('predict_labels', reuse=tf.AUTO_REUSE):
            self.flatten_M_T = tf.reshape(self.M_T, shape=[-1, self.r * 2 * self.n_hidden])
            self.label_dense_0 = tf.nn.relu(dense(self.flatten_M_T, hidden=2 * self.n_hidden, scope='dense_0'))
            if self.is_train:
                self.label_dense_0 = tf.nn.dropout(self.label_dense_0, rate=1 - self.dropout_keep_prob)

            self.output = dense(self.label_dense_0, hidden=self.num_class, scope='output_labels')

    def _compute_loss(self):
        self.pre_labels = tf.argmax(self.output, axis=1)
        if self.pos_weight > 0:
            self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(self.labels, 2),
                                                                                logits=self.output,
                                                                                pos_weight=self.pos_weight))
        else:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                      logits=self.output))
        self.loss += self.penalized_term
        # self.loss = self._focal_loss(tf.one_hot(self.labels, 2, axis=1), self.output)
        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _compute_accuracy(self):
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.pre_labels, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

    def _create_train_op(self):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            if self.opt_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.opt_type == 'adam':
                self.optimizer = tc.opt.LazyAdamOptimizer(self.lr)
            elif self.opt_type == 'rprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.lr)
            elif self.opt_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise NotImplementedError('Unsupported optimizer: {}'.format(self.opt_type))
            # self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.all_params), self.norm)
            # self.grads = tf.gradients(self.loss, self.all_params)
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = self.optimizer.apply_gradients(zip(capped_grads, variables),
                                                           global_step=self.global_step)
