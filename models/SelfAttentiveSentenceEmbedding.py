import tensorflow as tf
import time
from modules.rnn_module import rnn, dense


class SelfAttentive(object):
    def __init__(self, args, batch, max_len, token_embeddings, logger, trainable=True):
        self.logger = logger
        # basic config
        self.max_len = max_len
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.da = args.sa_da
        self.r = args.sa_r
        self.num_class = 2
        self.opt_type = args.optim
        self.dropout_keep_prob = args.dropout_keep_prob
        self.weight_decay = args.weight_decay
        self.is_train = trainable

        self.eid, self.token_ids, self.token_len, self.labels = batch.get_next()
        # self.batch_size = self.eid.get_shape().as_list()[0]
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
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
        self._compute_accuracy()
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
            self.H, _ = rnn('bi-lstm', self.token_emb, self.token_len, self.hidden_size, self.layer_num)
        if self.is_train:
            self.H = tf.nn.dropout(self.H, self.dropout_keep_prob)

    def _cal_attention(self):
        with tf.variable_scope('cal_attention', reuse=tf.AUTO_REUSE):
            self.W_s1 = tf.get_variable('Ws1', shape=[2 * self.hidden_size, self.da], initializer=self.initializer)
            self.W_s1_H = tf.nn.tanh(tf.matmul(tf.reshape(self.H, [-1, 2 * self.hidden_size]), self.W_s1))
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
            # self.I = tf.reshape(tf.tile(tf.diag(tf.ones([self.r]), name='diag_identity'), [self.batch_size, 1]),
            #                     [self.batch_size, self.r, self.r])
            self.I = tf.diag(tf.ones([self.r]), name='diag_identity')

            self.penalized_term = tf.reduce_mean(tf.square(tf.norm(self.AA_T - self.I, ord='euclidean', axis=[1, 2])))

    def _predict_label(self):
        with tf.variable_scope('predict_labels', reuse=tf.AUTO_REUSE):
            self.flatten_M_T = tf.reshape(self.M_T, shape=[-1, self.r * 2 * self.hidden_size])
            self.label_dense_0 = tf.nn.relu(dense(self.flatten_M_T, hidden=2 * self.hidden_size, scope='dense_0'))
            if self.is_train:
                self.label_dense_0 = tf.nn.dropout(self.label_dense_0, self.dropout_keep_prob)

            self.output = dense(self.label_dense_0, hidden=self.num_class, scope='output_labels')

    def _compute_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.output, labels=tf.stop_gradient(tf.one_hot(self.labels, 2, axis=1))))
        self.loss += self.penalized_term
        # self.loss = self._focal_loss(tf.one_hot(self.labels, 2, axis=1), self.output)
        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _compute_accuracy(self):
        with tf.name_scope('accuracy'):
            self.pre_labels = tf.argmax(self.output, axis=1)
            correct_predictions = tf.equal(self.pre_labels, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

    def _create_train_op(self):
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            if self.opt_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.lr)
            elif self.opt_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.opt_type == 'rprop':
                self.optimizer = tf.train.RMSPropOptimizer(self.lr)
            elif self.opt_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise NotImplementedError('Unsupported optimizer: {}'.format(self.opt_type))
            # self.train_op = self.optimizer.minimize(self.loss)
            grads = self.optimizer.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, 5)
            self.train_op = self.optimizer.apply_gradients(zip(capped_grads, variables))
