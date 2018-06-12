import tensorflow as tf
import logging
import time
from module import dense
from tensorflow.python.ops import array_ops


class BasicCNN(object):
    def __init__(self, args, batch, max_len, token_embeddings, trainable=True):
        # logging
        self.logger = logging.getLogger('basic cnn model')
        # basic config
        self.max_len = max_len
        self.embed_size = args.embed_size
        self.filter_sizes = args.filter_sizes
        self.num_filters = args.num_filters
        self.num_class = 2
        self.opt_type = args.optim
        self.dropout_keep_prob = args.dropout_keep_prob
        self.weight_decay = args.weight_decay
        self.is_train = trainable

        self.eid, self.token_ids, self.token_len, self.labels = batch.get_next()
        self.mask = tf.sequence_mask(self.token_len, self.max_len, dtype=tf.float32, name='masks')
        self.lr = tf.get_variable('lr', shape=[], dtype=tf.float32, trainable=False)
        self.is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)

        self._build_graph(token_embeddings)

    def _build_graph(self, token_embeddings):
        start_t = time.time()
        self._embed(token_embeddings)
        self._conv_pool()
        self._predict_label()
        self._compute_loss()
        self._compute_accuracy()
        # 选择优化算法
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))

    def _embed(self, token_embeddings):
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding', reuse=tf.AUTO_REUSE):
            word_embeddings = tf.get_variable('word_embeddings',
                                              initializer=tf.constant(token_embeddings, dtype=tf.float32),
                                              trainable=False)
            self.token_emb = tf.nn.embedding_lookup(word_embeddings, self.token_ids)
            self.token_emb = tf.expand_dims(self.token_emb, -1)

    def _conv_pool(self):
        with tf.variable_scope('conv_pool', reuse=tf.AUTO_REUSE):
            self.pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                filter_shape = [filter_size, self.embed_size, 1, self.num_filters]
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                # b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                # conv = tf.nn.conv2d(
                #     self.token_emb,
                #     W,
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name='conv')
                h = tf.contrib.layers.convolution2d(self.token_emb, self.num_filters, [filter_size, self.embed_size],
                                                    padding='VALID',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params={'decay': 0.9})
                # h = tf.nn.relu(conv)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")

                self.pooled_outputs.append(pooled)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(self.pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
        if self.is_train:
            self.h_pool_flat = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            # self.seq_encode = tf.reshape(self.seq_encode, [-1, 2 * self.hidden_size])

    def _predict_label(self):
        with tf.variable_scope('predict_labels', reuse=tf.AUTO_REUSE):
            self.label_dense_0 = tf.nn.relu(
                dense(self.h_pool_flat, hidden=int(self.num_filters_total / 2), scope='dense_0'))
            if self.is_train:
                self.label_dense_0 = tf.nn.dropout(self.label_dense_0, self.dropout_keep_prob)

            self.output = dense(self.label_dense_0, hidden=self.num_class, scope='output_labels')

    # def _focal_loss(self, prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    #     """Compute focal loss for predictions.
    #         Multi-labels Focal loss formula:
    #             FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
    #                  ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    #     Args:
    #      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
    #         num_classes] representing the predicted logits for each class
    #      target_tensor: A float tensor of shape [batch_size, num_anchors,
    #         num_classes] representing one-hot encoded classification targets
    #      weights: A float tensor of shape [batch_size, num_anchors]
    #      alpha: A scalar tensor for focal loss alpha hyper-parameter
    #      gamma: A scalar tensor for focal loss gamma hyper-parameter
    #     Returns:
    #         loss: A (scalar) tensor representing the value of the loss function
    #     """
    #     sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    #     zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    #
    #     # For positive prediction, only need consider front part loss, back part is 0;
    #     # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    #     pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    #
    #     # For negative prediction, only need consider back part loss, front part is 0;
    #     # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    #     neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    #     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
    #                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(
    #         tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    #     return tf.reduce_sum(per_entry_cross_ent)

    def _focal_loss(self, onehot_labels, cls_preds, alpha=0.75, gamma=4.0, name=None, scope=None):
        with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
            logits = tf.convert_to_tensor(cls_preds)
            onehot_labels = tf.convert_to_tensor(onehot_labels)

            precise_logits = tf.cast(logits, tf.float32) if (logits.dtype == tf.float16) else logits
            onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
            predictions = tf.nn.sigmoid(precise_logits)
            predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1. - predictions)
            # add small value to avoid 0
            epsilon = 1e-10
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1 - alpha_t)
            losses = tf.reduce_mean(
                tf.reduce_sum(-alpha_t * tf.pow(1. - predictions_pt, gamma) * tf.log(predictions_pt + epsilon),
                              name=name, axis=1))
            return losses

    def _compute_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.output, labels=tf.stop_gradient(tf.one_hot(self.labels, 2, axis=1))))
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
        """
        Selects the training algorithm and creates a train operation with it
        """
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
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
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss)