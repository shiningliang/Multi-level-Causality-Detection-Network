import os
import argparse
import logging
import ujson as json
import numpy as np
import tensorflow as tf
from preprocess import run_prepare
from .models.BiLSTM import BasicBiLSTM
from .models.CNN import BasicCNN
from util import get_record_parser, evaluate_batch, get_batch_dataset, get_dataset
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='tensorflow')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Causality identification on AltLex')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--filter_sizes', type=list, default=[3, 4, 5],
                                help='size of the filters')
    model_settings.add_argument('--num_filters', type=int, default=200,
                                help='num of the filters')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--layer_num', type=int, default=1,
                                help='num of layers')
    model_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')
    model_settings.add_argument('--capacity', type=int, default=150000,
                                help='Batch size of data set shuffle')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--raw_dir', default='data/raw_data',
                               help='the dir to store raw data')
    path_settings.add_argument('--processed_dir', default='data/processed_data',
                               help='the dir to store processed data')
    path_settings.add_argument('--model_dir', default='checkpoints/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def train(args, file_paths, max_len):
    logger = logging.getLogger('altlex')
    logger.info('Loading token embeddings...')
    with open(file_paths.token_emb_file, 'rb') as fh:
        token_embeddings = np.array(json.load(fh), dtype=np.float32)
    logger.info('Loading test eval file...')
    with open(file_paths.test_eval_file, "r") as fh:
        test_eval_file = json.load(fh)
    logger.info('Loading train meta...')
    with open(file_paths.train_meta, "r") as fh:
        train_meta = json.load(fh)
    logger.info('Loading test meta...')
    with open(file_paths.test_meta, "r") as fh:
        test_meta = json.load(fh)
    train_total = train_meta['total']
    test_total = test_meta['total']
    logger.info('Num of train examples {}'.format(train_total))
    logger.info('Num of test examples {}'.format(test_total))

    parser = get_record_parser(args, max_len)
    train_dataset = get_batch_dataset(file_paths.train_record_file, parser, args)
    test_dataset = get_dataset(file_paths.test_record_file, parser, args)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()
    logger.info('Initialize the model...')
    model = BasicCNN(args, iterator, max_len, token_embeddings, trainable=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    max_train_acc, max_test_acc, max_p, max_r, max_f = 0, 0, 0, 0, 0
    with tf.Session(config=sess_config) as sess:
        # writer = tf.summary.FileWriter(args.summary_dir)
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(test_iterator.string_handle())
        # max_acc, min_mse = 0, 1e20
        log_every_n_batch, n_batch_loss, n_batch_acc = 50, 0, 0
        for epoch in range(1, args.epochs + 1):
            logger.info('Training the model for epoch {}'.format(epoch))
            train_loss, train_acc = [], []
            # lr_decay = 0.9 ** max(epoch - 5, 0)
            sess.run(tf.assign(model.lr, tf.constant(args.learning_rate, dtype=tf.float32)))
            sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))

            # train
            for i in range(train_total // args.batch_size + 1):
                loss, acc, train_op = sess.run([model.loss, model.accuracy, model.train_op],
                                               feed_dict={handle: train_handle})
                bitx = i + 1
                n_batch_loss += loss
                n_batch_acc += acc
                if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                    logger.info('Batch {} to {} Average loss {} Average acc {}'.
                                format(bitx - log_every_n_batch + 1,
                                       bitx, n_batch_loss / log_every_n_batch, n_batch_acc / log_every_n_batch))
                    n_batch_loss = 0
                    n_batch_acc = 0
                train_loss.append(loss)
                train_acc.append(acc)
            train_loss = np.mean(train_loss)
            train_acc = np.mean(train_acc)
            if max_train_acc < train_acc:
                max_train_acc = train_acc
            logger.info('Epoch {} Average loss {} Average acc {}'.format(epoch, train_loss, train_acc))
            # loss_sum = tf.Summary(value=[tf.Summary.Value(tag='model/loss', simple_value=train_loss), ])
            # writer.add_summary(loss_sum, epoch)

            # eval
            sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
            logger.info('Evaluating the model after epoch {}'.format(epoch))
            eval_loss, eval_acc, precision, recall, f1, summ = evaluate_batch(model, test_total // args.batch_size + 1,
                                                                              test_eval_file, sess, 'test', handle,
                                                                              dev_handle)
            logger.info('Dev Loss {}'.format(eval_loss))
            logger.info('Dev Acc: {}'.format(eval_acc))
            logger.info('Dev Precision: {}'.format(precision))
            logger.info('Dev Recall: {}'.format(recall))
            logger.info('Dev F-measure: {}'.format(f1))
            if eval_acc > max_test_acc:
                max_test_acc = eval_acc
            if precision > max_p:
                max_p = precision
            if recall > max_r:
                max_r = recall
            if f1 > max_f:
                max_f = f1
            # for s in summ:
            #     writer.add_summary(s, epoch)
            # writer.flush()
            # filename = os.path.join(args.model_dir, "model_{}.ckpt".format(epoch))
            # if eval_acc > max_acc and eval_mse < min_mse:
            #     max_acc = eval_acc
            #     min_mse = eval_mse
            #     saver.save(sess, filename)
        logger.info('Max Acc {} Max Precision {} Max Recall {} Max F1 {}'.format(max_test_acc, max_p, max_r, max_f))


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger('altlex')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 是否存储日志
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('Preparing the directories...')
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    class FilePaths(object):
        def __init__(self):
            # 运行记录文件
            self.train_record_file = os.path.join(args.processed_dir, 'train.tfrecords')
            # self.dev_record_file = os.path.join(args.preprocessed_dir, 'dev.tfrecords')
            self.test_record_file = os.path.join(args.processed_dir, 'test.tfrecords')
            # 评估文件
            self.train_eval_file = os.path.join(args.processed_dir, 'train_eval.json')
            # self.dev_eval_file = os.path.join(args.preprocessed_dir, 'dev_eval.json')
            self.test_eval_file = os.path.join(args.processed_dir, 'test_eval.json')
            # 计数文件
            self.train_meta = os.path.join(args.processed_dir, 'train_meta.json')
            # self.dev_meta = os.path.join(args.preprocessed_dir, 'dev_meta.json')
            self.test_meta = os.path.join(args.processed_dir, 'test_meta.json')
            self.shape_meta = os.path.join(args.processed_dir, 'shape_meta.json')

            self.corpus_file = os.path.join(args.processed_dir, 'corpus.txt')
            self.w2v_file = os.path.join(args.processed_dir, 'wiki_en_model.pkl')
            self.token_emb_file = os.path.join(args.processed_dir, 'token_emb.json')
            self.token2id_file = os.path.join(args.processed_dir, 'token2id.json')

    file_paths = FilePaths()
    # max_seq_len, index_dim = 0, 0
    if args.prepare:
        run_prepare(args, file_paths)
    if args.train:
        with open(file_paths.shape_meta, 'r') as fh:
            shape_meta = json.load(fh)
        train(args, file_paths, shape_meta['max_len'])


if __name__ == '__main__':
    run()
