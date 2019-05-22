import os
import argparse
import logging
import ujson as json
import numpy as np
import tensorflow as tf
from preprocess import run_prepare
from models.SelfAttentiveSentenceEmbedding import SelfAttentive
from utils.tf_util import get_record_parser, evaluate_batch, get_batch_dataset, get_dataset, print_metrics
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
    parser.add_argument('--build', action='store_true',
                        help='whether to build word dict and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--num_steps', type=int, default=16000,
                                help='num of step')
    train_settings.add_argument('--period', type=int, default=400,
                                help='period to save batch loss')
    train_settings.add_argument('--checkpoint', type=int, default=1600,
                                help='checkpoint for evaluation')
    train_settings.add_argument('--eval_num_batches', type=int, default=10,
                                help='num of batches for evaluation')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--lr', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0.0001,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--global_norm', type=int, default=5,
                                help='clip gradient norm')
    train_settings.add_argument('--train_batch', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--valid_batch', type=int, default=32,
                                help='dev batch size')
    train_settings.add_argument('--epochs', type=int, default=20,
                                help='train epochs')
    train_settings.add_argument('--patience', type=int, default=2,
                                help='num of epochs for train patients')
    train_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')
    train_settings.add_argument('--capacity', type=int, default=150000,
                                help='Batch size of data set shuffle')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--max_len', type=int, default=200,
                                help='max sentence length')
    model_settings.add_argument('--n_emb', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--filter_sizes', type=list, default=[3, 4, 5],
                                help='size of the filters')
    model_settings.add_argument('--num_filters', type=int, default=200,
                                help='num of the filters')
    model_settings.add_argument('--n_hidden', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--n_layer', type=int, default=1,
                                help='num of layers')
    model_settings.add_argument('--sa_da', type=int, default=128,
                                help='dim of self attentive da')
    model_settings.add_argument('--sa_r', type=int, default=32,
                                help='dim of self attentive r')
    model_settings.add_argument('--pos_weight', type=int, default=2,
                                help='positive example weight')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--task', default='bootstrapped',
                               help='the task name')
    path_settings.add_argument('--raw_dir', default='data/raw_data',
                               help='the dir to store raw data')
    path_settings.add_argument('--train_file', default='altlex_train_bootstrapped.tsv',
                               help='the train file name')
    path_settings.add_argument('--valid_file', default='altlex_dev.tsv',
                               help='the valid file name')
    path_settings.add_argument('--test_file', default='altlex_gold.tsv',
                               help='the test file name')
    path_settings.add_argument('--processed_dir', default='data/processed_data/tf',
                               help='the dir to store processed data')
    path_settings.add_argument('--outputs_dir', default='outputs/',
                               help='the dir for outputs')
    path_settings.add_argument('--model_dir', default='models/',
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
    logger.info('Loading train eval file...')
    with open(file_paths.train_eval_file, 'r') as fh:
        train_eval_file = json.load(fh)
    with open(file_paths.test_eval_file, 'r') as fh:
        valid_eval_file = json.load(fh)
    logger.info('Loading train meta...')
    with open(file_paths.train_meta, 'r') as fh:
        train_meta = json.load(fh)
    logger.info('Loading valid meta...')
    with open(file_paths.test_meta, 'r') as fh:
        valid_meta = json.load(fh)
    train_total = train_meta['total']
    valid_total = valid_meta['total']
    logger.info('Num of train examples {}'.format(train_total))
    logger.info('Num of test examples {}'.format(valid_total))

    parser = get_record_parser(max_len)
    train_dataset = get_batch_dataset(file_paths.train_record_file, parser, args)
    valid_dataset = get_dataset(file_paths.test_record_file, parser, args)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = valid_dataset.make_one_shot_iterator()
    logger.info('Initialize the model...')
    # model = BasicBiLSTM(args, iterator, max_len, token_embeddings, trainable=True)
    model = SelfAttentive(args, iterator, token_embeddings, logger)
    sess_config = tf.ConfigProto(intra_op_parallelism_threads=8,
                                 inter_op_parallelism_threads=8,
                                 allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    max_acc, max_p, max_r, max_f, max_sum, max_epoch = 0, 0, 0, 0, 0, 0
    f1_save, patience = 0, 0
    lr = args.lr
    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(args.summary_dir)
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.n_batch, tf.constant(args.train_batch, dtype=tf.int32)))
        for step in range(1, args.num_steps + 1):
            sess.run(tf.assign(model.global_step, tf.constant(step, dtype=tf.int32)))
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
            if step % args.period == 0:
                logger.info('Period point {} Loss {}'.format(step, loss))
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag='model/loss', simple_value=loss), ])
                writer.add_summary(loss_sum, step)

            if step % args.checkpoint == 0:
                logger.info('Evaluating the model for epoch {}'.format(step // args.checkpoint))
                sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
                train_metrics, summ = evaluate_batch(model, args.eval_num_batches, train_eval_file, sess, 'train',
                                                     handle, train_handle)
                print_metrics(train_metrics, logger, 'Train')
                for s in summ:
                    writer.add_summary(s, step)

                sess.run(tf.assign(model.n_batch, tf.constant(args.valid_batch, dtype=tf.int32)))
                valid_metrics, summ = evaluate_batch(model, valid_total // args.valid_batch, valid_eval_file, sess,
                                                     'valid', handle, valid_handle)
                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
                print_metrics(valid_metrics, logger, 'Valid')
                for s in summ:
                    writer.add_summary(s, step)
                writer.flush()
                f1 = valid_metrics['f1']
                if f1 > f1_save:
                    f1_save = f1
                    patience = 0
                else:
                    patience += 1
                if patience >= args.patience:
                    lr /= 2.0
                    logger.info('Learning rate reduced to {}'.format(lr))
                    f1_save = f1
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                max_acc = max(valid_metrics['acc'], max_acc)
                max_p = max(valid_metrics['precision'], max_p)
                max_r = max(valid_metrics['recall'], max_r)
                max_f = max(valid_metrics['f1'], max_f)
                valid_sum = valid_metrics['precision'] + valid_metrics['recall'] + valid_metrics['f1']
                if valid_sum > max_sum:
                    max_sum = valid_sum
                    max_epoch = step // args.checkpoint
                    # filename = os.path.join(args.model_dir, "model_{}.ckpt".format(global_step))
                    # saver.save(sess, filename)
        logger.info('Max Acc {} Max Precision {} Max Recall {} Max F1 {}'.format(max_acc, max_p, max_r, max_f))
        logger.info('Max epoch {}'.format(max_epoch))


def run():
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
    args.processed_dir = os.path.join(args.processed_dir, args.task)
    args.model_dir = os.path.join(args.outputs_dir, args.task, args.model, args.model_dir)
    args.result_dir = os.path.join(args.outputs_dir, args.task, args.model, args.result_dir)
    args.summary_dir = os.path.join(args.outputs_dir, args.task, args.model, args.summary_dir)
    for dir_path in [args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    class FilePaths(object):
        def __init__(self):
            # 运行记录文件
            self.train_record_file = os.path.join(args.processed_dir, 'train.tfrecords')
            self.valid_record_file = os.path.join(args.processed_dir, 'valid.tfrecords')
            self.test_record_file = os.path.join(args.processed_dir, 'test.tfrecords')
            # 评估文件
            self.train_eval_file = os.path.join(args.processed_dir, 'train_eval.json')
            self.valid_eval_file = os.path.join(args.processed_dir, 'valid_eval.json')
            self.test_eval_file = os.path.join(args.processed_dir, 'test_eval.json')
            # 计数文件
            self.train_meta = os.path.join(args.processed_dir, 'train_meta.json')
            self.valid_meta = os.path.join(args.processed_dir, 'valid_meta.json')
            self.test_meta = os.path.join(args.processed_dir, 'test_meta.json')
            self.shape_meta = os.path.join(args.processed_dir, 'shape_meta.json')

            self.corpus_file = os.path.join(args.processed_dir, 'corpus.txt')
            self.w2v_file = './data/processed_data/wiki_en_model.pkl'
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
