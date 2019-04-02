import os
import argparse
import logging
import random
import ujson as json
import pickle as pkl
import numpy as np
import pandas as pd
import fastai
from fastai import *
from fastai.text import *
import torch
import torch.optim as optim
from ulmfit_preprocess import run_prepare
from torch_util import get_batch, evaluate_batch, FocalLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Causality')
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
    parser.add_argument('--seed', type=int, default=23333,
                        help='random seed (default: 23333)')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--disable_cuda', action='store_true',
                                help='Disable CUDA')
    train_settings.add_argument('--lr', type=float, default=0.0001,
                                help='learning rate')
    train_settings.add_argument('--clip', type=float, default=0.35,
                                help='gradient clip, -1 means no clip (default: 0.35)')
    train_settings.add_argument('--weight_decay', type=float, default=0.0003,
                                help='weight decay')
    train_settings.add_argument('--emb_dropout', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--layer_dropout', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_train', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--batch_eval', type=int, default=64,
                                help='dev batch size')
    train_settings.add_argument('--epochs', type=int, default=20,
                                help='train epochs')
    train_settings.add_argument('--optim', default='Adam',
                                help='optimizer type')
    train_settings.add_argument('--patience', type=int, default=2,
                                help='num of epochs for train patients')
    train_settings.add_argument('--period', type=int, default=1000,
                                help='period to save batch loss')
    train_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--max_len', type=dict, default={'full': 200, 'pre': 100, 'alt': 10, 'cur': 200},
                                help='max length of sequence')
    model_settings.add_argument('--n_emb', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--n_hidden', type=int, default=64,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--n_layer', type=int, default=2,
                                help='num of layers')
    model_settings.add_argument('--is_fc', type=bool, default=False,
                                help='whether to use focal loss')
    model_settings.add_argument('--is_atten', type=bool, default=False,
                                help='whether to use self attention')
    model_settings.add_argument('--is_gated', type=bool, default=False,
                                help='whether to use gated conv')
    model_settings.add_argument('--n_block', type=int, default=4,
                                help='attention block size (default: 2)')
    model_settings.add_argument('--n_head', type=int, default=4,
                                help='attention head size (default: 2)')
    model_settings.add_argument('--is_pos', type=bool, default=False,
                                help='whether to use position embedding')
    model_settings.add_argument('--is_sinusoid', type=bool, default=True,
                                help='whether to use sinusoid position embedding')
    model_settings.add_argument('--is_ffn', type=bool, default=True,
                                help='whether to use point-wise ffn')
    model_settings.add_argument('--n_kernel', type=int, default=3,
                                help='kernel size (default: 3)')
    model_settings.add_argument('--n_kernels', type=int, default=[2, 3, 4],
                                help='kernels size (default: 2, 3, 4)')
    model_settings.add_argument('--n_level', type=int, default=6,
                                help='# of levels (default: 10)')
    model_settings.add_argument('--n_filter', type=int, default=50,
                                help='number of hidden units per layer (default: 256)')
    model_settings.add_argument('--n_class', type=int, default=2,
                                help='class size (default: 2)')
    model_settings.add_argument('--kmax_pooling', type=int, default=2,
                                help='top-K max pooling')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--task', default='bootstrapped',
                               help='the task name')
    path_settings.add_argument('--model', default='ULMFIT',
                               help='the model name')
    path_settings.add_argument('--train_file', default='altlex_train.tsv',
                               help='the train file name')
    path_settings.add_argument('--valid_file', default='altlex_dev.tsv',
                               help='the valid file name')
    path_settings.add_argument('--test_file', default='altlex_gold.tsv',
                               help='the test file name')
    path_settings.add_argument('--raw_dir', default='data/raw_data/',
                               help='the dir to store raw data')
    path_settings.add_argument('--processed_dir', default='data/processed_data/fastai',
                               help='the dir to store prepared data')
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


def train_one_epoch(model, optimizer, train_num, train_file, args, logger):
    model.train()
    train_loss = []
    n_batch_loss = 0
    weight = torch.from_numpy(np.array([0.2, 0.8], dtype=np.float32)).to(args.device)
    for batch_idx, batch in enumerate(range(0, train_num, args.batch_train)):
        start_idx = batch
        end_idx = start_idx + args.batch_train
        # sentences, cau_labels, seq_lens = get_batch(train_file[start_idx:end_idx], args.device)
        tokens, tokens_pre, tokens_alt, tokens_cur, cau_labels, seq_lens, _ = get_batch(train_file[start_idx:end_idx],
                                                                                        args.device)

        optimizer.zero_grad()
        outputs = model(tokens, tokens_pre, tokens_alt, tokens_cur, seq_lens)
        # outputs = model(sentences)
        # loss = compute_loss(logits=outputs, target=labels, length=seq_lens)
        if args.is_fc:
            criterion = FocalLoss(gamma=2, alpha=0.75)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight)
        loss = criterion(outputs, cau_labels)
        # params = model.state_dict()
        # l2_reg = torch.autograd.Variable(torch.FloatTensor(1), requires_grad=True).cuda()
        # l2_reg = l2_reg + params['linear.weight'].norm(2) + params['linear.bias'].norm(2)
        # loss += l2_reg * args.weight_decay
        loss.backward()
        if args.clip > 0:
            # 梯度裁剪，输入是(NN参数，最大梯度范数，范数类型=2)，一般默认为L2范数
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        n_batch_loss += loss.item()
        bidx = batch_idx + 1
        if bidx % args.period == 0:
            logger.info('AvgLoss batch [{} {}] - {}'.format(bidx - args.period + 1, bidx, n_batch_loss / args.period))
            n_batch_loss = 0
        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    return avg_train_loss
    # avg_train_acc = np.mean(train_acc)
    # logger.info('Epoch {} Average Loss {} Average Acc {}'.format(ep, avg_train_loss, avg_train_acc))
    # loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=avg_train_loss), ])
    # acc_sum = tf.Summary(value=[tf.Summary.Value(tag="model/acc", simple_value=avg_train_acc), ])
    # writer.add_summary(loss_sum, epoch)
    # writer.add_summary(acc_sum, epoch)


def train(args, file_paths):
    logger = logging.getLogger('Causality')
    logger.info('Loading train file...')
    df_trn = pd.read_csv(file_paths.train_file)
    logger.info('Loading valid file...')
    df_val = pd.read_csv(file_paths.test_file)
    logger.info('Loading train meta...')
    with open(file_paths.train_meta, 'r') as fh:
        train_meta = json.load(fh)
    fh.close()
    logger.info('Loading valid meta...')
    with open(file_paths.test_meta, 'r') as fh:
        valid_meta = json.load(fh)
    fh.close()

    train_num = train_meta['total']
    valid_num = valid_meta['total']
    logger.info('Num train data {} Num valid data {}'.format(train_num, valid_num))

    dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    logger.info('Initialize the model...')
    max_acc, max_p, max_r, max_f, max_sum, max_epoch = 0, 0, 0, 0, 0, 0
    FALSE = {}

    # Language model data
    data_lm = TextLMDataBunch.from_df(train_df=df_trn, valid_df=df_val, path='')
    # Classifier model data
    data_clas = TextClasDataBunch.from_df(path="", train_df=df_trn, valid_df=df_val, vocab=data_lm.train_ds.vocab,
                                          bs=32)

    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    print(learn.model)
    # train the learner object with learning rate = 1e-2
    learn.unfreeze()
    learn.fit_one_cycle(4, 1e-2)
    # save encoder
    learn.save_encoder('ft_enc')

    # use the data_clas object we created earlier to build a classifier with our fine-tuned encoder
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    print(learn.model)
    learn.load_encoder('ft_enc')
    learn.freeze_to(-2)
    # learn.unfreeze()
    learn.fit_one_cycle(4, 1e-2)
    # get predictions
    preds, targets = learn.get_preds()
    predictions = np.argmax(preds, axis=1)
    print(pd.crosstab(predictions, targets))
    pred = predictions.numpy()
    targ = targets.numpy()
    acc = accuracy_score(targ, pred)
    precision = precision_score(targ, pred)
    recall = recall_score(targ, pred)
    f1 = f1_score(targ, pred)
    logger.info('Valid Acc - {}'.format(acc))
    logger.info('Valid Precision - {}'.format(precision))
    logger.info('Valid Recall - {}'.format(recall))
    logger.info('Valid F1 - {}'.format(f1))

    # for ep in range(1, args.epochs + 1):
    #     logger.info('Training the model for epoch {}'.format(ep))
    #     avg_loss = train_one_epoch(model, optimizer, train_num, train_file, args, logger)
    #     logger.info('Epoch {} AvgLoss {}'.format(ep, avg_loss))
    #
    #     logger.info('Evaluating the model for epoch {}'.format(ep))
    #     eval_metrics = evaluate_batch(model, valid_num, args.batch_eval, valid_file, args.device, args.is_fc,
    #                                   'valid', logger)
    #     logger.info('Valid Loss - {}'.format(eval_metrics['loss']))
    #     logger.info('Valid Acc - {}'.format(eval_metrics['acc']))
    #     logger.info('Valid Precision - {}'.format(eval_metrics['precision']))
    #     logger.info('Valid Recall - {}'.format(eval_metrics['recall']))
    #     logger.info('Valid F1 - {}'.format(eval_metrics['f1']))
    #     max_acc = max((eval_metrics['acc'], max_acc))
    #     max_p = max(eval_metrics['precision'], max_p)
    #     max_r = max(eval_metrics['recall'], max_r)
    #     max_f = max(eval_metrics['f1'], max_f)
    #     valid_sum = eval_metrics['precision'] + eval_metrics['recall'] + eval_metrics['f1']
    #     if valid_sum > max_sum:
    #         max_sum = valid_sum
    #         max_epoch = ep
    #         FALSE = {'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']}
    #     scheduler.step(metrics=eval_metrics['f1'])
    #     random.shuffle(train_file)
    #
    # logger.info('Max Acc - {}'.format(max_acc))
    # logger.info('Max Precision - {}'.format(max_p))
    # logger.info('Max Recall - {}'.format(max_r))
    # logger.info('Max F1 - {}'.format(max_f))
    # logger.info('Max Epoch - {}'.format(max_epoch))
    # logger.info('Max Sum - {}'.format(max_sum))
    # with open(os.path.join(args.result_dir, 'FALSE.json'), 'w') as f:
    #     f.write(json.dumps(FALSE) + '\n')
    # f.close()


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    logger = logging.getLogger('Causality')
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
    torch.manual_seed(args.seed)
    args.device = None
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    logger.info('Preparing the directories...')
    args.processed_dir = os.path.join(args.processed_dir, args.task)
    args.model_dir = os.path.join(args.outputs_dir, args.task, args.model, args.model_dir)
    args.result_dir = os.path.join(args.outputs_dir, args.task, args.model, args.result_dir)
    args.summary_dir = os.path.join(args.outputs_dir, args.task, args.model, args.summary_dir)
    for dir_path in [args.raw_dir, args.processed_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    class FilePaths(object):
        def __init__(self):
            # 运行记录文件
            self.train_file = os.path.join(args.processed_dir, 'train_training.csv')
            self.valid_file = os.path.join(args.processed_dir, 'valid.csv')
            self.test_file = os.path.join(args.processed_dir, 'test.csv')
            # 计数文件
            self.train_meta = os.path.join(args.processed_dir, 'train_meta.json')
            self.valid_meta = os.path.join(args.processed_dir, 'valid_meta.json')
            self.test_meta = os.path.join(args.processed_dir, 'test_meta.json')
            self.shape_meta = os.path.join(args.processed_dir, 'shape_meta.json')

            self.train_annotation = os.path.join(args.processed_dir, 'train_annotations.txt')
            self.valid_annotation = os.path.join(args.processed_dir, 'valid_annotations.txt')
            self.test_annotation = os.path.join(args.processed_dir, 'test_annotations.txt')

            self.corpus_file = os.path.join(args.processed_dir, 'corpus.txt')
            self.w2v_file = './data/processed_data/wiki.en.pkl'
            self.token_emb_file = os.path.join(args.processed_dir, 'token_emb.pkl')
            self.token2id_file = os.path.join(args.processed_dir, 'token2id.json')

    file_paths = FilePaths()
    if args.prepare:
        # max_seq_len, index_dim = run_prepare(args, file_paths)
        run_prepare(args, file_paths)
        # with open(file_paths.shape_meta, 'wb') as fh:
        #     pkl.dump({'max_len': max_seq_len, 'dim': index_dim}, fh)
        # fh.close()
    if args.train:
        train(args, file_paths)


if __name__ == '__main__':
    run()
