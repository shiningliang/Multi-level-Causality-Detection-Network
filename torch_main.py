import os
import argparse
import logging
import random
import ujson as json
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
from preprocess.torch_preprocess import run_prepare
import models

from models.torch_TextCNN import TextCNN
from models.torch_TextRNN import TextRNN
from models.torch_DPCNN import TextCNNDeep

from utils.torch_util import get_batch, evaluate_batch, FocalLoss, draw_att, draw_curve, save_loss, save_metrics

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
    parser.add_argument('--multi', type=str, default=5,
                        help='times for experiment')
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
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--emb_dropout', type=float, default=0,
                                help='dropout keep rate')
    train_settings.add_argument('--layer_dropout', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_train', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--batch_eval', type=int, default=64,
                                help='dev batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
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
    model_settings.add_argument('--max_len', type=dict, default={'full': 128, 'pre': 64, 'alt': 8, 'cur': 64},
                                help='max length of sequence')
    model_settings.add_argument('--w2v_type', type=str, default='wiki',
                                help='type of the embeddings')
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
    model_settings.add_argument('--n_filter', type=int, default=100,
                                help='number of hidden units per layer (default: 256)')
    model_settings.add_argument('--n_class', type=int, default=2,
                                help='class size (default: 2)')
    model_settings.add_argument('--kmax_pooling', type=int, default=2,
                                help='top-K max pooling')
    model_settings.add_argument('--dp_blocks', type=int, default=3,
                                help='dpcnn block size (default: 3)')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--task', default='training',
                               help='the task name')
    path_settings.add_argument('--model', default='MCDN',
                               help='the model name')
    path_settings.add_argument('--train_file', default='altlex_train_bootstrapped.tsv',
                               help='the train file name')
    path_settings.add_argument('--valid_file', default='altlex_gold.tsv',
                               help='the valid file name')
    path_settings.add_argument('--test_file', default='altlex_test.tsv',
                               help='the test file name')
    path_settings.add_argument('--transfer_file1', default='2010_random_filtered.json',
                               help='the transfer file name')
    path_settings.add_argument('--transfer_file2', default='2010_full_filtered.json',
                               help='the transfer file name')
    path_settings.add_argument('--raw_dir', default='data/raw_data/',
                               help='the dir to store raw data')
    path_settings.add_argument('--processed_dir', default='data/processed_data/torch',
                               help='the dir to store prepared data')
    path_settings.add_argument('--outputs_dir', default='outputs/',
                               help='the dir for outputs')
    path_settings.add_argument('--model_dir', default='models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='results/',
                               help='the dir to store the results')
    path_settings.add_argument('--pics_dir', default='pics/',
                               help='the dir to store the pictures')
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
    with open(file_paths.train_record_file, 'rb') as fh:
        train_file = pkl.load(fh)
    fh.close()
    logger.info('Loading valid file...')
    with open(file_paths.valid_record_file, 'rb') as fh:
        valid_file = pkl.load(fh)
    fh.close()
    logger.info('Loading train meta...')
    train_meta = load_json(file_paths.train_meta)
    logger.info('Loading valid meta...')
    valid_meta = load_json(file_paths.valid_meta)
    logger.info('Loading token embeddings...')
    with open(file_paths.token_emb_file, 'rb') as fh:
        token_embeddings = pkl.load(fh)
    fh.close()
    train_num = train_meta['total']
    valid_num = valid_meta['total']

    logger.info('Loading shape meta...')
    logger.info('Num train data {} valid data {}'.format(train_num, valid_num))

    args.dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    records = np.zeros((args.multi, 7), dtype=np.float)
    best_sum = -1
    for i in range(1, args.multi + 1):
        logger.info('Initialize the model...')
        model = getattr(models, args.model)(token_embeddings, args, logger).to(device=args.device)
        # model = TCN(token_embeddings, args.max_len['full'], args.n_class, n_channel=[args.n_filter] * args.n_level,
        #             n_kernel=args.n_kernel, n_block=args.n_block, n_head=args.n_head, dropout=dropout, logger=logger).
        #             to(device=args.device)
        # model = BiGRU(token_embeddings, args.max_len['full'], args.n_class, args.n_hidden, args.n_layer, args.n_block,
        #               args.n_head, args.is_sinusoid, args.is_ffn, dropout, logger).to(device=args.device)
        # model = TextCNN(token_embeddings, args.max_len, args.n_class, args.n_kernels, args.n_filter, args.is_pos,
        #                 args.is_sinusoid, args.dropout, logger).to(device=args.device)
        # model = TextCNNDeep(token_embeddings, args.max_len, args.n_class, args.n_kernels, args.n_filter,
        #                     args.dropout, logger).to(device=args.device)
        # model = TextRNN(token_embeddings, args.n_class, args.n_hidden, args.n_layer, args.kmax_pooling,
        #                 args.is_pos, args.is_sinusoid, args.dropout, logger).to(device=args.device)
        lr = args.lr
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=args.patience, verbose=True)
        logger.info('Turn {}'.format(i))
        max_acc, max_p, max_r, max_f, max_roc, max_prc, max_sum, max_epoch = np.zeros(8)
        FALSE, ROC, PRC = {}, {}, {}
        train_loss, valid_loss = [], []
        is_best = False
        for ep in range(1, args.epochs + 1):
            logger.info('Training the model for epoch {}'.format(ep))
            avg_loss = train_one_epoch(model, optimizer, train_num, train_file, args, logger)
            train_loss.append(avg_loss)
            logger.info('Epoch {} AvgLoss {}'.format(ep, avg_loss))

            logger.info('Evaluating the model for epoch {}'.format(ep))
            eval_metrics, fpr, tpr, precision, recall = evaluate_batch(model, valid_num, args.batch_eval, valid_file,
                                                                       args.device, args.is_fc, 'valid', logger)
            valid_loss.append(eval_metrics['loss'])
            logger.info('Valid Loss - {}'.format(eval_metrics['loss']))
            logger.info('Valid Acc - {}'.format(eval_metrics['acc']))
            logger.info('Valid Precision - {}'.format(eval_metrics['precision']))
            logger.info('Valid Recall - {}'.format(eval_metrics['recall']))
            logger.info('Valid F1 - {}'.format(eval_metrics['f1']))
            logger.info('Valid AUCROC - {}'.format(eval_metrics['auc_roc']))
            logger.info('Valid AUCPRC - {}'.format(eval_metrics['auc_prc']))
            valid_sum = eval_metrics['auc_roc'] + eval_metrics['auc_prc'] + eval_metrics['f1']
            if valid_sum > max_sum:
                max_acc = eval_metrics['acc']
                max_p = eval_metrics['precision']
                max_r = eval_metrics['recall']
                max_f = eval_metrics['f1']
                max_roc = eval_metrics['auc_roc']
                max_prc = eval_metrics['auc_prc']
                max_sum = valid_sum
                max_epoch = ep
                FALSE = {'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']}
                ROC = {'FPR': fpr, 'TPR': tpr}
                PRC = {'PRECISION': precision, 'RECALL': recall}
                # torch.save(model, os.path.join(args.model_dir, 'model.pth'))
                if max_sum > best_sum:
                    best_sum = max_sum
                    is_best = True
                    torch.save(model.state_dict(),
                               os.path.join(args.model_dir, 'model_turn' + str(i) + '_epoch' + str(ep) + '.bin'))

            scheduler.step(metrics=eval_metrics['f1'])
            random.shuffle(train_file)

        logger.info('Max Acc - {}'.format(max_acc))
        logger.info('Max Precision - {}'.format(max_p))
        logger.info('Max Recall - {}'.format(max_r))
        logger.info('Max F1 - {}'.format(max_f))
        logger.info('Max ROC - {}'.format(max_roc))
        logger.info('Max PRC - {}'.format(max_prc))
        logger.info('Max Epoch - {}'.format(max_epoch))
        logger.info('Max Sum - {}'.format(max_sum))

        if is_best:
            dump_json(os.path.join(args.result_dir, 'FALSE_valid.json'), FALSE)
            dump_json(os.path.join(args.result_dir, 'ROC_valid.json'), ROC)
            dump_json(os.path.join(args.result_dir, 'PRC_valid.json'), PRC)
            save_loss(train_loss, valid_loss, args.result_dir)
            draw_curve(ROC['FPR'], ROC['TPR'], PRC['PRECISION'], PRC['RECALL'], args.pics_dir)

        records[i - 1] = [max_acc, max_p, max_r, max_f, max_roc, max_prc, max_epoch]

    save_metrics(records, args.result_dir)


def evaluate(args, file_paths):
    logger = logging.getLogger('Causality')
    logger.info('Loading valid file...')
    with open(file_paths.valid_record_file, 'rb') as fh:
        valid_file = pkl.load(fh)
    fh.close()
    logger.info('Loading test file...')
    with open(file_paths.test_record_file, 'rb') as fh:
        test_file = pkl.load(fh)
    fh.close()
    logger.info('Loading valid meta...')
    with open(file_paths.valid_meta, 'r') as fh:
        valid_meta = json.load(fh)
    fh.close()
    logger.info('Loading test meta...')
    with open(file_paths.test_meta, 'r') as fh:
        test_meta = json.load(fh)
    fh.close()
    logger.info('Loading id to token file...')
    with open(file_paths.id2token_file, 'r') as fh:
        id2token_file = json.load(fh)
    fh.close()
    logger.info('Loading token embeddings...')
    with open(file_paths.token_emb_file, 'rb') as fh:
        token_embeddings = pkl.load(fh)
    fh.close()
    valid_num = valid_meta['total']
    test_num = test_meta['total']

    logger.info('Loading shape meta...')
    logger.info('Num valid data {} test data {}'.format(valid_num, test_num))

    # model = torch.load(os.path.join(args.model_dir, 'model.pth'))
    args.dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    model = getattr(models, args.model)(token_embeddings, args, logger).to(device=args.device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.bin')))

    eval_metrics, fpr, tpr, precision, recall = evaluate_batch(model, valid_num, args.batch_eval, valid_file,
                                                               args.device, args.is_fc, 'eval', logger)
    logger.info('Eval Loss - {}'.format(eval_metrics['loss']))
    logger.info('Eval Acc - {}'.format(eval_metrics['acc']))
    logger.info('Eval Precision - {}'.format(eval_metrics['precision']))
    logger.info('Eval Recall - {}'.format(eval_metrics['recall']))
    logger.info('Eval F1 - {}'.format(eval_metrics['f1']))
    logger.info('Eval AUCROC - {}'.format(eval_metrics['auc_roc']))
    logger.info('Eval AUCPRC - {}'.format(eval_metrics['auc_prc']))

    if args.model == 'MCDN' or args.model == 'TB':
        draw_att(model, test_num, args.batch_eval, test_file, args.device, id2token_file,
                 args.pics_dir, args.n_block, args.n_head, logger)

    FALSE = {'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']}
    ROC = {'FPR': fpr, 'TPR': tpr}
    PRC = {'PRECISION': precision, 'RECALL': recall}

    dump_json(os.path.join(args.result_dir, 'FALSE_transfer.json'), FALSE)
    dump_json(os.path.join(args.result_dir, 'ROC_transfer.json'), ROC)
    dump_json(os.path.join(args.result_dir, 'PRC_transfer.json'), PRC)
    draw_curve(ROC['FPR'], ROC['TPR'], PRC['PRECISION'], PRC['RECALL'], args.pics_dir)


def dump_json(file_path, obj):
    with open(file_path, 'w') as f:
        f.write(json.dumps(obj) + '\n')
    f.close()


def load_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    f.close()
    return obj


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
    args.processed_dir = os.path.join(args.processed_dir, args.task, str(args.max_len['full']))
    args.model_dir = os.path.join(args.outputs_dir, args.task, args.model, args.model_dir)
    args.result_dir = os.path.join(args.outputs_dir, args.task, args.model, args.result_dir)
    args.pics_dir = os.path.join(args.outputs_dir, args.task, args.model, args.pics_dir)
    args.summary_dir = os.path.join(args.outputs_dir, args.task, args.model, args.summary_dir)
    for dir_path in [args.raw_dir, args.processed_dir, args.model_dir, args.result_dir, args.pics_dir,
                     args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    class FilePaths(object):
        def __init__(self, w2v_type):
            # 运行记录文件
            self.train_record_file = os.path.join(args.processed_dir, 'train.pkl')
            self.valid_record_file = os.path.join(args.processed_dir, 'valid.pkl')
            self.test_record_file = os.path.join(args.processed_dir, 'test.pkl')
            self.transfer_record_file1 = os.path.join(args.processed_dir, 'transfer1.pkl')
            self.transfer_record_file2 = os.path.join(args.processed_dir, 'transfer2.pkl')
            # 计数文件
            self.train_meta = os.path.join(args.processed_dir, 'train_meta.json')
            self.valid_meta = os.path.join(args.processed_dir, 'valid_meta.json')
            self.test_meta = os.path.join(args.processed_dir, 'test_meta.json')
            self.transfer_meta1 = os.path.join(args.processed_dir, 'transfer_meta1.json')
            self.transfer_meta2 = os.path.join(args.processed_dir, 'transfer_meta2.json')
            self.shape_meta = os.path.join(args.processed_dir, 'shape_meta.json')

            self.train_annotation = os.path.join(args.processed_dir, 'train_annotations.txt')
            self.valid_annotation = os.path.join(args.processed_dir, 'valid_annotations.txt')
            self.test_annotation = os.path.join(args.processed_dir, 'test_annotations.txt')

            self.corpus_file = os.path.join(args.processed_dir, 'corpus.txt')
            self.token_emb_file = os.path.join(args.processed_dir, 'token_emb.pkl')
            self.token2id_file = os.path.join(args.processed_dir, 'token2id.json')
            self.id2token_file = os.path.join(args.processed_dir, 'id2token.json')

            if w2v_type == 'wiki':
                self.w2v_file = './data/processed_data/wiki.en.pkl'
            elif w2v_type == 'google':
                self.w2v_file = './data/processed_data/google.news.pkl'
            elif w2v_type == 'glove6':
                self.w2v_file = './data/processed_data/glove.6B.pkl'
            elif w2v_type == 'glove840':
                self.w2v_file = './data/processed_data/glove.840B.pkl'

    file_paths = FilePaths(args.w2v_type)
    if args.prepare:
        run_prepare(args)
    if args.train:
        train(args, file_paths)
    if args.evaluate:
        evaluate(args, file_paths)


if __name__ == '__main__':
    run()
