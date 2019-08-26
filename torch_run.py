import os
import logging
import random
import pickle as pkl
import numpy as np
import torch
import torch.optim as optim
from preprocess.torch_preprocess import run_prepare
import models
from config import opt


from utils.torch_util import get_batch, evaluate_batch, FocalLoss, draw_att, draw_curve, load_json, dump_json, save_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


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


def train(args):
    logger = logging.getLogger('Causality')
    logger.info('Loading train file...')
    with open(args.train_record_file, 'rb') as fh:
        train_file = pkl.load(fh)
    fh.close()
    logger.info('Loading valid file...')
    with open(args.valid_record_file, 'rb') as fh:
        valid_file = pkl.load(fh)
    fh.close()
    logger.info('Loading train meta...')
    train_meta = load_json(args.train_meta)
    logger.info('Loading valid meta...')
    valid_meta = load_json(args.valid_meta)
    logger.info('Loading token embeddings...')
    with open(args.token_emb_file, 'rb') as fh:
        token_embeddings = pkl.load(fh)
    fh.close()
    train_num = train_meta['total']
    valid_num = valid_meta['total']

    logger.info('Loading shape meta...')
    logger.info('Num train data {} valid data {}'.format(train_num, valid_num))

    args.dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    logger.info('Initialize the model...')
    model = getattr(models, args.model)(token_embeddings, args, logger).to(device=args.device)
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=args.patience, verbose=True)
    # torch.backends.cudnn.benchmark = True
    max_acc, max_p, max_r, max_f, max_roc, max_prc, max_sum, max_epoch = np.zeros(8)
    FALSE, ROC, PRC = {}, {}, {}
    train_loss, valid_loss = [], []
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
        max_acc = max((eval_metrics['acc'], max_acc))
        max_p = max(eval_metrics['precision'], max_p)
        max_r = max(eval_metrics['recall'], max_r)
        max_f = max(eval_metrics['f1'], max_f)
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
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.bin'))

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

    dump_json(os.path.join(args.result_dir, 'FALSE_valid.json'), FALSE)
    dump_json(os.path.join(args.result_dir, 'ROC_valid.json'), ROC)
    dump_json(os.path.join(args.result_dir, 'PRC_valid.json'), PRC)
    save_loss(train_loss, valid_loss, args.result_dir)
    draw_curve(ROC['FPR'], ROC['TPR'], PRC['PRECISION'], PRC['RECALL'], args.pics_dir)


def evaluate(args):
    logger = logging.getLogger('Causality')
    logger.info('Loading valid file...')
    with open(args.valid_record_file, 'rb') as fh:
        valid_file = pkl.load(fh)
    fh.close()
    logger.info('Loading test file...')
    with open(args.test_record_file, 'rb') as fh:
        test_file = pkl.load(fh)
    fh.close()
    logger.info('Loading valid meta...')
    valid_meta = load_json(args.valid_meta)
    logger.info('Loading test meta...')
    test_meta = load_json(args.test_meta)
    logger.info('Loading id to token file...')
    id2token_file = load_json(args.id2token_file)
    logger.info('Loading token embeddings...')
    with open(args.token_emb_file, 'rb') as fh:
        token_embeddings = pkl.load(fh)
    fh.close()
    valid_num = valid_meta['total']
    test_num = test_meta['total']

    logger.info('Loading shape meta...')
    logger.info('Num valid data {} test data {}'.format(valid_num, test_num))

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


def run(**kwargs):
    """
    Prepares and runs the whole system.
    """
    opt._parse(kwargs)

    logger = logging.getLogger('Causality')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 是否存储日志
    if opt.log_path:
        file_handler = logging.FileHandler(opt.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    # logger.info('Running with args : {}'.format(opt))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(opt.seed)
    opt.device = None
    if torch.cuda.is_available() and not opt.disable_cuda:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')

    if opt.prepare:
        run_prepare(opt)
    if opt.train:
        train(opt)
    if opt.evaluate:
        evaluate(opt)


if __name__ == '__main__':
    import fire
    fire.Fire()
