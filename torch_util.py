import numpy as np
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
plt.switch_backend('agg')


def get_batch(samples, device):
    ids, tokens, tokens_pre, tokens_alt, tokens_cur, cau_labels, seq_lens = [], [], [], [], [], [], []
    for sample in samples:
        ids.append(sample['id'])
        tokens.append(sample['tokens'])
        tokens_pre.append(sample['tokens_pre'])
        tokens_alt.append(sample['tokens_alt'])
        tokens_cur.append(sample['tokens_cur'])
        cau_labels.append(sample['cau_label'])
        seq_lens.append(sample['length'])
    tokens = np.asarray(tokens, dtype=np.int64)
    tokens_pre = np.asarray(tokens_pre, dtype=np.int64)
    tokens_alt = np.asarray(tokens_alt, dtype=np.int64)
    tokens_cur = np.asarray(tokens_cur, dtype=np.int64)
    cau_labels = np.asarray(cau_labels, dtype=np.int64)
    seq_lens = np.asarray(seq_lens, dtype=np.int64)
    return torch.from_numpy(tokens).to(device), torch.from_numpy(tokens_pre).to(device), \
           torch.from_numpy(tokens_alt).to(device), torch.from_numpy(tokens_cur).to(device),\
           torch.from_numpy(cau_labels).to(device), torch.from_numpy(seq_lens).to(device), ids


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))

    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length, weight):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat, dim=len(logits_flat.size()) - 1)
    log_probs_flat = torch.mul(log_probs_flat, weight)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()

    return loss


def evaluate_batch(model, data_num, batch_size, eval_file, device, is_fc, data_type, logger):
    losses = []
    fp, fn = [], []
    causality_preds, causality_labels = [], []
    metrics = {}
    model.eval()
    for batch_idx, batch in enumerate(range(0, data_num, batch_size)):
        start_idx = batch
        end_idx = start_idx + batch_size
        tokens, tokens_pre, tokens_alt, tokens_cur, cau_labels, seq_lens, eids = get_batch(eval_file[start_idx:end_idx],
                                                                                           device)
        cau_outputs = model(tokens, tokens_pre, tokens_alt, tokens_cur, seq_lens)
        cau_outputs = cau_outputs.detach()

        if is_fc:
            criterion = FocalLoss(gamma=2, alpha=0.75)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(cau_outputs, cau_labels)
        losses.append(loss.item())
        cau_preds = torch.max(cau_outputs.cpu(), 1)[1].numpy()
        cau_labels = cau_labels.cpu().numpy()
        causality_preds += cau_preds.tolist()
        causality_labels += cau_labels.tolist()
        if data_type == 'valid':
            for pred, label, eid in zip(cau_preds, cau_labels, eids):
                if label == 1 and pred == 0:
                    fn.append(eid)
                if label == 0 and pred == 1:
                    fp.append(eid)

    metrics['loss'] = np.mean(losses)
    metrics['acc'] = accuracy_score(causality_labels, causality_preds)
    metrics['precision'] = precision_score(causality_labels, causality_preds)
    metrics['recall'] = recall_score(causality_labels, causality_preds)
    metrics['f1'] = f1_score(causality_labels, causality_preds)
    if data_type == 'valid':
        metrics['fp'] = fp
        metrics['fn'] = fn
    logger.info('Full confusion matrix')
    logger.info(confusion_matrix(causality_labels, causality_preds))
    return metrics
    # tn, fp, fn, tp = confusion_matrix(auc_ref, auc_pre).ravel()
    # loss_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=metrics['loss']), ])
    # acc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=metrics['acc']), ])
    # auc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/roc'.format(data_type), simple_value=metrics['roc']), ])
    # prc_sum = tf.Summary(value=[tf.Summary.Value(tag='{}/prc'.format(data_type), simple_value=metrics['prc']), ])
    # return metrics, (loss_sum, acc_sum, auc_sum, prc_sum)


def train_one_epoch(model, optimizer, loader, args, logger):
    model.train()
    train_loss = []
    n_batch_loss = 0
    weight = torch.from_numpy(np.array([0.8, 0.2], dtype=np.float32)).to(args.device)
    for step, batch in enumerate(loader):
        indexes, medicines, labels, seq_lens = tuple(map(lambda x: x.to(args.device), batch))
        optimizer.zero_grad()
        outputs = model(indexes, medicines)
        # if args.is_fc:
        #     criterion = FocalLoss(gamma=2, alpha=0.75)
        # else:
        #     criterion = torch.nn.CrossEntropyLoss(weight)
        # loss = criterion(outputs.view(-1, args.n_class), labels.view(-1))
        loss = compute_loss(logits=outputs, target=labels, length=seq_lens, weight=weight)
        loss.backward()
        if args.clip > 0:
            # 梯度裁剪，输入是(NN参数，最大梯度范数，范数类型=2)，一般默认为L2范数
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        n_batch_loss += loss.item()
        bidx = step + 1
        if bidx % args.period == 0:
            logger.info('AvgLoss batch [{} {}] - {}'.format(bidx - args.period + 1, bidx, n_batch_loss / args.period))
            n_batch_loss = 0
        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    return avg_train_loss


def evaluate_one_epoch(model, loader, device, data_type, is_point, logger):
    losses = []
    pre_labels, pre_scores, ref = [], [], []
    fp = []
    fn = []
    metrics = {}
    pre_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    ref_points = {3: [], 18: [], 36: [], 72: [], 144: [], 216: []}
    model.eval()
    weight = torch.from_numpy(np.array([0.5, 0.5], dtype=np.float32)).to(device)
    for step, batch in enumerate(loader):
        indexes, medicines, labels, seq_lens = tuple(map(lambda x: x.to(device), batch))
        outputs = model(indexes, medicines)
        outputs = outputs.detach()
        loss = compute_loss(logits=outputs, target=labels, length=seq_lens, weight=weight).item()
        losses.append(loss)
        output_labels = torch.max(outputs.cpu(), 2)[1].numpy()
        output_scores = outputs.cpu()[:, :, 1].numpy()
        labels = labels.cpu().numpy()
        seq_lens = seq_lens.cpu().numpy()

        for pre_label, pre_score, label, seq_len in zip(output_labels, output_scores, labels, seq_lens):
            if is_point:
                pre_labels.append(pre_label[seq_len - 1])
                ref.append(label[seq_len - 1])
            else:
                pre_labels += pre_label[:seq_len].tolist()
                pre_scores += pre_score[:seq_len].tolist()
                ref += label[:seq_len].tolist()

            for k, v in pre_points.items():
                if seq_len >= k:
                    v.append(pre_label[k - 1])
                    ref_points[k].append(label[k - 1])

    metrics['loss'] = np.mean(losses)
    metrics['acc'] = accuracy_score(ref, pre_labels)
    metrics['roc'] = roc_auc_score(ref, pre_scores)
    (precisions, recalls, thresholds) = precision_recall_curve(ref, pre_scores)
    metrics['prc'] = auc(recalls, precisions)
    metrics['pse'] = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    if data_type == 'eval':
        metrics['fp'] = fp
        metrics['fn'] = fn
    for k, v in pre_points.items():
        logger.info('{} hour confusion matrix. AUCROC : {}'.format(int(k / 3), roc_auc_score(ref_points[k], v)))
        logger.info(confusion_matrix(ref_points[k], v))
    logger.info('Full confusion matrix')
    logger.info(confusion_matrix(ref, pre_labels))
    return metrics


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = functional.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def visulization(model, data_num, batch_size, test_file, device, id2token_file, logger):
    model.eval()
    for batch_idx, batch in enumerate(range(0, data_num, batch_size)):
        start_idx = batch
        end_idx = start_idx + batch_size
        tokens, tokens_pre, tokens_alt, tokens_cur, cau_labels, seq_lens, eids = get_batch(test_file[start_idx:end_idx],
                                                                                           device)
        cau_outputs = model(tokens, tokens_pre, tokens_alt, tokens_cur, seq_lens)

        nhead = 4
        nblock = 4
        tokens = tokens.cpu().numpy()
        seq_lens = seq_lens.cpu().numpy()
        nbatch = len(tokens)
        for block in range(nblock):
            # logger.info('Block - {}'.format(block + 1))
            fig, axs = plt.subplots(1, 4, figsize=(16, 9))
            for idx in range(nbatch):
                sample = trans_ids(tokens[idx][:seq_lens[idx]], id2token_file)
                # logger.info('Sample {} - {}'.format(eids[idx], sample))
                atten_weights = model.transformer.blocks[block].self_attn.attn[idx].detach().cpu().numpy()
                # atten_weights = model.__getattr__('self_attention_%d' % block).atten_weights[head_idx:tail_idx].detach()
                # atten_weights = atten_weights.cpu().numpy()
                for h in range(nhead):
                    # logger.info('Head - {}'.format(h + 1))
                    # print(atten_weights[h])
                    axs[h].set_title('head_' + str(h), fontsize=12)
                    axs[h].tick_params(axis='x', labelsize=10)
                    axs[h].tick_params(axis='y', labelsize=10)
                    draw(atten_weights[h][:seq_lens[idx], :seq_lens[idx]], sample, sample if h == 0 else [], axs[h])
                plt.savefig('./pictures/block_' + str(block + 1) + '_sample_' + str(eids[idx]))


def trans_ids(ids, id2token_file):
    tokens = []
    for tid in ids:
        # if tid > 0:
        tokens.append(id2token_file[str(tid)])
        # else:
        #     break

    return tokens


def draw(data, x, y, ax):
    seaborn.heatmap(data, linewidths=0.05, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                    cbar=False, ax=ax, cmap='Blues')
