import numpy as np
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve


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
    causality_preds, causality_scores, causality_labels = [], [], []
    metrics = {}
    model.eval()
    for batch_idx, batch in enumerate(range(0, data_num, batch_size)):
        start_idx = batch
        end_idx = start_idx + batch_size
        tokens, tokens_pre, tokens_alt, tokens_cur, cau_labels, seq_lens, eids = get_batch(eval_file[start_idx:end_idx],
                                                                                           device)
        cau_outputs, penal, weights = model(tokens, tokens_pre, tokens_alt, tokens_cur, seq_lens)
        cau_outputs = cau_outputs.detach()
        if is_fc:
            criterion = FocalLoss(gamma=2, alpha=0.75)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(cau_outputs, cau_labels)
        losses.append(loss.item())
        cau_preds = torch.max(cau_outputs.cpu(), 1)[1].numpy()
        cau_scores = cau_outputs.cpu()[:, 1].numpy()
        cau_labels = cau_labels.cpu().numpy()
        causality_preds += cau_preds.tolist()
        causality_scores += cau_scores.tolist()
        causality_labels += cau_labels.tolist()
        if data_type == 'valid' or data_type == 'eval':
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
    fpr, tpr, _ = roc_curve(causality_labels, causality_scores)
    (precisions, recalls, _) = precision_recall_curve(causality_labels, causality_scores)
    metrics['auc_roc'] = auc(fpr, tpr)
    metrics['auc_prc'] = auc(recalls, precisions)
    if data_type == 'valid' or data_type == 'eval':
        metrics['fp'] = fp
        metrics['fn'] = fn
    logger.info('Full confusion matrix')
    logger.info(confusion_matrix(causality_labels, causality_preds))
    return metrics, fpr, tpr, precisions, recalls


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
