import torch
from torch import nn
from torch.nn import utils as nn_utils
from modules.torch_CNN import TemporalConvNet
from modules.torch_attention import Multihead_Attention, FeedForward, PositionEmbedding, WordEmbedding, label_smoothing
from time import time


class TCN(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_channel, n_kernel, n_block, n_head, dropout, logger):
        super(TCN, self).__init__()
        self.n_hidden = n_channel[-1]
        self.max_len = max_len
        n_dict, n_emb = token_embeddings.shape
        self.n_block = n_block
        start_t = time()
        self.embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(token_embeddings))
        self.embedding.weight.requires_grad = False
        self.emb_dropout = nn.Dropout(dropout['emb'])
        self.tcn = TemporalConvNet(n_emb, n_channel, kernel_size=n_kernel, dropout=dropout['layer'])
        for i in range(self.n_block):
            self.__setattr__('self_attention_%d' % i, Multihead_Attention(self.n_hidden, n_head, dropout['layer']))
            # self.__setattr__('feed_forward_%d' % i, FeedForward(self.n_hidden, [4 * self.n_hidden, self.n_hidden]))
        self.linear = nn.Linear(self.max_len * self.n_hidden, output_size)
        self.init_weights()
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = self.emb_dropout(x_emb)
        y = self.tcn(x_emb.transpose(2, 1))
        y = y.transpose(1, 2)
        for i in range(self.n_block):
            y = self.__getattr__('self_attention_%d' % i)(y)
            # y = self.__getattr__('feed_forward_%d' % i)(y)
        y = torch.reshape(y, [-1, self.max_len * self.n_hidden])
        return self.linear(y)
