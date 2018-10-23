import torch
from torch import nn
from modules.torch_CNN import TemporalConvNet
from modules.torch_attention import Multihead_Attention, FeedForward
from time import time


class TCN(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_channel, n_kernel, dropout, logger):
        super(TCN, self).__init__()
        self.n_hidden = n_channel[-1]
        self.max_len = max_len
        n_dict, n_emb = token_embeddings.shape
        start_t = time()
        self.embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(token_embeddings))
        self.embedding.weight.requires_grad = False
        self.tcn = TemporalConvNet(n_emb, n_channel, kernel_size=n_kernel, dropout=dropout)
        self.linear = nn.Linear(self.max_len * self.n_hidden, output_size)
        self.init_weights()
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x_emb = self.embedding(x)
        y = self.tcn(x_emb.transpose(2, 1))
        y = y.transpose(1, 2)
        y = torch.reshape(y, [-1, self.max_len * self.n_hidden])
        return self.linear(y)
