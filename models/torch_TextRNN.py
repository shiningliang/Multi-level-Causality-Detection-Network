import torch
import torch.nn as nn
from torch.nn import utils as nn_utils
import torch.nn.functional as F
from time import time


# 有两个做法有待实验验证
# 1、kmax_pooling的使用，对所有RNN的输出做最大池化
# 2、分类器选用两层全连接层+BN层，还是直接使用一层全连接层
# 3、是否需要init_hidden

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]  # torch.Tensor.topk()的输出有两项，后一项为索引
    return x.gather(dim, index)


class TextRNN(nn.Module):
    def __init__(self, token_embeddings, output_size, n_hidden, n_layer, kmax_pooling,  is_pos, is_sinusoid,
                 dropout, logger):
        super(TextRNN, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.gru_hidden = 2 * n_hidden
        self.kmax_pooling = kmax_pooling
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout['emb'])
        self.bi_GRU = nn.GRU(n_emb, n_hidden, n_layer, dropout=dropout['layer'], bidirectional=True)

        # 两层全连接层，中间添加批标准化层
        # 全连接层隐藏元个数需要再做修改
        self.out_fc = nn.Sequential(nn.Linear(self.kmax_pooling * self.gru_hidden, self.gru_hidden),
                                    # nn.BatchNorm1d(self.gru_hidden),
                                    nn.ReLU(),
                                    nn.Linear(self.gru_hidden, output_size)
                                    )

        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    # 对LSTM所有隐含层的输出做kmax pooling
    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        # batch_size = x.shape[0]
        # sorted_seq_lens, indices = torch.sort(seq_lens, dim=0, descending=True)
        # _, desorted_indices = torch.sort(indices, descending=False)
        # x = x[indices]
        x_emb = self.word_embedding(x)
        # x_emb = nn_utils.rnn.pack_padded_sequence(x_emb, sorted_seq_lens, batch_first=True)
        output = self.bi_GRU(x_emb)[0].permute(0, 2, 1)  # batch * hidden * seq
        pooling = kmax_pooling(output, 2, self.kmax_pooling)  # batch * hidden * kmax

        # word+article
        flatten = pooling.view(pooling.size(0), -1)

        return self.out_fc(flatten)
