import torch
from torch import nn
from modules.torch_attention import Multihead_Attention, FeedForward
from modules.torch_TextCNNNet import TextCNNNet
from sru import SRU
from time import time


class CRN(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_hidden, n_layer, n_kernels, n_filter, topk, dropout,
                 logger):
        super(CRN, self).__init__()
        self.n_hidden = 4 * n_hidden
        self.max_len = max_len
        n_dict, n_emb = token_embeddings.shape
        start_t = time()
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout['emb'])
        # self.bi_gru = nn.GRU(n_emb, n_hidden, n_layer, dropout=dropout['layer'], bidirectional=True)
        self.sentence_encoder = SRU(n_emb, n_hidden, n_layer, dropout['layer'], bidirectional=True)
        self.pre_encoder = TextCNNNet(n_emb, max_len['pre'], n_filter, n_kernels)
        self.alt_encoder = TextCNNNet(n_emb, max_len['alt'], n_filter, n_kernels)
        self.cur_encoder = TextCNNNet(n_emb, max_len['cur'], n_filter, n_kernels)

        self.g_fc1 = nn.Linear(3 * n_filter + 2 * n_hidden, self.n_hidden)
        self.g_fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.f_fc1 = nn.Linear(self.n_hidden, self.n_hidden)

        self.init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False
        self.g_fc1.weight.data.normal_(0, 0.01)
        self.g_fc2.weight.data.normal_(0, 0.01)
        self.f_fc1.weight.data.normal_(0, 0.01)

    def forward(self, x, x_pre, x_alt, x_cur):
        x_word_emb = self.word_embedding(x)
        x_pre_word_emb = self.word_embedding(x_pre)
        x_alt_word_emb = self.word_embedding(x_alt)
        x_cur_word_emb = self.word_embedding(x_cur)
        x_word_emb = self.emb_dropout(x_word_emb)
        x_pre_word_emb = self.emb_dropout(x_pre_word_emb)
        x_alt_word_emb = self.emb_dropout(x_alt_word_emb)
        x_cur_word_emb = self.emb_dropout(x_cur_word_emb)

        y = self.sentence_encoder(x_word_emb.permute(1, 0, 2))[0].permute(1, 0, 2)
        y_pre = self.pre_encoder(x_pre_word_emb)
        y_alt = self.alt_encoder(x_alt_word_emb)
        y_cur = self.cur_encoder(x_cur_word_emb)
        pre_cur = torch.cat((y_pre, y_cur), dim=1)
        pre_alt = torch.cat((y_pre, y_alt), dim=1)
        alt_cur = torch.cat((y_alt, y_cur), dim=1)
        y_composed = torch.stack([pre_cur, pre_alt, alt_cur], dim=1)


        y = torch.reshape(y, [-1, self.max_len * self.n_hidden])
        return self.linear(y)
