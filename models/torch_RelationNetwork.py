import torch
from torch import nn
from torch.nn import utils as nn_utils
from modules.torch_attention import Multihead_Attention, FeedForward
from modules.torch_TextCNNNet import TextCNNNet
from sru import SRU
from time import time


class CRN(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_hidden, n_layer, n_kernels, n_filter, topk=1,
                 dropout=None, logger=None):
        super(CRN, self).__init__()
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.n_filter = n_filter
        self.n_linear = 4 * n_hidden
        n_dict, n_emb = token_embeddings.shape
        start_t = time()
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout['emb'])
        self.sentence_encoder = nn.GRU(n_emb, n_hidden, n_layer, dropout=dropout['layer'], batch_first=True,
                                       bidirectional=True)
        # self.sentence_encoder = SRU(n_emb, n_hidden, n_layer, dropout['layer'], bidirectional=True)
        self.pre_encoder = TextCNNNet(n_emb, max_len['pre'], n_filter, n_kernels)
        self.alt_encoder = TextCNNNet(n_emb, max_len['alt'], n_filter, n_kernels)
        self.cur_encoder = TextCNNNet(n_emb, max_len['cur'], n_filter, n_kernels)

        self.g_fc = nn.Sequential(nn.Linear(6 * n_filter + 2 * n_hidden, self.n_linear),
                                  nn.ReLU(),
                                  nn.Dropout(dropout['layer']),
                                  nn.Linear(self.n_linear, self.n_linear),
                                  nn.ReLU())
        self.f_fc = nn.Sequential(nn.Linear(self.n_linear, self.n_linear),
                                  nn.ReLU(),
                                  nn.Dropout(dropout['layer']),
                                  nn.Linear(self.n_linear, output_size))

        self.init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        batch_size = x.shape[0]
        sorted_seq_lens, indices = torch.sort(seq_lens, dim=0, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        x = x[indices]
        x_word_emb = self.word_embedding(x)
        x_pre_word_emb = self.word_embedding(x_pre)
        x_alt_word_emb = self.word_embedding(x_alt)
        x_cur_word_emb = self.word_embedding(x_cur)
        x_word_emb = self.emb_dropout(x_word_emb)
        x_pre_word_emb = self.emb_dropout(x_pre_word_emb)
        x_alt_word_emb = self.emb_dropout(x_alt_word_emb)
        x_cur_word_emb = self.emb_dropout(x_cur_word_emb)

        x_word_emb = nn_utils.rnn.pack_padded_sequence(x_word_emb, sorted_seq_lens, batch_first=True)
        output, state = self.sentence_encoder(x_word_emb)
        state = state.view(self.n_layer, 2, batch_size, self.n_hidden)
        forward_state, backward_state = state[-1][0], state[-1][1]
        y_state = torch.cat([forward_state, backward_state], dim=1)
        y_pre = self.pre_encoder(x_pre_word_emb)
        y_alt = self.alt_encoder(x_alt_word_emb)
        y_cur = self.cur_encoder(x_cur_word_emb)
        pre_cur = torch.cat((y_pre, y_cur), dim=1)
        cur_pre = torch.cat((y_cur, y_pre), dim=1)
        pre_alt = torch.cat((y_pre, y_alt), dim=1)
        alt_cur = torch.cat((y_alt, y_cur), dim=1)
        y_composed = torch.stack([pre_cur, cur_pre, pre_alt, alt_cur], dim=1)
        y_state = torch.unsqueeze(y_state, 1)
        y_state = y_state.repeat(1, 4, 1)
        y_pair = torch.cat([y_composed, y_state], 2)

        y_pair = y_pair.view(batch_size * 4, 6 * self.n_filter + 2 * self.n_hidden)
        y_pair = self.g_fc(y_pair)

        y_pair = y_pair.view(batch_size, 4, self.n_linear)
        y_pair = y_pair.sum(1).squeeze()

        y_pair = self.f_fc(y_pair)
        return y_pair
