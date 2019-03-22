import torch
from torch import nn
from torch.nn import utils as nn_utils
from modules.torch_attention import Multihead_Attention, FeedForward, PositionEmbedding, WordEmbedding, label_smoothing
from modules.torch_TextCNNNet import TextCNNNet
from sru import SRU
from time import time


class MCIN(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_hidden, n_layer, n_kernels, n_filter,
                 n_block, n_head, is_sinusoid, is_ffn, dropout, logger, is_test=None):
        super(MCIN, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.sinusoid = is_sinusoid
        self.gru_hidden = n_hidden
        self.att_hidden = n_emb
        self.crn_hidden = 4 * n_hidden
        self.max_len = max_len['full']
        self.n_block = n_block
        self.is_ffn = is_ffn
        self.n_layer = n_layer
        self.n_filter = n_filter
        self.is_test = is_test
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        if is_sinusoid:
            self.position_embedding = PositionEmbedding(n_emb, zeros_pad=False, scale=False)
        else:
            self.position_embedding = WordEmbedding(self.max_len, n_emb, zeros_pad=False, scale=False)
        self.emb_dropout = nn.Dropout(dropout['emb'])

        self.seg_encoder = nn.GRU(n_emb, n_hidden, n_layer, dropout=dropout['layer'], batch_first=True, bidirectional=True)
        for i in range(self.n_block):
            self.__setattr__('self_attention_%d' % i, Multihead_Attention(self.att_hidden, n_head, dropout['layer']))
            if self.is_ffn:
                self.__setattr__('feed_forward_%d' % i, FeedForward(self.att_hidden,
                                                                    [4 * self.att_hidden, self.att_hidden]))
        self.word_fc = nn.Linear(self.max_len * self.att_hidden, self.att_hidden)

        self.pre_encoder = TextCNNNet(n_emb, max_len['pre'], n_filter, n_kernels)
        self.alt_encoder = TextCNNNet(n_emb, max_len['alt'], n_filter, n_kernels)
        self.cur_encoder = TextCNNNet(n_emb, max_len['cur'], n_filter, n_kernels)
        self.g_fc = nn.Sequential(nn.Linear(6 * n_filter + 2 * n_hidden, self.crn_hidden),
                                  nn.ReLU(),
                                  nn.Dropout(dropout['layer']),
                                  nn.Linear(self.crn_hidden, self.crn_hidden),
                                  nn.ReLU())
        self.f_fc = nn.Sequential(nn.Linear(self.crn_hidden, self.crn_hidden),
                                  nn.ReLU(),
                                  nn.Dropout(dropout['layer']))

        self.out_fc = nn.Sequential(nn.Linear(self.att_hidden + self.crn_hidden, self.gru_hidden),
                                    nn.ReLU(),
                                    nn.Dropout(dropout['layer']),
                                    nn.Linear(self.gru_hidden, output_size))

        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
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
        x_pre_word_emb = self.emb_dropout(x_pre_word_emb)
        x_alt_word_emb = self.emb_dropout(x_alt_word_emb)
        x_cur_word_emb = self.emb_dropout(x_cur_word_emb)

        if self.sinusoid:
            x_word_emb += self.position_embedding(x)
        else:
            x_word_emb += self.position_embedding(torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long().cuda())
        y_embeded = self.emb_dropout(x_word_emb)
        for i in range(self.n_block):
            y_attended = self.__getattr__('self_attention_%d' % i)(y_embeded)
            if self.is_ffn:
                y_transformed = self.__getattr__('feed_forward_%d' % i)(y_attended)
        y_word = torch.reshape(y_transformed, [-1, self.max_len * self.att_hidden])
        y_word = self.word_fc(y_word)

        x_word_emb = nn_utils.rnn.pack_padded_sequence(x_word_emb, sorted_seq_lens, batch_first=True)
        output, state = self.seg_encoder(x_word_emb)
        state = state.view(self.n_layer, 2, batch_size, self.gru_hidden)
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

        y_pair = y_pair.view(batch_size * 4, 6 * self.n_filter + 2 * self.gru_hidden)
        y_pair = self.g_fc(y_pair)
        y_pair = y_pair.view(batch_size, 4, self.crn_hidden)
        y_pair = y_pair.sum(1).squeeze()
        y_segment = self.f_fc(y_pair)

        y_word_seg = torch.cat([y_word, y_segment], 1)
        return self.out_fc(y_word_seg)
