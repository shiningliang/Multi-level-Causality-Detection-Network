import torch
from torch import nn
from torch.nn import utils as nn_utils
from modules.torch_transformer import Encoder, PositionalEncoding
from modules.torch_TextCNNNet import TextCNNNet
from sru import SRU
from time import time


class MCDN(nn.Module):
    def __init__(self, token_embeddings, args, logger):
        super(MCDN, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.max_len = args.max_len['full']
        self.att_hidden = n_emb
        self.gru_hidden = args.n_hidden
        self.crn_hidden = 4 * args.n_hidden
        self.n_block = args.n_block
        self.n_head = args.n_head
        self.n_layer = args.n_layer
        self.n_filter = args.n_filter
        self.n_kernels = args.n_kernels
        self.is_sinusoid = args.is_sinusoid
        self.is_ffn = args.is_ffn
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        if self.is_sinusoid:
            self.pos_embedding = PositionalEncoding(n_emb, max_len=self.max_len)
        #     self.position_embedding = PositionEmbedding(n_emb, zeros_pad=False, scale=False)
        # else:
        #     self.position_embedding = WordEmbedding(self.max_len, n_emb, zeros_pad=False, scale=False)
        self.emb_dropout = nn.Dropout(args.dropout['emb'])
        self.transformer = Encoder(self.n_head, self.n_block, n_emb, args.dropout['layer'])
        self.seg_encoder = nn.GRU(n_emb, self.gru_hidden, self.n_layer, dropout=args.dropout['layer'], batch_first=True,
                                  bidirectional=True)

        self.word_fc = nn.Linear(self.max_len * self.att_hidden, self.att_hidden)

        self.pre_encoder = TextCNNNet(n_emb, args.max_len['pre'], self.n_filter, self.n_kernels)
        self.alt_encoder = TextCNNNet(n_emb, args.max_len['alt'], self.n_filter, self.n_kernels)
        self.cur_encoder = TextCNNNet(n_emb, args.max_len['cur'], self.n_filter, self.n_kernels)
        self.g_fc = nn.Sequential(nn.Linear(6 * self.n_filter + 2 * self.gru_hidden, self.crn_hidden),
                                  nn.ReLU(),
                                  nn.Dropout(args.dropout['layer']),
                                  nn.Linear(self.crn_hidden, self.crn_hidden),
                                  nn.ReLU())
        self.f_fc = nn.Sequential(nn.Linear(self.crn_hidden, self.crn_hidden),
                                  nn.ReLU(),
                                  nn.Dropout(args.dropout['layer']))

        self.out_fc = nn.Sequential(nn.Linear(self.att_hidden + self.crn_hidden, self.gru_hidden),
                                    nn.ReLU(),
                                    nn.Dropout(args.dropout['layer']),
                                    nn.Linear(self.gru_hidden, args.n_class))

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
        x_mask = (x != 0).unsqueeze(-2)
        x_word_emb = self.word_embedding(x)
        x_pre_word_emb = self.word_embedding(x_pre)
        x_alt_word_emb = self.word_embedding(x_alt)
        x_cur_word_emb = self.word_embedding(x_cur)
        x_pre_word_emb = self.emb_dropout(x_pre_word_emb)
        x_alt_word_emb = self.emb_dropout(x_alt_word_emb)
        x_cur_word_emb = self.emb_dropout(x_cur_word_emb)

        if self.is_sinusoid:
            x_word_emb = self.pos_embedding(x_word_emb)
        #     x_word_emb += self.position_embedding(x)
        # else:
        #     x_word_emb += self.position_embedding(torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long().cuda())
        x_embeded = self.emb_dropout(x_word_emb)
        y_transformed = self.transformer(x_embeded, x_mask)
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
