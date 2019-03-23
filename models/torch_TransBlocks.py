import torch
from torch import nn
from modules.torch_transformer import Encoder, PositionalEncoding
from time import time


class TB(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_hidden, n_layer, n_kernels, n_filter,
                 n_block, n_head, is_sinusoid, is_ffn, dropout, logger, is_test=None):
        super(TB, self).__init__()
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
        self.pos_embedding = PositionalEncoding(n_emb, max_len=self.max_len)
        self.emb_dropout = nn.Dropout(dropout['emb'])
        self.transformer = Encoder(n_head, n_block, n_emb, dropout['layer'])

        self.out_fc = nn.Sequential(nn.Linear(self.max_len * self.att_hidden, self.att_hidden),
                                    nn.ReLU(),
                                    nn.Dropout(dropout['layer']),
                                    nn.Linear(self.att_hidden, output_size))

        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        x_mask = (x != 0).unsqueeze(-2)
        x_word_emb = self.word_embedding(x)

        x_word_emb = self.pos_embedding(x_word_emb)
        x_embeded = self.emb_dropout(x_word_emb)
        y_transformed = self.transformer(x_embeded, x_mask)
        y_word = torch.reshape(y_transformed, [-1, self.max_len * self.att_hidden])

        return self.out_fc(y_word)
