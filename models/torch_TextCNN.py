import torch
from torch import nn
from modules.torch_TextCNNNet import TextCNNNet
from modules.torch_attention import PositionEmbedding, WordEmbedding
from time import time


class TextCNN(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_kernels, n_filter, is_pos, is_sinusoid,
                 dropout, logger):
        super(TextCNN, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.sinusoid = is_sinusoid
        self.max_len = max_len['full']
        self.n_filter = 3 * n_filter
        self.fc_hidden = n_filter
        self.is_pos = is_pos
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        if is_sinusoid:
            self.position_embedding = PositionEmbedding(n_emb, zeros_pad=False, scale=False)
        else:
            self.position_embedding = WordEmbedding(self.max_len, n_emb, zeros_pad=False, scale=False)
        self.emb_dropout = nn.Dropout(dropout['emb'])

        self.cnn_encoder = TextCNNNet(n_emb, max_len['full'], n_filter, n_kernels)
        self.out_fc = nn.Sequential(nn.Linear(self.n_filter, self.fc_hidden),
                                    nn.ReLU(),
                                    nn.Dropout(dropout['layer']),
                                    nn.Linear(self.fc_hidden, output_size))

        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        x_emb = self.word_embedding(x)
        x_emb = self.emb_dropout(x_emb)
        if self.is_pos:
            if self.sinusoid:
                x_emb += self.position_embedding(x)
            else:
                x_emb += self.position_embedding(
                    torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long().cuda())
        y = self.cnn_encoder(x_emb)

        return self.out_fc(y)
