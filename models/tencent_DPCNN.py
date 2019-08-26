import torch
from torch import nn
import torch.nn.functional as F
from time import time


class DPCNN(nn.Module):
    """
    Reference:
        Deep Pyramid Convolutional Neural Networks for Text Categorization
    """

    def __init__(self, token_embeddings, config, logger):
        super(DPCNN, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.num_kernels = config.n_filter
        self.pooling_stride = 2
        self.kernel_size = config.n_kernels[1]
        self.radius = int(self.kernel_size / 2)
        assert self.kernel_size % 2 == 1, "DPCNN kernel should be odd!"

        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.emb_dropout = nn.Dropout(config.dropout['emb'])
        self.convert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(n_emb, self.num_kernels, self.kernel_size, padding=self.radius))

        self.convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius)
        ) for _ in range(config.dp_blocks)])
        self.linear = torch.nn.Linear(self.num_kernels, config.n_class)
        self.out_dropout = nn.Dropout(config.dropout['layer'])
        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        embedding = self.word_embedding(x)
        embedding = self.emb_dropout(embedding)
        embedding = embedding.permute(0, 2, 1)
        conv_embedding = self.convert_conv(embedding)
        conv_features = self.convs[0](conv_embedding)
        conv_features = conv_embedding + conv_features
        for i in range(1, len(self.convs)):
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride)
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features
        doc_embedding = F.max_pool1d(
            conv_features, conv_features.size(2)).squeeze()
        return self.out_dropout(self.linear(doc_embedding))
