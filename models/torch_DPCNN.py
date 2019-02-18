import torch
import torch.nn as nn
from time import time

torch.manual_seed(1)


class TextCNNDeep(nn.Module):
    def __init__(self, token_embeddings, max_len, output_size, n_kernels, n_filter, dropout, logger):
        super(TextCNNDeep, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.n_filter = n_filter
        self.n_feature_map = 250
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)

        self.question_convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=n_emb, out_channels=n_filter, kernel_size=kernel_size),
                           nn.BatchNorm1d(n_filter),
                           nn.ReLU(inplace=True),

                           nn.Conv1d(in_channels=n_filter,
                                     out_channels=n_filter,
                                     kernel_size=kernel_size),
                           nn.BatchNorm1d(n_filter),
                           nn.ReLU(inplace=True),
                           nn.MaxPool1d(kernel_size=(max_len['full'] - kernel_size * 2 + 2))
                           ) for kernel_size in n_kernels])

        self.num_seq = len(n_kernels)
        self.change_dim_conv = nn.Conv1d(n_filter * self.num_seq, self.n_feature_map, kernel_size=1, stride=1)
        self.standard_pooling = nn.MaxPool1d(kernel_size=3, stride=2)
        self.standard_batchnm = nn.BatchNorm1d(num_features=self.n_feature_map)
        self.standard_act_fun = nn.ReLU()

        self.fc = nn.Sequential(
            nn.Linear(self.n_feature_map, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(output_size, output_size)
        )
        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def forward(self, inp, x_pre, x_alt, x_cur, seq_lens):
        inp_emb = self.word_embedding(inp)
        # question = self.encoder(question)
        # permute 的作用是交换维度，因为词嵌入的维度200要作为后面conv1的输入的channel，所以第二和三维交换
        x = [question_conv(inp_emb.permute(0, 2, 1)) for question_conv in self.question_convs]
        x = torch.cat(x, dim=1)
        xp = x
        xp = self.change_dim_conv(xp)
        x = self.conv3x3(in_channels=x.size(1), out_channels=self.n_feature_map)(x)
        x = self.standard_batchnm(x)
        x = self.standard_act_fun(x)
        x = self.conv3x3(self.n_feature_map, self.n_feature_map)(x)
        x = self.standard_batchnm(x)
        x = self.standard_act_fun(x)
        x = x + xp
        while x.size(2) > 2:
            x = self._block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def conv3x3(self, in_channels, out_channels, stride=1, padding=1):
        """3x3 convolution with padding"""
        _conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                          padding=padding, bias=False)
        return _conv.cuda()
        # if self.opt.USE_CUDA:
        #     return _conv.cuda()
        # else:
        #     return _conv

    def _block(self, x):
        x = self.standard_pooling(x)
        xp = x
        x = self.conv3x3(self.opt.NUM_ID_FEATURE_MAP, self.opt.NUM_ID_FEATURE_MAP)(x)
        x = self.standard_batchnm(x)
        x = self.standard_act_fun(x)
        x = self.conv3x3(self.opt.NUM_ID_FEATURE_MAP, self.opt.NUM_ID_FEATURE_MAP)(x)
        x = self.standard_batchnm(x)
        x = self.standard_act_fun(x)
        x += xp
        return x
