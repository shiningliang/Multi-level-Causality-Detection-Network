import torch
from torch import nn
from time import time
from modules.drnn_module import RNN
from modules.drnn_module import RNNType


class DRNN(nn.Module):
    def __init__(self, token_embeddings, config, logger):
        super(DRNN, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.rnn_type = "GRU"
        self.word_embedding = nn.Embedding(n_dict, n_emb, padding_idx=0)
        self.forward_rnn = RNN(n_emb, config.n_hidden, 1, batch_first=True, rnn_type=self.rnn_type)
        # if config.DRNN.bidirectional:
        self.backward_rnn = RNN(n_emb, config.n_hidden, 1, batch_first=True, rnn_type=self.rnn_type)
        self.window_size = config.window_size
        self.dropout = torch.nn.Dropout(p=config.dropout['layer'])
        self.hidden_dimension = config.n_hidden * 2
        # if config.DRNN.bidirectional:
        # self.hidden_dimension *= 2
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dimension)

        self.mlp = torch.nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.linear = torch.nn.Linear(self.hidden_dimension, config.n_class)
        self._init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False

    def get_parameter_optimizer_dict(self):
        params = super(DRNN, self).get_parameter_optimizer_dict()
        params.append({'params': self.forward_rnn.parameters()})
        if self.config.DRNN.bidirectional:
            params.append({'params': self.backward_rnn.parameters()})
        params.append({'params': self.batch_norm.parameters()})
        params.append({'params': self.mlp.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def get_embedding(self, batch, pad_shape=None, pad_value=0):
        # mask = torch.tensor(batch).gt(0).float()
        mask = torch.where(batch > 0, torch.full_like(batch, 1), batch).float()
        if pad_shape is not None:
            batch = torch.nn.functional.pad(batch, pad_shape, mode='constant', value=pad_value)
        embedding = self.word_embedding(batch)
        # length = batch[cDataset.DOC_TOKEN_LEN].to(self.config.device)
        return embedding, mask

    def forward(self, x, x_pre, x_alt, x_cur, seq_lens):
        front_pad_embedding, mask = self.get_embedding(x, [self.window_size - 1, 0], 0)
        # if self.config.DRNN.bidirectional:
        tail_pad_embedding, _ = self.get_embedding(x, [0, self.window_size - 1], 0)
        batch_size = front_pad_embedding.size(0)
        mask = mask.unsqueeze(2)

        front_slice_embedding_list = \
            [front_pad_embedding[:, i:i + self.window_size, :] for i in
             range(front_pad_embedding.size(1) - self.window_size + 1)]

        front_slice_embedding = torch.cat(front_slice_embedding_list, dim=0)

        state = None
        for i in range(front_slice_embedding.size(1)):
            _, state = self.forward_rnn(front_slice_embedding[:, i:i + 1, :],
                                        init_state=state, ori_state=True)
            if self.rnn_type == RNNType.LSTM:
                state[0] = self.dropout(state[0])
            else:
                state = self.dropout(state)
        front_state = state[0] if self.rnn_type == RNNType.LSTM else state
        front_state = front_state.transpose(0, 1)
        front_hidden = torch.cat(front_state.split(batch_size, dim=0), dim=1)
        front_hidden = front_hidden * mask

        hidden = front_hidden
        # if self.config.DRNN.bidirectional:
        tail_slice_embedding_list = list()
        for i in range(tail_pad_embedding.size(1) - self.window_size + 1):
            slice_embedding = \
                tail_pad_embedding[:, i:i + self.window_size, :]
            tail_slice_embedding_list.append(slice_embedding)
        tail_slice_embedding = torch.cat(tail_slice_embedding_list, dim=0)

        state = None
        for i in range(tail_slice_embedding.size(1), 0, -1):
            _, state = self.backward_rnn(
                tail_slice_embedding[:, i - 1:i, :],
                init_state=state, ori_state=True)
            if i != tail_slice_embedding.size(1) - 1:
                if self.rnn_type == RNNType.LSTM:
                    state[0] = self.dropout(state[0])
                else:
                    state = self.dropout(state)
        tail_state = state[0] if self.rnn_type == RNNType.LSTM else state
        tail_state = tail_state.transpose(0, 1)
        tail_hidden = torch.cat(tail_state.split(batch_size, dim=0), dim=1)
        tail_hidden = tail_hidden * mask
        hidden = torch.cat([hidden, tail_hidden], dim=2)

        hidden = hidden.transpose(1, 2).contiguous()

        batch_normed = self.batch_norm(hidden).transpose(1, 2)
        batch_normed = batch_normed * mask
        mlp_hidden = self.mlp(batch_normed)
        mlp_hidden = mlp_hidden * mask
        neg_mask = (mask - 1) * 65500.0
        mlp_hidden = mlp_hidden + neg_mask
        max_pooling = torch.nn.functional.max_pool1d(
            mlp_hidden.transpose(1, 2), mlp_hidden.size(1)).squeeze()
        return self.linear(self.dropout(max_pooling))
