from torch.nn import functional
import torch.nn as nn
import torch
from torch.autograd import Variable
from time import time


class SelfAttentive(nn.Module):

    def __init__(self, token_embeddings, output_size, n_hidden, n_layer, da, r, dropout, logger):
        super(SelfAttentive, self).__init__()
        start_t = time()
        n_dict, n_emb = token_embeddings.shape
        self.r = r
        self.n_hidden = n_hidden
        # Embedding Layer
        self.word_embedding = nn.Embedding(n_dict, n_emb)
        self.emb_dropout = nn.Dropout(dropout['emb'])
        # RNN type
        self.bi_lstm = nn.LSTM(n_emb, n_hidden, n_layer, bias=False, batch_first=True, bidirectional=True)

        # Self Attention Layers
        self.S1 = nn.Linear(self.n_hidden * 2, da, bias=False)
        self.S2 = nn.Linear(da, r, bias=False)

        # Final MLP Layers
        self.MLP = nn.Linear(r * self.n_hidden * 2, self.n_hidden)
        self.decoder = nn.Linear(self.n_hidden, output_size)

        self.init_weights(token_embeddings)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def init_weights(self, embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.word_embedding.weight.requires_grad = False
        initrange = 0.1
        self.S1.weight.data.uniform_(-initrange, initrange)
        self.S2.weight.data.uniform_(-initrange, initrange)
        self.MLP.weight.data.uniform_(-initrange, initrange)
        self.MLP.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)

    def forward(self, x, seq_lens):
        x_emb = self.word_embedding(x)
        sorted_seq_lens, indices = torch.sort(seq_lens, dim=0, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        rnn_x = torch.nn.utils.rnn.pack_padded_sequence(x_emb, sorted_seq_lens, batch_first=True)
        output, hidden = self.bi_lstm(rnn_x)

        depacked_output, lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        BM = Variable(torch.zeros(x.size(0), self.r * self.n_hidden * 2).cuda())
        penal = Variable(torch.zeros(1).cuda())
        I = Variable(torch.eye(self.r).cuda())
        weights = {}

        # Attention Block
        for i in range(x.size(0)):
            H = depacked_output[i, :lens[i], :]
            s1 = self.S1(H)
            s2 = self.S2(functional.tanh(s1))

            # Attention Weights and Embedding
            A = functional.softmax(s2.t())
            M = torch.mm(A, H)
            BM[i, :] = M.view(-1)

            # Penalization term
            AAT = torch.mm(A, A.t())
            P = torch.norm(AAT - I, 2)
            penal += P * P
            weights[i] = A

        # Penalization Term
        penal /= x.size(0)

        # MLP block for Classifier Feature
        MLPhidden = self.MLP(BM)
        decoded = self.decoder(functional.relu(MLPhidden))

        return decoded, penal, weights

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters()).data
    #
    #     return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
    #             Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))
