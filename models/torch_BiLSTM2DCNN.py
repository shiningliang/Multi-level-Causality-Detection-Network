import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
torch.manual_seed(233)
random.seed(233)


class BiLSTM2DCNN(nn.Module):
    def __init__(self, args):
        super(BiLSTM2DCNN, self).__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(args.embed_num, args.embedding_dim)

        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, bidirectional=True, dropout=args.dropout_model)

        self.hidden = self.init_hidden(args.batch_size)

        # self.convl3 = nn.Conv2d(1, args.kernel_num, (3, args.hidden_dim * 2))
        self.convsl = [nn.Conv2d(1, args.kernel_num, (K, args.hidden_dim * 2)) for K in args.kernel_sizes]

        self.hidden2label1 = nn.Linear(args.kernel_num * len(args.kernel_sizes), args.class_num)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return (Variable(torch.zeros(2, batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, batch_size, self.hidden_dim)))

    def forward(self, sentence):
        # print(sentence)                                     # [torch.LongTensor of size 44x64]
        x = self.word_embeddings(sentence)
        x = self.dropout_embed(x)
        # print(embeds.size())                                # torch.Size([44, 64, 100])
        # x = embeds.view(len(sentence), self.batch_size, -1)
        # print(x.size())                                     # torch.Size([44, 64, 100])
        lstm_out, self.hidden = self.lstm(x, self.hidden)   # lstm_out 10*5*50 hidden 1*5*50 *2
        # print(lstm_out)         # [torch.FloatTensor of size 44x64x400]
        # print(self.hidden)      # [torch.FloatTensor of size 2x64x200],[torch.FloatTensor of size 2x64x200]
        # lstm_out = [F.max_pool1d(i, len(lstm_out)).unsqueeze(2) for i in lstm_out]
        lstm_out = torch.transpose(lstm_out, 0, 1)          # 64*44*400
        # lstm_out = torch.transpose(lstm_out, 1, 2)          # 5*50*10
        # print(lstm_out)         # [torch.FloatTensor of size 64x400x44]
        lstm_out = lstm_out.unsqueeze(1)
        # lstm_out = F.relu(self.convl3(lstm_out).squeeze(3))     # 64*100*44
        lstm_out = [F.relu(conv(lstm_out)).squeeze(3) for conv in self.convsl]
        # print(lstm_out)             # [torch.FloatTensor of size 64x100x47] 46 45

        # lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2))
        # # print(lstm_out.size())  # torch.Size([64, 100, 1])
        # lstm_out = lstm_out.squeeze(2)  # 10*5  64*100

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in lstm_out]
        x = torch.cat(x, 1)
        # print(x)                    # [torch.FloatTensor of size 64x300]

        #lstm_out = torch.cat(lstm_out, 1)
        x = self.dropout(x)
        # lstm_out = lstm_out.view(len(sentence), -1)
        # print(x)                        # [torch.FloatTensor of size 64x300]
        y = self.hidden2label1(F.tanh(x))
        # print(y)                          # [torch.FloatTensor of size 64x2]
        # y = self.hidden2label2(F.tanh(y))
        log_probs = F.log_softmax(y)
        # log_probs = y
        return log_probs
