import torch
import torch.nn as nn


class TextCNNNet(nn.Module):
    def __init__(self, n_input, max_len, n_output, kernel_sizes, topk=1):
        super(TextCNNNet, self).__init__()
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(n_input, n_output, k, padding=k // 2),
                                                  nn.BatchNorm1d(n_output),
                                                  nn.ReLU(),
                                                  nn.MaxPool1d(max_len)) for k in kernel_sizes])

    def forward(self, x):
        x = x.transpose(2, 1)
        conv_x = [conv(x) for conv in self.convs]
        conv_x = torch.cat(conv_x, dim=1)
        conv_x = conv_x.squeeze()

        return conv_x