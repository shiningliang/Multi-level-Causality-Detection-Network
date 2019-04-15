import ujson as json
import numpy as np
import logging
import os

T = 2
NU = 5
NI = 5
NF = 32
demo_name = './data/raw_data/demo.json'

# records = [{"user": [[1, 3, 5], [2, 3, 4]], "item": [[2, 3, 4], [1, 2, 5]]},
#            {"user": [[1, 2, 4], [3, 4, 5]], "item": [[2, 4, 5], [1, 2, 3]]}]
#
# with open(demo_name, 'w') as f:
#     for i in range(len(records)):
#         f.write(json.dumps(records[i]) + '\n')
# f.close()
#


import torch
from torch import nn
from time import time


class COTEMP(nn.Module):
    def __init__(self, u_emb, i_emb, max_len, output_size, n_hidden, n_layer, dropout, logger):
        super(COTEMP, self).__init__()
        start_t = time()
        self.gru_hidden = n_hidden
        self.max_len = max_len
        self.n_layer = n_layer
        self.user_embedding = nn.Embedding(T * NU + 1, NF, padding_idx=0)
        self.item_embedding = nn.Embedding(T * NI + 1, NF, padding_idx=0)

        self.emb_dropout = nn.Dropout(dropout['emb'])
        self.user_encoder = nn.GRU(2 * NF, n_hidden, n_layer, dropout=dropout['layer'], batch_first=True,
                                   bidirectional=True)
        self.item_encoder = nn.GRU(2 * NF, n_hidden, n_layer, dropout=dropout['layer'], batch_first=True,
                                   bidirectional=True)
        self.out_fc = nn.Linear(4 * n_hidden, output_size)

        self._init_weights(u_emb, i_emb)
        logger.info('Time to build graph: {} s'.format(time() - start_t))

    def _init_weights(self, u_emb, i_emb):
        self.user_embedding.weight.data.copy_(torch.from_numpy(u_emb))
        self.item_embedding.weight.data.copy_(torch.from_numpy(i_emb))

    def forward(self, user_records, item_records, uids, iids):
        batch_size = user_records.shape[0]
        urecords = torch.reshape(user_records, (batch_size * T, self.max_len))
        irecords = torch.reshape(item_records, (batch_size * T, self.max_len))
        user_masks = torch.where(urecords > 0, torch.full_like(urecords, 1), urecords)
        item_masks = torch.where(irecords > 0, torch.full_like(irecords, 1), irecords)

        urecords_embs = self.item_embedding(urecords)  # user的record是item
        irecords_embs = self.user_embedding(irecords)
        uid_embs = self.user_embedding(uids)  # user每个月的embedding
        # uid_embs = torch.unsqueeze(uid_embs, 1)
        # uid_embs = uid_embs.repeat(1, T, 1)
        iid_embs = self.item_embedding(iids)
        # iid_embs = torch.unsqueeze(1, T, 1)
        # iid_embs = iid_embs.repeat(1, T)

        urecords_embs = torch.sum(urecords_embs, dim=1)
        irecords_embs = torch.sum(irecords_embs, dim=1)
        user_masks = torch.sum(user_masks, dim=-1, keepdim=True)
        item_masks = torch.sum(item_masks, dim=-1, keepdim=True)
        user_masks = user_masks.repeat(1, NF).float()
        item_masks = item_masks.repeat(1, NF).float()
        user_avgs = torch.div(urecords_embs, user_masks)
        item_avgs = torch.div(irecords_embs, item_masks)
        user_avgs = self.emb_dropout(user_avgs)
        user_avgs = torch.reshape(user_avgs, (batch_size, T, NF))
        item_avgs = self.emb_dropout(item_avgs)
        item_avgs = torch.reshape(item_avgs, (batch_size, T, NF))

        user_avgs = torch.cat([user_avgs, uid_embs], dim=2)
        item_avgs = torch.cat([item_avgs, iid_embs], dim=2)

        uout, ustate = self.user_encoder(user_avgs)
        iout, istate = self.item_encoder(item_avgs)

        ustate = ustate.view(self.n_layer, 2, batch_size, self.gru_hidden)
        uforward_state, ubackward_state = ustate[-1][0], ustate[-1][1]
        u_final = torch.cat([uforward_state, ubackward_state], dim=1)
        istate = istate.view(self.n_layer, 2, batch_size, self.gru_hidden)
        iforward_state, ibackward_state = istate[-1][0], istate[-1][1]
        i_final = torch.cat([iforward_state, ibackward_state], dim=1)
        y_final = torch.cat([u_final, i_final], dim=1)
        return self.out_fc(y_final)


def get_batch(records, uids, iids, labels, device):
    u_records = [record['user'] for record in records]
    i_records = [record['item'] for record in records]
    u_records = np.asarray(u_records, dtype=np.int64)
    i_records = np.asarray(i_records, dtype=np.int64)
    uids = np.asarray(uids, dtype=np.int64)
    iids = np.asarray(iids, dtype=np.int64)
    labels = np.asarray(labels, np.int64)

    return torch.from_numpy(u_records).to(device), torch.from_numpy(i_records).to(device), \
           torch.from_numpy(uids).to(device), torch.from_numpy(iids).to(device), torch.from_numpy(labels).to(device)


if __name__ == '__main__':
    logger = logging.getLogger('Rec')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(23333)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    lines = open(demo_name, 'r').readlines()
    records = []
    for line in lines:
        records.append(json.loads(line))

    uids = [1, 2]
    iids = [2, 1]

    for record in records:
        user = record['user']
        item = record['item']
        for t in range(T):
            ut = user[t]  # 第t月与user有交互的item list
            for idx, ut_i in enumerate(ut):
                ut[idx] = t * NI + ut_i  # 按第t月寻找组 按id偏移
                # ut[idx] = (ut_i - 1) * T + t + 1
            it = item[t]
            for idx, it_u in enumerate(it):
                it[idx] = t * NU + it_u
                # it[idx] = (it_u - 1) * T + t + 1

    UEM = np.random.normal(0., 0.01, (T * NU + 1, NF))
    UEM[0] = 0.
    IEM = np.random.normal(0., 0.01, (T * NI + 1, NF))
    IEM[0] = 0.

    labels = [1, 0]
    T_uids, T_iids = [], []
    for uid in uids:
        T_uids.append([t * NU + uid for t in range(T)])

    for iid in iids:
        T_iids.append([t * NI + iid for t in range(T)])

    dropout = {'emb': 0.3, 'layer': 0.3}
    model = COTEMP(UEM, IEM, 3, 2, 32, 2, dropout, logger).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_num = len(records)
    train_batch = 2
    clip = 0.2
    period = 1
    model.train()
    train_loss = []
    n_batch_loss = 0
    for batch_idx, batch in enumerate(range(0, train_num, train_batch)):
        start_idx = batch
        end_idx = start_idx + train_batch
        b_user_records, b_item_records, b_uids, b_iids, b_labels = get_batch(records[start_idx:end_idx],
                                                                             T_uids, T_iids, labels, device)

        optimizer.zero_grad()
        outputs = model(b_user_records, b_item_records, b_uids, b_iids)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, b_labels)
        loss.backward()
        if clip > 0:
            # 梯度裁剪，输入是(NN参数，最大梯度范数，范数类型=2)，一般默认为L2范数
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        n_batch_loss += loss.item()
        bidx = batch_idx + 1
        if bidx % period == 0:
            logger.info('AvgLoss batch [{} {}] - {}'.format(bidx - period + 1, bidx, n_batch_loss / period))
            n_batch_loss = 0
        train_loss.append(loss.item())


    print('Hello World')
