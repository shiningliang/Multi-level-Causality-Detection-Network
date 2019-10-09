import warnings
import os
import os.path as op


class DefaultConfig(object):
    def __init__(self):
        self.prepare = False
        self.build = False
        self.train = False
        self.evaluate = False
        self.case = False
        self.gpu = '0'
        self.seed = 23333

        self.disable_cuda = False
        self.warmup = 0.5
        self.lr = 0.0001
        self.weight_decay = 0.0003
        self.clip = 0.35
        self.emb_dropout = 0.3
        self.layer_dropout = 0.3
        self.batch_train = 32
        self.batch_eval = 64
        self.epochs = 10
        self.optim = 'Adam'
        self.patience = 2
        self.period = 1000
        self.num_threads = 8
        self.max_len = {'full': 128, 'pre': 64, 'alt': 8, 'cur': 64}
        self.w2v_type = 'wiki'
        self.n_emb = 300
        self.n_hidden = 64
        self.n_layer = 2
        self.n_block = 4
        self.n_head = 4
        self.is_pos = False
        self.is_sinusoid = True
        self.is_ffn = True
        self.is_fc = True
        self.n_kernel = 3
        self.n_kernels = [2, 3, 4]
        self.n_filter = 50
        self.kmax_pooling = 2
        self.window_size = 10
        self.n_class = 2

        self.task = 'bootstrapped'
        self.model = 'MCDN'
        self.train_file = 'altlex_train_bootstrapped.tsv'
        self.valid_file = 'altlex_dev.tsv'
        self.test_file = 'altlex_gold.tsv'
        self.transfer_file1 = '2010_random_filtered.json'
        self.transfer_file2 = '2010_full_filtered.json'
        self.raw_dir = 'data/raw_data/'
        self.processed_dir = 'data/processed_data/torch'
        self.outputs_dir = 'outputs/'
        self.model_dir = 'models/'
        self.result_dir = 'results/'
        self.pics_dir = 'pics/'
        self.summary_dir = 'summary/'
        self.log_path = None

    def _parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('Config Space:')
        for k, v in vars(self).items():
            if not k.startswith('__'):
                print('%s=%s' % (k, v))

        self.processed_dir = op.join(self.processed_dir, self.task, str(self.max_len['full']))
        self.model_dir = op.join(self.outputs_dir, self.task, self.model, self.model_dir)
        self.result_dir = op.join(self.outputs_dir, self.task, self.model, self.result_dir)
        self.pics_dir = op.join(self.outputs_dir, self.task, self.model, self.pics_dir)
        self.summary_dir = op.join(self.outputs_dir, self.task, self.model, self.summary_dir)
        for dir_path in [self.raw_dir, self.processed_dir, self.model_dir, self.result_dir, self.pics_dir,
                         self.summary_dir]:
            if not op.exists(dir_path):
                os.makedirs(dir_path)

        # 运行记录文件
        self.train_record_file = op.join(self.processed_dir, 'train.pkl')
        self.valid_record_file = op.join(self.processed_dir, 'valid.pkl')
        self.test_record_file = op.join(self.processed_dir, 'test.pkl')
        self.transfer_record_file1 = op.join(self.processed_dir, 'transfer1.pkl')
        self.transfer_record_file2 = op.join(self.processed_dir, 'transfer2.pkl')
        # 计数文件
        self.train_meta = op.join(self.processed_dir, 'train_meta.json')
        self.valid_meta = op.join(self.processed_dir, 'valid_meta.json')
        self.test_meta = op.join(self.processed_dir, 'test_meta.json')
        self.transfer_meta1 = op.join(self.processed_dir, 'transfer_meta1.json')
        self.transfer_meta2 = op.join(self.processed_dir, 'transfer_meta2.json')
        self.shape_meta = op.join(self.processed_dir, 'shape_meta.json')

        self.train_annotation = op.join(self.processed_dir, 'train_annotations.txt')
        self.valid_annotation = op.join(self.processed_dir, 'valid_annotations.txt')
        self.test_annotation = op.join(self.processed_dir, 'test_annotations.txt')

        self.corpus_file = op.join(self.processed_dir, 'corpus.txt')
        self.token_emb_file = op.join(self.processed_dir, 'token_emb.pkl')
        self.token2id_file = op.join(self.processed_dir, 'token2id.json')
        self.id2token_file = op.join(self.processed_dir, 'id2token.json')

        if self.w2v_type == 'wiki':
            self.w2v_file = './data/processed_data/wiki.en.pkl'
        elif self.w2v_type == 'google':
            self.w2v_file = './data/processed_data/google.news.pkl'
        elif self.w2v_type == 'glove6':
            self.w2v_file = './data/processed_data/glove.6B.pkl'
        elif self.w2v_type == 'glove840':
            self.w2v_file = './data/processed_data/glove.840B.pkl'
        elif self.w2v_type == 'fastText':
            self.w2v_file = './data/processed_data/fastText.pkl'


opt = DefaultConfig()
