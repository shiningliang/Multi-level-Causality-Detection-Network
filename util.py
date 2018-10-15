import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_record_parser(max_len):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               'eid': tf.FixedLenFeature([], tf.int64),
                                               'token_ids': tf.FixedLenFeature([], tf.string),
                                               'token_len': tf.FixedLenFeature([], tf.int64),
                                               'label': tf.FixedLenFeature([], tf.int64)
                                           })
        eid = features['eid']
        token_ids = tf.reshape(tf.decode_raw(features['token_ids'], tf.int32), [max_len])
        token_len = tf.to_int32(features['token_len'])
        label = tf.to_int32(features['label'])
        return eid, token_ids, token_len, label
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(
        config.capacity).batch(config.train_batch).repeat()
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).batch(config.valid_batch).repeat(config.epochs)
    return dataset


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    losses, refs, pres = [], [], []
    metrics = {}
    for i in range(num_batches):
        eids, loss, labels = sess.run([model.eid, model.loss, model.pre_labels],
                                      feed_dict={handle: str_handle} if handle is not None else None)
        losses.append(loss)
        pres += labels.tolist()
        for pid in eids:
            refs.append(eval_file[str(pid)])

    refs = np.asarray(refs, dtype=np.int32)
    pres = np.asarray(pres, dtype=np.int32)
    metrics['loss'] = np.mean(losses)
    metrics['acc'] = accuracy_score(refs, pres)
    metrics['precision'] = precision_score(refs, pres)
    metrics['recall'] = recall_score(refs, pres)
    metrics['f1'] = f1_score(refs, pres)
    print(confusion_matrix(refs, pres))
    # target_names = ['non-causal', 'causal']
    # print(classification_report(refs, pres, target_names))

    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=metrics['loss']), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=metrics['acc']), ])
    pre_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=metrics['precision']), ])
    rec_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=metrics['recall']), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=metrics['f1']), ])
    return metrics, [loss_sum, acc_sum, pre_sum, rec_sum, f1_sum]


def print_metrics(metrics, logger, type):
    logger.info('{} metrics'.format(type))
    logger.info('Loss - {}'.format(metrics['loss']))
    logger.info('Acc - {}'.format(metrics['acc']))
    logger.info('Precision - {}'.format(metrics['precision']))
    logger.info('Recall - {}'.format(metrics['recall']))
    logger.info('F1 - {}'.format(metrics['f1']))