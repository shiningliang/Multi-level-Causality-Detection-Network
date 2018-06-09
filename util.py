import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_record_parser(config, max_len):
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
        token_len = features['token_len']
        label = features['label']
        return eid, token_ids, token_len, label

    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(
        config.capacity).batch(config.batch_size).repeat(config.epochs)

    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).batch(config.batch_size).repeat()

    return dataset


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    losses, refs, pres = [], [], []
    for i in range(num_batches):
        eids, loss, labels = sess.run([model.eid, model.loss, model.pre_labels],
                                      feed_dict={handle: str_handle} if handle is not None else None)
        losses.append(loss)
        pres += labels.tolist()
        for pid in eids:
            refs.append(eval_file[str(pid)])

    refs = np.asarray(refs, dtype=np.int32)
    pres = np.asarray(pres, dtype=np.int32)
    avg_loss = np.mean(losses)
    acc = accuracy_score(refs, pres)
    precision = precision_score(refs, pres)
    recall = recall_score(refs, pres)
    f1 = f1_score(refs, pres)
    print(confusion_matrix(refs, pres))
    # target_names = ['non-causal', 'causal']
    # print(classification_report(refs, pres, target_names))

    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=avg_loss), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=acc), ])
    pre_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=precision), ])
    rec_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=recall), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/acc".format(data_type), simple_value=f1), ])
    return avg_loss, acc, precision, recall, f1, [loss_sum, acc_sum, pre_sum, rec_sum, f1_sum]
