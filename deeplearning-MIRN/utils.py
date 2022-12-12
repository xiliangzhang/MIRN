#!/usr/bin/env python
# encoding: utf-8


import json
import logging
from threading import Thread
from itertools import chain

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, create_embedding_matrix, embedding_lookup, \
    get_dense_input, varlen_embedding_lookup, get_varlen_pooling_list, mergeDict


import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.metrics import top_k_categorical_accuracy

def recall_N(y_true, y_pred, N=50):
    y_pred1=K.eval(y_pred)
    return len(set(y_pred1[:N]) & set(y_true)) * 1.0 / len(y_true)

def top_3_metric(y_true,y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,k=3)

def recall_N_metric(y_true,y_pred,N=50):
    tp=K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    pp = K.sum(K.round(K.clip(y_true,0,1)))
    recall=tp/(pp+K.epsilon)
    return recall

def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)

def get_item_embedding(item_embedding, item_input_layer):
    return Lambda(lambda x: tf.squeeze(tf.gather(item_embedding, x), axis=1))(
        item_input_layer)

def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.python.org/pypi/deepmatch/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if ver.is_prerelease or ver.is_postrelease:
                        continue
                    latest_version = max(latest_version, ver)
                if latest_version > version:
                    logging.warning(
                        '\nDeepMatch version {0} detected. Your version is {1}.\nUse `pip install -U deepmatch` to upgrade.Changelog: https://github.com/shenweichen/DeepMatch/releases/tag/v{0}'.format(
                            latest_version, version))
        except:
            print("Please check the latest version manually on https://pypi.org/project/deepmatch/#history")
            return

    Thread(target=check, args=(version,)).start()

def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False, embedding_matrix_dict=None):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    if embedding_matrix_dict is None:
        embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                        seq_mask_zero=seq_mask_zero)

    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list
