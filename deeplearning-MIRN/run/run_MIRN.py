#!/usr/bin/env python
# encoding: utf-8


import pandas as pd
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from deepctr.layers import NoMask
from sklearn.preprocessing import LabelEncoder

from deeplearning.layers import core, interaction, sequence
from preprocess import SingleInputModulePreprocess
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from deeplearning.models import MIRN
from deeplearning.utils import sampledsoftmaxloss, recall_N
from tensorflow.python.keras import Model
from deeplearning import LossHistory
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.utils import plot_model

model_path = r''
embedding_dim = 32
epochs = 100
amazon_path = r'G:\MyProject\dataset\ratings_All_Electronics_final.csv'
movielen_path = r'G:\MyProject\dataset\movielens.txt'
test_path = r'E:\MyProject\Test\movielens_sample.txt'
validation_split = 0.01
weight_name = r'G:\MyProject\weights\MIRN_weight.h5'
MIRN_object = {}
is_training = True


def read_data_from_file(data_set):
    if data_set == 'movielen':
        return read_movielen_from_file(test_path)
    elif data_set == 'amazon':
        return read_amazon_from_file(amazon_path)


def read_movielen_from_file(path):
    data = pd.read_csv(path)
    features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "genres"]
    feature_max_idx = {}
    for feature in features:
        print('========================================================= ')
        print("start encoding data!")
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
    print("==============================================================")
    print('read data finished!')
    return data, feature_max_idx


def read_amazon_from_file(path):
    data = pd.read_csv(path)
    features = ['user_id', 'asin', 'rating', 'review_time', 'price', 'sales_rank']
    feature_max_idx = {}

    for feature in features:
        print('========================================================= ')
        print("start encoding data!")
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
    print("==============================================================")
    print('read data finished!')
    return data, feature_max_idx


def amazon2input(data):
    # sparse_feature
    # ['asin','user_id','rating','price','sales_ranking']
    item_profile = data[['asin']].drop_duplicates('asin')
    print("data条数: ", len(data))
    print("item_profile: ", len(item_profile))

    print('=================================================================')
    print('start convert data to model input!')
    train_set, test_set = SingleInputModulePreprocess.gen_amazon_data_set(data)

    train_model_input, train_label = SingleInputModulePreprocess.gen_amazon_model_input(train_set)
    test_model_input, test_label = SingleInputModulePreprocess.gen_amazon_model_input(test_set)
    print('=================================================================')
    print('data convert finished!')
    return train_model_input, train_label, test_model_input, test_label


def movielen2input(data):
    # sparse_features:
    # ["movie_id", "user_id", "gender", "age", "occupation","zip", "genres"]
    user_profile = data[["user_id", "gender", "age", "occupation", "zip", "genres"]].drop_duplicates("user_id")

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')
    print("data条数: ", len(data))
    print("user_profile: ", len(user_profile))
    print("item_profile: ", len(item_profile))

    user_profile.set_index("user_id", inplace=True)
    print('=================================================================')
    print('start convert data to model input!')
    train_set, test_set = SingleInputModulePreprocess.gen_movielen_data_set(data)

    train_model_input, train_label = SingleInputModulePreprocess.gen_movielen_model_input(train_set, user_profile, 5)
    test_model_input, test_label = SingleInputModulePreprocess.gen_movielen_model_input(test_set, user_profile, 5)
    print('=================================================================')
    print('data convert finished!')
    return train_model_input, train_label, test_model_input, test_label


def define_amazon_embedding_shape(feature_max_idx):
    print('=================================================================')
    print("start convert feature columns!")
    user_behavior_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                             VarLenSparseFeat(SparseFeat('hist_asin', feature_max_idx['asin'], 16,
                                                         embedding_name='asin'), 50, 'mean', 'hist_len'),

                             ]
    item_feature_columns = [SparseFeat('asin', feature_max_idx['asin'], 16),
                            SparseFeat('price', feature_max_idx['price'], 16),
                            SparseFeat('rating', feature_max_idx['rating'], 16),
                            SparseFeat('sales_rank', feature_max_idx['sales_rank'], 16),
                            ]
    print('=================================================================')
    print("feature_columns finished!")
    return user_behavior_columns, item_feature_columns


def define_movielen_embedding_shape(feature_max_idx):
    print(feature_max_idx)
    print('=================================================================')
    print("start define feature columns!")
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
                            SparseFeat("gender", feature_max_idx['gender'], 16),
                            SparseFeat("age", feature_max_idx['age'], 16),
                            SparseFeat("occupation", feature_max_idx['occupation'], 16),
                            SparseFeat("zip", feature_max_idx['zip'], 16),
                            VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
                                                        embedding_name="movie_id"), 5, 'mean',
                                             'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]
    print('=================================================================')
    print("feature_columns finished!")
    return user_feature_columns, item_feature_columns


def train_model(train_model_input, train_label, user_feature_columns, item_feature_columns):
    print('=================================================================')
    print("start loading model!")
    model = MIRN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32))
    model.summary()
    model.compile(optimizer='adam', loss=sampledsoftmaxloss, metrics=['accuracy'])

    callbacks = LossHistory()
    history = model.fit(train_model_input, train_label,
                        batch_size=256, epochs=epochs, verbose=1, validation_split=validation_split,
                        callbacks=[callbacks])
    print('=================================================================')
    print("training finished! start save weights")
    print(callbacks.losses)
    print(callbacks.acc)
    with open(r'F:\csw\MyProject\log\lossLog.txt', 'a', encoding='utf-8') as f:
        f.write("\n")
        f.write("======================================================")
        for loss in callbacks.losses:
            f.write('\n')
            f.write(loss)
        f.write("======================================================")
        for acc in callbacks.acc:
            f.write('\n')
            f.write(acc)
    f.close()
    model.save(weight_name)


def MIRN_predict(data,test_model_input,test_label):
    import faiss
    model = load_model(model_path,custom_objects=MIRN_object)
    item_profile = data[['asin']].drop_duplicates('asin')
    all_item_model_input={'asin':item_profile['asin'].values}

    user_embedding_model = Model(inputs=model.user_input,outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input,outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_model_input,batch_size=2**12)
    item_embs = item_embedding_model.predict(all_item_model_input,batch_size=2**12)

    # print(user_embs.shape)
    # print(item_embs.shape)

    index=faiss.IndexFlatIP(embedding_dim)
    index.add(item_embs)
    D, I = index.search(np.ascontiguousarray(user_embs), 50)
    s = []
    hit = 0
    for i, uid in enumerate(test_model_input['user_id']):
        try:
            pred = [item_profile['movie_id'].values[x] for x in I[i]]
            filter_item = None
            recall_score = recall_N(test_label[uid], pred, N=50)
            s.append(recall_score)
            if test_label[uid] in pred:
                hit += 1
        except:
            print(i)
    print("recall", np.mean(s))
    print("hr", hit / len(test_model_input['user_id']))


def run(data_name):
    data, feature_max_idx = read_data_from_file(data_name)
    if data_name == 'amazon':
        train_model_input, train_label, test_model_input, test_label = amazon2input(data)
        user_feature_column, item_feature_columns = define_amazon_embedding_shape(feature_max_idx)
    elif data_name == 'movielen':
        train_model_input, train_label, test_model_input, test_label = movielen2input(data)
        user_feature_column, item_feature_columns = define_movielen_embedding_shape(feature_max_idx)
    if is_training:
        K.set_learning_phase(True)
        train_model(train_model_input, train_label, user_feature_column, item_feature_columns)
        K.set_learning_phase(False)
        print('<==================================================>')
        print('training down！')
    else:
        pass
        # MIRN_predict(data, test_model_input, test_label)


if __name__ == '__main__':
    run('movielen')
