#!/usr/bin/env python
# encoding: utf-8
import time

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import preprocess.SingleInputModulePreprocess as sip
import preprocess.DoubleInputModulePreprocess as dip
from collections import defaultdict


def read_movielen_from_file(path, inputType='two'):
    data = pd.read_csv(path)
    sparse_feature = ['movie_id', 'user_id', 'gender',
                      'age', 'occupation', 'zip', 'genres']
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
    user_profile = data[["user_id", "gender", "age", "occupation", "zip", "genres"]].drop_duplicates('user_id')
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)
    if inputType == 'two':
        train_set, test_set = dip.gen_movielen_data_set(data)

        train_model_input, train_label = dip.gen_movielen_model_input(train_set, user_profile)
        test_model_input, test_label = dip.gen_movielen_model_input(test_set, user_profile)
    elif inputType == 'one':
        # TODO
        pass
    else:
        train_model_input = None
        train_label = None
        test_model_input = None
        test_label = None
    print("train_model_input: ", train_model_input)
    print("train_label: ", train_label)
    print("test_model_input: ", test_model_input)
    print("test_label: ", test_label)
    return train_model_input, train_label, test_model_input, test_label


def Multi_input_model_test():
    path = r'F:\csw\MyProject\dataset\movielens.txt'
    data = pd.read_csv(path)
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
    user_profile = data[["user_id", "gender", "age", "occupation", "zip", "genres"]]
    print(user_profile)
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')
    print("user_profile: ", len(user_profile))
    print("item_profile: ", len(item_profile))


    user_profile.set_index("user_id", inplace=True)
    print(user_profile)


def read_amazon_from_file(rating_path):
    sparse_features = ["user_id", "asin", "rating", "review_time", "price", "salesRank", "bought_together",
                       "also_bought", "also_viewed", "buy_after_viewing"]

    rating_data = pd.read_csv(rating_path)
    print(rating_data)
    item_profile = rating_data[["user_id",'asin', "price", "sales_rank", "bought_together", "also_bought",
                              "buy_after_viewing"]].drop_duplicates("asin")
    item_profile.set_index("user_id", inplace=True)
    print(item_profile)

    rating_data = pd.read_csv(rating_path)
    print(rating_data)


def test():
    item_path = r'G:\MyProject\dataset\meta_All_Electronics_test.csv'
    rating_path = r'G:\MyProject\dataset\ratings_All_Electronics.csv'
    # data=pd.read_csv(path,error_bad_lines=False,lineterminator='\n')
    item_data = pd.read_csv(item_path)
    rating_data = pd.read_csv(rating_path)
    rating_item = rating_data['asin']
    item_data.replace("$$$$", np.nan, inplace=True)
    # print(rating_item)
    # data['user_id']=[]
    price = {}
    bought_together = {}
    sales_rank = {}
    also_bought = {}
    also_viewed = {}
    buy_after_viewing = {}
    bad_item = set()
    already_item = []
    for asin in rating_item:
        # # df.isin({'A': [1, 3], 'B': [4, 7, 12]})
        try:
            tmp_data = item_data[item_data['asin'].isin([asin])]
            if tmp_data.values.size == 0:
                bad_item.add(asin)
            else:
                if already_item.__contains__(asin):
                    continue
                already_item.append(asin)
                price[asin] = tmp_data['price'].get_values().tolist()[0]
                bought_together[asin] = tmp_data['bought_together'].get_values().tolist()[0]
                sales_rank[asin] = tmp_data['salesRank'].get_values().tolist()[0]
                also_bought[asin] = tmp_data['also_bought'].get_values().tolist()[0]
                also_viewed[asin] = tmp_data["also_viewed"].get_values().tolist()[0]
                buy_after_viewing[asin] = tmp_data["buy_after_viewing"].get_values().tolist()[0]
        except:
            print(asin)

    print(price)
    print(bad_item)
    tmp_item_data = rating_data.drop(rating_data[rating_data['asin'].isin(bad_item)].index)
    print(tmp_item_data)
    tmp_item_data.to_csv(r'G:\MyProject\dataset\ratings_All_Electronics_clean.csv')
    return price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing


def merge_rating_and_item(price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing, rating_path):
    rating_data = pd.read_csv(rating_path)
    rating_data['price'] = 0
    rating_data['bought_together'] = 0
    rating_data['sales_rank'] = 0
    rating_data['also_bought'] = 0
    rating_data['also_view'] = 0
    rating_data["buy_after_viewing"] = 0
    for index, data in rating_data.iterrows():
        print(index)
        asin = data[2]
        rating_data.loc[index,'price']=price.get(asin)
        rating_data.loc[index,'price'] = price.get(asin)
        rating_data.loc[index,"bought_together"] = bought_together.get(asin)
        rating_data.loc[index,"sales_rank"] = sales_rank.get(asin)
        rating_data.loc[index,"also_bought"] = also_bought.get(asin)
        rating_data.loc[index,"also_viewed"] = also_viewed.get(asin)
        rating_data.loc[index,"buy_after_viewing"] = buy_after_viewing.get(asin)
    print(rating_data)
    print("finish")
    rating_data.to_csv(r'G:\MyProject\dataset\ratings_All_Electronics_final.csv')


if __name__ == '__main__':
    price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing = test()
    print(price)
    rating_path_clean = r'G:\MyProject\dataset\ratings_All_Electronics_clean.csv'
    merge_rating_and_item(price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing,
                          rating_path_clean)
