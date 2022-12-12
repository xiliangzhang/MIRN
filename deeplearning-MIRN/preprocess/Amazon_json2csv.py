#!/usr/bin/env python
# encoding: utf-8

import json
import pandas as pd
import numpy as np
from openpyxl import Workbook

def read_json_test(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.replace("'", "\"")
                d = json.loads(line)
                print(d)
            except:
                continue

def review_json2csv(input_path, output_path):
    wb = Workbook()
    sheet = wb.active
    sheet.title = 'review_data'
    sheet["A1"].value = 'reviewerID'
    sheet["B1"].value = 'asin'
    sheet["C1"].value = 'reviewerName'
    sheet["D1"].value = 'vote'
    sheet["E1"].value = 'style'
    sheet["F1"].value = 'reviewText'
    sheet["G1"].value = 'overall'
    sheet["H1"].value = 'summary'
    sheet["I1"].value = 'unixReviewTime'
    sheet["J1"].value = 'reviewTime'
    sheet["K1"].value = 'image'
    count = 2
    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.replace("'", "\"")
                d = json.loads(line)
                d = add_review_keys(d)
                sheet["A" + str(count)].value = d['reviewerID']
                sheet["B" + str(count)].value = d['asin']
                sheet["C" + str(count)].value = d['reviewerName']
                sheet["D" + str(count)].value = d['vote']
                sheet["E" + str(count)].value = d['style']
                sheet["F" + str(count)].value = d['reviewText']
                sheet["G" + str(count)].value = d['overall']
                sheet["H" + str(count)].value = d['summary']
                sheet["I" + str(count)].value = d['unixReviewTime']
                sheet["J" + str(count)].value = d['reviewTime']
                sheet["K" + str(count)].value = d['image']
                count += 1
            except:
                continue
    wb.save(output_path)

def meta_json2csv(input_path, output_path):
    wb = Workbook()
    sheet = wb.active
    sheet.title = 'metadata'
    sheet["A1"].value = 'asin'
    sheet["B1"].value = 'title'
    sheet["C1"].value = 'feature'
    sheet["D1"].value = 'description'
    sheet["E1"].value = 'price'
    sheet["F1"].value = 'image'
    sheet["G1"].value = 'bought_together'
    sheet["H1"].value = 'salesRank'
    sheet["I1"].value = 'brand'
    sheet["J1"].value = 'categories'
    sheet["K1"].value = 'tech1'
    sheet["L1"].value = 'tech2'
    sheet["M1"].value = 'similar'
    sheet["N1"].value = 'also_bought'
    sheet["O1"].value = 'also_viewed'
    sheet["P1"].value = 'buy_after_viewing'
    count = 2
    cal_err=0
    with open(input_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.replace("'", "\"")
                d = json.loads(line)
                d = add_meta_keys(d)
                sheet["A" + str(count)].value = d['asin']
                sheet["B" + str(count)].value = d['title']
                sheet["C" + str(count)].value = d['feature']
                sheet["D" + str(count)].value = d['description']
                sheet["E" + str(count)].value = d['price']
                sheet["F" + str(count)].value = d['imUrl']
                sheet["G" + str(count)].value = d['bought_together']
                sheet["H" + str(count)].value = d['salesRank']
                sheet["I" + str(count)].value = d['brand']
                sheet["J" + str(count)].value = d['categories']
                sheet["K" + str(count)].value = d['tech1']
                sheet["L" + str(count)].value = d['tech2']
                sheet["M" + str(count)].value = d['similar']
                sheet["N" + str(count)].value = d['also_bought']
                sheet["O" + str(count)].value = d['also_viewed']
                sheet["P" + str(count)].value = d['buy_after_viewing']
                count += 1
            except:
                cal_err+=1
                continue
    wb.save(output_path)
    wb.close()
    print("cal_err",cal_err)
    print("count",count)

def add_review_keys(Ndict):
    Keys = ['reviewerID', 'asin', 'reviewerName', 'vote', 'style', 'reviewText', 'overall', 'summary',
            'unixReviewTime', 'reviewTime', 'image']
    keySet = Ndict.keys()
    for key in Keys:
        if key not in keySet:
            Ndict[key] = "$$$$"
    return Ndict

def add_meta_keys(Ndict):
    Keys = ['asin', 'title', 'feature', 'description', 'price', 'imUrl', 'salesRank',
            'brand', 'categories', 'tech1', 'tech2', 'similar','related']
    keySet = Ndict.keys()
    for key in Keys:
        if key not in keySet:
            Ndict[key] = "$$$$"
    Ndict['brand'] = '$$$$'
    categoriesList = Ndict['categories'][0]
    Ndict['categories']=''
    for category in categoriesList:
        Ndict['categories']+=category
    rankDict = Ndict['salesRank']
    tmp=str(rankDict)[1:-2]
    Ndict['salesRank'] = tmp
    Ndict['also_viewed']='$$$$'
    Ndict['buy_after_viewing']='$$$$'
    Ndict['also_bought']='$$$$'
    Ndict['bought_together']='$$$$'
    # Ndict['related'].keys()=['bought_together', 'also_bought', 'also_viewed', 'buy_after_viewing']
    try:
        for key in Ndict['related'].keys():
            tmp=str(Ndict['related'][key])
            tmp=tmp.lstrip('[')
            tmp=tmp.rstrip(']')
            Ndict[key]=tmp
            # print(Ndict[key])
    except:
        # print("related 转换失败！")
        # print(Ndict['related'])
        pass
    return Ndict

def read_item_and_rating_file(item_path,rating_path,output_path):
    item_data = pd.read_csv(item_path)
    rating_data = pd.read_csv(rating_path)
    rating_item = rating_data['asin']
    item_data.replace("$$$$", np.nan, inplace=True)
    price = {}
    bought_together = {}
    sales_rank = {}
    also_bought = {}
    also_viewed = {}
    buy_after_viewing = {}
    bad_item = set()
    already_item = []
    for asin in rating_item:
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
    tmp_item_data.to_csv(output_path)
    return price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing

def merge_rating_and_item(price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing, rating_path,final_output_path):
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
    rating_data.to_csv(final_output_path)

if __name__ == '__main__':
    # read_json(review_path)
    review_path = r'G:\MyProject\dataset\reviews_All_Electronics.json'
    review_path_out = r'G:\MyProject\dataset\reviews_All_Electronics.csv'
    review_json2csv(review_path, review_path_out)

    meta_path = r'G:\MyProject\dataset\meta_Clothing_Shoes_and_Jewelry.json'
    meta_path_out = r'G:\MyProject\dataset\meta_Clothing_Shoes_and_Jewelry.csv'
    meta_json2csv(meta_path, meta_path_out)

    item_path = r'G:\MyProject\dataset\meta_All_Electronics_test.csv'
    rating_path = r'G:\MyProject\dataset\ratings_All_Electronics.csv'
    output_path = r'G:\MyProject\dataset\ratings_All_Electronics_clean.csv'
    price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing = read_item_and_rating_file(item_path,rating_path,output_path)

    rating_path_clean = r'G:\MyProject\dataset\ratings_All_Electronics_clean.csv'
    final_output_path=r'G:\MyProject\dataset\ratings_All_Electronics_final.csv'
    merge_rating_and_item(price, bought_together, sales_rank, also_bought, also_viewed, buy_after_viewing,
                          rating_path_clean,final_output_path)
