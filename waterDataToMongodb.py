# author    : Charles
# time      ：2020/7/16  11:55 
# file      ：waterDataToMongodb.PY
# project   ：WaterAnalysis
# IDE       : PyCharm

import pandas as pd
import pymongo
import json
import WaterQualityDetection3


def update_raw_data():
    water = pd.read_csv('2018_2019xuhui.csv')
    water = water[water['testtime'] > '2019-07-25 16:00:00.000']
    water_json_raw_list = json.loads(water.T.to_json()).values()
    my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@10.200.43.91:27017")
    my_db = my_client['water']
    my_col = my_db['dataRaw']
    x = my_col.insert_many(water_json_raw_list)
    print(len(x.inserted_ids), '个文档已插入')


def write_from_json_to_mongodb(source='data.json', destination='waterAndAD2'):
    my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@10.200.43.91:27017")
    my_db = my_client['water']
    my_col2 = my_db[destination]
    data = json.load(source)
    my_col2.delete_many({})
    x = my_col2.insert_many(data)
    print(len(x.inserted_ids), '个文档已插入')


def data_analysis():
    my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@10.200.43.91:27017")
    my_db = my_client['water']
    my_col2 = my_db['dataRaw']
    # y = my_col2.find({"testtime": {"$gt": "2019-07-24 16:00:00.000"}})
    # 这里需要对全量数据进行处理
    y = my_col2.find({})
    water_data_raw = pd.DataFrame(list(y))
    order = ['siteno', 'temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N', 'flux',
             'testtime']
    water_data_raw = water_data_raw[order]
    #  对原始数据进行异常分析
    WQD = WaterQualityDetection3.WaterQualityDetection(water_data_raw)
    WQD.CoreAnalysis()


if __name__ == '__main__':
    # update_data()
    print("done")
