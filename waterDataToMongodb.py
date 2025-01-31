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
    my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@210.14.69.108:27017")
    my_db = my_client['water']
    my_col = my_db['dataRaw']
    x = my_col.insert_many(water_json_raw_list)
    print(len(x.inserted_ids), '个文档已插入')


def rename_collection(origin='waterAndAD2', destination='waterAndAD3'):
    my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@210.14.69.108:27017")
    my_db = my_client['water']
    my_col = my_db[origin]
    my_col.rename(destination)


def write_from_json_to_mongodb(source='./data/data_column_list', destination='waterAndAD2'):
    my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@210.14.69.108:27017")
    my_db = my_client['water']
    my_col2 = my_db[destination]
    my_col2.delete_many({})
    count = 0
    file = open(source)
    while 1:
        line = file.readline()
        if not line:
            break
        line = line[:-2]
        data = json.loads(line)
        my_col2.insert_one(data)
        count += 1
    print(count, '个文档已插入')


# 测试文件 用的方法 实际不需要使用
def demo_t(source='./data/data_column_list'):
    file = open(source)
    while 1:
        line = file.readline()
        if not line:
            break
        line = line[:-2]
        print(json.loads(line))
        break


# 这是理论上针对数据从分析训练模型训练建模到入库的过程，实际并未使用
def data_analysis():
    my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@210.14.69.108:27017")
    my_db = my_client['water']
    my_col2 = my_db['dataRaw']
    # y = my_col2.find({"testtime": {"$gt": "2019-07-24 16:00:00.000"}})
    # 这里需要对全量数据进行处理
    y = my_col2.find({})
    water_data_raw = pd.DataFrame(list(y))
    order = ['siteno', 'temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N', 'flux',
             'testtime']
    water_data_raw = water_data_raw[order]
    # water_data_raw是原始数据
    water2 = water_data_raw[water_data_raw['temperature'].notnull()]
    water2 = water2.fillna(method='ffill')
    # water2 是进行处理后理论上无空数据的部分
    #  对原始数据进行异常分析
    WQD = WaterQualityDetection3.WaterQualityDetection(water2)
    WQD.CoreAnalysis()


if __name__ == '__main__':
    # 本文件建议不好直接运行，选择有需要的函数进行执行
    # update_raw_data()
    # demo_t()
    # write_from_json_to_mongodb()
    # rename_collection('waterAndAD2', 'waterAndAD3')
    # rename_collection('waterAndAD', 'waterAndAD2')
    # rename_collection('waterAndAD3', 'waterAndAD')
    print("done")
