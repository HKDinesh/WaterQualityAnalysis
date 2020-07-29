# author    : Charles
# time      ：2020/7/16  11:51 
# file      ：WaterQualityDetection3.py.PY
# project   ：WaterAnalysis
# IDE       : PyCharm

import json
import logging
import os

import numpy as np
import pandas as pd
import pymongo
from dateutil.parser import parse
from fbprophet import Prophet
from sklearn.externals import joblib
from sklearn import preprocessing
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN


class WaterQualityDetection:

    def __init__(self, water_data1):
        order = ['siteno', 'temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N',
                 'flux', 'testtime']
        water_data1 = water_data1[order]
        water_data1['testtime'] = water_data1['testtime'].apply(parse)
        water_data1['siteno'] = water_data1['siteno'].astype(np.str)
        # 利用有无.copy()的性质，简化了代码，但同时降低了代码的可读性
        self.water_data = water_data1.copy()
        self.result = water_data1.copy()
        self.SS3 = pd.DataFrame(
            columns=['temperature3', 'pH3', 'EC3', 'ORP3', 'DO3', 'turbidity3', 'transparency3', 'COD3', 'P3', 'NH3N3',
                     'flux3'])
        self.Graph1 = {}
        self.Graph2 = {}
        self.Graph3 = {}
        self.S2Model = {}
        self.S3Model = {}

    def S1(self):

        water_data = self.water_data
        result = self.result

        # 设备异常，0表示异常，1表示正常
        water_data['S1_temperature'] = 0
        water_data['S1_pH'] = 0
        water_data['S1_EC'] = 0
        water_data['S1_DO'] = 0
        water_data['S1_turbidity'] = 0
        water_data['S1_transparency'] = 0
        water_data['S1_COD'] = 0
        water_data['S1_NH3N'] = 0
        water_data['S1_P'] = 0

        water_data.loc[(water_data.temperature >= 0) & (water_data.temperature <= 80), ['S1_temperature']] = 1
        water_data.loc[(water_data.pH >= 2) & (water_data.pH <= 17), ['S1_pH']] = 1
        water_data.loc[(water_data.EC >= 0) & (water_data.EC <= 4000), ['S1_EC']] = 1
        water_data.loc[(water_data.DO >= 0) & (water_data.DO <= 80), ['S1_DO']] = 1
        water_data.loc[(water_data.turbidity >= 0) & (water_data.turbidity <= 1000), ['S1_turbidity']] = 1
        water_data.loc[(water_data.transparency >= 0) & (water_data.transparency <= 200), ['S1_transparency']] = 1
        water_data.loc[(water_data.COD >= 0) & (water_data.COD <= 100), ['S1_COD']] = 1
        water_data.loc[(water_data.NH3N >= 0) & (water_data.NH3N <= 20), ['S1_NH3N']] = 1
        water_data.loc[(water_data.P <= 3), ['S1_P']] = 1

        water_data['S1'] = water_data['S1_temperature'] * water_data['S1_pH'] * water_data['S1_EC'] * water_data[
            'S1_DO'] * water_data['S1_turbidity'] * water_data['S1_transparency'] * water_data['S1_COD'] * water_data[
                               'S1_NH3N'] * water_data['S1_P']

        # 转换一级数据信号
        water_data.loc[water_data['S1'] == 0, ['S1']] = 2
        water_data.loc[water_data['S1'] == 1, ['S1']] = 0
        water_data.loc[water_data['S1'] == 2, ['S1']] = 1

        result['设备异常'] = water_data['S1'].values

        def Anomaly1_Names(data):
            dataT = data[13:22]
            dataT.index = ['temperature', 'pH', 'EC', 'DO', 'turbidity', 'transparency', 'COD', 'NH3N', 'P']
            data['S1_names'] = dataT[dataT == 0].index.tolist()
            return data

        water_data = water_data.apply(Anomaly1_Names, axis=1)
        result['设备异常维度'] = water_data['S1_names'].values

    def S2(self):

        self.S1()
        water_data = self.water_data
        result = self.result

        # 数据预处理及模型训练
        clean_data = water_data[water_data['S1'] == 0]
        Y = pd.DataFrame(index=clean_data.index, columns=['S2'])

        X_train = np.array(clean_data.iloc[:, 1:12])
        name = list(clean_data.iloc[:, 1:12].columns.values)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)

        clf1 = IForest(contamination=0.05, max_features=11, bootstrap=True)
        clf2 = KNN(contamination=0.05, n_neighbors=100)
        clf3 = HBOS(contamination=0.05, n_bins=10)
        clf4 = PCA(contamination=0.05)

        clf1.fit(X_train)
        clf2.fit(X_train)
        clf3.fit(X_train)
        clf4.fit(X_train)

        Y['S2'] = clf1.labels_ * clf2.labels_ * clf3.labels_ * clf4.labels_
        water_data = pd.concat([water_data, Y], axis=1)
        # water_data.loc[water_data['S2'].isna(),['S2']]=0，将S1中异常的，在S2中标注为0；

        result['统计异常'] = water_data['S2'].values

        # 寻找异常维度
        from sklearn.neighbors import KernelDensity
        clean_data = water_data[water_data['S1'] == 0]
        dens = pd.DataFrame(index=clean_data.index,
                            columns=['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P',
                                     'NH3N', 'flux'])

        for i in dens.columns:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(clean_data[i].values.reshape(-1, 1))
            dens[i] = np.exp(kde.score_samples(clean_data[i].values.reshape(-1, 1)))
        dens = dens.iloc[:, 0:11].rank()
        dens['S2_names'] = dens.idxmin(axis=1)
        water_data = pd.concat([water_data, dens['S2_names']], axis=1)
        self.water_data = water_data
        result['统计异常维度'] = water_data['S2_names'].values

        # 存储模型
        joblib.dump(scaler, "./water_model/S2_scaler")
        joblib.dump(clf1, "./water_model/S2_Iforest")

    def TsAnomalyDetect(self, dfT, Site, Factor, Alpha=0.95, N_changepoints=25, Range=0.99, year="auto", week="auto",
                        day="auto", season=10.0, changepoint=0.05):
        """
        注意：
        1.输入的数据为两列的DataFrame，第一列表示时间，第二列表示数据
        2.建议对数据进行等距切分，不等距数据也可以使用
        """

        # exec在函数内调用上的坑
        loc = locals()
        exec('dfT = dfT.rename(columns={{ "{}":"ds", "{}":"y" }})'.format(dfT.columns[0], dfT.columns[1]))
        dfT = loc['dfT']

        # 时间序列数据标准化
        dfT['y'] = (dfT['y'] - dfT['y'].mean()) / (dfT['y'].std())

        # 模型训练及预测
        m = Prophet(interval_width=Alpha, n_changepoints=N_changepoints, changepoint_range=Range,  # 趋势的置信区间，而不是实际值的预测区间
                    yearly_seasonality=year, weekly_seasonality=week, daily_seasonality=day,
                    seasonality_prior_scale=season, changepoint_prior_scale=changepoint)
        m.fit(dfT)
        future = m.make_future_dataframe(periods=0)
        forecast = m.predict(future)
        forecast['y'] = dfT['y'].values

        # 同行做运算
        def Anomaly(data):
            if (data['y'] > data['yhat_lower']) & (data['y'] < data['yhat_upper']):
                data['anomaly'] = 0
            else:
                data['anomaly'] = 1
            return data

        joblib.dump(m, "./water_model/" + Site + Factor)

        return forecast.apply(Anomaly, axis=1)

    def S3(self):

        self.S2()
        water_data = self.water_data
        result = self.result
        S3 = self.SS3

        for i in water_data.siteno.unique():

            SitenoT = water_data[water_data.siteno.isin([i])]
            SitenoT = SitenoT.loc[
                (SitenoT['S1'] == 0) & (SitenoT['S2'] == 0), ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity',
                                                              'transparency', 'COD', 'P', 'NH3N', 'flux', 'testtime']]

            count = 0
            Temp = pd.DataFrame(index=SitenoT.index,
                                columns=['temperature3', 'pH3', 'EC3', 'ORP3', 'DO3', 'turbidity3', 'transparency3',
                                         'COD3', 'P3', 'NH3N3', 'flux3'])

            for j in ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N', 'flux']:

                print("model： " + str(i) + " " + str(j))
                dfT = SitenoT[['testtime', j]]
                dfT = dfT.sort_values(by="testtime", ascending=True)
                try:
                    result3 = self.TsAnomalyDetect(dfT=dfT, Site=i, Factor=j, day=2, changepoint=0.5, week=False)
                    Temp[Temp.columns[count]] = result3['anomaly'].values
                except Exception as e:
                    logging.exception(e)
                    print("模型训练失败，被训练数据展示：")
                    print(dfT)

                count = count + 1

            S3 = S3.append(Temp)

        def Anomaly3_Names(data):
            dataT = data.copy()
            dataT.index = ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N',
                           'flux']
            data['S3_names'] = dataT[dataT == 1].index.tolist()
            if np.sum(dataT) > 1:
                data['S3'] = 1
            else:
                data['S3'] = 0
            return data

        S3 = S3.apply(Anomaly3_Names, axis=1)
        water_data = pd.concat([water_data, S3[['S3', 'S3_names']]], axis=1)
        self.water_data = water_data
        result['时间序列异常'] = water_data['S3'].values
        result['时间序列异常维度'] = water_data['S3_names'].values

    def graph1(self):

        water_data = self.water_data

        D = water_data.loc[
            water_data.S1 == 0, ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P',
                                 'NH3N', 'flux', 'testtime', 'siteno']]
        D.index = D['testtime']

        CorrData = self.Graph1
        for i in water_data.siteno.unique():

            SitenoT = D[D.siteno.isin([i])]
            new_d_month = list(SitenoT['testtime'].dt.strftime('%Y-%m').unique())
            SitenoT = SitenoT[
                ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N', 'flux']]
            Site = {}
            for j in new_d_month:
                Site[j] = SitenoT[j].corr()
            CorrData[i] = Site

    def graph2(self):

        water_data = self.water_data
        D = water_data[['S1', 'S2', 'S3', 'testtime', 'siteno']]

        D.index = D['testtime']

        Summary = self.Graph2

        for i in water_data.siteno.unique():

            SitenoT = D[D.siteno.isin([i])]
            new_d_month = list(SitenoT['testtime'].dt.strftime('%Y-%m').unique())
            SitenoT = SitenoT[['S1', 'S2', 'S3']]

            Site = {}
            for j in new_d_month:
                Site[j] = SitenoT[j].sum(axis=0)  # 这里有问题，待修正
            #     当数据有空数据的时候，可能出错并停止运行
            Summary[i] = Site

    def graph3(self):

        S3 = self.SS3
        water_data = self.water_data
        D = water_data[
            ['S1_temperature', 'S1_pH', 'S1_EC', 'S1_DO', 'S1_turbidity', 'S1_transparency', 'S1_COD', 'S1_NH3N',
             'S1_P', 'siteno', 'testtime', 'S2_names']]
        D = pd.concat([D, S3[
            ['temperature3', 'pH3', 'EC3', 'ORP3', 'DO3', 'turbidity3', 'transparency3', 'COD3', 'P3', 'NH3N3',
             'flux3']]], axis=1)

        # 辅助变量
        D['S1_ORP'] = 1
        D['S1_flux'] = 1

        def Anomaly_Count(data):
            Temp = ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N', 'flux']
            for i in Temp:
                if (data['S1_' + i] == 0) | (data['S2_names'] == i) | (data[i + '3'] == 1):
                    data[i] = 1
                else:
                    data[i] = 0
            return data

        D = D.apply(Anomaly_Count, axis=1)

        D.index = D['testtime']

        Summary1 = self.Graph3
        for i in water_data.siteno.unique():

            SitenoT = D[D.siteno.isin([i])]
            new_d_month = list(SitenoT['testtime'].dt.strftime('%Y-%m').unique())
            SitenoT = SitenoT[
                ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N', 'flux']]

            Site = {}
            for j in new_d_month:
                Site[j] = SitenoT[j].sum(axis=0)
            Summary1[i] = Site

    def predict1(self, water_data):
        # water_data为包站点名，列名的一个DataFrame

        result = water_data.copy()
        # 设备异常，0表示异常，1表示正常
        water_data['S1_temperature'] = 0
        water_data['S1_pH'] = 0
        water_data['S1_EC'] = 0
        water_data['S1_DO'] = 0
        water_data['S1_turbidity'] = 0
        water_data['S1_transparency'] = 0
        water_data['S1_COD'] = 0
        water_data['S1_NH3N'] = 0
        water_data['S1_P'] = 0

        water_data.loc[(water_data.temperature > 0) & (water_data.temperature < 80), ['S1_temperature']] = 1
        water_data.loc[(water_data.pH > 2) & (water_data.pH < 17), ['S1_pH']] = 1
        water_data.loc[(water_data.EC > 0) & (water_data.EC < 4000), ['S1_EC']] = 1
        water_data.loc[(water_data.DO > 0) & (water_data.DO < 80), ['S1_DO']] = 1
        water_data.loc[(water_data.turbidity > 0) & (water_data.turbidity < 1000), ['S1_turbidity']] = 1
        water_data.loc[(water_data.transparency > 0) & (water_data.transparency < 200), ['S1_transparency']] = 1
        water_data.loc[(water_data.COD > 0) & (water_data.COD < 100), ['S1_COD']] = 1
        water_data.loc[(water_data.NH3N > 0) & (water_data.NH3N < 20), ['S1_NH3N']] = 1
        water_data.loc[(water_data.P < 3), ['S1_P']] = 1

        water_data['S1'] = water_data['S1_temperature'] * water_data['S1_pH'] * water_data['S1_EC'] * water_data[
            'S1_DO'] * water_data['S1_turbidity'] * water_data['S1_transparency'] * water_data['S1_COD'] * water_data[
                               'S1_NH3N'] * water_data['S1_P']
        # 转换一级数据信号
        water_data.loc[water_data['S1'] == 0, ['S1']] = 2
        water_data.loc[water_data['S1'] == 1, ['S1']] = 0
        water_data.loc[water_data['S1'] == 2, ['S1']] = 1

        result['设备异常'] = water_data['S1'].values

        def Anomly1_Names(data):
            dataT = data[13:22]
            dataT.index = ['temperature', 'pH', 'EC', 'DO', 'turbidity', 'transparency', 'COD', 'NH3N', 'P']
            data['S1_names'] = dataT[dataT == 0].index.tolist()
            return data

        water_data = water_data.apply(Anomly1_Names, axis=1)
        result['设备异常维度'] = water_data['S1_names'].values

        return result[['siteno', '设备异常', '设备异常维度']]

    def predict2(self, data):
        # Data是一个11维度的array

        return self.S2Model["IF"].predict(self.S2Model["SC"].transform((data)))

    def predict3(self, Time, Y, Site, Factor):
        # Time为指定类型，Y为一个数值,Site为站点名,Factor为维度名

        try:
            forecast = self.S3Model[Site + Factor].predict(Time)
            if (Y > forecast['yhat_lower'].values) & (Y < forecast['yhat_upper'].values):
                result = 0
            else:
                result = 1
        except Exception as e:
            logging.exception(e)
            result = "由于历史数据原因，Model:" + Site + Factor + "未被训练"

        return result

    def LoadModel(self):

        self.water_data = joblib.load("./model/water_data")
        self.S2Model["SC"] = joblib.load("./water_model/S2_scaler")
        self.S2Model["IF"] = joblib.load("./water_model/S2_Iforest")
        for i in self.water_data.siteno.unique():
            for j in ['temperature', 'pH', 'EC', 'ORP', 'DO', 'turbidity', 'transparency', 'COD', 'P', 'NH3N', 'flux']:
                try:
                    self.S3Model[i + j] = joblib.load("./water_model/" + i + j)
                except Exception as e:
                    logging.exception(e)
                    print("Model:" + i + j + " 未被训练")

    def LoadData(self):

        self.water_data = joblib.load("./model/water_data")
        self.result = joblib.load("./model/result")
        self.Graph1 = joblib.load("./model/Graph1")
        self.Graph2 = joblib.load("./model/Graph2")
        self.Graph3 = joblib.load("./model/Graph3")

    def preprocess(self):
        water_data = self.result

        water_data.dropna(thresh=3, axis=0, inplace=True)
        water_data.reset_index(inplace=True, drop=True)
        water_data.iloc[:, 1:12] = water_data.iloc[:, 1:12].fillna(method='ffill')

        # def Normalization(water_data):
        #
        #     def MaxMinNormalization(x, Max, Min):
        #         x = (x - Min) / (Max - Min)
        #         return x
        #
        #     dataStandard = np.array (water_data.iloc[:, 1:12])
        #     for i in range (11):
        #         a = dataStandard[:, i]
        #         b = np.sort (a)
        #         len_std = len (a)
        #         if i != 9:
        #             max_99 = b[int (len_std * 0.99)]
        #         else:
        #             max_99 = b[int (len_std * 0.97)]
        #         min_99 = b[0]
        #         c = MaxMinNormalization (a, max_99, min_99)
        #         c = [i if i <= 1 else 1 for i in c]
        #         water_data.iloc[:, i + 1] = c
        #         return water_data

        # 是否归一化
        # water_data = Normalization(water_data)

        # 构造 月份数据
        water_data['monthTime'] = self.water_data.testtime.dt.strftime('%Y-%m')

        def Anomaly2_Names(data):
            if data.isnull()["统计异常维度"] == False:
                # print('aaa',[data['math']])
                data['统计异常维度'] = [data['统计异常维度']]
            else:
                data['统计异常维度'] = []
            if data.isnull()["时间序列异常维度"] == True:
                data['时间序列异常维度'] = []
            return data

        # 每条数据的异常
        water_data = water_data.apply(Anomaly2_Names, axis=1)
        water_data['error'] = water_data['时间序列异常维度'] + water_data['统计异常维度'] + water_data['设备异常维度']
        # self.water_data_2 预处理完成
        water_data['marker'] = None
        self.water_data_2 = water_data

    def write_to_mongodb(self):
        my_client_name = 'water'
        my_db_name = 'waterAndAD'
        my_client = pymongo.MongoClient("mongodb://inesa_water:inesa2019@210.14.69.108:27017")
        my_db = my_client[my_client_name]
        my_col = my_db[my_db_name]
        water_data_pre_list = json.loads(self.water_data_2.T.to_json()).values()

        my_col.delete_many({})
        x = my_col.insert_many(water_data_pre_list)
        print(len(x.inserted_ids), '个文档已插入')

    def write_to_json(self):
        s = SaveJson()
        path = "data_column_list"
        water_data_pre_list = json.loads(self.water_data_2.T.to_json()).values()
        # 这里产生的是数据的列表
        s.save_file(path, water_data_pre_list)
        print("json file has been done.")

    def CoreAnalysis(self):

        self.S3()
        self.graph1()
        self.graph2()
        self.graph3()
        self.preprocess()
        self.write_to_json()
        # self.write_to_mongodb()
        joblib.dump(self.water_data, "./model/water_data")
        joblib.dump(self.result, "./model/result")
        joblib.dump(self.Graph1, "./model/Graph1")
        joblib.dump(self.Graph2, "./model/Graph2")
        joblib.dump(self.Graph3, "./model/Graph3")


class SaveJson(object):

    def save_file(self, path, item):
        # 先将list中的字典对象转化为可写入文本的字符串
        for i in item:
            json_i = json.dumps(i)
            try:
                if not os.path.exists(path):
                    with open(path, "w", encoding='utf-8') as f:
                        f.write(json_i + ",\n")
                else:
                    with open(path, "a", encoding='utf-8') as f:
                        f.write(json_i + ",\n")
            except Exception as e:
                print("write error==>", e)
