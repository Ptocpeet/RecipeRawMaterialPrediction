import csv
import re
import numpy as np


def data_read(path):  # 从文件中读取数据
    train_data = []
    with open(path, encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            train_data.append(row)
    return train_data


def isFloat(x):
    try:
        float(x)
        return True
    except:
        return False


def data_split2(datas1, datas2):  # 将读取的数据处理为字典列表
    total_sum = 0
    n_ = 0
    for zb, data in enumerate(datas2):
        for index in range(2, len(data)):
            dec1 = data[index].split('#')
            # print(data)
            # print(datas3[zb])
            # print("")
            dec2 = datas1[zb][index].split('#')
            # dec3 = datas3[zb][index].split('#')
            if not dec2[1] or not dec1[1]:
                continue
            if (not isFloat(dec2[1])) or (not isFloat(dec1[1])):
                continue
            # x_ = (float(dec3[1]) + float(dec1[1])) / 2
            total_sum += (float(dec1[1]) - float(dec2[1])) ** 2
            n_ += 1
    RMSE = (total_sum / n_) ** 0.5
    print("均方根误差为:", RMSE)


def count_RMSE():
    datas1 = data_read('../data/recipe1.csv')  # 训练食谱
    datas2 = data_read('../data/result2.csv')  # 测试食谱
    # datas3 = data_read('../data/recipe6.csv')  # 测试食谱
    data_split2(datas1, datas2)
