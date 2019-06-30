# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:30:31 2019

@author: xuefei
"""

#根据DeliciousDataset数据集统计标签的分布
from os.path import realpath, dirname
from tqdm import tqdm
import collections
from pandas import Series
import matplotlib.pyplot as plt

#读取数据
def LoadData():
    data = []
    address = dirname(realpath(__file__)) + r"\DeliciousDataset.dat"
    print("正在读取数据集")
    with open(address,"r", encoding='UTF-8') as file:
        for line in tqdm(file.readlines()):
            temp = line.strip().split("\t")
            if len(temp) != 3:
                continue
            data.extend(temp[-1].split(" "))
    return data








data = LoadData()
obj = Series(collections.Counter(data)).sort_values(ascending = False)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
obj.plot(logy=True,logx=True)    #logy=True：y轴坐标为10的n次方
plt.show()

