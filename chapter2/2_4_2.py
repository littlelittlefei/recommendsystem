# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:22:08 2019

@author: xuefei
"""

#2.4节 基于物品的协同过滤算法ItemCF

import random
from os.path import realpath, dirname
from tqdm import tqdm
import math

#将数据分成训练集和测试集
def SplitData(data,M,k,seed):
    test = []
    train = []
    random.seed(seed)
    for user,item in data:
        if random.randint(0,M) == k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train,test


#读取数据
def LoadData():
    data = []
    address = dirname(realpath(__file__)) + r"\ratings.dat"
    with open(address,"r") as file:
        for line in file.readlines():
            data.append([int(i) for i in line.strip().split("::")[:2]])
    return data


#将列表转换成字典
def dict_update(l):
    users = list(set([i[0] for i in l]))
    dic = dict(zip(users,[[] for _ in range(len(users))]))
    for key,value in l:
        dic[key].append(value)
    return dic



#准确率和召回率
def Recall(train, test, w, k, N): 
    hit = 0 
    all = 0 
    print('计算召回率')
    for user in tqdm(train.keys()): 
        if user not in test:
            continue
        tu = test[user] 
        rank = GetRecommendation(user,train,w,k,N) 
        for item in rank: 
            if item in tu: 
                hit += 1 
        all += len(tu) 
    return hit / (all * 1.0) 

def Precision(train, test, w, k, N): 
    hit = 0 
    all = 0 
    print('计算准确率')
    for user in tqdm(train.keys()): 
        if user not in test:
            continue
        tu = test[user] 
        rank = GetRecommendation(user,train,w,k,N) 
        for item in rank: 
            if item in tu: 
                hit += 1 
        all += N 
    return hit / (all * 1.0)


#计算物品相似度
def ItemSimilarity(train):
    n = dict()
    c = dict()
    print('求解物品相似度')
    for user,items in tqdm(train.items()):
        for item in items:
            if not item  in n:
                n[item] = 0
            n[item] += 1
            for item2 in items:
                if item == item2:
                    continue
                if not item in c:
                    c[item] = {}
                if not item2 in c[item]:
                    c[item][item2] = 0
                c[item][item2] += 1
    w = dict()
    for i,related_items in c.items():
        if not i in w:
            w[i] = {}
        for item in related_items:
            if i == item:
                continue
            w[i][item] = c[i][item] / math.sqrt(n[i] * n[item])  
    
    return w
    


#计算修正后物品相似度：惩罚活跃用户
def ItemSimilarity_re(train):
    n = dict()
    c = dict()
    print('求解物品相似度')
    for user,items in tqdm(train.items()):
        for item in items:
            if not item  in n:
                n[item] = 0
            n[item] += 1
            for item2 in items:
                if item == item2:
                    continue
                if not item in c:
                    c[item] = {}
                if not item2 in c[item]:
                    c[item][item2] = 0
                c[item][item2] += 1 / math.log(1 + len(items) * 1.0)
    w = dict()
    
    
    
    for i,related_items in c.items():
        if not i in w:
            w[i] = {}
        for item in related_items:
            if i == item:
                continue
            w[i][item] = c[i][item] / math.sqrt(n[i] * n[item])         
            
    return w


#给出给定用户对于每一个未使用商品的推荐评分
def GetRecommendation(user_id,train,w,k,N):
    item_list = train[user_id]
    rank = dict()
    for item in item_list:
        for item_sim,score in sorted(w[item].items(),key = lambda x:x[1],reverse = True)[0:k]:
            if item_sim == item:
                continue
            if item_sim in item_list:
                continue
            if not item_sim in rank:
                rank[item_sim] = 0
            rank[item_sim] += score
    rank = sorted(rank.items(),key=lambda x:x[1],reverse = True)
    return [i[0] for i in rank[:N]]





#训练集/测试集参数
M = 8
k = 5
seed = 0
N = 5
#关联物品个数
K = 10
data = LoadData()
train,test = SplitData(data,M,k,seed)
train = dict_update(train)
test = dict_update(test)
#w = ItemSimilarity(train) 
w = ItemSimilarity_re(train) 

   
#召回率：
recall = Recall(train, test, w, K, N)
#准确率：
precision = Precision(train, test, w, K, N)















