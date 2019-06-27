# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:56:44 2019

@author: xuefei
"""

import random
from os.path import realpath, dirname
from tqdm import tqdm


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
def Recall(train, test, N, P, Q): 
    hit = 0 
    all = 0 
    print('计算召回率')
    for user in tqdm(train.keys()): 
        if user not in test:
            continue
        tu = test[user] 
        rank = GetRecommendation(user,P,Q,N)
        for item in rank: 
            if item in tu: 
                hit += 1 
        all += len(tu) 
    return hit / (all * 1.0) 

def Precision(train, test, N, P, Q): 
    hit = 0 
    all = 0 
    print('计算准确率')
    for user in tqdm(train.keys()): 
        if user not in test:
            continue
        tu = test[user] 
        rank = GetRecommendation(user,P,Q,N)
        for item in rank: 
            if item in tu: 
                hit += 1 
        all += N 
    return hit / (all * 1.0)



#实现负样本采样
def RandSelectNegativeSamples(items,items_pool):
    ret = dict()
    length = len(items)
    #正样本
    for i in items:
        ret[i] = 1
    #抽取负样本
    for i in range(0,3 * len(items)):
        item = items_pool[random.randint(0,len(items_pool) - 1)]
        if item in ret:
            continue
        if item in items:
            continue
        ret[item] = 0
        if len(ret) > length:
            break
    return ret
        
    
#实现最速下降
def LatentFactorModel(user_items, F, NN, alpha, lamb, items_pool): 
    #初始化矩阵
    [P, Q] = InitModel(user_items, F)
    print('开始迭代')
    for step in tqdm(range(0,NN)): 
        for user, items in user_items.items(): 
            samples = RandSelectNegativeSamples(items,items_pool) 
            for item, rui in samples.items(): 
                eui = rui - sum([P[user][f] * Q[item][f] for f in range(F)])
                for f in range(0, F): 
                    P[user][f] += alpha * (eui * Q[item][f] - lamb * P[user][f]) 
                    Q[item][f] += alpha * (eui * P[user][f] - lamb * Q[item][f]) 
        #自适应地调整学习率
        alpha *= 0.9
    return P,Q



#初始化矩阵，由于书里没有明确给出怎么初始化，这里就使用常数初始化
def InitModel(user_items, F):
    users = list(user_items.keys())
    items = list()
    for user,item in user_items.items():
        items.extend(item)
    items = list(set(items))
    temp = dict(zip([i for i in range(F)],[0.5] * F))
    P = dict(zip(users,[temp.copy() for i in range(len(users))]))
    Q = dict(zip(items,[temp.copy() for i in range(len(items))]))
    return [P,Q]


#推荐
def GetRecommendation(user,P,Q,N):
    rank = dict()
    for f,puf in P[user].items():
        if f not in Q:
            continue
        for i,qfi in Q[f].items():
            if i not in rank:
                rank[i] = 0
            rank[i] += puf * qfi
    #print(rank)
    rank = sorted(rank.items(),key=lambda x:x[1],reverse = True)
    return [i[0] for i in rank[:N]]



M = 8
k = 5
seed = 0
F = 100
alpha = 0.02
lamb = 0.01
#迭代步数
NN = 10
N = 3
data = LoadData()
#data = data[:1000]
train,test = SplitData(data,M,k,seed)
train = dict_update(train)
test = dict_update(test)
items_pool = list()
for user,item in train.items():
    items_pool.extend(item)
items_pool = list(set(items_pool))


P,Q = LatentFactorModel(train, F, NN, alpha, lamb, items_pool)

#召回率：
recall = Recall(train, test, N, P, Q)
#准确率：
precision = Precision(train, test, N, P, Q)
































