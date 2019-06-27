# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:26:40 2019

@author: xuefei
"""
#2.4节 基于用户的协同过滤算法userCF

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


#计算余弦相似度
def UserSimilarity(train):
    #建立倒序表
    item_users = dict()
    for key,values in train.items():
        for value in values:
            if value not in item_users:
                item_users[value] = []
            item_users[value].append(key)
    #建立用户爱好数量字典和用户对爱好数量字典
    users = list(train.keys())
    n = dict(zip(users,[0 for _ in range(len(users))]))
    c = dict(zip(users,[n.copy() for _ in range(len(users))]))
    print('求解用户相似度')
    for key,values in tqdm(item_users.items()):
    #for key,values in item_users.items():
        for u in values:
            n[u] += 1
            for v in values:               
                if u == v:
                    continue
                #n[u] = len(train[u])
                #print([u,v])
                c[u][v] += 1

    #遍历刚才建立的字典，求相似度
    w = dict(zip(users,[{} for _ in range(len(users))]))
    for u,others in c.items():
        for v,times in others.items():
            w[u][v] = times / math.sqrt(n[u] * n[v])
    return w
            
#计算修正后的余弦相似度
def UserSimilarity_re(train):
    #建立倒序表
    item_users = dict()
    for key,values in train.items():
        for value in values:
            if value not in item_users:
                item_users[value] = []
            item_users[value].append(key)
    #建立用户爱好数量字典和用户对爱好数量字典
    users = list(train.keys())
    n = dict(zip(users,[0 for _ in range(len(users))]))
    c = dict(zip(users,[n.copy() for _ in range(len(users))]))
    for key,values in tqdm(item_users.items()):
    #for key,values in item_users.items():
        for u in values:
            n[u] += 1
            for v in values:               
                if u == v:
                    continue
                #n[u] = len(train[u])
                #print([u,v])
                c[u][v] += 1 / (math.log(1 + len(values)))

    #遍历刚才建立的字典，求相似度
    w = dict(zip(users,[{} for _ in range(len(users))]))
    for u,others in c.items():
        for v,times in others.items():
            w[u][v] = times / math.sqrt(n[u] * n[v])
    return w


#给出给定用户对于每一个未使用商品的推荐评分
def GetRecommendation(user,train,w,k,N):
    rank = dict()
    item_exist = train[user]
    for u,sim in sorted(w[user].items(),key = lambda x:x[1],reverse = True)[0:k]:
        for item in train[u]:
            if item in item_exist:
                continue
            if not item in rank:
                rank[item] = 0
            rank[item] += sim
    rank = sorted(rank.items(),key=lambda x:x[1],reverse = True)
    return [i[0] for i in rank[:N]]

        
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
        





#训练集/测试集参数
M = 8
k = 5
seed = 0
N = 5
#关联用户个数
K = 80
data = LoadData()
train,test = SplitData(data,M,k,seed)
train = dict_update(train)
test = dict_update(test)
#w = UserSimilarity(train) 
#修正之后稍微优秀了一点，但是也没有优秀太多
w = UserSimilarity_re(train) 

   
#召回率：
recall = Recall(train, test, w, K, N)
#准确率：
precision = Precision(train, test, w, K, N)



