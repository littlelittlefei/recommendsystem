# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:16:51 2019

@author: xuefei
"""
#4_3 基于用户标签的推荐算法
from os.path import realpath, dirname
from tqdm import tqdm
from pandas import Series
import matplotlib.pyplot as plt
import math
import random



#衡量两个物品的余弦相似度
def CosineSim(item_tags, i, j): 
    ret = 0 
    for b,wib in item_tags[i].items(): 
        if b in item_tags[j]: 
            ret += wib * item_tags[j][b] 
    ni = 0 
    nj = 0 
    for b, w in item_tags[i].items(): 
        ni += w * w 
    for b, w in item_tags[j].items(): 
        nj += w * w 
    if ret == 0: 
        return 0 
    return ret / math.sqrt(ni * nj)


#衡量一个推荐列表的多样性
def Diversity(item_tags, recommend_items): 
    ret = 0 
    n = 0 
    for i in recommend_items.keys(): 
        for j in recommend_items.keys(): 
            if i == j: 
                continue 
            ret += CosineSim(item_tags, i, j) 
            n += 1 
    return ret / (n * 1.0)

#载入数据
def LoadData():
    data = []
    address = dirname(realpath(__file__)) + r"\DeliciousDataset.dat"
    print("正在读取数据集")
    with open(address,"r", encoding='UTF-8') as file:
        for line in tqdm(file.readlines()):
            a = line.strip().split("\t")
            if (len(a) != 3) or (a[2] == ""):
                continue
            tags = a[2].split(" ")
            data.extend([[a[0],a[1],i] for i in tags])
    return data

#将数据分成训练集和测试集
def SplitData(data,M,k,seed):
    test = []
    train = []
    random.seed(seed)
    for item in data:
        if random.randint(0,M) == k:
            test.append(item)
        else:
            train.append(item)
    return train,test

#统计用户和标签之间的关系
def InitStat(records): 
    #建立用户和标签的关系
    print("建立用户和标签的关系...")
    for user, item, tag in tqdm(records): 
        addValueToMat(user_tags, user, tag, 1) 
        addValueToMat(tag_items, tag, item, 1) 
        addValueToMat(user_items, user, item, 1)
        addValueToMat(tag_users, tag, user, 1)
def addValueToMat(dic,key1,key2,value):
    if key1 not in dic:
        dic[key1] = {}
    if key2 not in dic[key1]:
        dic[key1][key2] = 0
    dic[key1][key2] += 1
    
#统计用户和标签之间的关系
def InitStat_test(records): 
    #建立用户和标签的关系
    print("建立用户和标签的关系...")
    for user, item, tag in tqdm(records): 
        addValueToMat(user_items_test, user, item, 1) 



#利用标签进行个性化推荐      
def Recommend(user,user_items,tag_items,N): 
    recommend_items = dict() 
    tagged_items = user_items[user] 
    for tag, wut in user_tags[user].items(): 
        for item, wti in tag_items[tag].items(): 
            #if items have been tagged, do not recommend them 
            if item in tagged_items: 
                continue 
            if item not in recommend_items: 
                recommend_items[item] = wut * wti 
            else: 
                recommend_items[item] += wut * wti
    recommend_items = sorted(recommend_items.items(),key=lambda x:x[1],reverse = True)
    return [i[0] for i in recommend_items[:N]]


#利用标签进行个性化推荐修正_tfidf     
def Recommend_tf(user,user_items,tag_items,N,tag_users): 
    recommend_items = dict() 
    tagged_items = user_items[user] 
    for tag, wut in user_tags[user].items(): 
        for item, wti in tag_items[tag].items(): 
            #if items have been tagged, do not recommend them 
            if item in tagged_items: 
                continue 
            if not tag in tag_users: 
                continue            
            if item not in recommend_items: 
                recommend_items[item] = wut * wti / (1 + math.log(1 + len(tag_users[tag])))
            else: 
                recommend_items[item] += wut * wti
    recommend_items = sorted(recommend_items.items(),key=lambda x:x[1],reverse = True)
    return [i[0] for i in recommend_items[:N]]    



    
#准确率和召回率
def Recall(user_tags_test, user_items,tag_items, N,tag_users): 
    hit = 0 
    all = 0 
    print('计算召回率')
    for user in tqdm(user_items.keys()): 
        if user not in user_tags_test:
            continue
        tu = user_tags_test[user] 
        rank = Recommend(user,user_items,tag_items,N) 
        #rank = Recommend_tf(user,user_items,tag_items,N,tag_users)
        for item in rank: 
            if item in tu: 
                hit += 1 
        all += len(tu) 
    return hit / (all * 1.0) 
def Precision(user_tags_test, user_items,tag_items, N,tag_users): 
    hit = 0 
    all = 0 
    print('计算准确率')
    for user in tqdm(user_items.keys()): 
        if user not in user_tags_test:
            continue
        tu = user_tags_test[user] 
        rank = Recommend(user,user_items,tag_items,N)
        #rank = Recommend_tf(user,user_items,tag_items,N,tag_users)
        for item in rank: 
            if item in tu: 
                hit += 1 
        all += N 
    return hit / (all * 1.0)



#载入数据
#注意record的第二列是url，应该是做了脱敏处理
#只取了前200w个，因为全部都跑实在是太久了
records = LoadData()[:2000000]
#随机选择训练/测试集
M = 10
seed = 0
k = 5
train,test = SplitData(records,M,k,seed)

user_tags = dict() 
user_items_test = dict() 
tag_items = dict() 
user_items = dict() 
tag_users = dict()
InitStat(train)
InitStat_test(test)

N = 50
#召回率：
recall = Recall(user_items_test, user_items,tag_items, N, tag_users)
#准确率：
precision = Precision(user_items_test, user_items,tag_items, N ,tag_users)


















