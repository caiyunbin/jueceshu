# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:07:59 2019

@author: Caiyunbin
"""

import math

def create_data():                #这是一个数据集
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [0,1,'no'],
               [1,0,'no'],
               [0,1,'no']]
    labels = ['no sufacing','flippers']
    return dataset , labels


def shannon(dataset):
    ent = len(dataset)         #首先计算数据集的长度
    labelcounts = {}           #创建一个字典，用以存储类别的数量
    for featvec in dataset:    #将特征列一个一个取出，计算类别数量
        currentlabel = featvec[-1] #数据最后一列
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] +=1
    shannon = 0                #定义累加E = -P*log(p,2)
    for key in labelcounts:     
            p = float(labelcounts[key])/ent
            shannon -=p*math.log(p,2)
    return shannon

def split_dataset(dataset,axis,value):  ###需要传入三个参数数据集、特征、返回值
    split = []                          #将特征列删除，返回剩余数据集
    for featvec in dataset:
        if featvec[axis] == value:
            reduced = featvec[:axis]                 #[ : n] 代表列表中的第一项到第n项
            reduced.extend(featvec[axis+1:])         #[m : ] 代表列表中的第m+1项到最后一项
            split.append(reduced)                    #list[start:end:step]
    return split        
            
                
def bestsplit(dataset):
    column_num = len(dataset[0])-1 #列表中第一个元素的长度
    baseentropy = shannon(dataset) #计算初始的香浓熵
    bestshan = 0.0
    bestfeature = -1              #表示最后一列标签列是最佳
    for i in range(column_num):   #循环两次0,1 不循环标签列
        featlist = [item[i] for item in dataset] #取出嵌套列表指定列
        uniquevals = set(featlist)
        newentropy = 0.0
        for value in uniquevals:
            subdataset = split_dataset(dataset,i,value)
            P = len(subdataset)/float(len(dataset))
            newentropy += P*shannon(subdataset)          #类别的权重
        infogain = baseentropy - newentropy
        if (infogain>bestshan):   #将每一列的信息增益进行比较
            bestshan = infogain
            bestfeature = i
    return bestfeature

def majority(classlist):
    classcount ={}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote]=0
        classcount[vote]+=1
    sort_vote = sorted(vote.items(),key =lambda vote:vote[1],reverse = False)
    return sort_vote[0][0]
    
def create_tree(dataset,label):
    classlist = [item[-1] for item in dataset]
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    if len(dataset[0])==1:
        return majority(classlist)
    bestf = bestsplit(dataset)
    bestlabel = label[bestf]
    mytree = {bestlabel:{}}
    del(label[bestf])
    featvalues = [item[bestf] for item in dataset]
    uniquv = set(featvalues)
    for value in uniquv:
        sublabel = label[:]
        mytree[bestlabel][value] = create_tree(split_dataset(dataset,bestf,value),sublabel)
    return mytree
    
    
def main():
    dataset,label = create_data()
    a=create_tree(dataset,label)
    print(a)
    
if __name__ == '__main__':
    main()
    
