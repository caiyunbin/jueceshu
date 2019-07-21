# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:47:14 2019

@author: Caiyunbin
"""

import pandas as pd 
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz


#定义一个数据集或者说导入一个数据集
def create_data():
    row_data = {'no surfacing':[1,1,1,0,0],
                'flippers':[1,1,0,1,1],
                'fish':['yes','yes','no','no','no']}
    dataset = pd.DataFrame(row_data)
    return dataset

#定义香浓熵的算法
def calent(dataset):
    n = dataset.shape[0]           #注意shape的中括号
    nclass = dataset.iloc[:,-1].value_counts()
    p = nclass/n
    ent = (-p*np.log2(p)).sum()
    return ent

#定义最佳分割的列方法
def best_split(dataset):
    baseent = calent(dataset)
    bestgain = 0
    axis = -1
    for i in range(dataset.shape[1]-1):
        levels = dataset.iloc[:,i].value_counts().index
        ents = 0
        for j in levels:
            childset = dataset[dataset.iloc[:,i]==j]
            ent = calent(childset)
            ents += (childset.shape[0]/dataset.shape[0])*ent
        infogain = baseent - ents
        if (infogain>bestgain):
            bestgain = infogain
            axis = i
    return axis

#定义一个划分数据集的方法
def mysplit(dataset,axis,value):
    col = dataset.columns[axis]
    redataset = dataset.loc[dataset[col]==value,:].drop(col,axis=1)
    return redataset

#使用递归的方法构建一棵树
def createtree(dataset):
    colum_all_feature = list(dataset.columns)
    classlist = dataset.iloc[:,-1].value_counts()  #会按照从高到低的顺序进行排列
    if classlist[0] == dataset.shape[0] or dataset.shape[1]==1:
        return classlist.index[0]
    axis = best_split(dataset)
    bestfeature = colum_all_feature[axis]
    mytree = {bestfeature:{}}
    del colum_all_feature[axis]
    valuelist = set(dataset.iloc[:,axis])
    for value in valuelist:
        mytree[bestfeature][value] = createtree(mysplit(dataset,axis,value))
    return mytree

#关于决策树的存储，格式为numpy库npy文件
def save_tree(mytree):
    np.save('mytree.npy',mytree)
    
    
#将数在测试集上进行测试
def classy(inputtree,labels,testvec):            #判断其中的一条向量
    firststr = next(iter(inputtree))
    seconddict = inputtree[firststr]
    featindex = labels.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex]==key:
            if type(seconddict[key])==dict:
                classlabel = classy(seconddict[key],labels,testvec)
            else:
                classlabel = seconddict[key]
    return classlabel

#查看该模型的性能
def acc_classify(train,test):
    inputtree = createtree(train)
    labels = list(train.columns)
    result = []
    for i in range(test.shape[0]):
        testvec = test.iloc[i,:-1]
        classlabel = classy(inputtree,labels,testvec)
        result.append(classlabel)
    test['predict']=result
    acc = (test.iloc[:,-1]==test.iloc[:-2]).mean()
    print('模型的准确率为',acc)

#对决策树进行画图
xtrain = dataset.iloc[:,:-1]
Ytrain = dataset.iloc[:,-1]
labels = Ytrain.unique().tolist
Ytrain = Ytrain.apply(lambda x:labels.index(x))

clf = DecisionTreeClassifier()
clf = clf.fit(xtrain,Ytrain)
tree.export_graphviz(clf)
dot_data = tree.export_graphviz(clf,out_file = None,
                                feature_names =['no surfacing','flippers'],
                                class_names=['fish','not fish'],
                                filled = True,rounded = True,
                                special_characters = True,
                                graphviz.source(dot_data))
               
 
   
def main():
    dat = create_data()
    mytree = createtree(dat)
    print(mytree)
    
if __name__ == '__main__':
    main()



mytree=createtree(dataset)
inputtree = mytree


