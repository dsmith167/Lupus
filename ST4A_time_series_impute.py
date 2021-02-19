#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:39:59 2020

@author: dylansmith
"""

import statistics as st
import pandas as pd
import math as math
import numpy as np
from sklearn import linear_model as lin
from sklearn.neighbors import KNeighborsClassifier as knn
#sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs
import random as rand
from sklearn.preprocessing import MinMaxScaler,StandardScaler

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format

def norm_df(df):
  df_norm=pd.DataFrame(MinMaxScaler().fit_transform(df),columns=df.columns)
  df_norm.index=df.index

  return df_norm

def bounding(x,m,M):
    #apply method
    x=list(x)[0]
    return max(min(M,x),m)

def rounding(x,m,M):
    #apply method
    x=list(x)[0]
    if np.isnan(x):
        return x
    if abs(x-m)<abs(x-M):
        return m
    elif abs(x-m)>=abs(x-M):
        return M
    
def hd(df,x0=5,x1=5,end=True):
    if end!=True:
        return df.iloc[0:x0,0:x1]
    else:
        return df.iloc[0:x0,-x1:]

def rl(x):
    return range(len(x))

D1=pd.read_csv('data_3_1_.csv').drop(['new random ID.1'],axis=1)
D1.index=D1['new random ID']

DB=D1.copy()

Missing_co=[];Missing_in=[]
for i in range(len(DB.columns)):
    Missing_co.append(1-len(DB[DB.columns[i]].dropna(0))/len(DB))
Miss_co=pd.DataFrame(Missing_co,index=DB.columns,columns=['% Missing'])

## delete columns that have a certain level of missing values in period 1

cutoff=.5
feature_drop=[]

for i in rl(Miss_co):
    if Miss_co.iloc[i,0]>cutoff and Miss_co.index[i][0]=='1':
        feature_drop.append(Miss_co.index[i][2:])
    if Miss_co.index[i][2:] in feature_drop:
        DB=DB.drop([Miss_co.index[i]],axis=1)

Miss_co=Miss_co.T[list(DB.columns)].T

for i in range(len(DB.index)):
    Missing_in.append(1-DB.iloc[i:i+1,:].dropna(1).shape[1]/DB.shape[1])
Miss_in=pd.DataFrame(Missing_in,index=DB.index,columns=['% Missing'])

#343-->331 columns
for c in rl(DB.columns):
    if DB.columns[c][0]=='1':
        break

x=int((DB.shape[1]-c-1)/6)
coi=['new random ID','DOB']
co0=list(DB.columns[0:c]);co0.remove(coi[0]);co0.remove(coi[1])
c=c;co1=list(DB.columns[c:c+x])
c+=x;co2=list(DB.columns[c:c+x])
c+=x;co3=list(DB.columns[c:c+x])
c+=x;co4=list(DB.columns[c:c+x])
c+=x;co5=list(DB.columns[c:c+x])
c+=x;co6=list(DB.columns[c:c+x])
C=[co0,co1,co2,co3,co4,co5,co6]
binary=['Blood - UA_0', 'Blood - UA_1', 'Blood - UA_2', 'Blood - UA_3', 
 'GFR_combined',
 'Protein - UA_0', 'Protein - UA_1', 'Protein - UA_2', 'Protein - UA_3',
 'RBC - UA_1', 'RBC - UA_2', 'RBC - UA_3', 'RBC - UA_4', 'RBC - UA_0']

#pre-cleaning
for i in rl(DB.columns):
    if DB.columns[i][2:] in binary:
        DB[DB.columns[i]]=DB[DB.columns[i:i+1]].apply(lambda x: rounding(x,0,1) ,axis=1)
    try:
        DB[DB.columns[i]]=DB[DB.columns[i]].apply(lambda x: max(x,0))
    except:
        pass

for i in range(1,len(C)):
    se=[]
    for j in range(0,i+1):se+=C[j] #only choose data at or before period i
    Set=DB[se]
    tr=[];te=[]
    MT=Miss_co.T
    for j in rl(se):
        if MT[se[j]][0]>0 and se[j][0]==str(i): te.append(se[j])
        else: tr.append(se[j])
    Train_Full=Set[tr];Test_Full=Set[te]
    for j in rl(te):
        Train=norm_df(Train_Full)
        test=te[j:j+1];Test=Test_Full[test]
        trX=[];teX=[]
        for x in rl(Test): #find complete Test indices
            if np.isnan(Test.iloc[x,0]): teX.append(Test.index[x])
            else: trX.append(Test.index[x])
        TrainX=Train.T[trX].T;TestX=Train.T[teX].T #filter by indices
        TrainY=Test.T[trX].T;TestY=Test.T[teX].T #filter by indices
        counts=DB.groupby(test)[test].count();Min=np.min(counts.iloc[:,0]);Max=np.max(counts.iloc[:,0])
        if test[0][2:]=='Payor' or len(counts)<=2:
            n=min(max(Min,10),50)
            impute=knn(n_neighbors=n,weights='distance')
            impute.fit(TrainX,np.array(TrainY).reshape(len(TrainY)))
            TestY[test]=pd.DataFrame(impute.predict(TestX),index=TestY.index)
        else:
            impute=lin.Ridge()
            impute.fit(TrainX,TrainY)
            TestY[test]=pd.DataFrame(impute.predict(TestX),index=TestY.index)
            if Max<=1.15 and Min>=-.15: #percentage columns
                TestY=TestY.apply(lambda x: bounding(x,0,1),axis=1)
        DB[test]=pd.concat([TrainY,TestY]).sort_index() ## set new values


for i in rl(DB.columns):
    try:DB[DB.columns[i]]=DB[DB.columns[i]].apply(lambda x: max(x,0))
    except:pass

exp=DB.to_csv('cleaned_data_3_1.csv')
   