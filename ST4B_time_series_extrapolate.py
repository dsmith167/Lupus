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

from scipy.interpolate import interp1d as interp
def extrapolate(x,Type=1,k=2):
    #Type: 0 is in before
    #Type: 1 is after
    #k is num of entries to use for extrapolation surrounding NA
    #k>=2
    if np.sum(np.isnan(x))>4:return x
    elif np.sum(np.isnan(x))==4: k=2
    s=False
    if type(x)==pd.core.series.Series:I=x.index;x=list(x);s=True
    X=x.copy()
    if Type==None:Types=[0,1]
    else: Types=[Type]
    for Type in Types:
        data=[]
        count_na=0
        if Type==1:
            x=x[-1::-1]
        i=0
        while len(data)<k:
            if np.isnan(x[i]):
                count_na+=1
            else:
                data.append(x[i])
            i+=1
            if i==len(x):break
        data=data[-1::-1]
        f=interp(list(range(len(data))),data,fill_value='extrapolate')
        i=0;y=k+count_na-1
        if Type==1:i=len(X)-count_na
        while count_na!=0:
            X[i]=float(f(y))
            y+=1; i+=1; count_na-=1
            if Type==0:y-=2
    if s==True:X=pd.Series(X,index=I)
    return X

def forward(x):
    s=False
    if type(x)==pd.core.series.Series:I=x.index;x=list(x);s=True
    X=x.copy()
    data=[];di=[]
    ni=[]
    for i in range(len(X)):
        if np.isnan(x[i]):
            ni.append(i)
        else:
            data.append(x[i])
            di.append(i)
    f=interp(di,data,kind='previous',fill_value='extrapolate')
    for i in ni:
        if i>di[0]:
            X[i]=float(f(i))
    if s==True:X=pd.Series(X,index=I)
    return X

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

#cleaning
for i in rl(DB.columns):
    if DB.columns[i][2:] in binary:
        DB[DB.columns[i]]=DB[DB.columns[i:i+1]].apply(lambda x: rounding(x,0,1) ,axis=1)
    try:
        DB[DB.columns[i]]=DB[DB.columns[i]].apply(lambda x: max(x,0))
    except:
        pass

feat_list=[]
for i in rl(co1): feat_list.append(co1[i][2:])
for i in rl(feat_list):
    cols=[]; feat=feat_list[i]
    for j in range(1,7):cols.append(str(j)+"."+feat)
    temp=DB[cols]
    if feat in binary:
        temp=temp.apply(lambda x: extrapolate(x,None,k=3),axis=1)
    else:
        temp=temp.apply(lambda x: extrapolate(x,None),axis=1)
    DB[cols]=temp

#cleaning
for i in rl(DB.columns):
    if DB.columns[i][2:] in binary:
        DB[DB.columns[i]]=DB[DB.columns[i:i+1]].apply(lambda x: rounding(x,0,1) ,axis=1)
    try:
        DB[DB.columns[i]]=DB[DB.columns[i]].apply(lambda x: max(x,0))
    except:
        pass

exp=DB.dropna(0).to_csv('cleaned_data_3_1_extrapolate.csv')
   