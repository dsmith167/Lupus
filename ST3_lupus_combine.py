#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:29:43 2020

@author: dylansmith

combine inputs based on encounter
"""

import pandas as pd
import numpy as np
from scipy import stats

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format

def hd(df,x0=5,x1=5):
    return df.iloc[0:x0,0:x1]

def rl(x):
    return range(len(x))

D1=pd.read_csv('0.enc2_3_1.csv').drop(['new random ID.1'],axis=1)
D2=pd.read_csv('0.lab2_3_1.csv')
D3=pd.read_csv('0.med2_3_1.csv')

D=[D1,D2,D3]
for j in rl(D):
    d=D[j]
    cols=list(d.columns)
    
    for i in rl(cols):
        try: 
            if j==0:x=str(int(cols[i][-1])+1)+".";cols[i]=x+cols[i][:-2]
            elif j==1:x=str(int(cols[i][7]))+".";cols[i]=x+cols[i][9:]
            elif j==2:x=str(int(cols[i][-1]))+".";cols[i]=x+cols[i][:-2]
        except:
            pass
    
    d.columns=cols


for i in range(3):
    D=[D1,D2,D3]
    I=D[i]
    Keep=I['new random ID']

    D1=D1[D1['new random ID'].isin(Keep)]
    D2=D2[D2['new random ID'].isin(Keep)]
    D3=D3[D3['new random ID'].isin(Keep)]

D2,D3=D2.iloc[:,1:],D3.iloc[:,1:]
D1.index,D2.index,D3.index=Keep,Keep,Keep

DB=pd.concat([D1,D2,D3],axis=1)
DB.rename(columns={'1.Age':'0.Age'},inplace=True)

##order database
cols=list(DB.columns[0:17])+sorted(list(DB.columns[17:]))
DB=DB[cols]

Missing_co=[];Missing_in=[]
for i in range(len(DB.columns)):
    Missing_co.append(1-len(DB[DB.columns[i]].dropna(0))/len(DB))
for i in range(len(DB.index)):
    Missing_in.append(1-DB.iloc[i:i+1,:].dropna(1).shape[1]/DB.shape[1])
Miss_co=pd.DataFrame(Missing_co,index=DB.columns,columns=['% Missing'])
Miss_in=pd.DataFrame(Missing_in,index=DB.index,columns=['% Missing'])

exp=pd.DataFrame([Miss_in.index,Missing_in,Miss_co.index,Missing_co],index=['r_ID','%','c_ID','%']).T.to_csv('missing_3_1.csv')
exp=DB.to_csv('data_3_1_.csv')

#### DB_ Setup ###
Miss_in_2=pd.DataFrame(Missing_in+[np.nan],index=list(DB.index)+['%Missing_co'],columns=['% Missing'])
DB_=DB.copy();Miss_co_2=Miss_co.T
DB_=pd.concat([DB_,Miss_co],axis=0);DB_.index=list(DB.index)+['%Missing_co']
DB_['%Missing_in']=Miss_in_2
##################

exp=DB_.to_csv('data_3_1_w_miss.csv')

#visualize missing vals on columns

'''import matplotlib.pylab as plt
plt.figure(figsize=(12,10))
T=[]
for i in rl(Miss_co):
    try:
        x=int(Miss_co.index[i][0])
        T.append(x)
    except:
        T.append(0)

plt.scatter(T,Missing_co,alpha=0)

for i in rl(Missing_co):
    fontsize=7
    if T[i]==0:
        color='black'
    elif Missing_co[i]<.25:
        color='green'
    elif Missing_co[i]<.5:
        color='yellow'
        plt.text(T[i],Missing_co[i],Miss_co.index[i],color=color,alpha=1,fontsize=8)
    else:
        color='red'
        plt.text(T[i],Missing_co[i],Miss_co.index[i],color=color,alpha=1,fontsize=8)
    plt.scatter(T[i],Missing_co[i],color=color,alpha=1)
    
plt.draw();plt.savefig('fig.png')'''