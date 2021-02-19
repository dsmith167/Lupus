#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 00:04:19 2020

@author: dylansmith
"""

import sys
import os
import pandas as pd
import numpy as np
import scipy.io
# rm/ svm/ lr/ dt/ nn
from sklearn.model_selection import KFold
# from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc as AUC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from scipy.stats import entropy

def rl(x):
    return range(len(x))

def hd(df,n=5,m=5):
    print(df.iloc[:n,:m])


def auc_score(y,pred):
    mt,mf,mh=roc_curve(y,pred.T[1])
    return AUC(mt,mf)

def norm_df(df,scale=None,full=False,pct=5):
  #Scale DataFrame based on ss: Standard Scaler or mm: MinMax Scaler
  ## Full ==True :: scale all numeric columns and duplicate non-numeric
  ## Full ==False :: scale all numeric columns and remove non-numeric
  if scale=='ss':
      scale=StandardScaler()
  elif scale=='mm':
      scale=MinMaxScaler()
  if full!=None:
      D=df.copy()
      N=np.array(D).T;l=[]
      for n in rl(N):
          try:np.sum(np.transpose(N[n]))/1.0
          except:l.append(n)
      columns=list(df.columns);L=[];C=columns.copy()
      for i in l: C.remove(columns[i]);L.append(columns[i])
      D0=df.drop(C,axis=1)
      D=df.drop(L,axis=1)
      df=D
  if scale==None:
      df_norm=df
  elif scale=='pct':
      m=pd.Series(np.percentile(df,pct,0),index=df.columns); M=pd.Series(np.percentile(df,100-pct,0),index=df.columns)
      for i in rl(m):
          M_=M.iloc[i]; m_=m.iloc[i]
          if M_-m_==0: M.iloc[i]=m_+1
      df_norm=(df-m)/(M-m)
  else:
      df_norm=pd.DataFrame(scale.fit_transform(df),index=df.index,columns=df.columns)
  if full==True:
      d=pd.DataFrame([],columns=columns,index=df.index)
      for i in rl(D0.columns):
          d[D0.columns[i]]=D0[D0.columns[i]]
      for i in rl(df_norm.columns):
          d[df_norm.columns[i]]=df_norm[df_norm.columns[i]]
      return d
  return df_norm

def sub(x,s=0,t=2):
    if x==s: x=t
    return x
    
def intersect(l,l2,outer=None):
    l=list(l);l2=list(l2);I=[]
    for i in rl(l):
        if l[i] in l2:
            I.append(l[i])
    for i in rl(I):l.remove(I[i]);l2.remove(I[i])
    if outer==0:return l
    elif outer==1:return l2
    return I
            

def sumavg(X):
    ind=list(X.index); avg=list(X.index); tot={}; Z=pd.DataFrame([],index=X.columns)
    for i in rl(ind): 
        avg[i]=avg[i].split('.')[-1]
        if avg[i] in tot.keys(): tot[avg[i]].append(ind[i])
        else: tot[avg[i]]=[ind[i]]
    avg=list(set(avg))
    for i in rl(avg):
        Z[avg[i]]=np.nanmean(X.loc[tot[avg[i]],:],axis=0)
    return Z.T

def get_bags(df,number):
    temp1 = df[df['Label'] == 1 ]
    temp0 = df[df['Label'] == 0 ]
    lenth = temp1.shape[0]
    list_bags = []
    for i in range(number):
        if len(temp0) < lenth: #70
            temp4 = pd.concat([temp0,temp1],axis = 0)
            list_bags.append(temp4)
        else:
            temp3 = temp0.sample(n = lenth,replace = False,random_state=i)
            temp4 = pd.concat([temp3,temp1],axis = 0)
#             temp0 = temp0.drop(temp3.index) #this would be without replacement
            list_bags.append(temp4)
    return(list_bags)
    
def thr_round(pred, thresh, index=1,b0=0,b1=1):
    ## input np array Nx2
    y_pred=[]
    for i in pred:
        p0=i[index]
        if p0<thresh: p0=b0
        else: p0=b1
        y_pred.append(p0)
    return y_pred

def doPCA (data,n=None,R_=None):
  #scaler=StandardScaler(); norm_data=scaler.fit_transform(data)
  norm_data=data
  if R_==None: R_=.95
  if n==None:
      pca=PCA();pca.fit_transform(norm_data)
      R=pca.explained_variance_ratio_
      for i in rl(R):
          if i>0:
              R[i]=R[i]+R[i-1]
      for r in R:
          if r>R_:
              break
      c=list(R).index(r)
      n=c+1
  pca=PCA(n_components=n)
  newPCAData=pca.fit_transform(norm_data)
  return newPCAData,pca



########################################################################################################
    
########################################################################################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

def d_add(d1,d2):
    #add dictionary entries by like keys
    key1=list(d1.keys()); key2=list(d2.keys())
    setk=sorted(list(set(key1+key2)))
    dfin={}
    for i in rl(setk):
        item=setk[i]; dfin[item]=0
        if item in key2: dfin[item]=np.add(d2[item],dfin[item])
        if item in key1: dfin[item]=np.add(d1[item],dfin[item])
    return dfin

def d_combine(d1,d2):
    #combine dictionaries
    #priority to d1 if there are duplicate keys
    key1=list(d1.keys()); key2=list(d2.keys())
    setk=sorted(list(set(key1+key2)))
    dfin={}
    for i in rl(setk):
        item=setk[i]
        if item in key2: dfin[item]=d2[item]
        if item in key1: dfin[item]=d1[item]
    return dfin

def d_inner(d1,d2=None):
    #Combined dictionaries by common inner items with lists
    d1=d1.copy()
    for i in d1.keys():
        if type(d1[i])!=list:d1[i]=[d1[i]]
    if d2==None: return d1
    d2=d2.copy()
    for i in d2.keys():
        if type(d2[i])!=list:d2[i]=[d2[i]]    
    key1=list(d1.keys()); key2=list(d2.keys())
    setk=list(set(key1+key2))
    dfin={}
    for i in rl(setk):
        item=setk[i]; i1,i2=[],[]
        if item in key2: i2=d2[item]
        if item in key1: i1=d1[item]
        dfin[item]=i1+i2
    return dfin

def d_most(d):
    #majority vote for each list within d.values
    key=list(d.keys()); dfin={}
    for k in key:
        temp={}; choice=[]
        for i in d[k]:
            if i not in temp: temp[i]=1
            else: temp[i]+=1
        M=max(temp.values())
        for i in temp:
            if temp[i]==M: choice.append(i)
        dfin[k]=random.choice(choice)
    return dfin

def d_multiply(d1,x):
    #multiply inner entries of a dictionary
    key=list(d1.keys())
    dfin={}
    for i in rl(key):
        item=key[i]
        dfin[item]=np.multiply(d1[item],x)
    return dfin

def d_top(d,N=3):
    #top N value/count pairs for each list within d.values
    key=list(d.keys()); dfin={}
    for k in key:
        temp={}; temp2={}
        for i in d[k]:
            if i not in temp: temp[i]=1
            else: temp[i]+=1
        M=sorted(temp.values(),reverse=True)[:N]
        for i in temp:
            if temp[i] in M: temp2[i]=temp[i]
        dfin[k]=temp2
    return dfin

def unzip(d, st=False):
    keys=list(d.keys())
    List=[]
    for k in keys:
        if st==True and type(d[k])==dict:List.append(str(d[k]))
        else:List.append(d[k])
    return List,keys

def reshape_add(a1,a2):
    D1,D2=a1.shape,a2.shape
    if len(D2)<3:a2=np.reshape(a2,(1,D2[0],D2[1])); D2=a2.shape
    D=np.add(D1,D2)
    D[1:]=np.max([D1,D2],0)[1:]
    N=np.zeros(D)
    N[:D1[0],:D1[1],:D1[2]]=a1
    N[D1[0]:,:D2[1],:D2[2]]=a2
    return N

def Percentile(train,data=None,pct=5):
    x_train=pd.DataFrame(train)
    m=pd.Series(np.percentile(x_train,pct,0),index=x_train.columns)
    M=pd.Series(np.percentile(x_train,100-pct,0),index=x_train.columns)
    for i in rl(m):
        M_=M.iloc[i]; m_=m.iloc[i]
        if M_-m_==0: M.iloc[i]=m_+1
    try: data==None; data=x_train
    except: data=pd.DataFrame(data,columns=x_train.columns)
    df_norm=(data-m)/(M-m)
    return df_norm.values

def compute_measure(true_label,predicted_probability,threshold=.5):
    predicted_label=thr_round(predicted_probability, threshold)
    t_id=(true_label==predicted_label) # truely predicted
    f_id=np.logical_not(t_id) # falsely predicted

    p_id=(true_label>0) # positive targets
    n_id=np.logical_not(p_id) # negative targets

    tp=np.sum(np.logical_and(t_id, p_id)) # TP
    tn=np.sum(np.logical_and(t_id, n_id)) # TN

    fp=np.sum(n_id) -tn
    fn=np.sum(p_id) -tp

    tp_fp_tn_fn_list=[]
    tp_fp_tn_fn_list.append(tp)
    tp_fp_tn_fn_list.append(fp)
    tp_fp_tn_fn_list.append(tn)
    tp_fp_tn_fn_list.append(fn)
    tp_fp_tn_fn_list= np.array(tp_fp_tn_fn_list) #.74, .46, .29

    tp,fp,tn,fn=tp_fp_tn_fn_list

    with np.errstate(divide='ignore'):
        sen = (1.0*tp)/(tp+fn)
    with np.errstate(divide='ignore'):
        spec = (1.0*tn)/(tn+fp)
    with np.errstate(divide='ignore'):
        ppr = (1.0*tp)/(tp+fp)
    with np.errstate(divide='ignore'):
        npr = (1.0*tn)/(tn+fn)
    with np.errstate(divide='ignore'):
        acc= (tp+tn)*1.0/(tp+fp+tn+fn)
    with np.errstate(divide='ignore'):
        f1=tp/(tp+.5*(fn+fp))
    with np.errstate(divide='ignore'):
        f0=tn/(tn+.5*(fn+fp))

    diagnInd= np.log2(1+acc) + np.log2(1+(sen+spec)/2)
    ans=[]
    
    #"f1","f0","acc1","acc0","acc","auc"
    ans.append(f1)
    ans.append(f0)
    ans.append(sen)
    ans.append(spec)
    ans.append(acc)
    ans.append(auc_score(true_label,predicted_probability))
    ans.append(ppr)
    ans.append(npr)
    ans.append(diagnInd)
    
    
    
    return ans


def Classify(y_value,data,scaler=None,model='et',pars={}, split=None,
            search=None,scoring='roc_auc', stop=True , folds=10, pct=5,threshold=.5,stat=9):
    ## search should be None or a dictionary with list values
    ## if search!=None, pars will be ignored
    ## split should pass in an iterable of Pandas DataFrames [Train,Test]
    ## stop=True is used to avoid running the model once parameters are found
    ## split should pass in an iterable of Pandas DataFrames [Train,Test], used for preset splits

    y=data[y_value].values
    x=data.drop([y_value],axis=1).values
    
    if split==None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds)
    else: 
        x_train=split[0].drop([y_value],axis=1).values; x_test=split[1].drop([y_value],axis=1).values
        y_train=split[0][y_value].values; y_test=split[1][y_value].values
    
    if stop==True and search!=None: x_train, x_test, y_train, y_test = x,x,y,y; folds=folds+1 # ignore outer loop splitting
    if scaler=='ss': scaler=StandardScaler() ; scaler.fit(x_train);  x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)##STANDARD SCALER
    elif scaler=='mm': scaler=MinMaxScaler() ; scaler.fit(x_train);  x_train,x_test = scaler.transform(x_train),scaler.transform(x_test)##MIN-MAX SCALER  
    elif scaler=='pct': x_train,x_test = Percentile(x_train,None,pct),Percentile(x_train,x_test,pct) #PERCENTILE SCALER
    
    
    if model=='et': 
        mname='ExtraTrees Classifier'
        ##vars: 'n_estimators'
        model=ExtraTreesClassifier #ExtraTrees Classifier
    elif model=='knn':
        mname='KNeighbors Classifier'
        ##vars: 'n_neighbors'
        model=KNeighborsClassifier #KNeighbors Classifier
    elif model=='nn': 
        mname='MLP Classifier'
        ##vars: 'alpha', 'hidden_layer_sizes'[tuple], 'random_state'
        model=MLPClassifier #Neural Network
    elif model=='gb':
        mname='Gradient Boosting Classifier'
        ##vars: 'n_estimators','max_depth','min_samples_split','learning_rate','loss'
        model=GradientBoostingClassifier #Gradient Boost
    elif model=='lg':
        mname='Logistic Regression'
        model=LogisticRegression
    elif model=='svc':
        mname='Support Vector Machines'
        model=SVC
    elif model=='rf':
        mname='Random Forest Classifier'
        model=RandomForestClassifier
    elif model=='dt':
        mname='Decision Tree'
        model=tree.DecisionTreeClassifier
    elif model=='gnb':
        mname='Gaussian Naive Bayes Classifier'
        model=GaussianNB
    elif model=='bnb':
        mname='Bernoulli Naive Bayes Classifier'
        model=BernoulliNB
        
        
    if search==None:
        model=model(**pars)
    else:
        ## hyper parameters
        grid=GridSearchCV(model(),d_combine(search,d_inner(pars)),cv=folds-1,scoring=scoring) ##fix list of pars
        grid.fit(x_train,y_train)
        pars=grid.best_params_
        model=model(**pars)
        ## test stats of model through intra parameter
        skf=StratifiedKFold(n_splits=folds-1); valid_record=0
        if type(threshold)==list: valid_record=dict(zip(threshold,[0,]*len(threshold)))
        w_train,w_test,z_train,z_test,long_pred=[],[],[],[],{}
        for train_index, test_index in skf.split(x_train, y_train):
            w_train.append([]),w_test.append([]),z_train.append([]),z_test.append([])
            w_train[-1], w_test[-1] = x_train[train_index], x_train[test_index]
            z_train[-1], z_test[-1] = y_train[train_index], y_train[test_index]
        for i in range(folds-1):
            model.fit(w_train[i],z_train[i])
            pred=model.predict_proba(w_test[i])
            if i==0:long_pred['pred']=pred; long_pred['test']=z_test[i]
            else:
                long_pred['pred']=np.concatenate([long_pred['pred'],pred])
                long_pred['test']=np.concatenate([long_pred['test'],z_test[i]])
        if type(threshold)==list:
            for thresh in threshold:
                valid_record[thresh]=compute_measure(long_pred['test'],long_pred['pred'],threshold=thresh)[:stat]
        else:
            valid_record=compute_measure(long_pred['test'],long_pred['pred'],threshold=threshold)[:stat]
    ### training accuracy
    
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_train)
    if type(threshold)==list:
        train_record=dict(zip(threshold,[0,]*len(threshold)))
        for thresh in threshold:
            train_record[thresh]=compute_measure(y_train,pred,threshold=thresh)[:stat]
    else: train_record=compute_measure(y_train,pred,threshold=threshold)[:stat]
    
    if search!=None and stop==True:
        return model,pars,train_record,valid_record,pred     #Stop Here For Validation Sets
    
    pred = model.predict_proba(x_test)
    if type(threshold)==list:
        test_record=dict(zip(threshold,[0,]*len(threshold)))
        for thresh in threshold:
            test_record[thresh]=compute_measure(y_test,pred,threshold=thresh)[:stat]
    else: test_record=compute_measure(y_test,pred,threshold=threshold)[:stat]
    #MSE = mean_squared_error(y_test,y_pred)
    #print("The",mname,"ROC AUC Score on test set: {:.4f}".format(MSE))
    #train=[x_train,y_train]
    #test=[x_test,y_test,y_pred,y_prob]
    
    return model, pars, test_record #MSE,train,test,pars,model

def unravel(x):
    #takes in a two-tiered list, output single
    D=len(x)*len(x[0])
    return list(np.array(x).reshape(D))
