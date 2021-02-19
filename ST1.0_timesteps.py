#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 03:38:08 2020

@author: dylansmith
"""

import pandas as pd
import numpy as np
import math

record=0

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format

DB1=pd.read_csv('./1.Encounter.csv')
DB2=pd.read_csv('./2.LabTesting.csv')
DB2_=pd.read_csv('./2.LabTestingExp.csv')
DB3=pd.read_csv('./3.Medication_full.csv')
DB3_=pd.read_csv('./3.Medication_grouping.csv')
DB=[DB1,DB2,DB3,DB3_]

#cut down csvs to id and time
db1=np.array(DB1)[2:].T[0:2].T
db2=np.array(DB2)[2:].T[0:2].T ##Separate Entries by Group ID
db2_=db2=np.array(DB2_)[2:].T[0:2].T
db3=np.array(DB3)[:].T[0:2].T 
db3_=np.array(DB3_)[:].T[0:2].T 

def rl(x):
    return range(len(x))

def time(arr,dupl=False,decimal=False):
    X=np.array(arr) #N x 2
    print("starting iteration for",X.shape[0],"x",X.shape[1],"array")
    if dupl==True:
        XX=[]
        for i in rl(X):
            if len(XX)==0 :
                XX.append([])
            elif int(X[i][0])<int(X[i-1][0]):
                XX.append([])
                print("length of XX is",len(XX)-1,"; iter number is",i,"; and iter-",len(XX)-2,"has size",len(XX[-2]))
            XX[-1].append(X[i])
        #now XX list of lists
        L=[]
        for i in rl(XX):
            L.append(time(XX[i]))
    else:
        L={}
        for i in rl(X): #instance x id
            if int(X[i][0]) in L:
                if decimal==False:
                    L[int(X[i][0])].append(X[i][1]//1-X[i-1][1]//1)
                else:
                    L[int(X[i][0])].append(X[i][1]-X[i-1][1])
            else:
                L[int(X[i][0])]=[]
    return L

def mid(stats,N=80,R=True):
    #Find Range of Middle N% of list
    l=len(stats);n=round((100-N)/200,2)
    l0=int(round(l*(n),0))
    le=l-l0
    L=sorted(stats.copy())[l0:le]
    if R==True:
        try:
            L=[L[0],L[-1]]
        except:
            pass
    return L
    
    
def short(X,I=2):
    #round numbers in a list to i digits
    S=[]
    for i in rl(X):
        S.append(round(X[i],I))
        if S[-1]==int(S[-1]):
            S[-1]=int(S[-1])
    return S
    

def stats(time,dupl=False,I=2,K=200,k=False,sep=False,ni=1,piecewise='avg'):
    X=time;m0=80;m1=50
    if type(X)==list:
        ##Filter Categories with fewer than K Entries
        Xi={}
        for i in rl(X):
            if len(X[i])<K:
                continue
            else:
                Xi[i]=[]
                Ke=list(X[i].keys())
                Va=list(X[i].values())
                for j in rl(X[i]): ##Use only keys that have k instances
                    if k==False:
                        Xi[i]+=Va[j]
                    elif k=='inverse':
                        if str(Ke[j]) in Xi:
                            Xi[str(Ke[j])]+=Va[j]
                        else:
                            Xi[str(Ke[j])]=Va[j]
                    elif len(Va[j])>=k:
                        M=[];M+=mid(Va[j],80);M+=mid(Va[j],50);M+=mid(Va[j],20);M=list(set(M))
                        Xi[i]+=M
                if k=='inverse':
                    if i in Xi:
                        del Xi[i]
        X=Xi
    def S0():
        if type(X)==dict:
            piece=list(X.values())
            nz_piece=[]
            ni_piece=[]
            whole=[]
            def filteri(X,i=ni):
                if X>i:
                    return True
                else:
                    return False
            for i in rl(piece):
                whole+=piece[i]
                nz_piece.append(list(filter(None,piece[i])))
                ni_piece.append(list(filter(filteri,piece[i])))
            nz_whole=list(filter(None,whole))
            ni_whole=list(filter(filteri,whole))
            if nz_whole==[] or ni_whole==[]:
                return None
            ####BASIC
            print("BASIC:\n\tcombined stats on dataset:")
            P=short([np.average(whole),min(whole),max(whole),np.median(whole),np.std(whole),mid(whole,m0)[0],mid(whole,m0)[1],mid(whole,m1)[0],mid(whole,m1)[1]],I)
            print("      avg:",P[0],"\n      med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
            
            Min=[];Max=[];Avg=[];Med=[];SD=[];St=[];En=[];St2=[];En2=[]
            for i in rl(piece):
                if len(piece[i])>0:
                    Min.append(min(piece[i]));Max.append(max(piece[i]));Avg.append(np.average(piece[i]));Med.append(np.median(piece[i]));SD.append(np.std(piece[i]));    
                    St.append(mid(piece[i],m0)[0]);En.append(mid(piece[i],m0)[1]);St2.append(mid(piece[i],m1)[0]);En2.append(mid(piece[i],m1)[1])
            print("\tpiecewise stats on dataset:")
            if piecewise=='avg':P=short([np.average(Avg),np.average(Min),np.average(Max),np.average(Med),np.average(SD),np.average(St),np.average(En),np.average(St2),np.average(En2)],I)
            elif piecewise=='med':P=short([np.median(Avg),np.median(Min),np.median(Max),np.median(Med),np.median(SD),np.median(St),np.median(En),np.median(St2),np.median(En2)],I)
            print("  "+piecewise+" avg:",P[0],"\n  "+piecewise+" med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
            
            ####GREATER THAN I EXCLUSION
            print("\nGREATERTHAN_"+str(ni)+":\n\tcombined stats on dataset:")
            P=short([np.average(ni_whole),min(ni_whole),max(ni_whole),np.median(ni_whole),np.std(ni_whole),mid(ni_whole,m0)[0],mid(ni_whole,m0)[1],mid(ni_whole,m1)[0],mid(ni_whole,m1)[1]],I)
            print("      avg:",P[0],"\n      med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
            
            Min=[];Max=[];Avg=[];Med=[];SD=[];St=[];En=[];St2=[];En2=[]
            for i in rl(ni_piece):
                if len(ni_piece[i])>0:
                    Min.append(min(ni_piece[i]));Max.append(max(ni_piece[i]));Avg.append(np.average(ni_piece[i]));Med.append(np.median(ni_piece[i]));SD.append(np.std(ni_piece[i]));    
                    St.append(mid(ni_piece[i],m0)[0]);En.append(mid(ni_piece[i],m0)[1]);St2.append(mid(ni_piece[i],m1)[0]);En2.append(mid(ni_piece[i],m1)[1])
            print("\tpiecewise stats on dataset:")
            if piecewise=='avg':P=short([np.average(Avg),np.average(Min),np.average(Max),np.average(Med),np.average(SD),np.average(St),np.average(En),np.average(St2),np.average(En2)],I)
            elif piecewise=='med':P=short([np.median(Avg),np.median(Min),np.median(Max),np.median(Med),np.median(SD),np.median(St),np.median(En),np.median(St2),np.median(En2)],I)
            print("  "+piecewise+" avg:",P[0],"\n  "+piecewise+" med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
            
            ####ZERO EXCLUSION
            print("\nNONZERO:\n\tcombined stats on dataset:")
            P=short([np.average(nz_whole),min(nz_whole),max(nz_whole),np.median(nz_whole),np.std(nz_whole),mid(nz_whole,m0)[0],mid(nz_whole,m0)[1],mid(nz_whole,m1)[0],mid(nz_whole,m1)[1]],I)
            print("      avg:",P[0],"\n      med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
            
            Min=[];Max=[];Avg=[];Med=[];SD=[];St=[];En=[];St2=[];En2=[]
            for i in rl(nz_piece):
                if len(nz_piece[i])>0:
                    Min.append(min(nz_piece[i]));Max.append(max(nz_piece[i]));Avg.append(np.average(nz_piece[i]));Med.append(np.median(nz_piece[i]));SD.append(np.std(nz_piece[i]));    
                    St.append(mid(nz_piece[i],m0)[0]);En.append(mid(nz_piece[i],m0)[1]);St2.append(mid(nz_piece[i],m1)[0]);En2.append(mid(nz_piece[i],m1)[1])
            print("\tpiecewise stats on dataset:")
            if piecewise=='avg':P=short([np.average(Avg),np.average(Min),np.average(Max),np.average(Med),np.average(SD),np.average(St),np.average(En),np.average(St2),np.average(En2)],I)
            elif piecewise=='med':P=short([np.median(Avg),np.median(Min),np.median(Max),np.median(Med),np.median(SD),np.median(St),np.median(En),np.median(St2),np.median(En2)],I)
            print("  "+piecewise+" avg:",P[0],"\n  "+piecewise+" med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
            return [P[0],P[3],P[4],(P[5],P[6]),(P[7],P[8]),(P[1],P[2])]
    return S0()

    
##HARD CODE

T=[time(db1),time(db2,True),time(db3_,True),time(db3),time(db2_)]
##Writing Log to File
import sys
if record==1:
    stdout=sys.stdout
    sys.stdout=open("log.txt","w")
print("___Timestep Analysis for Encounters:___\n\n")
stats(T[0])
print("\n\n___Timestep Analysis for Lab Testing with min 1000 Samples, by group:___\n\n")
N=stats(T[1],K=1000)
print("\n\n___                                                       , by individual:___\n\n")
stats(T[1],K=100,k='inverse')
print("\n\n___Timestep Analysis for Medications, by group:___\n\n")
N=stats(T[2])
print("\n\n___                                 , by individual:___\n\n")
stats(T[2],k='inverse')
print("\n\n___Timestep for Medications:___\n\n")
stats(T[3])
if record==1:
    sys.stdout.close()
    sys.stdout=stdout

stdout=sys.stdout
sys.stdout=open("empty.txt","w")

CI=[]
for I in range(1,3):
    P=[]
    for i in rl(T[I]):
        X={}
        TI=list(T[I][i].keys())
        L=len(T[I][i])
        for t in TI:
            if T[I][i][t]==[]:
                del T[I][i][t]
        if len(T[I][i])>0:
            P.append([i,L,stats(T[I][i],K=1000)])
    if I==1:
        nn=list(np.array(DB2['Group_Id'])[2:]);NN={};count=0;IN=[]
    elif I==2:
        nn=list(np.array(DB3_['Group_Id'])[:]);NN={};count=0;IN=[]
    for i in rl(nn):
        if nn[i] in NN.values():
            pass
        else:
            NN[count]=nn[i]
            count+=1
    
    C=[];N100=21.06 ## num patients /100
    for i in rl(P):
        g=NN[P[i][0]];IN.append(str(P[i][0]))
        try:
            len(P[i][2])
            c=[g,P[i][1]/N100]
            c.extend(P[i][2][0:3])
            c.extend([P[i][2][3][0],P[i][2][3][1],P[i][2][4][0],P[i][2][4][1],P[i][2][5][0],P[i][2][5][1]])
            C.append(c)
        except:
            C.append([g,P[i][1]/N100,0,0,0,0,0,0,0,0,0])
    
    sys.stdout.close()
    sys.stdout=stdout

    C=pd.DataFrame(C,index=IN,columns=['group','% clients',"wgt avg","avg med:","avg sd:","mid 80%(L)","mid 80%(H)","mid 50%(L)","mid 50%(H)","avg min","avg max"])
    C=C.sort_values(by="% clients",ascending=False)
    
    if record==1:
        stdout=sys.stdout
        sys.stdout=open("log.txt","a")
    
    print()
    print(C)
    print()
    CI.append(C)

    if record==1:
        sys.stdout.close()
        sys.stdout=stdout

    stdout=sys.stdout
    sys.stdout=open("empty.txt","w")
    
sys.stdout.close()
sys.stdout=stdout

stdout=sys.stdout
sys.stdout=open("empty.txt","w")

print()

sys.stdout.close()
sys.stdout=stdout

T=[time(db1,decimal=True),0,0,time(db3,decimal=True),time(db2_,decimal=True)]

Di=[];II=[0,3,4];In=['Encounters',0,0,'Medications','Lab Testing']
for x in range(2):
    for ii in II:
        D=[]
        for i in rl(T[ii]):
            D.append(max(sum(T[ii][list(T[ii].keys())[i]])-365,0)/365)
            if x==1 and D[-1]==0:
                D.pop(-1)
                
        Di.append(D)
        
        if x==1:
            print("NonZero:")
        else:
            print("Complete")
        print("\n\nLength of Record Stats on dataset, Excluding 12mo:",In[ii])
        print("In Years:")
        whole=D;m0=80;m1=50
        P=short([np.average(whole),min(whole),max(whole),np.median(whole),np.std(whole),mid(whole,m0)[0],mid(whole,m0)[1],mid(whole,m1)[0],mid(whole,m1)[1]],I)
        print("      avg:",P[0],"\n      med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
        if ii==0:
            c=[In[ii],'Years',str(len(D))];c.extend(P)
            C=pd.DataFrame(c,index=['Record','Interval','Patients',"wgt avg","min","max","med:","sd:","mid 80%(L)","mid 80%(H)","mid 50%(L)","mid 50%(H)"]).T
        else:
            c=[In[ii],'Years',str(len(D))];c.extend(P)
            C=pd.concat([C,pd.DataFrame(c,index=['Record','Interval','Patients',"wgt avg","min","max","med:","sd:","mid 80%(L)","mid 80%(H)","mid 50%(L)","mid 50%(H)"]).T],axis=0)
    
        print("\n\n")
        print("In 6 Month Intervals:")
        whole=D;m0=80;m1=50
        P=short([np.average(whole),min(whole),max(whole),np.median(whole),np.std(whole),mid(whole,m0)[0],mid(whole,m0)[1],mid(whole,m1)[0],mid(whole,m1)[1]],I)
        P=list(np.round(np.multiply(P,2),2))
        print("      avg:",P[0],"\n      med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
        c=[In[ii],'1/2 Years',''];c.extend(P)
        C=pd.concat([C,pd.DataFrame(c,index=['Record','Interval','Patients',"wgt avg","min","max","med:","sd:","mid 80%(L)","mid 80%(H)","mid 50%(L)","mid 50%(H)"]).T],axis=0)
    
        print("\n\n")
        print("In Months:")
        whole=D;m0=80;m1=50
        P=short([np.average(whole),min(whole),max(whole),np.median(whole),np.std(whole),mid(whole,m0)[0],mid(whole,m0)[1],mid(whole,m1)[0],mid(whole,m1)[1]],I)
        P=list(np.round(np.multiply(P,12),2))
        print("      avg:",P[0],"\n      med:",P[3],"\n      sd:",P[4],"\n      mid 80%: (",P[5],"\t,",P[6],")","\n      mid 50%: (",P[7],"\t,",P[8],")","\n      min,max: (",P[1],"\t,",P[2],")")
        c=[In[ii],'Months',''];c.extend(P)
        C=pd.concat([C,pd.DataFrame(c,index=['Record','Interval','Patients',"wgt avg","min","max","med:","sd:","mid 80%(L)","mid 80%(H)","mid 50%(L)","mid 50%(H)"]).T],axis=0)
        
    
    
    exp=C.to_csv(str(x)+'Record_length.csv')

## setup

import xlrd
edit=0
D_=[DB1.iloc[2:,0:2],DB3.iloc[:,0:2],DB2_.iloc[2:,0:2]];d=pd.DataFrame([])

## min
    
for i in range(3):
    DB=D_[i].copy()
    I=In[II[i]]
    DB.columns=['new random ID',I]
    DB['new random ID']=DB['new random ID'].astype(int)
    DB[I]=DB[I].astype(float)
    D=DB.groupby('new random ID')
    d=pd.concat([D.min(),d],axis=1)

d['Date.Min (max)'],d['Record_0']=d.max(axis=1,skipna=True).astype(int),d.idxmax(axis=1, skipna=True)
d['Date Text_0']=d['Date.Min (max)'].apply(lambda x: "/".join([str(elem) for elem in (xlrd.xldate_as_tuple(x,0)[1],xlrd.xldate_as_tuple(x,0)[2],xlrd.xldate_as_tuple(x,0)[0])]))
d=pd.concat([d.iloc[:,3:],d.iloc[:,2:3],d.iloc[:,0:2]],axis=1)
#exp=d.to_csv('Record_start_set_full.csv')

## max
imaginary=1 #encounter dates based on end-365 (opt: im=0)
d1=pd.DataFrame([])
for i in [0,2]: ## exclude medication due to nature of imputing
    DB=D_[i].copy()
    I=In[II[i]]
    DB.columns=['new random ID',I]
    DB['new random ID']=DB['new random ID'].astype(int)
    DB[I]=DB[I].astype(float)
    D=DB.groupby('new random ID') 
    if i==0: #and edit==0
        M=D.apply(max);M[I]=M[I]-365; 
        if imaginary==0:
            M=dict(zip(M.index,M['Encounters']))
            DB=D_[0][D_[0]['Admit_Date']<=D_[0]['random number- study ID'].apply(lambda x: M[int(x)])]
            ## above limits to less than defined end of record
            ## reset temporary variables
            DB.columns=['new random ID',I];DB['new random ID']=DB['new random ID'].astype(int);DB[I]=DB[I].astype(float)
            D=DB.groupby('new random ID') #latest record before end-1
        else:d1=pd.concat([M.iloc[:,1:2],d1],axis=1) #forced date (based on end)
    if (i==2 or imaginary==0):d1=pd.concat([D.max(),d1],axis=1)

count=1
def temp(x,m=max,ind=None): ## max (max) or max (min) with 0 if nan
    if ~(x[1]>=0 or x[1]<0):
        mm=0;I=1
    elif ~(x[0]>=0 or x[0]<0):
        mm=0;I=0
    else:
        if m==max: ## upper limit of encounter date
            mm=min(max(x),x[1])
        else:
            mm=min(x)
    if ind==None:
        return mm
    else:
        x=list(x)
        if mm==1 or mm==0:
            return ind[I]
        else:
            return ind[x.index(mm)]

def date(x):
    if x==0:
        return "n/a"
    else:
        return ("/".join([str(elem) for elem in (xlrd.xldate_as_tuple(x,0)[1],xlrd.xldate_as_tuple(x,0)[2],xlrd.xldate_as_tuple(x,0)[0])]))

ind=['Lab Testing','Encounters']
for X in ['Date.Max (min)','Date.Max (max)']:
    m=max
    if count==1:
        m=min
    d[X]=d1.apply(lambda x: temp(x,m=m),axis=1)
    d[X][d[X].isna()]=0
    R='Record_'+str(count)
    d[R]=d1.apply(lambda x: temp(x,m=m,ind=ind),axis=1)
    d[R][d[R].isna()]="Both"
    d['Date Text_'+str(count)]=d[X].apply(lambda x:  date(x))
    d['Periods_'+str(count)]=(d.iloc[:,0]-d[X]).apply(lambda x: abs(min(x,0))/365.25)
    count+=1

d=d.drop(['Encounters',  'Lab Testing',  'Medications'],axis=1)

if imaginary==1:exp=d.to_csv('Record_start_end_derived.csv')
else:exp=d.to_csv('Record_start_end.csv')