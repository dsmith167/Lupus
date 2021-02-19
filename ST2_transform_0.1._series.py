#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:31:20 2020

@author: dylansmith
"""

import pandas as pd
import numpy as np
from scipy import stats

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format


Record=pd.read_csv('Record_start_end_derived.csv')
#Record=pd.read_csv('Record_start_end.csv')

DB0=pd.read_csv('lupus_sort_imputed_f.csv')
DB1=pd.read_csv('0.Demographic.csv')
DB2=pd.read_csv('1.Encounter_2.csv').iloc[2:,:]
DB2['random number- study ID']=DB2['random number- study ID'].astype(int)
DB2=DB2[DB2['random number- study ID'].isin(DB0['new random ID'].astype(int))]
DB2=DB2.drop(['Hospital','Service_Line','Discharge_Date', 'LOS_Days', 'Admit_Source', 'Discharge_Disposition','Patient_Type', 'Referrer_Discipline'],axis=1)
DB2.columns=['new random ID']+list(DB2.columns)[1:]

#DB0 #49 to 64

D=pd.concat([DB0['new random ID'].astype(int),DB0.iloc[:,49:65]],axis=1)
D.columns=['new random ID','Gender_Female',
 'DOB',
 'Race_American Indian',
 'Race_Asian',
 'Race_Black',
 'Race_Pacific Islander',
 'Race_Hispanic',
 'Race_Other',
 'Race_White',
 'Marital_status_Divorced',
 'Marital_status_Legally Separated',
 'Marital_status_Married',
 'Marital_status_Other',
 'Marital_status_Partner',
 'Marital_status_Single',
 'Marital_status_Widowed']

temp=DB1['Date_of_Birth'][DB1['new random ID'].astype(int).isin(D['new random ID'])]
temp.index=range(len(temp))
D['DOB']=temp

dlist=list(D['DOB'].apply(lambda x: x.split('/')))
tlist=list(Record['Date Text_0'][Record['new random ID'].astype(int).isin(D['new random ID'])].apply(lambda x: x.split('/')))

def rl(x):
    return range(len(x))

def d_ext(year,ext=None):
    if len(year)==2:
        if ext!=None:
            year=ext+year
        elif int(year)<=10:
            year='20'+year
        else:
            year='19'+year
    return year

def age_comp(time_0,time_1,ext=None):
    #takes [mm,dd,yy] or [mm,dd,yyyy] inputs
    year=int(d_ext(time_0[2],ext=ext)),int(d_ext(time_1[2],ext=ext))
    month=int(time_0[0]),int(time_1[0])
    day=int(time_0[1]),int(time_1[1])
    age=year[1]-year[0]-1
    if (month[1]==month[0] and day[1]>=day[0]) or month[1]>month[0]:
        age+=1
    return age

def filter_(x,r_starts,r_dates,split=True):
        if split==True:
            return (age_comp(r_starts[x[0]],x[2].split("/"),ext='20')>=0 and age_comp(x[2].split("/"),r_dates[x[0]],ext='20')>=0)
        else:
            ## x is within df of indices
            return (age_comp(r_starts[x[0]],r_dates[x[0]],ext='20')>=0)

def period_plus(r_starts,periodNumber=1,end=True):
    p=periodNumber
    r_starts=r_starts.apply(lambda x: [str(((int(x[0])-1+6*p)%12)+1),x[1],str(int(x[2])+(int(x[0])-1+6*p)//12)])
    if end==True:
        return r_starts
    else:
        return r_starts.apply(lambda x: [x[0],str(int(x[1])-1),x[2]])

def replace(x,Dict):
    for d in Dict.keys():
        for v in Dict[d]:
            try:
                if v in x: return d
            except:
                pass

def replace_all(x,List):
    count=0
    for i in rl(x):
        for l in List:
            try:
                if l in x[i]: count=1
            except:pass
    return count

def zero(x):
    if np.isnan(x):
        return 0
    elif x>1:
        return 2
    return x

def empty(x):
    try:
        len(x)==0
        return np.nan
    except:
        return x

def prop(x,df,column):
    i1=list(df.columns).index(column)
    if int(column[-1])==0:
        return x[i1]
    column_0=column[:-1]+str(int(column[-1])-1)
    ## indices
    i0=list(df.columns).index(column_0)
    if np.isnan(x[i1]):
        return x[i0]
    return x[i1]

def default_empty(df,n):
    n=n-1
    I=list(df.columns)
    T=[]
    for i in rl(I):
        try:
            if int(I[i][-1])==n:
                T.append(I[i])
        except:
            pass
    m=I.index(T[0]); M=I.index(T[-1])
    return T,m,M
        
D['Age_0']=pd.DataFrame([0,]*len(D.index),index=D.index)

##### Diagnosis Columns

n_periods=6
#det=['Periods_1','Date Text_1']

det=['Periods_2','Date Text_2']
## 1 : Min
## 2 : Max

Record_=Record[Record[det[0]]!=0]
r_periods=Record_[det[0]]*2 #6 month intervals
r_starts=Record_['Date Text_0'].apply(lambda x: x.split("/"))
r_starts=pd.concat([r_starts[r_starts.apply(lambda x: (x[1]=='29' or x[1]=='30' or x[1]=='31'))].apply(lambda x: [x[0],'28',x[2]]),
                             r_starts[~r_starts.apply(lambda x: (x[1]=='29' or x[1]=='30' or x[1]=='31'))]])
r_dates=Record_[det[1]].apply(lambda x: x.split("/"))
r_periods.index,r_dates.index,r_starts.index=[Record_['new random ID'],]*3

############################## reset starting dates
r_old_starts=r_starts
r_starts=period_plus(r_dates,-6)
##############################

## Set Ages

## Set Patients
D_=D[D['new random ID'].isin(Record_['new random ID'])]
D_.index=D_['new random ID']
## Set Data
Data=DB2[DB2['new random ID'].isin(Record_['new random ID'])]
## Clear instances of Cancelled or duplicate appointments
Data=Data[~(Data['Encounter_Status']=='Cancelled')]


ids=list(D_['new random ID'])
plist=[]

Clinics={'Rheumatology':[], #1
        'Dermatology':[], #2
        'Nephrology':[], #3
        'Other Visits':[]} #else
clinics=['Rheumatology','Dermatology','Nephrology','Other Visits']

Payor={1:[],
       2:[],
       3:[],
       4:[],
       5:[]}

#### fill dictionaries ####

C=np.array(pd.read_csv('preprocess_clinic.csv').iloc[:,3:6])
P=np.array(pd.read_csv('preprocess_payor.csv').iloc[:,3:6])

for i in rl(P):
    for j in range(1,6): 
        if P[i][1]==j: Payor[j].append(P[i][0])

for i in rl(C):
    for j in rl(clinics): 
        if C[i][1]==j+1: Clinics[clinics[j]].append(C[i][0])

###########################

Diagnoses={'Arterial Thrombosis': ['I74.','444.'],
           'Hemolytic Anemia': ['D59.','283.'],
           'Vasculitis': ['I77.6','M31.8','447.6'],
           'Pleurisy': ['511.9','511.0','511.89','R09.1'],
           'Pericarditis': ['I30.','I32.','I31.','420.','M32.12'],
           'acute lupus rash': ['L93.0','L93.1','L93.2','695.4'],
           'renal impairment other than ESRD': ['584','N17.'],
           'Myocarditis': ['I40.8','I40.9','I51.4','429.0','422.0','422.9'],
           'Autoimmune Hepatitis': ['573.3','571.42','K75.4','K75.9'],
           'Nephritis': ['580.','583.','N05.','N03.2','791.0','R80.','M32.14','M32.15'],
           'Venous Thrombosis': ['415.1','673.2','639.6','O88.2','I26','453.','I82.4','I82.9'],
           'Demyelinating Syndrome or Myelitis': ['G37.3','G04.89']}




# Set Ages at starting period
tlist=[]
for i in rl(r_starts): 
    if r_starts.index[i] in list(D_['new random ID']):
        tlist.append(r_starts[r_starts.index[i]])
alist=[];dlist=list(D_['DOB'].apply(lambda x: x.split('/')))
for i in range(len(tlist)):
    age=age_comp(dlist[i],tlist[i]);alist.append(age)
D_['Age_0']=alist
#####


List=[]
for n in range(1,n_periods+1):
    print("starting period",n)
    ## set date vectors
    start=period_plus(r_starts,n-1)
    end=period_plus(r_starts,n,end=False)
    ## trim data
    Data_n=Data[Data.apply(lambda x: filter_(x,start,end),axis=1)]
    ## replace out Payor/Clinic Names for their group name/number
    Data_n['Payor']=Data_n['Payor'].apply(lambda x: replace(x,Payor))
    Data_n['Clinic_Name']=Data_n['Clinic_Name'].apply(lambda x: replace(x,Clinics))
    grouping=Data_n.groupby('new random ID')
    
    ##PAYOR
    name='Payor_'+str(n-1)
    D_[name]=Data_n[Data_n['Payor'].isin([1,2,3,4,5])].groupby('new random ID')['Payor'].agg(lambda x: stats.mode(x)[0])
    D_[name]=D_.apply(lambda x: prop(x,D_,name),axis=1)
    ##CLINICS
    for m in rl(clinics):
        name=clinics[m]+"_"+str(n-1)
        if m!=3: #named clinic types
            D_[name]=Data_n[Data_n['Clinic_Name']==clinics[m]].groupby('new random ID')['Clinic_Name'].apply(len)
        else: #other type
            D_[name]=Data_n[~Data_n['Clinic_Name'].isin(clinics[0:3])].groupby('new random ID')['Clinic_Name'].apply(len)
        D_[name]=D_[name].apply(zero) 
    ##DIAGNOSES
    di=list(Diagnoses.keys())
    #Data_n.columns[-13:-1]
    for m in rl(di):
        name=di[m]+"_"+str(n-1)
        temp=Data_n.copy();temp=pd.concat([temp.iloc[:,0:1],temp.iloc[:,-13:-1]],axis=1)
        temp=pd.concat([temp['new random ID'],temp.apply(lambda x: replace_all(x,Diagnoses[di[m]]),axis=1)],axis=1);temp.columns=['new random ID','det']
        temp=temp.groupby('new random ID').max()
        D_[name]=temp
        D_[name]=D_[name].apply(zero)
        
    ## get rid of entries that are past the end of their record
    ## complete periods (CHOOSE ONE)
    #Lose=start.index.to_frame()[start.index.to_frame().apply(lambda x: filter_(x,r_dates,end,split=None),axis=1)]['new random ID']
    #Keep=start.index.to_frame()[start.index.to_frame().apply(lambda x: filter_(x,end,r_dates,split=None),axis=1)]['new random ID']
    ## incomplete periods (CHOOSE ONE)
    #Lose=start.index.to_frame()[start.index.to_frame().apply(lambda x: filter_(x,r_dates,start,split=None),axis=1)]['new random ID']
    #Keep=start.index.to_frame()[start.index.to_frame().apply(lambda x: filter_(x,start,r_dates,split=None),axis=1)]['new random ID']
    #T=default_empty(D_,n)[0]
    #D_.loc[D_['new random ID'].isin(Lose),T] = np.nan
    #List.append([Lose,Keep])
    
## Labels

import warnings
warnings. simplefilter(action='ignore', category=Warning)

## Set periods and trim data (t=t_end to t_end+12mo.)
start=period_plus(r_starts,n) #r_dates
end=period_plus(r_starts,n+2) #pp(r_dates,2)
Data_n=Data[Data.apply(lambda x: filter_(x,start,end),axis=1)]

## list of complete IDs for later #whether r_old_starts<r_starts
Keep=start.index.to_frame()[start.index.to_frame().apply(lambda x: filter_(x,r_old_starts,r_starts,split=None),axis=1)]['new random ID']

##LABEL
Label={'Label_Lupus Flareup':['710.0','M32']}

step=3
di='Label_Lupus Flareup_{}_I'.format(step) #Diagnoses 1-N #Filtered by inpatient
name=di
##temp data set containing random ID and diagnoses
temp=Data_n.copy();temp2=pd.concat([temp.iloc[:,0:3],temp.iloc[:,4:5],temp.iloc[:,-13:-13+step]],axis=1);temp=pd.concat([temp.iloc[:,0:1],temp.iloc[:,-13:-13+step]],axis=1)
temp=temp[Data_n['Inpatient_Outpatient']=='Inpatient']
##sub out Dagnosis for group name, then group to determine if 1 exists in grouping
temp=pd.concat([temp['new random ID'],temp.apply(lambda x: replace_all(x,Label[di[:19]]),axis=1)],axis=1);temp.columns=['new random ID','det']
temp=temp.groupby('new random ID').max()
class_1=temp[temp.index.isin(Keep)].groupby('det')['det'].count()[1]
print(di,"\n columns: 3\n filter: Inp\n class_1: ",class_1,"\n class_0: ",len(Keep)-class_1)
class_1=temp.groupby('det')['det'].count()[1]
print("_whole dataset:\n class_1: ",class_1,"\n class_0: ",len(D_)-class_1)
## place column in complete data and replace na with 0
D_[name]=temp
D_[name]=D_[name].apply(zero)

## remove all entries in final data set that are not in the list of complete IDs
D_=D_.loc[D_['new random ID'].isin(Keep)]
#D_.rename(columns={'Age_0':'Age Start'},inplace=True)
exp=D_.to_csv('0.enc_3_1.csv')

## remove all entries in final data set that are not in the list of complete IDs
#D_=D_.loc[D_['new random ID'].isin(Keep)]

## exported start dates 7/14/20
## Filtered Start Dates for new Scheme. Exclusion of incomplete records.
#exp=r_starts
#exp=pd.concat([exp,pd.DataFrame(exp.index,index=exp.index)],axis=1)
#exp=exp.loc[exp['new random ID'].isin(Keep)].drop(['new random ID'],axis=1)
#X=exp.to_csv('new_start.csv')

#from scipy.interpolate import interp1d as interp
#x=pd.DataFrame([np.nan,1,np.nan,np.nan,2,3,2,np.nan,4,np.nan,np.nan],index=list(range(11))).T
#x.interpolate(axis=1,limit_area='inside')
##now we have complete inner data
#

'''from scipy.interpolate import interp1d as interp
def extrapolate(x,Type=1,k=2):
    #Type: 0 is in before
    #Type: 1 is after
    #k is num of entries to use for extrapolation surrounding NA
    #k>=2
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
        i=0;y=k
        if Type==1:i=len(X)-count_na
        while count_na!=0:
            X[i]=float(f(y))
            y+=1; i+=1; count_na-=1
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
    
### example 1-- averaging
x=pd.DataFrame([np.nan,1,np.nan,np.nan,2,3,2,np.nan,4,np.nan,np.nan],index=list(range(11))).T

# first fill inner values (interpolation)
x=x.interpolate(axis=1,limit_area='inside')
# then fill outer values (extrapolation)
x=x.apply(lambda x: extrapolate(x,1),axis=1)

# Done, Result:

#    0    1    2    3    4    5    6    7    8    9    10
#0 0.67 1.00 1.33 1.67 2.00 3.00 2.00 3.00 4.00 5.00 6.00

### example 2-- previous
x=pd.DataFrame([np.nan,1,np.nan,np.nan,2,3,2,np.nan,4,np.nan,np.nan],index=list(range(11))).T

# fill inner and following values (interp./extrap.)
x.apply(lambda x: forward(x),axis=1)

# Done, Result:

#   0    1    2    3    4    5    6    7    8    9    10
#0 nan 1.00 1.00 1.00 2.00 3.00 2.00 2.00 4.00 4.00 4.00
'''
