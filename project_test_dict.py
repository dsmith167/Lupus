#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 04:21:05 2020

@author: dylansmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:48:01 2020

@author: dylansmith
"""

from project_functions import *

sources=['../cleaned_data_3_1.csv',
         '../cleaned_data_3_1_extrapolate.csv',
         '../data_3_1_.csv',
         '../impute_data_avg.csv',
         '../cleaned_data_old.csv']

outputs=['saved_results','output','feat_importance']

arg=sys.argv
i=1; sd_,pr_,pc_,no_,lm_,md_=-1,-1,-1,-1,-1,-1
while i<len(arg):
    if i==1: sd_=arg[1].lower() #y/n to small dataset
    if i==2: pr_=arg[2].lower() #0: Average DataSet, 1:Time-Series DataSet
    if i==3: pc_=arg[3].lower() #y/n to PCA
    if i==4: no_=arg[4].lower() #normalization: mm:0, ss:1
    if i==5: lm_=arg[5].lower() #pca: 0:thr .95//1:n_feat=bag size//dec:n_feat=% bag size//int:n_feat=int
    if i==6: md_=arg[6].lower() #0 for all, -1 for ensemble, or input for only run selected model
    i+=1

models=['dt','lg','svc','rf','gnb','bnb','knn','nn','et','gb']

write=0

#.74, .46, .29

if True:#write==1:

    Sum=10 #Sum=7
    K=Sum
    if Sum==0: K=1
    c=0
    for i in [0]: #pca
        for m in [0,1]: #average/time-series
            for j in [1]: #scaler
                for k in range(K):
                    for l in [10]:
                        c+=1
                        temp=models[k]; tem="nohup python3"; te=">log"+str(c)+".txt &"
                        if Sum==0: temp=-1; tem="!python"; te=""
                        print(tem+" project_test_dict.py",2,m,i,j,l,temp,te)

#for i in {1..35}; do rm "log$i.txt"; done

#https://linuxhint.com/nohup_command_linux/

df = pd.read_csv(sources[0],low_memory = False).drop(['new random ID.1'],axis=1)
df2= pd.read_csv(sources[1],low_memory = False).drop(['new random ID.1'],axis=1)
df0= pd.read_csv(sources[2])
df.index=df['new random ID'];df0.index=df0['new random ID'];df2.index=df2['new random ID']
if type(sd_)==str: sd_=int(sd_)
elif sd_==-1:
    sd_=int(input("smaller dataset: \n0 : impute:1723 items\n1 : forward_extr:643 items\n2 : two-way_extr:925 items\n3 : full impute:1996 items\nchoice : "))
if sd_==1:
    df=df0[df.columns]
    df=df.dropna(0)
elif sd_==2:
    df=df2
elif sd_==3:
    df=pd.read_csv(sources[4],low_memory = False);df.index=df['new random ID']

#Average/TimeSeries

if pr_=='1' or pr_=='y': pr_=1
elif pr_=='0' or pr_=='n': pr_=0
elif pr_==-1:
    prg=input("0/n:Average Model, 1/y:Time Series: 0/1: ").lower()
    if prg=='y' or prg=='1': pr_=1
    else: pr_=0
if sd_==3:df=df.drop(columns=['new random ID']);pr_==0
elif pr_==0:
    DF=pd.read_csv(sources[3],low_memory = False);DF.index=DF['new random ID']
    DF=DF.drop(['new random ID','Creatinine, random urine','ESR'],axis=1); df=DF.T[df.index].T
else:df=df.drop(columns=['new random ID','DOB']);df0=df0.drop(columns=['new random ID','DOB'])

df.columns=list(df.columns)[:-1]+['Label'];df0.columns=list(df0.columns)[:-1]+['Label']

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import warnings
warnings. simplefilter(action='ignore', category=Warning)

##drop HCT
#cols=list(df.columns)
#for i in rl(cols):
#    if "HCT" in df.columns[i]: cols.remove(df.columns[i])
#df=df[cols]

print(df['Label'].value_counts())

encounter=['Arterial Thrombosis','Hemolytic Anemia','Vasculitis','Pleurisy','Pericarditis','acute lupus rash','renal impairment other than ESRD','Myocarditis',
 'Autoimmune Hepatitis','Nephritis','Venous Thrombosis','Demyelinating Syndrome or Myelitis','Rheumatology','Dermatology','Nephrology','Other Visits','Payor']

medication=['Hydroxychloroquine','Azathioprine','Belimumab','Cyclosphamide','Methotrexate','Mycophenolate','Oral corticosteroids','Other SLE immunsuppressant','Rituximab']

if pr_==1:
    ##Create Change Columns
    I=[]
    for i in rl(df.columns):
        if df.columns[i][0]=='1':I.append(df.columns[i][1:])
    
    for i in range(5):
        for j in rl(I):
            name=str(i+1.5)+I[j]
            df[name]=df[str(i+2)+I[j]]-df[str(i+1)+I[j]]

    ##Organize
    cols=list(df.columns[0:17])+sorted(list(df.columns[17:]))
    df=df[cols]

    ##Drop Features
    
    for i in rl(I):I[i]=I[i][1:]
    
    drop=medication
    cols=list(df.columns)
    for i in rl(cols):
        dfc=df.columns[i]
        if dfc.split('.')[-1] in drop:
            cols.remove(dfc)
    df2=df[cols]
    
    drop=medication+encounter
    cols=list(df.columns)
    for i in rl(cols):
        dfc=df.columns[i]
        if dfc.split('.')[-1] in drop:
            cols.remove(dfc)
    df3=df[cols]


feat_labels = list(df.columns)[:-1]


    
X = df.drop(columns=['Label'])
y = df['Label'].values

# Constant number of records
num_instances = df.shape[0]

# Number of k folds
fold_number= 10
# Number of bags
bag_number = 100

bag_size=list(df['Label']).count(1)*2

pca_=0
if pc_=='1' or pc_=='y': pc_=1
elif pc_=='0' or pc_=='n': pc_=0
elif pc_==-1:
    pct=input("PCA? y/n: ").lower()
    if pct=='y' or pct=='1':pc_=1
if pc_==1:pca_=1

if no_=='1' or no_=='ss': no_=1
elif no_=='0' or no_=='mm': no_=0
elif no_=='2' or no_=='pct': no_=2
else:
    no_=input("Normalization Method? mm/ss/pct: ").lower()
    try: no_=int(no_)
    except: inp=no_
if no_==1:inp='ss'
elif no_==0:inp='mm'
elif no_==2:inp='pct'


if inp=='ss' or inp=='mm' or inp=='pct':
    if pca_==1:pca_=inp; scale=inp
    elif pca_==0:scale=inp



##Threshold level
#Threshold=[.35,.4,.45,.5,.55]
Threshold=[.3,.35,.4,.45,.5,.55,.6]

##Models and Statistics Used

Model_Names=['dt','lg','svc','rf','gnb','bnb','knn','nn','et','gb']
stats=['f1', 'f0', 'acc1', 'acc0', 'acc', 'auc', 'ppr', 'npr', 'diagI']
stat_num=len(stats)


if md_==-1:I=input("Model Selection 0 for all, index+1 for model, or by name: ")
else: I=md_
try: 
    I=int(I); md_=I
    if I==0:pass
    elif I==-1: Model_Names=[]
    else:Model_Names=[Model_Names[I-1]]
except:
    if I in Model_Names: md_=Model_Names.index(I)+1;Model_Names=[I]; 
    else:pass; md_=0


##Parameters

kn=[5,7,9,20,30]
if bag_size<60:
    for k in rl(kn): kn[k]=max(1,round(kn[k]*(bag_size/60)))
    kn=sorted(list(set(kn)))
mss=[2,3,4]
msl=[2,4,8]
mxd=[2,4,6,8]
nes=[100,200,350,500]
if pr_==1:temp=[100,1000,10000]
else: temp=[1,5,10]


Search={
        'dt': {"min_samples_split":mss,"min_samples_leaf":msl,"max_depth":mxd},
        'lg': {"C":temp},
        'svc': {"kernel":['linear','rbf'],"C":[.01,10,100]},
        'rf': {"min_samples_split":mss,"min_samples_leaf":msl,"max_depth":mxd,"n_estimators":nes[0:2]},
        'gnb': {},
        'bnb': {},
        'knn': {"n_neighbors":kn,"metric":['minkowski','manhattan']},
        'nn': {"alpha":[1,2,3],"hidden_layer_sizes":[(100,50),(128,128),(256,128),(256,256)]},
        'et': {"min_samples_split":mss,"min_samples_leaf":msl,"max_depth":mxd,"n_estimators":nes[-1:]},
        'gb': {"max_depth":mxd,"n_estimators":nes[-1:],'min_samples_split':mss,"learning_rate":[.01,.1,.5]}}
Given_Pars={
        'dt': {},
        'lg': {"penalty":'l2',"solver":'lbfgs'},
        'svc': {"probability":True, "gamma":0.01},
        'rf': {"random_state" : 42},
        'gnb': {"var_smoothing" : 10}, ##maybe change to search
        'bnb': {"alpha" :1},           ##maybe change to search
        'knn': {"weights":'distance'},
        'nn': {},
        'et': {"random_state" : 42},
        'gb': {"random_state" : 42,"subsample":.5,"loss":'deviance'}}


if pca_==False: X = norm_df(X,scale=scale); df[X.columns]=X; X=X.values; scale_=scale
else:
    X=norm_df(X,scale=scale)
    
    if lm_=='1' or lm_=='y': NewData,pca=doPCA(X,n=bag_size)
    elif lm_=='0' or lm_=='n': NewData,pca=doPCA(X,R_=.95)
    else:
        if lm_==-1:temp_inp=input("Limit Features according to bag size? y/n (or type int/decimal for # of features/ % of bag size): ").lower()
        else:temp_inp=lm_
        if temp_inp=='y': #default is size of bag
            NewData,pca=doPCA(X,n=bag_size)
        try: 
            f_inp=float(temp_inp)
            itemp_inp=int(f_inp) #feature size according to input
            if itemp_inp==f_inp:
                NewData,pca=doPCA(X,n=itemp_inp)
            else:
                itemp_inp=f_inp #decimal of bag size e.g. ".5" == 50%
                iti=round(bag_size*itemp_inp)
                NewData,pca=doPCA(X,n=iti)
        except: NewData,pca=doPCA(X,R_=.95)
    X = NewData
    N=np.ndarray.tolist(X.T);N.append(list(y));N=np.array(N).T
    df=pd.DataFrame(N,index=df.index)
    L=list(df.columns)
    for i in rl(L):L[i]="PC_"+str(i+1)
    L[-1]='Label'
    df.columns=L
    print("explained variance:     ",round(np.sum(pca.explained_variance_ratio_*100),3),"%")

print("shape of feature set:  ",X.shape)
print("shape of data set:     ",df.shape)

Par_Performance,Performance,Tr_Performance,Vd_Performance,Model_Lists,Parameters=[],[],[],[],[],[]





#################################################################################################################
#################################################################################################################


          
THRESHOLD=Threshold

####################
#Model_Names=['dt','lg','svc','rf','gnb','bnb','knn','nn','et','gb']
####################

Model_List,Prediction_List,Majority_Param,Top_Param={},{},{},{}
for thresh in Threshold: 
    if thresh==Threshold[0]:Perform,Tr_Perform,Vd_Perform=dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold)))
    Perform[thresh],Tr_Perform[thresh],Vd_Perform[thresh]={},{},{}

for w in rl(Model_Names):
    
    MNAME=Model_Names[w];  Classifier,Prediction=[],[]; fold_index=-1
    GIVENPARS=Given_Pars[MNAME]; SEARCH= Search[MNAME]; Choice=d_combine(GIVENPARS,SEARCH)
    skfolds = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=42)
    Training_Rec,Testing_Rec,Validation_Rec=dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold)))
    for i in Choice.keys(): Choice[i]=[] #set up dictionary of parameters
    if w==0: Bag_List=[]
    for train_index, test_index in skfolds.split(X,y):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
        test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
        print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
        
        TRAINING = df.iloc[train_index]
        
        fold_index+=1
        
        if w==0: bag_list = get_bags(TRAINING,bag_number); Bag_List.append(bag_list)
        else: bag_list= Bag_List[fold_index]
        
        classifier,prediction = [],[]
        ParChoice=Choice.copy(); TRAINING_REC,VALIDATION_REC=dict(zip(Threshold,[0,]*len(Threshold))),dict(zip(Threshold,[0,]*len(Threshold)))
        for i in range(bag_number):   ### for each bag, do folds-1 fold CV for parameter tuning and training acc/stats
            BAG=bag_list[i]
            C=Classify('Label',BAG,scaler=None,model=MNAME,pars=GIVENPARS,search=SEARCH,scoring='roc_auc', 
                       folds=fold_number-1, pct=5,threshold=THRESHOLD,stat=stat_num)
            MODEL,PARS,Train_Rec,Valid_Rec,Pred=C
            TRAINING_REC=d_add(TRAINING_REC,Train_Rec)
            VALIDATION_REC=d_add(VALIDATION_REC,Valid_Rec)
            classifier.append(MODEL); prediction.append(Pred)
            ParChoice=d_inner(ParChoice,PARS)
        b=1/bag_number
        TRAINING_REC,VALIDATION_REC=d_multiply(TRAINING_REC,b), d_multiply(VALIDATION_REC,b) #average
        pred=0
        for i in rl(classifier): pred+=classifier[i].predict_proba(test_X)/bag_number
        
        TESTING_REC=dict(zip(Threshold,[0,]*len(Threshold)))
        for thresh in Threshold:
            TESTING_REC[thresh]=compute_measure(test_y,pred,thresh)[:stat_num]
        
        Testing_Rec=d_add(Testing_Rec,d_multiply(TESTING_REC,1/fold_number))
        Training_Rec=d_add(Training_Rec,d_multiply(TRAINING_REC,1/fold_number))
        Validation_Rec=d_add(Validation_Rec,d_multiply(VALIDATION_REC,1/fold_number))

        Choice=d_inner(Choice,ParChoice)
        Classifier.append(classifier); Prediction.append(reshape_add(np.array(prediction),pred))
    Model_List[MNAME]=Classifier;  Prediction_List[MNAME]=Prediction
    Majority_Param[MNAME]=d_most(Choice);  Top_Param[MNAME]=d_top(Choice)
    for thresh in Threshold:
        if w==0: Perform[thresh],Tr_Perform[thresh],Vd_Perform[thresh]={},{},{}
        Perform[thresh][MNAME]=Testing_Rec[thresh]
        Tr_Perform[thresh][MNAME]=Training_Rec[thresh]
        Vd_Perform[thresh][MNAME]=Validation_Rec[thresh]
        a,b,c,d,e,f=Testing_Rec[thresh][:6];     g,h,i,j,k,l=Training_Rec[thresh][:6]
        print("Threshold {}, Testing: {} -- \nf1={}\nf0={}\nacc1={}\nacc0={}\nacc={}\nauc={}\n".format(thresh,MNAME.upper(),a,b,c,d,e,f))
        print("Threshold {}, Training: {} -- \nf1={}\nf0={}\nacc1={}\nacc0={}\nacc={}\nauc={}\n".format(thresh,MNAME.upper(),g,h,i,j,k,l))

if md_!=-1:
    Model_Lists.append(Model_List)
    for thresh in Threshold:
        Parameters.append(Top_Param)
        #Par_Performance.append(Majority_Param)
        Performance.append(Perform[thresh])
        Tr_Performance.append(Tr_Perform[thresh])
        Vd_Performance.append(Vd_Perform[thresh])

#################################################################################################################
#################################################################################################################

sni=str(num_instances)+"_"
xs=str(X.shape[1])+"_"
if "PC_1" in df.columns:pc=str(1)+"_"
else:pc=str(0)+"_"
inp=inp+"_"
str_1=pc+sni+xs+inp #predictions
str_2=pc+sni+inp+xs #stats

if md_!=0 and md_!=-1:
    str_2_=pc+sni+inp+xs+Model_Names[0]
    p=len(Prediction); output={}
    for i in rl(Prediction): output[str(i)]=Prediction[i]
    scipy.io.savemat("saved_results/"+str_1+Model_Names[0]+".mat", mdict=output, oned_as='row')


for i in range(len(Threshold)):
    print("\nThreshold:",Threshold[i])
    for J in rl(Model_Names):
        name=Model_Names[J]
        keys=list(Parameters[i][name].keys())
        print("\nResults:",name)
        print("\nTesting:")
        for H in rl(Performance[i][name]):
            print(stats[H]+":("+str(round(Performance[i][name][H],5)), end=")  ")
        print("\nTraining:")
        for H in rl(Tr_Performance[i][name]):
            print(stats[H]+":("+str(round(Tr_Performance[i][name][H],5)), end=")  ")
        print("\nValidation:")
        for H in rl(Vd_Performance[i][name]):
            print(stats[H]+":("+str(round(Vd_Performance[i][name][H],5)), end=")  ")  
        print("\nHyper Parameters:")
        for key in keys:
            print(key+":"+str((Parameters[i][name][key])), end="  ")
        print()
        
            
L0=['dt','lg','svc','rf','gnb','bnb','knn','nn','et','gb']
Model_Names=['dt','lg','svc','rf','gnb','bnb','knn','nn','et','gb']
Scaling='auc' #'f1', 'f0', 'acc1', 'acc0', 'acc', 'auc'
#ensemble

def roc(y,pred):
    mt,mf,mh=roc_curve(y,pred.T[1])
    return mt,mf #AUC(mt,mf) #np.concatente([A,B])

ROC={}

print()
print()
if md_==0 or md_==-1:
    for i in range(len(Threshold)):
        I=i
        if md_==-1:   #load saved results
            Performance.append({})
            Tr_Performance.append({})
            Vd_Performance.append({})
            Parameters.append({})
            Stats={}; xs=str(X.shape[1]); route="saved_results/"
            contents=os.listdir(route);L02=[]
            for i in rl(L0):
                name=L0[i]
                st_1,st_2=str_1+name+".mat",str_2+name+".csv"
                if (st_2 not in contents) or (st_2 not in contents): print("missing file:",name);continue
                else:L02.append(name)
                temp0=scipy.io.loadmat(route+st_1) #predictions
                temp=pd.read_csv(route+st_2).iloc[I:I+1,1:];tlist=list(temp.columns);temp=list(np.array(temp)[0]);ch=0 #stats
                #########
                while ch<len(temp)-1:
                    if 'Choice' in tlist[ch]: del temp[ch]; del tlist[ch]
                    else: ch+=1
                #########
                Prediction_List[name]=temp0
                Performance[-1][name]=np.array(temp[0:stat_num])
                Tr_Performance[-1][name]=np.array(temp[stat_num:2*stat_num])
                Vd_Performance[-1][name]=np.array(temp[2*stat_num:3*stat_num])
                Parameters[-1][name]=temp[-1]
            L0=L02
        THRESHOLD=Threshold[I]
        stat_ind=stats.index(Scaling)
        Perform=Performance[I]
        MNAME='ens'
        
        if md_==-1: Bag_List=[]
        
        #['NN','KNN','DT','SVC','Logistic','RF','ET','GB','BNB']
        L1=[]
        
        for n in rl(L0): 
            if L0[n] not in ['dt','rf','lg','bnb','nn']:L1.append(0) #gnb
            else:L1.append(Perform[L0[n]][stat_ind]*100)
        total_scale=np.sum(L1)
        
        Testing_Rec,Training_Rec,Validation_Rec=0,0,0
        for name in L0: scale=L1[L0.index(name)];Validation_Rec+=Vd_Performance[I][name]*scale/total_scale
        fold_index=-1
        for train_index, test_index in skfolds.split(X,y):
            train_X, test_X = X[train_index], X[test_index]
            train_y, test_y = y[train_index], y[test_index]
            train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
            test_0, test_1 = len(test_y[test_y==0]), len(test_y[test_y==1])
            print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
        
            fold_index+=1
            
            TRAINING = df.iloc[train_index]
            
            if md_==-1:bag_list = get_bags(TRAINING,bag_number); Bag_List.append(bag_list)
            else:bag_list= Bag_List[fold_index]
            
            pred=0; TRAINING_REC,VALIDATION_REC=0,0
            
            predict_list={};pred_list={};j=-1
            for p in rl(L0):
                name=L0[p]; temp=Prediction_List[name][str(fold_index)]; t0=temp[0]
                if j==-1:
                    for j in rl(t0):
                        if np.max(t0[j])==0:break
                #if j!=80: print("broken:",name,fold_index)
                predict_list[name]=temp[:bag_number,:j,:]; pred_list[name]=temp[-1,:,:]
                
            for i in range(bag_number):
                pred_tr=0
                for p in rl(L0):
                    name,scale=L0[p],L1[p]
                    BAG=bag_list[i]
                    pred_tr+=predict_list[name][i]*scale
                    if i==0:
                        if fold_index==0:ROC[name]={}; ROC[name]["pred"]=pred_list[name]
                        else: ROC[name]["pred"]=np.concatenate([ROC[name]["pred"],pred_list[name]])
                        
                        pred+=pred_list[name]*scale
                Train_Rec=compute_measure(BAG['Label'],pred_tr/total_scale,THRESHOLD)[:stat_num]
                TRAINING_REC=np.add(TRAINING_REC,Train_Rec)
            TRAINING_REC=TRAINING_REC/bag_number
            pred=pred/total_scale
            
            if fold_index==0:ROC[MNAME]={}; ROC[MNAME]["pred"]=pred; ROC["Y"]=test_y
            else: ROC[MNAME]["pred"]=np.concatenate([ROC[MNAME]["pred"],pred]); ROC["Y"]=np.concatenate([ROC["Y"],test_y])
            
            TESTING_REC=np.array(compute_measure(test_y,pred,THRESHOLD)[:stat_num])
            
            Testing_Rec=Testing_Rec+TESTING_REC/fold_number
            Training_Rec=Training_Rec+TRAINING_REC/fold_number
        a,b,c,d,e,f=Testing_Rec[:6];     g,h,i,j,k,l=Training_Rec[:6]
        print("Testing: {} -- \nf1={}\nf0={}\nacc1={}\nacc0={}\nacc={}\nauc={}\n".format(MNAME.upper(),a,b,c,d,e,f))
        print("Training: {} -- \nf1={}\nf0={}\nacc1={}\nacc0={}\nacc={}\nauc={}\n".format(MNAME.upper(),g,h,i,j,k,l))
        Performance[I][MNAME]=Testing_Rec
        Tr_Performance[I][MNAME]=Training_Rec
        Vd_Performance[I][MNAME]=Validation_Rec
        Parameters[I][MNAME]="{}"

## Exporting Performance ################

for I in range(3):
    Level=[Performance,Tr_Performance,Vd_Performance][I]
    level=['','tr_','vd_'][I]
    for i in rl(Threshold):
        temp=unzip(Level[i]);  pd4=pd.DataFrame(temp[0])
        pd4.index=temp[1]
        if I==2:
            temp2=unzip(Parameters[i],1); pd2=pd.DataFrame(temp2[0]); pd2.index=temp2[1]
            pd4=pd.concat([pd4,pd2],axis=1)
            cols=list(pd4.columns[:-1])+['Top_Parameters']
        else:cols=list(pd4.columns)
        ind=list(pd4.index)
        ti=abs(4-len(str(Threshold[i])))
        for i_ in rl(ind): ind[i_]=str(Threshold[i])[1:]+"0"*ti+"_"+ind[i_]
        for c in rl(cols):
            if c!=stat_num:
                cols[c]=level+stats[c]
            else: pass
        pd4.columns=cols; pd4.index=ind
        if i==0:pd3=pd4
        else:pd3=pd.concat([pd3,pd4],axis=0)
    if I==0: pd1=pd3
    else: pd1=pd.concat([pd1,pd3],axis=1)


############################
import matplotlib as mpl

def roc2(y,pred,n=2000):
    #n is number of sampled thresholds-1; 1/n is grain of search
    A=[];thr=[];B=[]
    for i in range(n+1):
        th=i/n
        tp,tn=compute_measure(y,pred,th)[2:4]; fp=1-tn
        A.append((fp,tp));thr.append(th)
    S=np.array(list(set(A))).T
    for i in range(len(S.T)):
        B.append((S.T[i][0],S.T[i][1],thr[A.index(tuple(S.T[i]))]))
    B.sort(key=(lambda x:x[2]))
    return np.array(B)


if md_==0 or md_==-1:
    style="default" #seaborn
    mpl.style.use(style); color="white"
    plt.figure(figsize=(10,8)); P=Performance[0]; ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for mod,name in zip(['dt','lg', 'rf', 'bnb', 'nn', 'ens'],['Decision Tree','Logistic Regression', 'Random Forest', 'Bernoulli N.B.', 'Neural Network', 'Ensemble Method']):
        R=roc2(ROC['Y'],ROC[mod]['pred'])[:,:2].T
        if mod!='ens':plt.plot(R[0],R[1],label=name+': '+' (AUC={})'.format(str(P[mod][5])[:5]),alpha=.8,lw=1.5)
        else:plt.plot(R[0],R[1],label=name+': '+' (AUC={})'.format(str(P[mod][5])[:5]),alpha=.8,lw=1.5,color='black',linestyle='--')
    if X.shape[1]==588:title="Progressive"
    else:title="Average"
    plt.title('Receiver Operating Characteristic Curves for dataset : {}, {}'.format(X.shape[0],title),fontsize=13)
    plt.ylabel('True positive rate',fontsize=11); plt.xlabel('False positive rate',fontsize=11)
    plt.xticks(ticks);plt.yticks(ticks)
    plt.plot([0, 1],[0, 1],linestyle=':', lw=1, color='r', alpha=.8)
    #if style=="default":color="lightgrey"
    plt.grid(color=color)
    plt.legend(fontsize=11,loc="lower right")
    plt.draw();plt.savefig("ROC curves/"+str(X.shape[0])+"."+title+".png");plt.close()

#dna=".5.dsDNA Ab,binary"
#ccols=['1.5.dsDNA Ab,binary','2.5.dsDNA Ab,binary','3.5.dsDNA Ab,binary','4.5.dsDNA Ab,binary','5.5.dsDNA Ab,binary','1.dsDNA Ab,binary','2.dsDNA Ab,binary','3.dsDNA Ab,binary','4.dsDNA Ab,binary','5.dsDNA Ab,binary','6.dsDNA Ab,binary','Label']
#df[ccols].iloc[:5,5:]
#np.mean(df[ccols][df['Label']==1])

#smooth ROC curve; ranking with combined HCT and HGB
#export DNA data


import datetime as dt
prefix="output/"
today=str(dt.datetime.today()).split(" ")[0][5:]+".csv"

## export

if pca_==0: pca_=scale_.upper()+"_"
else: pca_="_PCA_"+pca_.upper()+"_"
bags=str(bag_number)+"-bags"

if md_==0: md_=""
elif md_==-1: md_="ens"
else: md_=Model_Names[0]

NN=str(num_instances)+"."+str(X.shape[1])+"."

## ensure unique csvs

lines=os.listdir(prefix)
count=0
for l in lines:
    if ((today[:-4] in l) and (".csv" in l)) and (('compl_'+NN+pca_+bags+"_"+today) in l):count+=1
if count>0:today=today[:-4]+'__'+str(count)+".csv"

#####

if md_=="" or md_=="ens":
    exp=pd1.to_csv(prefix+NN+pca_+bags+"_"+today)
else:
    exp=pd1.to_csv("saved_results/"+str_2_+".csv")

#export data

#exp=df.to_csv("../"+NN+pca_+'_post_differences_cleaned_data.csv')
#exp=df2.to_csv("../"+NN+'_no_med_pdcd.csv')
#exp=df3.to_csv("../"+NN+'_no_enc_no_med_pdcd.csv')


#[dsmith167@erdos ~]$ "nohup python file.py > log1.txt %"


'''
#########################################################
##FEATURE IMPORTANCE

models=list(Model_List.keys())
target="feat_importance/"

for M in ['lg','svc','bnb']:
    model=M
    if model in models:
        md=unravel(Model_List[model])
    
        feature_importances = pd.DataFrame(columns = ['feature', 'importance'])
        for i in range(len(md)):
            feature_importance=pd.DataFrame(np.hstack((np.array([feat_labels]).T, md[i].coef_.T)), columns=['feature', 'importance'])
            feature_importances = feature_importances.append(feature_importance)
        
        
        feature_importances['importance']=pd.to_numeric(feature_importances['importance'])
        feature_importances = feature_importances.groupby('feature').sum()
        exp=feature_importances.sort_values(by='importance', ascending=False).to_csv(target+model+'_feat_'+str_2+'.csv')  

for M in ['dt','rf']:#,'et','gb']:
    model=M
    if model in models:
        md=unravel(Model_List[model])
        
        a = []
        for i in range(len(md)):
            for feature in zip(feat_labels, md[i].feature_importances_):
                a.append(feature)
        my_set = {x[0] for x in a}
        my_sums = [(i,sum(x[1] for x in a if x[0] == i)/20) for i in my_set]
        my_sums.sort(key = lambda x: x[1],reverse = True)
        feature_importances=pd.DataFrame(my_sums,columns=["feature","importance"])
        exp=feature_importances.to_csv(target+model+'_feat_'+str_2+'.csv',index=False)
'''
'''
def replace_in(x,Y=["HGB","HCT"]):
    x=list(x)
    for xi in rl(x):
        for y in Y:
            try:
                if y in x[xi]:x[xi]=x[xi].replace(y,"/".join(Y));break
            except: pass
    return pd.Series(x)

source='feat_importance/'
contents=os.listdir(source[:-1])
partial="_feat_"+'0_925_ss_588_.csv'
files={};F=0
s_group={};S=0
a_group={};A=0

for M in ['dt','rf','lg']:
    file=M+partial
    if file in contents:
        load=pd.read_csv(source+file).apply(lambda x: replace_in(x))
        load['importance']=np.abs(load['importance'])
        files[M]=load.sort_values("importance",ascending=False)
        load['feature']=load['feature'].apply(lambda x: ".".join(x.split(".")[1:]))
        s_group[M]=load.groupby('feature').sum().sort_values("importance",ascending=False)
        a_group[M]=load.groupby('feature').mean().sort_values("importance",ascending=False)

for M in s_group.keys():
    try: S=pd.concat([S,pd.DataFrame(list(s_group[M].index),columns=[M+" rank"])],axis=1);A=pd.concat([A,pd.DataFrame(list(a_group[M].index),columns=[M+" rank"])],axis=1)
    except: S=pd.DataFrame(list(s_group[M].index),columns=[M+" rank"]);A=pd.DataFrame(list(a_group[M].index),columns=[M+" rank"])
        
#exp=S.to_csv(source+'s_group'+partial)
exp=A.to_csv(source+'a_group'+partial)'''