#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 01:30:41 2020

@author: dylansmith
"""

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format

DB1=pd.read_csv('./Medication_Features.csv')
DB2=pd.read_csv('./3.Medication.csv',low_memory=False).iloc[2:]

categories=dict(zip(DB1.iloc[:11,1],DB1.iloc[:11,0]))
grouping=pd.concat([DB1['Medications'],DB1['Index.1']],axis=1).dropna(0)
grouping=dict(zip(grouping.iloc[:,0],grouping.iloc[:,1]))

Med=pd.concat([DB2['new random number'],DB2['Medication_Date'],DB2['Medication']],axis=1)
med=np.array(Med)

M=[]
for i in range(len(med)):
    if med[i][2] in grouping:
        M.append(categories[grouping[med[i][2]]])
    else:
        M.append(0)
        
Med['Group_Id']=M

#exp=Med[Med['Group_Id']!=0].to_csv('./3.Medication_grouping.csv',index=False)
#exp=Med.to_csv('./3.Medication_grouping_narrow.csv',index=False)
#exp=pd.concat([Med,DB2.iloc[:,15:18]],axis=1).to_csv('./3.Medication_grouping_full.csv',index=False)
