# -*- coding: utf-8 -*-
# My libs
import libs.data_manager.data_manager as dm
import libs.data_manager.model as mod
import libs.data_manager.profile_miner as pm
# downloaded libs
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# plot libs
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from  sklearn .manifold import TSNE 
import seaborn as sns
# data libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
# other
from bson import json_util
import pickle as pk
from time import time
from copy import deepcopy
from functools import reduce

def saveObject(obj,filename):
    outfile = open(filename,'wb')
    pk.dump(obj,outfile)
    outfile.close()

def loadObject(filename):
    infile = open(filename,'rb')
    obj = pk.load(infile,encoding='latin1')
    infile.close()
    return obj

"""
Notes about my libs:
    Data Manager is a top layer embedding the dataset to add operations on it.
    Profile miner use categorical data to exract a tree of profiles


Few observations : 
    data types : why keep measures as int and not float? I used float to detect
        data nature (categorical vs continuous) so I converted some column into
        float types.
        Id don't interested me in the first place.
        However it's not difficult at all to change this.
        Categorical data in the files are all string (object in pandas).
        So I need to collect the int values to cast all of the with "astype"
"""

savePicklePath = "save/pickle_objects/"

#=========================[Battery 0 ]=========================================

battery_path0 = "data/batteries_00.csv"
with open(battery_path0,"r") as csv_file :
    d0 = dm.DataManager(pd.DataFrame(pd.read_csv(csv_file)))
print(d0.dataset.columns)

""" Some info """
desc0 = d0.dataset.describe()
h0 = d0.dataset.head()
t0 = d0.dataset.tail()
ty0 = d0.dataset.dtypes
v0 = {}
cols0 = d0.dataset.columns.values
for c in d0.dataset.columns:
    v0[c] = d0.dataset[c].value_counts()
sh0 = d0.dataset.shape
nan_counts0 = d0.dataset.isna().sum() # pas de NaN, ca c'est que du bonheur <3
# see plot in the vizu folder as pairplot.png
#sns.pairplot(d0.dataset)

#=========================[Battery 1 ]=========================================

battery_path1 = "data/batteries_01.csv"
with open(battery_path1,"r") as csv_file :
    d1 = dm.DataManager(pd.DataFrame(pd.read_csv(csv_file)))
print(d1.dataset.columns)

""" Some info """
d1.dataset.describe()
h1 = d1.dataset.head()
t1 = d1.dataset.tail()
ty1 = d1.dataset.dtypes
v1 = {}
cols1 = d1.dataset.columns.values
for c in d1.dataset.columns:
    v1[c] = d1.dataset[c].value_counts()
sh1 = d1.dataset.shape
nan_counts1 = d1.dataset.isna().sum() # software version missing
# see plot in the vizu folder as pairplot.png
#sns.pairplot(d1.dataset)


#=========================[Battery 2 ]=========================================

battery_path2 = "data/batteries_02.csv"
with open(battery_path2,"r") as csv_file :
    d2 = dm.DataManager(pd.DataFrame(pd.read_csv(csv_file)))
print(d2.dataset.columns)

""" Some info """
d2.dataset.describe()
h2 = d2.dataset.head()
t2 = d2.dataset.tail()
ty2 = d2.dataset.dtypes
v2 = {}
cols2 = d2.dataset.columns.values
for c in d2.dataset.columns:
    v2[c] = d2.dataset[c].value_counts()
sh2 = d2.dataset.shape
nan_counts2 = d2.dataset.isna().sum() # 90% events NaN
# not working
#sns.pairplot(d2.dataset)

#==============================================================================

d2Fault=d2.dataset[ ['batteryfaultmonitoring_hvbatstatucat4derate',
       'batteryfaultmonitoring_hvbatstatuscat6dlybpo',
       'batteryfaultmonitoring_hvbatstatuscat7nowbpo']]

countFault = d2Fault.sum() # 9376, 1112, 258
indexFault_derate = d2Fault[d2Fault.batteryfaultmonitoring_hvbatstatucat4derate==True].index
indexFault_dlybpo = d2Fault[d2Fault.batteryfaultmonitoring_hvbatstatuscat6dlybpo==True].index
indexFault_nowbpo = d2Fault[d2Fault.batteryfaultmonitoring_hvbatstatuscat7nowbpo==True].index

# 278, 258, 258
idx_intersect_derate_dlybpo = indexFault_derate.intersection(indexFault_dlybpo)
count_intersect_derate_dlybpo = idx_intersect_derate_dlybpo.shape[0]
idx_intersect_derate_nowbpo = indexFault_derate.intersection(indexFault_nowbpo)
count_intersect_derate_nowbpo = idx_intersect_derate_nowbpo.shape[0]
idx_intersect_dlybpo_nowbpo = indexFault_dlybpo.intersection(indexFault_nowbpo)
count_intersect_dlybpo_nowbpo = idx_intersect_dlybpo_nowbpo.shape[0]
# 258
count_all_three_codes = idx_intersect_dlybpo_nowbpo.intersection(indexFault_derate).shape[0]
# if nowbpo happend -> derate and dlybpo both happend (but nowbpo is scarcer than the others)
# this could be verified by an apriori on these 3 columns


#=================== [A priori] ===============================================

df = deepcopy(d2Fault)
df['batteryfaultmonitoring_hvbatstatucat4derate'] = df['batteryfaultmonitoring_hvbatstatucat4derate'].apply(lambda d : "derate_"+str(d))
df['batteryfaultmonitoring_hvbatstatuscat6dlybpo'] = df['batteryfaultmonitoring_hvbatstatuscat6dlybpo'].apply(lambda d : "dlybpo_"+str(d))
df['batteryfaultmonitoring_hvbatstatuscat7nowbpo'] = df['batteryfaultmonitoring_hvbatstatuscat7nowbpo'].apply(lambda d : "nowbpo_"+str(d))
df_values = df.values
te = TransactionEncoder()
te_ary = te.fit(df_values).transform(df_values)

supp = count_all_three_codes/d2Fault.shape[0]
frequent_itemsets = apriori(d2Fault, min_support=0.9*supp, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
saveObject(rules,savePicklePath+"Fault_rules")

#=================== [derate case] ============================================

derateOnlyIndex = indexFault_derate.difference(indexFault_dlybpo).difference(indexFault_nowbpo)
dfDerateOnly = d2.dataset.iloc[derateOnlyIndex,:]
descDerate = dfDerateOnly.describe()

healthyIndex = d2Fault.index.difference(indexFault_derate).difference(indexFault_dlybpo).difference(indexFault_nowbpo)
dfHealthy = d2.dataset.iloc[healthyIndex,:]
descHealthy = dfHealthy.describe()

diffDesc = descDerate - descHealthy
# sort by interest. First look the most interesting according to the description
interestingColumns = ['batterycellmeasurements_hvbattcurrentextx1000',
                      'batterycellmeasurements_hvbattcellvoltagemaxx1000',
                      'thermalmanagement_hvbattinletcoolanttemp',
                'batteryhealthmonitoring_hvbatstateofhealthmaxx10']

healthyCor = dfHealthy.corr()
derateCor = dfDerateOnly.corr()
diffCor = healthyCor - derateCor


# plot distributions#
for col in interestingColumns:
    fig, ax = plt.subplots()
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("frequency")
    health_heights, health_bins = np.histogram(dfHealthy[col])
    derate_heights, derate_bins = np.histogram(dfDerateOnly[col], bins=health_bins)
    width = (health_bins[1] - [0])/10
    ax.bar(health_bins[:-1], health_heights/dfHealthy.shape[0], width=width, facecolor='cornflowerblue', label='health')
    ax.bar(derate_bins[:-1]+width, derate_heights/dfDerateOnly.shape[0], width=width, facecolor='seagreen',label='derate')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
# see the vizu folder

subCor1 = d2.dataset[['thermalmanagement_hvbattinletcoolanttemp',
                'batteryhealthmonitoring_hvbatstateofhealthmaxx10',
                'batteryfaultmonitoring_hvbatstatucat4derate']]
subCor1['batteryfaultmonitoring_hvbatstatucat4derate'] = subCor1['batteryfaultmonitoring_hvbatstatucat4derate'].astype('int64')
subCor1.dtypes
sns.pairplot(subCor1,height=4)

subCor2 = d2.dataset[['batterycellmeasurements_hvbattcurrentextx1000',
                      'batterycellmeasurements_hvbattcellvoltagemaxx1000',
                'batteryfaultmonitoring_hvbatstatucat4derate']]
subCor2['batteryfaultmonitoring_hvbatstatucat4derate'] = subCor2['batteryfaultmonitoring_hvbatstatucat4derate'].astype('int64')
subCor2.dtypes
sns.pairplot(subCor2)

for col in interestingColumns:
    cols = [col]+ ['batteryfaultmonitoring_hvbatstatucat4derate']
    fig = plt.figure()
    ax = sns.boxplot(y=col,x="batteryfaultmonitoring_hvbatstatucat4derate",   
                     data=d2.dataset[cols].sample(frac=0.01), linewidth=2.5)
    
#===============================[Profile mining]===============================
colsId = []
for col in list(d2.dataset.columns.values):
    if(col[-2:]=="id"):
        colsId.append(col)
statuscat = ['batteryfaultmonitoring_hvbatstatucat4derate',
       'batteryfaultmonitoring_hvbatstatuscat6dlybpo',
       'batteryfaultmonitoring_hvbatstatuscat7nowbpo']
modesAndStatus = list(d2.typesSummary()[0]["object"])
d2.select(colsId+modesAndStatus+statuscat) # part of my lib : refresh informations about columns
d2.dataset.columns

# since it is still to huge in terms of combination of values, we need to del some id
keepIdCol = ["cellbalancing_hvbattblncngtrgcellid"]
d2.delColumns([x for x in colsId if not x in keepIdCol])
d2.dataset.columns

 
upperBound = reduce((lambda x, y: x * y), [v2[col].shape[0] for col in d2.dataset.columns.values])
# maximum combinations : 94176 if I keep only "cellbalancing_hvbattblncngtrgcellid" among the id columns.
# however the tree in this case give only 1744 profiles which means that most of the combinations doesn't
# appears. Then the calculus will be very fast.

profileTree = pm.ProfileTree(d2.dataset) # needs the dataset, not the data manager
# not really usefull but illustrates well how it works. Numbers are the conditionnal probabilities
profileTree.disp() 

profiles = profileTree.getProfiles()
profiles.sort() # sort is needed to increase the following line
profile_col = d2.dataset.apply(lambda r : pm.find(r,profiles,isSorted=True), axis=1)
countProfiles = profile_col.value_counts()
sizes_count = countProfiles.value_counts()
# too much class of 1 element
# As you can see, some of the profiles contains only few samples, sometimes only one.
# We could only delete the profiles and the corresponding line in the DB, but
# although it is questionnable we will try to merge some profiles with a Kmeans.
# We need to find a metric. For a profile we assign the frequency column of the 3 status


# Let's load the original data again
battery_path2 = "data/batteries_02.csv"
with open(battery_path2,"r") as csv_file :
    d2 = dm.DataManager(pd.DataFrame(pd.read_csv(csv_file)))
print(d2.dataset.columns)

d2.select(colsId+modesAndStatus+statuscat)
print(d2.dataset.columns)
# we need now to merge nodes at each level before creating another level
""" 
colsValues > profile = [id:[id1,id4],mode:[mode1,mode2]]...
isReduced : do the dataset contain only rows matching with the profile?
return [freq(4derate),freq(6dlybpo),freq(7nowbpo)]
"""
def mergeMetric(colsValues,dataset,isReduced=False):
    if(not isReduced):
        statuscat = ['batteryfaultmonitoring_hvbatstatucat4derate',
           'batteryfaultmonitoring_hvbatstatuscat6dlybpo',
           'batteryfaultmonitoring_hvbatstatuscat7nowbpo']
        dselect = dataset[list(colsValues.keys())+statuscat]
        # for each key check if the value is in the list, then multiply the booleans
        match = lambda r : bool(reduce((lambda x, y: x * y), [r[key] in value for key,value in colsValues.items()]))
        dselect = dselect[ dselect.apply(match, axis=1) ]
    else :
        dselect = dataset
    length = dselect.shape[0]
    return np.array([dselect['batteryfaultmonitoring_hvbatstatucat4derate'].sum()/length,
                     dselect['batteryfaultmonitoring_hvbatstatuscat6dlybpo'].sum()/length,
                     dselect['batteryfaultmonitoring_hvbatstatuscat7nowbpo'].sum()/length])
    
    
def metric(colsValues,dataset,isReduced=False):
    if(not isReduced):
        statuscat = ['batteryfaultmonitoring_hvbatstatucat4derate',
           'batteryfaultmonitoring_hvbatstatuscat6dlybpo',
           'batteryfaultmonitoring_hvbatstatuscat7nowbpo']
        dselect = dataset[list(colsValues.keys())+statuscat]
        # for each key check if the value is in the list, then multiply the booleans
        match = lambda r : bool(reduce((lambda x, y: x * y), [r[key]==value for key,value in colsValues.items()]))
        dselect = dselect[ dselect.apply(match, axis=1) ]
    else :
        dselect = dataset
    length = dselect.shape[0]
    return np.array([dselect['batteryfaultmonitoring_hvbatstatucat4derate'].sum()/length,
                     dselect['batteryfaultmonitoring_hvbatstatuscat6dlybpo'].sum()/length,
                     dselect['batteryfaultmonitoring_hvbatstatuscat7nowbpo'].sum()/length])
""" test : this is what we wan t to achieve """
mergeProfileValues = {'cellbalancing_hvbattbalancingstatus': ['noBalancing','initialValue'],
 'thermalmanagement_hvbattthrmlmngrmode': ['idle'],
 'other_powermode': ['accessory1','running'],
 'cellbalancing_hvbattblncngtrgcellid': [0,11]}
# array([0., 0.0106383, 0.]) very long time execution : 
# dataset will be reduced in the tree building at each step
mergeMetric(mergeProfileValues,d2.dataset,isReduced=False)
# obviously it is wrong to set isReduced here, but this give an idea about the speed
mergeMetric(mergeProfileValues,d2.dataset,isReduced=True)

import libs.data_manager.profile_metric as pm2
""" let's consider an object measuring the reduced dataframe """


class frequencyMetric(pm2.Metric):
    def __init__(self,df,cols):
        super(frequencyMetric,self).__init__(df,cols)
    def measure(self,df):
        length = df.shape[0]
        return np.array([df[c].sum()/length for c in self.cols])
    
import profile_v2 as pm2
d00 = deepcopy(d2.dataset)
metCols = ['batteryfaultmonitoring_hvbatstatucat4derate',
                     'batteryfaultmonitoring_hvbatstatuscat6dlybpo',
                     'batteryfaultmonitoring_hvbatstatuscat7nowbpo']
profileTree2 = pm2.ProfileTree(d00,metricClass=frequencyMetric,
                metricCols=metCols) 
profileTree2.disp() 
profiles2 = profileTree2.getProfiles()
profiles2.sort()
profile_col2 = pm2.apply_profiles(d2.dataset,profileTree2,metCols)
countProfiles2 = profile_col2.value_counts()
sizes_count2 = countProfiles2.value_counts()

for ev in [p.events for p in profiles2]:
    for k,v in ev.items():
        assert(type(v)==list)

p50 = d00.iloc[50,:]
a = [x for x in profileTree2.tree.children[0].children]
print("match with root",pm2._match(profileTree2.tree,p50))
for el in a:
    print(pm2._match(el,p50))