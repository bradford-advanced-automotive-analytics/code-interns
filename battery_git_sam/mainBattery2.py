# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:07:03 2019

@author: ssquilla
"""
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
import matplotlib.colors
# data libs
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
# other
from bson import json_util
import pickle as pk
from time import time
from copy import deepcopy
from functools import reduce
from random import shuffle
from random import randint

def saveObject(obj,filename):
    outfile = open(filename,'wb')
    pk.dump(obj,outfile)
    outfile.close()

def loadObject(filename):
    infile = open(filename,'rb')
    obj = pk.load(infile,encoding='latin1')
    infile.close()
    return obj

savePicklePath = "save/pickle_objects/"

#=========================[CAVA ID ]===========================================

battery_path ="data/CAVA-13_5m_rowID.csv"
with open(battery_path,"r") as csv_file :
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
nan_counts0 = d0.dataset.isna().sum()
# see plot in the vizu folder as pairplot.png
#sns.pairplot(d0.dataset)


d0.delColumns(["bmselectricalmodulemonitoring_hvbattfusetemperature",
 "messageheader_messagebodytype",
				"thermalmanagement_hvbattcoolingenergyusd"])

print(d0.typesSummary())
intToFloat = ['other_odometermastervalue',
       'other_recordedtime_seconds',
       'batterycellmeasurements_hvbattvoltageext',
       'bmselectricalmodulemonitoring_hvbattmemtemperature',
       'thermalmanagement_hvbattinletcoolanttemp',
       'thermalmanagement_hvbattoutletcoolanttemp']
d0.dataset[intToFloat] = d0.dataset[intToFloat].astype("float64")
stringToDate = ['_actualtime',
       'produceddatetime_WirelessCar',
       'messageheader_timestamp']
for date_att in stringToDate:
    d0.dataset[date_att] = pd.to_datetime(d0.dataset[date_att])
d0.setInfos() # my lib needs to refresh information when types change    
print(d0.typesSummary())

# chose some columns to study
cols_select = ['encryptedvin', 'analyticvin','_cycle','_num_cycle', # groupby cols
               # some measurments
               'IMBALANCE_DELTA_actual',
               'batterycellmeasurements_hvbattcurrentext',
               'batteryhealthmonitoring_hvbattstateofhealthpwr',
               'batteryhealthmonitoring_hvbatstateofhealthmax',
               'batterycellmeasurements_hvbattcelltemphottest',
               'bmselectricalmodulemonitoring_hvbattfusetemperature',
               'bmselectricalmodulemonitoring_hvbattmemtemperature',
               # car mode
               'other_powermode', 'other_odometermastervalue',
               'other_ambienttemp', 'other_vehiclespeed',
               'other_recordedtime_seconds','cellbalancing_hvbattbalancingstatus',
               # errors
               'batteryfaultmonitoring_hvbatstatucat4derate',
               'batteryfaultmonitoring_hvbatstatuscat6dlybpo',
               'batteryfaultmonitoring_hvbatstatuscat7nowbpo']
d0.select(cols_select)
vin_groups = d0.dataset.groupby(by=['encryptedvin', 'analyticvin','_cycle',
               '_num_cycle'])

"""
filtered = vin_groups.filter(lambda x: x["IMBALANCE_Percent"].max() > 0.05 and x.shape[0]>5)
filtered = filtered.reset_index()

regroup = filtered.groupby(by=['encryptedvin', 'analyticvin','_cycle',
               '_num_cycle'])
"""

n_groups = len(list(vin_groups.groups.keys()))
idxGroup = randint(0,n_groups)
key = list(vin_groups.groups.keys())[idxGroup]
n_points = 10
group = d0.dataset.iloc[vin_groups.groups[key]][0:n_points]

plt.figure()
plt.title(str(key))
X = group['other_recordedtime_seconds']
plt.plot(X,1000*np.array(group['IMBALANCE_DELTA_actual']),color="r")
plt.plot(X,group['batterycellmeasurements_hvbattcurrentext'],color="y")
plt.plot(X,group['batteryhealthmonitoring_hvbattstateofhealthpwr'],color="g")
plt.plot(X,group['batteryhealthmonitoring_hvbatstateofhealthmax'],color="b")
plt.plot(X,group['batterycellmeasurements_hvbattcelltemphottest'],color="c")
plt.plot(X,group['bmselectricalmodulemonitoring_hvbattfusetemperature'],color="m")
plt.plot(X,group['bmselectricalmodulemonitoring_hvbattmemtemperature'],color="r")
plt.plot(X,[0]*len(X),'--k')
plt.plot(X,[30]*len(X),'--k')
plt.plot(X,[50]*len(X),'--k')
plt.legend(['1000*IMBALANCE_Percent',
               'batterycellmeasurements_hvbattcurrentext',
               'batteryhealthmonitoring_hvbattstateofhealthpwr',
               'batteryhealthmonitoring_hvbatstateofhealthmax',
               'batterycellmeasurements_hvbattcelltemphottest',
               'bmselectricalmodulemonitoring_hvbattfusetemperature',
               'bmselectricalmodulemonitoring_hvbattmemtemperature'])
plt.xlabel("record time (s)")
plt.ylabel("measure")
plt.show()






