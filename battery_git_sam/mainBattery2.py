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
import pywt as wv
# other
from bson import json_util
import pickle as pk
from time import time
from copy import deepcopy
from functools import reduce
from random import shuffle
from random import randint
from copy import deepcopy
from math import floor

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


""" 1 value only """
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

""" better than reload the data when error """
dd = deepcopy(d0)

# chose some columns to study
cols_select = ['encryptedvin', 'analyticvin','_cycle','_num_cycle', # groupby cols
               # some measurments
               'IMBALANCE_DELTA_actual',
               'batterycellmeasurements_hvbattcurrentext',
               'batteryhealthmonitoring_hvbattstateofhealthpwr',
               'batteryhealthmonitoring_hvbatstateofhealthmax',
               'batterycellmeasurements_hvbattcelltemphottest',
               'bmselectricalmodulemonitoring_hvbattmemtemperature',
               'other_odometermastervalue', # different scale
               'other_ambienttemp',
               'other_vehiclespeed',
               # car mode
               'other_powermode', 'other_odometermastervalue',
               'other_recordedtime_seconds','cellbalancing_hvbattbalancingstatus',
               # errors
               'batteryfaultmonitoring_hvbatstatucat4derate',
               'batteryfaultmonitoring_hvbatstatuscat6dlybpo',
               'batteryfaultmonitoring_hvbatstatuscat7nowbpo']
d0.select(cols_select)
vin_groups = d0.dataset.groupby(by=['analyticvin','_cycle','_num_cycle'])
# num necessary to have scale for wavelets

n_groups = len(list(vin_groups.groups.keys()))
idxGroup = randint(0,n_groups)
key = list(vin_groups.groups.keys())[idxGroup]
groups = d0.dataset.iloc[vin_groups.groups[key]].reset_index()

sub_groups=groups.groupby(by=['encryptedvin'])


def group_analysis_plot(group):
    X = group['other_recordedtime_seconds']
    n_points = floor(len(X)*1)
   
    X = X.iloc[:n_points+1]
    X -= X[0]
    plotsAttribute = ['IMBALANCE_DELTA_actual',
                   'bmselectricalmodulemonitoring_hvbattmemtemperature',
                   'other_vehiclespeed',
                   'other_ambienttemp',
                   'batterycellmeasurements_hvbattcurrentext']
    """ delta*1000 """
    fig,axes = plt.subplots(nrows=2,ncols=2)
    axes[0,0].set_title(str(key)+" n_points="+str(n_points))
    axes[0,0].plot(X,1000*np.array(group.loc[:n_points,plotsAttribute[0]]),color="r")
    axes[0,0].plot(X,group.loc[:n_points,plotsAttribute[1]],color="y")
    axes[0,0].plot(X,group.loc[:n_points,plotsAttribute[2]],color="g")
    axes[0,0].plot(X,group.loc[:n_points,plotsAttribute[3]],color="b")
    axes[0,0].plot(X,group.loc[:n_points,plotsAttribute[4]],color="c")
    axes[0,0].plot(X,[0]*len(X),'--k')
    axes[0,0].plot(X,[30]*len(X),'--k')
    axes[0,0].plot(X,[50]*len(X),'--k')
    axes[0,0].legend(plotsAttribute)
    axes[0,0].set_xlabel("record time (s)")
    axes[0,0].set_ylabel("measure")
    
    """
    we want to compare wavelets transform signals from every group with a unique scale for each signals
    """
    
    # first try on a group
    def waveCalcul(val,times,win_size,offset,wave):
        times -= times[0]
        times2=times[offset:min(offset+win_size,len(times))]
        val2=val[offset:min(offset+win_size,len(val))].values
        return wv.cwt(val2,np.arange(1,len(times2)),wave)
    
    wave = wv.wavelist("gaus")[0]
    coeffs0,freq0= waveCalcul(group.loc[:,plotsAttribute[0]],X,len(X),0,wave)
    coeffs1,freq1= waveCalcul(group.loc[:,plotsAttribute[1]],X,len(X),0,wave)
    coeffs2,freq2= waveCalcul(group.loc[:,plotsAttribute[2]],X,len(X),0,wave)
    coeffs3,freq3= waveCalcul(group.loc[:,plotsAttribute[3]],X,len(X),0,wave)
    coeffs4,freq4= waveCalcul(group.loc[:,plotsAttribute[4]],X,len(X),0,wave)
    
    
    axes[0,1].matshow(coeffs0)
    axes[0,1].set_title(plotsAttribute[0])
    """
    plt.matshow(coeffs1)
    plt.title(plotsAttribute[1])
    plt.show()
    """
    axes[1,0].matshow(coeffs2)
    axes[1,0].set_title(plotsAttribute[2])
    """
    plt.matshow(coeffs3)
    plt.title(plotsAttribute[3])
    plt.show()
    """
    axes[1,1].matshow(coeffs4)
    axes[1,1].set_title(plotsAttribute[4])
    
    plt.show()


