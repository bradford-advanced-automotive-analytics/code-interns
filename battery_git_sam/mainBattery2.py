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
