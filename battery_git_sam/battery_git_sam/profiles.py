# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 02:20:12 2019

@author: samue
"""

import data_manager as dm
from bson import json_util
import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import copy
from sklearn import linear_model
from sklearn.cluster import KMeans
import numpy as np
import profile_miner as pm
from anytree.exporter import DotExporter

d = d.reset_index()
path = "save/pre_profiled_join.csv"
with open(path,"r") as csv_file :
    d = pd.DataFrame(pd.read_csv(csv_file))

path = "save/diagDup.csv"
with open(path,"r") as csv_file :
    d = pd.DataFrame(pd.read_csv(csv_file))
d.columns
d = d.drop(columns=[])
d.columns
d = d.drop(columns=["Brand","Engine Family","Fuel Type","Transmission Family"])
d = d.drop(columns=['WCC Desc'])
d.to_csv(r'save/profile_dic',index=False)
profileTree = pm.ProfileTree(d_col)
profiles = profileTree.getProfiles()
profiles.sort()
d_cop.iloc[0:10,:].apply(lambda r : pm.find(r,profiles,isSorted=True), axis=1)
d_cop = copy.deepcopy(d)
d_cop = d_cop.drop(columns=["VIN"])
cat = d_cop.apply(lambda r : pm.find(r,profiles,isSorted=True), axis=1)
profileTree.disp()
profileTree.export("save/profileTree.dot")


d["Profile"] = col

saveObject(profileTree,"save/treeReduced.pk")
saveObject(profiles,"save/profilesReduced.pk")
profileTree = loadObject("save/treeReduced.pk")
profiles = loadObject("save/profilesReduced.pk")

d = d.dropna()


path = "save/profiles_mining_final.csv"
with open(path,"r") as csv_file :
    d = pd.DataFrame(pd.read_csv(csv_file))
DM = dm.DataManager(d)
DM.dataset.to_csv(r'save/profiles_mining_final.csv',index=False)
DM.dataset.dtypes
""" DATE MODEL """
dpModel = dm.dateProcessor(DM,["Session Date","Warranty Start Date"])
""" CENTER BY PROD DATE = AGE : apply on session date,warranty start """
dpModel.center(DM,"Production Date")

DM.dataset.to_csv(r'save/profiles_centered.csv',index=False)

profile_groups = DM.dataset.groupby(by=["Profile"])

dico_dtc_freq = {}
for name,group in profile_groups:
    dico_dtc_freq[name] = group["DTC"].value_counts()
dtc_lists = sorted(list(DM.dataset["DTC"].unique()))

profiles_process_vect = {}
for k,v in dico_dtc_freq.items():
    vect = {}
    s = v.sum()
    for el in dtc_lists:
        vect[el] = 0
    for index,value in v.items():
        vect[index] = value/s
    profiles_process_vect[k] = vect
    
