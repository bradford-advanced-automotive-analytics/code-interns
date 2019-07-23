# -*- coding: utf-8 -*-
# My libs
import libs.data_manager.data_manager as dm
import libs.data_manager.model as mod
import libs.data_manager.profile_miner as pm
# plot libs
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from  sklearn .manifold import TSNE 
# data libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
# 
from bson import json_util
import pickle as pk
from time import time


def saveObject(obj,filename):
    outfile = open(filename,'wb')
    pk.dump(obj,outfile)
    outfile.close()

def loadObject(filename):
    infile = open(filename,'rb')
    obj = pk.load(infile,encoding='latin1')
    infile.close()
    return obj


data_path = "data/warranty.json"
with open(data_path,"r") as json_file :
    DM = dm.DataManager(pd.DataFrame(json_util.loads(json_file.read())))
print(DM.dataset.columns)


print("elements :",np.shape(DM.dataset))

# keep only the data relative to the architecture of the car
DM.select(['Production Date','VIN','Warranty Start Date'])
DM.dataset.to_csv(r'save/pre_profiled_join.csv',index=False)
DM.select(['Model Year', 'Brand', 'Vehicle', 'Mileage', 'Engine Family',
           'Fuel Type', 'Transmission Family', 'Fault Code', 'Start Date'])


freq = loadObject("save/profiles_process_vect.pk")

from sklearn.cluster import KMeans
import numpy as np

liste = [list(x.values()) for x in freq.values()]
X = np.array(liste)
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)


embedded_data = TSNE(n_components=3).fit_transform(liste)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(embedded_data[:,0],embedded_data[:,1],embedded_data[:,2],c=kmeans.labels_.astype(float))
plt.title("t-sne on high-dimension frequency vectors, K=10")
plt.show()  



