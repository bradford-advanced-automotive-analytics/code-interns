# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:20:51 2019

@author: sdybella
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import random




#Allowed to select only interesting sensors
#input : list with values, list_values
#        list of key, columns
#output: a list of hoops values which contains only sensors which have their key in columns
def select_list_channels(list_values,columns) :
    sortie = []
    for i in range(0,len(list_values)) :
        sortie.append(list_values[i][columns])
    return sortie
        

#Return a dataframe wich contains all stat vectors of hoops
#input : list of dataframe, dflistValues
#output : a list of vector
def toStatVectors(dflistValues) :
    sortie = []
    for i in range(0,len(dflistValues)) :
        df = dflistValues[i]
        row = [df.max().values] + [df.mean().values] + [df.std().values] + [df.min().values] + [df.median().values]
        row = np.ravel(row)
        sortie.append(row)
    return sortie


#Draw some hoops of each cluster
#input : list of hoop with onnly 2 sensors, df_list_vS
#       a vector which containe the clusterization, repartition
#       Maximal number of graph by cluster, nbGraphMax
#output : drawed hoops
def drawHoopsClust(df_list_vS,repartition,nbGraphMax) :
    nbCluster = np.unique(repartition).size
    list_cluster = []
    for i in range(0,nbCluster) :
        list_cluster.append([])
    for i in range(0,len(repartition)) :
        if len(list_cluster[repartition[i]]) < nbGraphMax :
            list_cluster[repartition[i]].append(i)
        elif random.randint(0,1) == 0 :
            loser = random.randint(0,nbGraphMax-1)
            list_cluster[repartition[i]][loser] = i         
    for i in range(0,nbCluster) :
        data = list_cluster[i]
        fig, axs = plt.subplots(ceil(len(data)/2), 2)
        fig.suptitle('Cluster '+str(i))
        for j in range(0,ceil(len(data)/2)):
            val = []
            name = []
            for key in df_list_vS[data[j]].columns :
                val.append(df_list_vS[data[j]][key].values)
                name.append(key)
            axs[j,0].plot(val[0],val[1])
            axs[j,0].plot(val[0][0],val[1][0],'o')
            axs[j,0].set_title('Hoops '+ str(j), fontsize=10)
            axs[j,0].set_xlabel(name[0])
            axs[j,0].set_ylabel(name[1])
        for j in range(ceil(len(data)/2),len(data)):
            val = []
            name = []
            jbis = j - ceil(len(data)/2)
            for key in df_list_vS[data[j]].columns :
                val.append(df_list_vS[data[j]][key].values)
                name.append(key)
            axs[jbis,1].plot(val[0],val[1])
            axs[jbis,1].plot(val[0][0],val[1][0],'o')
            axs[jbis,1].set_title('Hoops '+ str(j), fontsize=10)
            axs[jbis,1].set_xlabel(name[0])
            axs[jbis,1].set_ylabel(name[1])
        fig.tight_layout()
        plt.show()

    
