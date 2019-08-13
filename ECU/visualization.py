# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:49:23 2019

@author: sdybella
"""

import numpy as np
import pandas as pd
import treatment as t
import hoops as h
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import random
from sklearn.linear_model import LinearRegression


#Choose files that you want to study to convert it (.csv !!!, if not the case, use pre-treatment.py) in a dataframe without NaN or Infinite value
file = '06_drivingCycle00.dat.csv'
df = pd.read_csv(file,index_col = [0])
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='ffill')
df = df.dropna(axis=1)

#Choose columns you absolutely want to keep
savedColumns = ['time','ActMod_trqClth','VehV_v','PFltLd_facTempSimRgn_mp','PFltLd_mSotMeas','PFltLd_facSotSimRgnNO2_mp','PFltLd_mSotOfsNO2_mp','CoEOM_numOpModeActTSync','ASMod_dmSotEngNrm1_mp','PFltLd_mSot','SCRCat_mfNoxScrUsSnsr','PFltLd_pDiffSot_mp','PFltRgn_lSnceRgn','Exh_pAdapPPFltDiff','SCRCat_mfExhDpfDs','SCRCat_mfNoxScrDsSnsr','ETCtl_tDeCatDsMdlZon3_mp','PFltLd_dmSotSim','PFltRgn_stOpMode','Exh_dmNOxNoCat2Ds','ETCtl_tAdpnCatDs','ETCtl_tCatDsExh_mp','Exh_tAdapTOxiCatUs', 'ASMod_tPFltSurfSim','Exh_tPFltDs','Exh_tPFltUs','ETCtl_tCatUsExh_mp','PFltLd_mSotSim','SCRAd_rNH3','Exh_pPFltDiff','SCRCat_mfExhScrUs','ASMod_dmSotEngRgn1_mp','ASMod_dmSotEngNrm3_mp','SCRMod_dmEstNOxDs','DStgy_dmRdcAgDes','Exh_tOxiCatUs','SCRMod_rNO2NOxDs','CoEOM_stOpModeReq','ExhMod_ratNOX_mp','SCRT_tAvrg','SCRMod_rNO2NOx','UDC_mRdcAgDosQnt','SCRAd_stSlip','SCRMod_rEstNOxDs','Hegn_mFlowNoxB1S2','SCRFFC_dmNOxUs','ASMod_tIntMnfDs','Exh_tPFltUs','Exh_tOxiCatUsB2']


#Make a picture from the symbolic representation
#Input : dataframe with letter, dfc
#        integer, number of differents symbols
#Output : image
def topicture(dfc,nbletter) :
    ndfc = dfc.copy()
    ndfc = ndfc.applymap(lambda x : nbletter + 1 -(int(x,36)-9))
    picture = ndfc.values.transpose()
    plt.imshow(picture)
    plt.show()

#Plot curbs of column versus time, one with the values, the other with his linear regression
# input : dataframe, df
#         string, column
#         integer, segmentSize
#output : a plot 
def plotLR(df,column,segmentSize):
    nbSegments = ceil(len(df)/segmentSize)
    segments = np.array_split(df,nbSegments)
    for i in range(len(segments)):
        regression_model = LinearRegression()
        regression_model.fit(segments[i]['time'].values.reshape(-1,1),segments[i][column].values)
        plt.plot(segments[i]['time'],regression_model.predict(segments[i]['time'].values.reshape(-1,1)),color='r')
        plt.plot(segments[i]['time'],segments[i][column],color='b')
        pylab.legend(loc='upper left')
        
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
        if ceil(len(data)/2) != 1 :
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
        else :
            for j in range(0,len(data)):
                val = []
                name = []
                for key in df_list_vS[data[j]].columns :
                    val.append(df_list_vS[data[j]][key].values)
                    name.append(key)
                axs[j].plot(val[0],val[1])
                axs[j].plot(val[0][0],val[1][0],'o')
                axs[j].set_title('Hoops '+ str(j), fontsize=10)
                axs[j].set_xlabel(name[0])
                axs[j].set_ylabel(name[1])
                fig.tight_layout()
                plt.show()
