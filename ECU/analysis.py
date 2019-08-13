# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:25:35 2019

@author: sdybella
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import treatment as t
import hoops as h
import clusterHoops as c
import visualization as v
from bio import Cluster



#Firstly, you need to initialize some object
acp = PCA(svd_solver='full')
robustScaler = RobustScaler()
minMaxScaler = MinMaxScaler()
standardScaler = StandardScaler()


#Choose files that you want to study to convert it (.csv !!!, if not the case, use pre-treatment.py) in a dataframe without NaN or Infinite value
file = '06_drivingCycle00.dat.csv'
df = pd.read_csv(file,index_col = [0])
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='ffill')
df = df.dropna(axis=1)

#Choose columns you absolutely want to keep
savedColumns = ['time','ActMod_trqClth','VehV_v','PFltLd_facTempSimRgn_mp','PFltLd_mSotMeas','PFltLd_facSotSimRgnNO2_mp','PFltLd_mSotOfsNO2_mp','CoEOM_numOpModeActTSync','ASMod_dmSotEngNrm1_mp','PFltLd_mSot','SCRCat_mfNoxScrUsSnsr','PFltLd_pDiffSot_mp','PFltRgn_lSnceRgn','Exh_pAdapPPFltDiff','SCRCat_mfExhDpfDs','SCRCat_mfNoxScrDsSnsr','ETCtl_tDeCatDsMdlZon3_mp','PFltLd_dmSotSim','PFltRgn_stOpMode','Exh_dmNOxNoCat2Ds','ETCtl_tAdpnCatDs','ETCtl_tCatDsExh_mp','Exh_tAdapTOxiCatUs', 'ASMod_tPFltSurfSim','Exh_tPFltDs','Exh_tPFltUs','ETCtl_tCatUsExh_mp','PFltLd_mSotSim','SCRAd_rNH3','Exh_pPFltDiff','SCRCat_mfExhScrUs','ASMod_dmSotEngRgn1_mp','ASMod_dmSotEngNrm3_mp','SCRMod_dmEstNOxDs','DStgy_dmRdcAgDes','Exh_tOxiCatUs','SCRMod_rNO2NOxDs','CoEOM_stOpModeReq','ExhMod_ratNOX_mp','SCRT_tAvrg','SCRMod_rNO2NOx','UDC_mRdcAgDosQnt','SCRAd_stSlip','SCRMod_rEstNOxDs','Hegn_mFlowNoxB1S2','SCRFFC_dmNOxUs','ASMod_tIntMnfDs','Exh_tPFltUs','Exh_tOxiCatUsB2']

#You can delete strongly correlated sensors
df = t.deleteCorrelations(df,0.95,savedColumns)

#Choose a Discretisation function from treatment.py which you will use to sum up the data
dfdiscret = t.RegDiscretisation(df,10)

#Choose an symbolic representation function from hoops.py
dfsymbolic = h.tree_encoding2(dfdiscret,[0.8,0.5,0.3])

v.topicture(dfsymbolic[['VehV_v','ActMod_trqClth']],9)


#Make some probabilities
p1 = h.probahoops(dfsymbolic,3,1)
p2 = h.probahoops(dfsymbolic,3,2)
p3 = h.probahoops(dfsymbolic,3,3)

#Then, split your hoops with splitHoops function. Be sure to chosse your reference key and your method to split it correctly
df_list_symbolic = h.splitHoops(dfsymbolic,'ActMod_trqClth',h.splitEngineTorqueIncrease2)
df_list_values = h.getValuesHoops(df_list_symbolic,df,10)

#Choose the sensors that you want to focus
focusedColumns = ['VehV_v','ActMod_trqClth']
df_list_values_focused = c.select_list_channels(df_list_values,focusedColumns)



#Use your clusterization method (for example, statVector/acp/kmeans)

test = c.toStatVectors(df_list_values_focused)
test = standardScaler.fit_transform(test)
test = pd.DataFrame(test).fillna(method='ffill').values
output = acp.fit_transform(test)
kmeans = KMeans(n_clusters=2).fit(output.transpose()[0:3].transpose())

    

#Draw your clusterization
#In 2D
fig = plt.figure()
plt.plot(acp.explained_variance_ratio_)
plt.title('Percentage of information for each eigen value')

#In 3D
fig2 = plt.figure()
ax = Axes3D(fig2)  
ax.scatter(output[:,0], output[:,1], output[:,2],c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.show()

#Choose a pair of sensors on which you want to draw their hoops
pair = ['VehV_v','ActMod_trqClth']
listS = c.select_list_channels(df_list_values,pair)

#Finally, draw some hoops on each cluster and enjoy 
v.drawHoopsClust(listS,kmeans.labels_,4)