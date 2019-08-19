# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:05:19 2019

@author: ssquilla
"""

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
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor
from math import sqrt
from sklearn.linear_model import LassoCV,LassoLarsIC


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

battery_path ="data/CAVA-13_5m_rowID.csv"
with open(battery_path,"r") as csv_file :
    d0 = dm.DataManager(pd.DataFrame(pd.read_csv(csv_file)))
    
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
       'thermalmanagement_hvbattoutletcoolanttemp',
       'batteryhealthmonitoring_hvbattfastchgcounter']
d0.dataset[intToFloat] = d0.dataset[intToFloat].astype("float64")
stringToDate = ['_actualtime',
       'produceddatetime_WirelessCar',
       'messageheader_timestamp']
for date_att in stringToDate:
    d0.dataset[date_att] = pd.to_datetime(d0.dataset[date_att])
d0.setInfos() # my lib needs to refresh information when types change    
print(d0.typesSummary())

d0.dataset['IMBALANCE_DELTA_actual'] = 1000*d0.dataset['IMBALANCE_DELTA_actual']

""" better than reload the data when error """
dd = deepcopy(d0)

#==============================================================================

def regression(df,d0,perc_trainset,bycols,alpha=0.1):
    df = df.sample(frac=1).reset_index(drop=True)
    n = int(dcopy.shape[0]*perc_trainset)
    train_data,test_data = df.loc[:n],df.loc[n:]
    #print(train_data.shape,test_data.shape)
    reg = linear_model.Ridge(alpha=alpha,fit_intercept=True)
    reg.fit(train_data[inputColumns].values,train_data[outputColumn].values)
    score = reg.score(test_data[inputColumns].values,test_data[outputColumn].values)
    # prediction vizualisation
    df = deepcopy(d0.dataset).loc[n:].reset_index()
    #df = dcopy.sort_values(by='_actualtime').reset_index()
    groups = df.groupby(by=bycols)
    keys = groups.groups.keys()
    return (np.random.permutation([(df.iloc[groups.groups[key]].reset_index(),key) for key in keys]),reg,n,score)

def lassoAicBic(df,cont,d0,perc_trainset,bycols):
    df = df.sample(frac=1).reset_index(drop=True)
    n = int(dcopy.shape[0]*perc_trainset)
    train_data,test_data = df.loc[:n,cont],df.loc[n:,cont]
    #print(train_data.shape,test_data.shape)
    model_bic = LassoLarsIC(criterion='bic',max_iter=1000)
    model_bic.fit(train_data[inputColumns].values, train_data[outputColumn].values)
    model_aic = LassoLarsIC(criterion='aic',max_iter=1000)
    model_aic.fit(train_data[inputColumns].values, train_data[outputColumn].values)
    groups =  df.loc[n:].groupby(by=bycols)
    keys = groups.groups.keys()
    return (np.random.permutation([(df.iloc[groups.groups[key]].reset_index(),key) for key in keys]),n,model_aic,model_bic)

"""
dscore/dperc
dscore/dalpha
"""
def derivativeScore(df,d0,bycols,perc_trainset,alpha):
    h = sqrt(np.finfo(np.float32).eps)
    (groupsCyclesRight,regRight,nRight,scoreRight) = regression(df,d0,perc_trainset-h,bycols,alpha)
    (groupsCyclesLeft,regLeft,nLeft,scoreLeft) = regression(df,d0,perc_trainset+h,bycols,alpha)
    df_dp = (scoreRight - scoreLeft)/(2*h)
    (groupsCyclesRight,regRight,nRight,scoreRight) = regression(df,d0,perc_trainset,bycols,alpha+h)
    (groupsCyclesLeft,regLeft,nLeft,scoreLeft) = regression(df,d0,perc_trainset,bycols,alpha-h)
    df_da = (scoreRight - scoreLeft)/(2*h)
    return np.array([df_dp,df_da])

# prediction vizualisation
def searchRate(df,d0,bycols,perc_0=0.5,alpha_0=0.2,maxiter=100,rate=0.01,eps_erreur=1e-3,max_perc=0.8):
    try:
        perc_trainset = perc_0
        alpha = alpha_0
        d_init = derivativeScore(df,d0,bycols,perc_trainset,alpha)
        print()
        dscore = np.copy(d_init)
        iter_ = 0
        X_i = np.array([perc_0,alpha_0])
        while(iter_ < maxiter and np.linalg.norm(dscore)>=eps_erreur*np.linalg.norm(d_init)
            and X_i[0]<max_perc and X_i[0]>0.1 and X_i[1]>0):
           X_i = X_i + rate*dscore
           if(iter_ < maxiter - 1):
               dscore = derivativeScore(df,d0,bycols,X_i[0],X_i[1])
           iter_ += 1
        return regression(df,d0,X_i[0],bycols,X_i[1])
    except ValueError:
        return regression(df,d0,0.8,bycols,alpha_0)

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_ + EPSILON
    alphas_ = model.alphas_ + EPSILON
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')


d0 = deepcopy(dd)
cycle,balancing,powermode,thermalmode="drive","noBalancing","keyOut","idle"
#d0.dataset = d0.dataset[d0.dataset["_cycle"]==cycle]
#d0.dataset = d0.dataset[d0.dataset['cellbalancing_hvbattbalancingstatus']==balancing]
#d0.dataset = d0.dataset[d0.dataset['other_powermode']==powermode]
#d0.dataset = d0.dataset[d0.dataset['thermalmanagement_hvbattthrmlmngrmode']==thermalmode]

d0.allowCopy() # allows to work on a copy. usefull to be sure that we dont work on dd
#d0.dataset = d0.dataset[d0.dataset["IMBALANCE_DELTA_actual"]<30].reset_index()

analytic_subgroups = pm.cutOnValues(d0.dataset,"analyticvin")
analytic = list(analytic_subgroups.keys())[1]
d0.dataset = analytic_subgroups[analytic]


cont = [         
       'other_vehiclespeed',
       'other_ambienttemp',
       'batterycellmeasurements_hvbattcelltempcoolest',
       'batterycellmeasurements_hvbattcurrentext',
       'batterycellmeasurements_hvbattvoltageext',
       'batterycellmeasurements_hvbattcelltemphottest',
       'bmselectricalmodulemonitoring_hvbattmemtemperature',
       'thermalmanagement_hvbattinletcoolanttemp',
       'thermalmanagement_hvbattoutletcoolanttemp',
       'batterycellmeasurements_hvbattcellvoltagemin',
       'batterycellmeasurements_hvbattcellvoltagemax',
       'batteryhealthmonitoring_hvbattstateofhealthpwr',
       'batteryhealthmonitoring_hvbatstateofhealthmin',
       'batteryhealthmonitoring_hvbatstateofhealth',
       'batteryhealthmonitoring_hvbatstateofhealthmax',
       
       #output
       'IMBALANCE_DELTA_actual',
       ]
"""
[array(['noBalancing', 'passiveBalancing', 'initialValue'], dtype=object),
 array(['running2', 'keyOut', 'ignitionOn2'], dtype=object),
 array(['idle', 'activeHeating', 'passingCooling', 'thermalBalancing',
'initialValue'], dtype=object)]
"""
cat = [   'cellbalancing_hvbattbalancingstatus',
       'other_powermode',
       'thermalmanagement_hvbattthrmlmngrmode'
       ] + ["analyticvin","_cycle"]
groupby =  ['_cycle','_num_cycle','encryptedvin',
        '_actualtime']
ids = [ # id
       'cellbalancing_hvbattblncngtrgcellid',
       'batterycellmeasurements_hvbattcelltempcoldcellid',
       'batterycellmeasurements_hvbattvoltmincellid',
       'batterycellmeasurements_hvbattvoltmaxcellid',
       'batteryhealthmonitoring_hvbatsochcmaxcellid',
       'batteryhealthmonitoring_hvbatsochcmincellid',
       'batterycellmeasurements_hvbatttemphotcellid',
       ]
selectCols = groupby + cont + ids + cat
outputColumn = 'IMBALANCE_DELTA_actual'
inputColumns = deepcopy(cont); inputColumns.remove(outputColumn)
print([d0.dataset[c].unique() for c in [   'cellbalancing_hvbattbalancingstatus',
       'other_powermode',
       'thermalmanagement_hvbattthrmlmngrmode'
       ]])
d0.select(selectCols)
 
#================= SCALING => NaN =============================================
#dcopy[cont].hist(bins=9)
#scaler = preprocessing.MinMaxScaler() # create NaNs...
dcopy = deepcopy(d0.dataset)
#dcopy[cont] = pd.DataFrame(scaler.fit_transform(dcopy[cont]),columns=dcopy[cont].columns)
dcopy = dcopy.reset_index()
corr = dcopy[cont].corr()  

# ================ RIDGE REGRESSION 5e-4 ======================================
plt.figure("ridge regression, alpha = 5e-4 " + analytic)
(groupsCycles,reg,n,score) = regression(dcopy[cont],d0,0.1,["encryptedvin","_cycle","_num_cycle"],alpha=5e-4)
for i in range(min(20,len(groupsCycles))):
    plt.subplot(4, 5, i+1)
    tit = deepcopy(cat) ; tit.remove("analyticvin")
    plt.title(str(groupsCycles[i][0][tit].loc[0].values))
    random_group = groupsCycles[i][0]
    #plt.xlabel("time (s)")
    plt.tick_params(axis='both', which='both', bottom='off',labelbottom='off')
    plt.ylabel("Imbalance (mV)")
    plt.plot(random_group['_actualtime'],random_group[outputColumn],"-b",
             random_group['_actualtime'],reg.predict(random_group[inputColumns]),"+k")
    plt.legend(["reality","prediction"])
# ================ AIC BIC REGRESSION =========================================
(random_groups,n,model_aic,model_bic) = lassoAicBic(dcopy,cont,d0,0.1,["encryptedvin","_cycle","_num_cycle"])
EPSILON = 1e-4
plt.figure("aic bic regression " + analytic)
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection')
for i in range(min(20,len(groupsCycles))):
    plt.subplot(4, 5, i+1)
    tit = deepcopy(cat) ; tit.remove("analyticvin")
    plt.title(str(groupsCycles[i][0][tit].loc[0].values))
    random_group = groupsCycles[i][0]
    #plt.xlabel("time (s)")
    plt.tick_params(axis='both', which='both', bottom='off',labelbottom='off')
    plt.ylabel("Imbalance (mV)")
    plt.plot(random_group['_actualtime'],random_group[outputColumn],"b",
             random_group['_actualtime'],model_aic.predict(random_group[inputColumns]),"+r",
             random_group['_actualtime'],model_bic.predict(random_group[inputColumns]),"+g")
    plt.legend(["reality","aic","bic"])
# =========== RIDGE WITH AIC ALPHA ============================================
plt.figure("ridge regression, aic alpha " + analytic)
(groupsCycles,reg,n,score) = regression(dcopy[cont],d0,0.1,["encryptedvin","_cycle","_num_cycle"],alpha=model_aic.alpha_)
for i in range(min(20,len(groupsCycles))):
    plt.subplot(4, 5, i+1)
    tit = deepcopy(cat) ; tit.remove("analyticvin")
    plt.title(str(groupsCycles[i][0][tit].loc[0].values))
    plt.tick_params(axis='both', which='both', bottom='off',labelbottom='off')
    random_group = groupsCycles[i][0]
    #plt.xlabel("time (s)")
    plt.ylabel("Imbalance (mV)")
    plt.plot(random_group['_actualtime'],random_group[outputColumn],"b",random_group['_actualtime'],reg.predict(random_group[inputColumns]),"+r")
    plt.legend(["reality","prediction"])

#============ RIDGE WITH ALPHA = 0 ============================================
plt.figure("rdige regression, alpha = 0 " + analytic)
(groupsCycles,reg,n,score) = regression(dcopy[cont],d0,0.1,["encryptedvin","_cycle","_num_cycle"],alpha=0)
for i in range(min(20,len(groupsCycles))):
    plt.subplot(4, 5, i+1)
    tit = deepcopy(cat) ; tit.remove("analyticvin")
    plt.title(str(groupsCycles[i][0][tit].loc[0].values))
    random_group = groupsCycles[i][0]
    #plt.xlabel("time (s)")
    plt.tick_params(axis='both', which='both', bottom='off',labelbottom='off')
    plt.ylabel("Imbalance (mV)")
    plt.plot(random_group['_actualtime'],random_group[outputColumn],"b",random_group['_actualtime'],reg.predict(random_group[inputColumns]),"+r")
    plt.legend(["reality","prediction"])
  