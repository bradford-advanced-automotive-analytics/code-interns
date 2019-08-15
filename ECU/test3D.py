# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:25:25 2019

@author: sdybella
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
import string


df = pd.read_csv('01_drivingCycle00.dat.csv')
df = df.fillna(method='ffill')
df = df.drop(columns=['CALC_Lambda'])

def StandardScaler(df):
    newDf = df.copy()
    newDf = (newDf-newDf.mean())/newDf.std()
    return newDf


def countHoop(listHoops, height):
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
   
    df = pd.DataFrame(columns=pd.MultiIndex.from_product([listHoops[0].columns,letters[:2**height+1]]))

    for i in range(len(listHoops)):
        hoop = listHoops[i]
        for column in hoop.columns:
            dicLetter = dict((key, 0) for key in letters[:2**height+1])
            for letter in hoop[column]:
                dicLetter[letter] += 1
               
            for key in dicLetter.keys():
                df.loc[i, (column, key)] = dicLetter[key]
                             
    #df.colums = pd.MultiIndex.from_product([['my new column'], ['A', 'B', 'C']])
    #print(df)
   
    return df


# linear regression based discretisation
def RegDiscretisation(df,segmentSize):
    newDf = pd.DataFrame(columns=df.columns)
    nbSegments = math.ceil(len(df)/segmentSize)
    segments = np.array_split(df,nbSegments)
    for i in range(len(segments)):
        regression_model = LinearRegression()
        for c in segments[i].columns :
            regression_model.fit(segments[i]['time'].values.reshape(-1,1),segments[i][c].values)
            newDf.loc[i,c] = float(regression_model.coef_)
    return newDf


def convertLR2(x,thresholdList):
    letters =[i for i in string.ascii_uppercase]
    n = len(thresholdList) - 1
    letters = letters[0:n]
    for i in range(n):
        if x>thresholdList[i] and x<thresholdList[i+1]:
            return letters[n-i-1]


def LRtransform2(df,thresholdList):
    return df.applymap(lambda x : convertLR2(x,thresholdList))

thresholds = [-np.inf,-250,-50,30,250,np.inf]
dfNormalLR = LRtransform2(RegDiscretisation(StandardScaler(df[['time','Epm_nEng','APP_r','ActMod_trqClth']]),10),thresholds)

def splitEngineTorqueIncrease2(chaine) :
    if ('C' in chaine) or ('D' in chaine) or ('E' in chaine) :
        if 'A' == chaine[-1] or 'B' == chaine[-1]:
            return True
        else :
            return False
    else :
        return False

def splitHoops(df,referenceKey,splitFunction):
    listdf = []
    curseur = 0
    listValeurAct = []
    for value in df[referenceKey]:
        listValeurAct.append(value)
        if splitFunction(listValeurAct):
            # split
            listdf.append(df[curseur:curseur + len(listValeurAct) - 1])
            curseur = curseur + len(listValeurAct) - 1
            listValeurAct = [value]
    listdf.append(df[curseur:df.shape[0]])
    return listdf

def getValuesHoops(listHoops,df,segmentSize):
    hoopsValues = []
    for hoop in listHoops:
        index = [i for i in hoop.index]
        start = index[0]*segmentSize
        end = index[-1]*segmentSize
        hoopsValues.append(df.loc[start:end])
    return hoopsValues

listHoops = splitHoops(dfNormalLR,'ActMod_trqClth',splitEngineTorqueIncrease2)
listValuesHoops = getValuesHoops(listHoops,df,10)

def getFeatures(listHoops,listValuesHoops):
    featuresMatrix = []
    nb = countHoop(listHoops, 2)
    for i in range(len(listHoops)):
        # nb of appearance of symbols for torque LR
        features = nb.loc[i,'ActMod_trqClth'].tolist()
        nbLetters = sum(features)
        # nb of appearance of symbols for engine speed
        features = features + nb.loc[i,'Epm_nEng'].tolist()
        # nb of appearance of symbols for pedal position
        features = features + nb.loc[i,'APP_r'].tolist()
        # duration of hoop
        features = (np.array(features)/nbLetters).tolist()
        features = features + [listValuesHoops[i].loc[listValuesHoops[i].index[-1],'time'] - listValuesHoops[i].loc[listValuesHoops[i].index[0],'time']]
        featuresMatrix.append(features)
    return pd.DataFrame.from_records(featuresMatrix)

from sklearn.decomposition import PCA as sklPCA
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

test = getFeatures(listHoops,listValuesHoops)

# We run PCA
sklPCA = sklPCA(n_components=8)
test2 = MinMaxScaler().fit_transform(test)
output = sklPCA.fit_transform(test2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(output[:,0], output[:,1], output[:,2], c='b', marker='o')
plt.show()