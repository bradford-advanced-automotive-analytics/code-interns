import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize


#data to test
file = '00_drivingCycle00.dat.csv'
df = pd.read_csv(file)
df = df.fillna(method='ffill')
    
    
robustScaler = RobustScaler()
minMaxScaler = MinMaxScaler() 
standardScaler = StandardScaler()

#mean based discretisation
#input : dataframe, df
#        integer, segmentSize
#output : discretized dataframe'''
def AvgDiscretisation(df,segmentSize):
    newDf = pd.DataFrame(columns=df.columns)
    nbSegments = math.ceil(len(df)/segmentSize)
    segments = np.array_split(df,nbSegments)
    for i in range(len(segments)):
        newDf.loc[i] = segments[i].mean().values
    return newDf

#linear regression based discretisation
#input : dataframe, df
#        integer, segmentSize
#output : discretized dataframe'''
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




#data should be normalised before use
#input : dataframe, df
#        integer, nbCuts, segmentSize
#output : normalized dataframe'''
def SAXtransform(df,nbCuts,segmentSize):
    ndf = pd.DataFrame()
    df = AvgDiscretisation(df,segmentSize)
    cuts = cuts_for_asize(nbCuts)
    for c in df.columns:
        ndf[c] = list(ts_to_string(df[c].values, cuts))      
    return ndf
    
#Recursive function which delete all sensors with a correlation factor above threshold
#input : dataframe, df
#        float, threshold
#output : dataframe '''
def deleteCorrelations(df, threshold, keepSensors):
    corr = df.corr()
    pairs = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
    pairs = pairs.where(pairs>threshold).dropna()
    columnsToKeep = []
    columnsToDelete = []
    listP = []
    for key in pairs.keys() :
        if key[0] != key[1] :
            listP.append([key[0],key[1]])
    if listP == []:
        return df
    else:
        for pair in listP:
            if (pair[0] not in columnsToDelete) and (pair[0] not in columnsToKeep):
                columnsToKeep.append(pair[0])
            if (pair[1] not in columnsToDelete) and (pair[1] not in columnsToKeep):
                if pair[1] in keepSensors:
                    columnsToKeep.append(pair[1])
                else:
                    columnsToDelete.append(pair[1])
            if columnsToDelete == []:
                return df
        return deleteCorrelations(df.drop(columns = columnsToDelete), threshold, keepSensors)

#Plot curbs of column versus time, one with the values, the other with his linear regression
# input : dataframe, df
#         string, column
#         integer, segmentSize
#output : a plot 
def plotLR(df,column,segmentSize):
    newDf = pd.DataFrame(columns = ['time',column])
    nbSegments = math.ceil(len(df)/segmentSize)
    segments = np.array_split(df,nbSegments)
    for i in range(len(segments)):
        regression_model = LinearRegression()
        regression_model.fit(segments[i]['time'].values.reshape(-1,1),segments[i][column].values)
        plt.plot(segments[i]['time'],regression_model.predict(segments[i]['time'].values.reshape(-1,1)),color='r')
        plt.plot(segments[i]['time'],segments[i][column],color='b')
        pylab.legend(loc='upper left')






