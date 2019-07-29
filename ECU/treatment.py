import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize

#Min max scaling
#input : dataframe, df
#       columns, list if columns we want to scale
#output : sclaled dataframe '''
def MinMaxScaler(df,columns):
    for column in columns:
        df[column] -= df[column].min()
        df[column] /= df[column].max()
    return df
        
#Min max scaling
#input : dataframe, df
#output : sclaled dataframe '''
def MinMaxScaler(df):
    df -= df.min()
    df /= df.max()
    return df
        
#Min max scaling
#input : dataframe, df
#        columns, list if columns we want to scale
#output : sclaled dataframe '''
def StandardScaler(df,columns):
    for column in columns:
        df[column] = (df[column]-df[column].mean())/df[column].std()
        return df
        
#Min max scaling
#input : dataframe, df
#output : scaled dataframe '''
def StandardScaler(df):
    df = (df-df.mean())/df.std()
    return df

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
            newDf.loc[i,c] = regression_model.coef_
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
def deleteCorrelations(df, threshold):
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
        for pair1 in listP:
            if (pair1[0] not in columnsToDelete) and (pair1[0] not in columnsToKeep):
                columnsToKeep.append(pair1[0])
            if (pair1[1] not in columnsToDelete) and (pair1[1] not in columnsToKeep):
                columnsToDelete.append(pair1[1])
        return deleteCorrelations(df.drop(columns = columnsToDelete),threshold)






