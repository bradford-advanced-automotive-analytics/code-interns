# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:20:51 2019

@author: sdybella
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from bio import Cluster

#Apply a MinMaxScaler on a dataframe
#input : dataframe, df
#output : scaled dataframe
def MinMaxScaler(df):
    newDf = df.copy()
    newDf -= newDf.min()
    newDf /= newDf.max()
    return newDf



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

# get longest hoop for number of points of interpolation
#input : listHoopsValues : list of values
#output : length of the biggest hoops 
def maxLength(listHoopsValues):
    maxLength = 0
    for hoop in listHoopsValues:
        if len(hoop) > maxLength:
            maxLength = len(hoop)
    return maxLength

# interpolate every hoops
#input : listHoopsValues : list of values
#       list of string, focused column
#output : interpolated hoops
def convertHoops(listHoopsValues,focusedColumn):
    newList = []
    newLength = maxLength(listHoopsValues)
    time = np.linspace(0, 1, newLength)
    for hoop in listHoopsValues:
        if hoop[focusedColumn[0]].size > 1 :
            points = MinMaxScaler(hoop[focusedColumn]).values
            distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            interpolator =  interp1d(distance, points, kind='slinear', axis=0)
            interpolated_points = interpolator(time)
            newDf = pd.DataFrame()
            newDf['time'] = time
            newDf[focusedColumn[0]] = interpolated_points[:,0]
            newDf[focusedColumn[1]] = interpolated_points[:,1]
            newList.append(newDf)
    return newList

# compute distance between two hoops:
#input : two interpolated hoops, hoop1, hoop2
#       two string, column1, column2
#output : diatance between
def computeDistance(hoop1,hoop2,column1,column2):
    return sum(abs(hoop1[column1]-hoop2[column1])**2+abs(hoop1[column2]-hoop2[column2])**2)

# Distance matrix between hoops using interpolation and sum of distance between points
def distanceMatrix3(listHoopsValuesInterpolated,focusedColumn):
    n = len(listHoopsValuesInterpolated)
    matrix = pd.DataFrame(0.0, index=np.arange(n), columns=np.arange(n))
    for i in range(n):
        matrix.iloc[i][i] = 0
        hoop1 = listHoopsValuesInterpolated[i][focusedColumn]
        for j in range(i+1,n):
            hoop2 = listHoopsValuesInterpolated[j][focusedColumn]
            distance = computeDistance(hoop1,hoop2,focusedColumn[0],focusedColumn[1])
            matrix.iloc[i][j] = distance
            matrix.iloc[j][i] = distance
    return matrix





    
