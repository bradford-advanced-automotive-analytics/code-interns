# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:38:35 2019

@author: sdybella
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import treatment as t
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
import string

alphabet = string.ascii_uppercase + string.ascii_lowercase


#data to test
file = '00_drivingCycle00.dat.csv'
df = pd.read_csv(file)
df = df.fillna(method='ffill')
    
    
robustScaler = RobustScaler()
minMaxScaler = MinMaxScaler() 
standardScaler = StandardScaler()


#describe x position with seuil
#input : float : x, seuil
#output : string which describe the result
def convertLR(x,seuil):
    if x<-seuil:
        return 'c'
    elif x>seuil:
        return 'a'
    else:
        return 'b'
    

#Call the function above on each element
#input : dataframe, df
#       float, thresh
#output : dataframe of character
def LRtransform(df,thresh):
    return df.applymap(lambda x : convertLR(x,thresh))

#Subfonction for tree_encoding function which return the correct letter
#input : float : x
#       integer : long, nbAlpha
#output : character
def alpha(x,long,nbAlpha) :
    result = x/long
    if result == 0 :
        ind = (ceil(nbAlpha/2) - 1)
    elif result < 0 :
        ind = max(0,(ceil(nbAlpha/2) - 1) + round(result))
    else :
        ind = min(nbAlpha-1,(ceil(nbAlpha/2) - 1) + round(result))
    ind = nbAlpha - 1 - ind 
    return alphabet[int(ind)]

#Realise a tree_encoding, on the given dataframe with the given height
#input : dataframe, df
#       integer, height
#output : emcoded dataframe
def tree_encoding(df,height):
    ndf = df.drop(columns=['time']).copy()
    ndf[ndf.columns] = robustScaler.fit_transform(ndf[ndf.columns])
    amaxMed = abs(ndf.max().min())
    aminMed = abs(ndf.min().max())
    nbAlpha = 2**height + 1
    longueur = (amaxMed + aminMed)/nbAlpha
    return ndf.applymap(lambda x : alpha(x,longueur,nbAlpha))


#Sunfonction of splitHoops, which ttell if we must split now or not
#input : list of character, chaine
#output : boolean
def splitEngineTorqueIncrease(chaine) :
    if ('C' in chaine) or ('B' in chaine) :
        if 'A' == chaine[-1] :
            return True
        else :
            return False
    else :
        return False

#function which split a dataframe with a split function and a column knowed as a reference
#input : dataframe, df
#       string, referenceKey
#       function, splitFunction
#output : list of subDataframe
def splitHoops(df,referenceKey,splitFunction):
    listdf = []
    curseur = 0
    listValeurAct = []
    for value in df[referenceKey] :
        listValeurAct.append(value)
        if splitFunction(listValeurAct) :
            # split
            listdf.append(df[curseur:curseur + len(listValeurAct) - 1])
            curseur = curseur + len(listValeurAct) - 1
            listValeurAct = [value]
    listdf.append(df[curseur:df.shape[0]])
    return listdf


#Give the list of hoops with the corresponding values
#input : list of dataframe of character, listhoops
#        dataframe, df
#       integer, segmentSize
#output : list of dataframe with values
def getValuesHoops(listHoops,df,segmentSize):
    hoopsValues = []
    for hoop in listHoops:
        index = list(hoop.index)
        start = index[0]*segmentSize
        end = index[-1]*segmentSize
        hoopsValues.append(df.loc[start:end])
    return hoopsValues
