# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:38:35 2019

@author: sdybella
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from saxpy.znorm import znorm
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize

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


#Subfonction for tree_encoding function which return the correct letter
#input : float : x, minG
#       integer : long, nbAlpha
#       boolean : sens
#output : character
def alpha(x,minG,long,sens,nbAlpha) :
    if sens :
        return alphabet[min([nbAlpha-1,int((x - minG)//long)])]
    else :
        return alphabet[max([0,int((x - minG)//long)])]

#Realise a tree_encoding, on the given dataframe with the given height
#input : dataframe, df
#       integer, height
#output : emcoded dataframe
def tree_encoding(df,height):
    ndf = df.copy()
    minG = ndf.min().min()
    aminG = abs(minG)
    if minG < 0 :
        ndf = ndf.applymap(lambda x : x + abs(minG))
        minG = 0
    maxG = ndf.max().max()
    amaxG = abs(maxG)
    nbAlpha = 2**height + 1
    if aminG < amaxG :
        longueur = (2*aminG)/nbAlpha
        sens = True
    else :
        longueur = (2*amaxG)/nbAlpha
        sens = False
    return ndf.applymap(lambda x :alpha(x,minG,longueur,sens,nbAlpha)) 


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

