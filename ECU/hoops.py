# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:38:35 2019

@author: sdybella
"""
import pandas as pd
from math import ceil
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import string
from collections import Counter
import treatment as t

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
        return 'C'
    elif x>seuil:
        return 'A'
    else:
        return 'B'

#The same with a list
def convertLR2(x,thresholdList):
    letters = list(string.ascii_uppercase)
    n = len(thresholdList) - 1
    letters = letters[0:n]
    for i in range(n):
        mini = thresholdList[i]
        maxi = thresholdList[i+1]
        if x>mini and  x<maxi :
            return letters[n-i-1]
    

#Call the function above on each element
#input : dataframe, df
#       float, thresh
#output : dataframe of character
def LRtransform(df,thresh):
    return df.applymap(lambda x : convertLR(x,thresh))

#The same with a list
def LRtransform2(df,thresholdList):
    return df.apply(lambda x : convertLR2(x,thresholdList))

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

#Realise a tree_encoding, on the given dataframe which use pourcentages to split the dataframe
#input : dataframe, df
#       list, pourcentage
#output : emcoded dataframe
def tree_encoding2(df,percentages,eps=None):
    eps= eps or 0.05
    ndf = df.drop(columns=['time']).copy()
    ndf[ndf.columns] = standardScaler.fit_transform(ndf[ndf.columns])
    for c in ndf.columns :
        try :
            seuils = [-np.Inf]
            seuils2 = []
            for p in percentages :
                rslt = t.findOptimum(ndf,p,c,eps=eps)
                seuils.append(-rslt)
                seuils2.append(rslt)
            seuils2.reverse()
            seuils2.append(np.inf)
            seuils = seuils + seuils2
        except RecursionError :
            seuils = [-np.Inf]
            seuils2 = []
            maxi = abs(ndf[c].max())
            mini = abs(ndf[c].min())
            extremum = (maxi + mini)/2
            for p in percentages :
                rslt = p*extremum
                seuils.append(-rslt)
                seuils2.append(rslt)
            seuils2.reverse()
            seuils2.append(np.inf)
            seuils = seuils + seuils2
        ndf[c] = ndf[c].apply(lambda x : convertLR2(x,seuils))
        print('colonne ' + c + ' finie')
    return ndf



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

# split function with 5 Letters
def splitEngineTorqueIncrease2(chaine) :
    if ('C' in chaine) or ('D' in chaine) or ('E' in chaine) :
        if 'A' == chaine[-1] or 'B' == chaine[-1]:
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

# Transforming a list of letters into a simplified form : letter and occurences
# input : list of letters
# output : l1 list of unique values in order of appearance
#                 l2 list matching l1 with the number of appaerance of the corresponding letter
def transformLetters(letters):
    l1 = []
    l2 = []
    currentLetter = letters[0]
    occurences = 1
    l1.append(currentLetter)
    for l in letters[1:]:
        if l == currentLetter:
            occurences += 1
        else:
            currentLetter = l
            l1.append(currentLetter)
            l2.append(occurences)
            occurences = 1
    l2.append(occurences)
    return l1, l2


# Get frequency of every possible association
# input : symbolicRepresentation
#             windowSize the number of letters we associate
# output : dictionnary with unique values and number of appearance
def frequencyAssociation(symbolicRepresentation,windowSize):
    symbolicRepresentationTransformed = [''.join(symbolicRepresentation[i:i+windowSize]) for i in range(len(symbolicRepresentation)-windowSize+1)]
    return Counter(symbolicRepresentationTransformed)

#Return a dataframe of probability
#Input : dataframe, dfs
#       Integer, nbletterKonowed
#output a dataframe
def probahoops(dfs,nbletterKnowed,nbletterGuessed):
    index = np.unique(dfs.values)
    indexbis = index.copy()
    #We create the new index of our table of proba
    for i in range(0,nbletterKnowed + nbletterGuessed -1) :
        newIndex = []
        for i in range(len(index)):
            for j in range(len(indexbis)):
                newIndex.append(index[i]+indexbis[j])
        index = newIndex.copy()
    dfp = pd.DataFrame(index=index,columns=dfs.columns)
    #Now we compute probabilities for each columns
    for c in dfs.columns :
        valeurs = dfs[c].values
        countA = frequencyAssociation(valeurs,nbletterKnowed)
        count = frequencyAssociation(valeurs,nbletterKnowed + nbletterGuessed)
        for asso in count :
            dfp[c][asso] = count[asso]/countA[asso[0:nbletterKnowed]]
    index2 = {}
    for ind in index :
        index2[ind]=ind[0:nbletterKnowed]+' -> '+ ind[nbletterKnowed:nbletterKnowed+nbletterGuessed]
    dfp = dfp.rename(index=index2)
    dfp = dfp.replace(np.NaN,0)
    return dfp

            
            
        








































