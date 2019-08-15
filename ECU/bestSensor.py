# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:33:16 2019

@author: sdybella
"""

import numpy as np
import pandas as pd
import math
import treatment as t
import random
from sklearn.linear_model import Ridge

#Give all combinaison in a list
#input : list, L
#       int : N,k
#output: list of list of combinaison
def combinaisons(L, N, k):
    h = 0
    i = 0
    j = 0
    n = [0] * (N - 1)
    G = []
    s = []
    if len(L) < N:
        return G
    elif N == 1:
        return L
    elif len(L) == N:
        while i < len(L):
            s.append(L[i])
            i = i + 1
        G.append(s)
    elif len(L) > N:
        l = math.factorial(len(L) - 1)/(math.factorial(N - 1)
             * math.factorial((len(L) - 1) - (N - 1)));
        while i < l:
            s = [L[len(L) - 1]]
            while h < len(n):
                if j > 0 and j < len(n):
                    n[j] = n[j - 1] + 1
                s.append(L[n[h]])
                h = h + 1
                j = j + 1
            G.append(s)
            h = 0
            j = 0
            while j < len(n) and n[j] != j + k:
                j = j + 1
            if j > 0:
                n[j - 1] = n[j - 1] + 1
            i = i + 1
        L.pop()
        G = G + combinaisons(L, N, k - 1)
    return G

#Make a standard scaaled data
def StandardScaler(df):
    newDf = df.copy()
    newDf = (newDf-newDf.mean())/newDf.std()
    return newDf

#Delete constant sensor
#input : dataframe, df
#output : scaled dataframe
def delconst(df) :
    listeViable = []
    for c in df.columns :
        valable = False
        moy = df[c].mean()
        for v in df[c] :
            if abs(v-moy) > 0.001 :
                valable = True
        if valable :
            listeViable.append(c)
    return df[listeViable]

#Genereate two sample of data (one for test, the other for learn)
#input : dataframe, df
#       integer, size
#output : two samples of data
def EnsembleTestApp(df,size) :
    appListe = []
    while len(appListe) != size :
        rand = random.randint(0,df.shape[0])
        if not (rand in appListe) :
            appListe.append(rand)
    testListe = []
    while len(testListe) != size :
        rand = random.randint(0,df.shape[0])
        if not (rand in appListe) and not (rand in testListe) :
            testListe.append(rand)
    app = df.iloc[appListe]
    test = df.iloc[testListe]
    return app,test
    


df = pd.read_csv('ECUdata\\csvCycle1\\ALL.csv')
N = 4
df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(axis = 1)
df = df.drop(columns=['time','Unnamed: 0'])
df = StandardScaler(df)
ndf = t.deleteCorrelations(df,0.50,['VehV_v','ActMod_trqClth'])
ndf = StandardScaler(ndf)
ndf.replace([np.inf, -np.inf], np.nan)
ndf = ndf.dropna(axis = 1)
ndf = delconst(ndf)
G = combinaisons(ndf.columns.values.tolist(), N, len(ndf.columns.values.tolist()) - N)
print('combi fini')

#generation of training data
app,test = EnsembleTestApp(ndf,5470)

#loop to make all simulation
bestScore = np.inf
bestCapt = G[0]
i = 1
for sensors in G :
    Xa = app[sensors].values
    Xt = test[sensors].values
    score = 0
    for s in ndf :
        ya = app[s].values
        yt = app[s].values
        clf = Ridge(alpha=0.005)
        clf.fit(Xa,ya)
        prediction = clf.predict(Xt)
        score = score + np.sum((yt-prediction)**2)
    if score < bestScore :
        bestScore = score
        bestCapt = sensors
    if i%100 == 0 :
        print(str((i/len(G))*100)+'%')
    i = i + 1
print(bestCapt)
        
        
        
        
    



