# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:42:51 2019

@author: sdybella
"""

import mdfreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' Give the mdfreader.Mdf of the file associate to filename
input : string, filename
output : mdfreader.Mdf, raw_dict'''  
def read_data(fileName):
    #counter = 0
    # read data from an mdf file
    # and print value for every key
    raw_dict = mdfreader.Mdf(fileName)
    #for key in raw_dict.keys():
        #print(key)
        #print(raw_dict[key])
        #print(type(raw_dict[key]['data']))
        #print(raw_dict[key]['data'])
        #counter+=1
    #print(counter)
    return raw_dict

'''Convert the dict to a pandas dataframe which keep only captors associate to time + an other dataframe with description and unit
input : mdfreader.Mdf, dic
        string, time
output : pandas.Dataframe, df1, df2'''
def toDataframe(dic,time):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    for key in dic.keys():
        if (dic[key]['master'] == time):
            df1[key] = dic[key]['data']
            df2[key] = [dic[key]['description'],dic[key]['unit']]
    return df1, df2


'''Find all key which correspond to time values and give their number of values
input : mdfreader.Mdf, dic
output : list, time_list'''
def find_time_s(dic) :
    time_list = []
    for key in dic.keys():
         if key[0:4] == 'time' :
             time_list.append([key,dic[key]['data'].size])
    return time_list

'''Find the nearest value of value in array
input : numpy.array, array
        scalar, value
output : the nearest value'''
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

'''Return true if a and b are similar
input : float , a, n, rel_tol, abs_tol
output : boolean'''
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

'''Select only the channels from channelList in dic
input : mdfreader, dic
        list, hannelList
output : mdfreader, ndic (contains a mdfreader which is linked to a dictionnary which contains only the selected attribute)'''
def selectChannels(dic,channelList):
    newDic = {}
    keys = dic.keys()
    times = []
    for channel in channelList:
        if (channel in keys):
            times.append(dic[channel]['master'])
    times = list(dict.fromkeys(times))
    for time in times:
        newDic[time] = dic[time]
    for channel in channelList:
        if (channel in keys):
            newDic[channel] = dic[channel]
    return newDic

'''Make an aproximation where each captor not associated with the more precise clock are associated to the nearest value of time for each value
input : mdfreader.Mdf, dic
output : pandas.Dataframe, df, dfbis'''
def sum_up_time(dic) :
    #Fist, I choose the time that I will keep
    df = pd.DataFrame()
    dfbis = pd.DataFrame()
    time_list = find_time_s(dic)
    time_select = time_list[0][0]
    mini = time_list[0][1]
    for time in time_list :
        if time[1] < mini and time[1] > 1000 :
            time_select,mini = time[0],time[1]
    time_values = dic[time_select]['data']
    #Secondly, we need to built the two final dataframe
    for key in dic.keys():
        if (dic[key]['master'] == time_select): #this captor is already associate with the good time
            if key[0:4] != 'time' :
                df[key] = dic[key]['data']
                dfbis[key] = [dic[key]['description'],dic[key]['unit']]
            else :
                df['time'] = dic[key]['data']
                dfbis['time'] = [dic[key]['description'],dic[key]['unit']]
        elif key[0:4] == 'time' or key[0] == '$' : # We ignored other time
            None
        else : #We studied a captor with an other time 
            time_studied = dic[dic[key]['master']]['data']
            capt_studied = dic[key]['data']
            result =[[],[]]
            const = [find_nearest(time_values,time_studied[0]),capt_studied[0]]
            for i in range(1,len(time_studied)) :
                nearest = find_nearest(time_values,time_studied[i])
                if nearest == const[0] :
                    const.append(capt_studied[i])
                else : 
                    result[0].append(sum(const[1:len(const)])/(len(const)-1))
                    result[1].append(const[0])
                    const = [nearest,capt_studied[i]]
            result[0].append(sum(const[1:len(const)])/(len(const)-1))
            result[1].append(const[0])
            # We create the compatible array to write it in our dataframe
            array = np.zeros(len(time_values))
            array.fill(np.nan)
            continuer = True
            t1,t2 = 0,0
            while continuer :
                if time_values[t1] == result[1][t2]:
                    array[t1] = result[0][t2]
                    t1 += 1
                    t2 += 1
                elif time_values[t1] < result[1][t2] :
                    t1 += 1
                else :
                    t2 += 1
                if t1 == len(time_values) or t2 == len(result[1]) :
                    continuer = False
            df[key] = array
            dfbis[key] = [dic[key]['description'],dic[key]['unit']]
        #print(key + ' done')
    cols = df.columns.tolist()
    cols.remove('time')
    cols = ['time'] + cols
    df = df[cols]
    return df, dfbis

'''Delete column with more than ratio NaN
input : mdfreader, dic
        float, ratio
ourput : Dataframe, df simplified'''
def dropAttNan(dic,ratio) :
    listViable = []
    for key in dic.keys() :
        tableau = dic[key]['data']
        nbTotal = tableau.size
        nbNaN = np.count_nonzero(pd.isnull(tableau))
        if nbNaN/nbTotal < ratio :
            listViable.append(key)
    ndic = selectChannels(dic,listViable)
    return ndic
    #print('data simplified')

'''Delete column with more than ratio NaN
input : mdfreader, dic
        float, ratio
ourput : Dataframe, df simplified'''
def dropAttZeros(dic,ratio) :
    listViable = []
    for key in dic.keys() :
        tableau = dic[key]['data']
        nbTotal = tableau.size
        nbNZero = np.count_nonzero(tableau)
        if (nbTotal - nbNZero)/nbTotal < ratio :
            listViable.append(key)
    ndic = selectChannels(dic,listViable)
    return ndic

'''Compute a dictionnary which tell us how many times a sensor appear in a family of data. nan and zero are boolean which can tell if we igmored or not nan and zero
input : mdfreader, dic
        boolean, nan, zero
output : dictionnay, occ'''
def findOcc(cycle,nan,zero) :
    occ = {}
    for file in cycle:
        dic = mdfreader.Mdf(file)
        if nan :
            dic = dropAttNan(dic,0.5)
        if zero :
            dic = dropAttZeros(dic,0.99)
        keys = dic.keys()
        for key in keys:
            if key in occ.keys():
                occ[key] += 1
            else :
                occ[key] = 1
    return occ
   
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #MAIN PROGRAM

# Select only the columns we understand
channelList = ['FlSys_volFlPlausChkd','BattU_u','PFltPOp_vFlt','PthLead_trqInrCurr','CoETS_trqInrLim','ActMod_trqClth','EngPrt_trqLim','VehV_v','InjSys_qTot','Epm_nEng','PthSet_trqInrSet']

fileNames0 = ['ECUdata\\drivingCycle00\\00_drivingCycle00.dat','ECUdata\\drivingCycle00\\01_drivingCycle00.dat','ECUdata\\drivingCycle00\\02_drivingCycle00.dat','ECUdata\\drivingCycle00\\03_drivingCycle00.dat','ECUdata\\drivingCycle00\\04_drivingCycle00.dat','ECUdata\\drivingCycle00\\05_drivingCycle00.dat','ECUdata\\drivingCycle00\\06_drivingCycle00.dat','ECUdata\\drivingCycle00\\07_drivingCycle00.dat','ECUdata\\drivingCycle00\\08_drivingCycle00.dat','ECUdata\\drivingCycle00\\09_drivingCycle00.dat','ECUdata\\drivingCycle00\\10_drivingCycle00.dat','ECUdata\\drivingCycle00\\11_drivingCycle00.dat','ECUdata\\drivingCycle00\\12_drivingCycle00.dat']
fileNames1 = ['ECUdata\\drivingCycle01\\00_drivingCycle01.dat','ECUdata\\drivingCycle01\\01_drivingCycle01.dat','ECUdata\\drivingCycle01\\02_drivingCycle01.dat','ECUdata\\drivingCycle01\\03_drivingCycle01.dat','ECUdata\\drivingCycle01\\04_drivingCycle01.dat','ECUdata\\drivingCycle01\\05_drivingCycle01.dat','ECUdata\\drivingCycle01\\06_drivingCycle01.dat','ECUdata\\drivingCycle01\\07_drivingCycle01.dat','ECUdata\\drivingCycle01\\08_drivingCycle01.dat','ECUdata\\drivingCycle01\\09_drivingCycle01.dat','ECUdata\\drivingCycle01\\10_drivingCycle01.dat']
fileNames2 = ['ECUdata\\drivingCycle02\\00_drivingCycle02.dat','ECUdata\\drivingCycle02\\01_drivingCycle02.dat','ECUdata\\drivingCycle02\\02_drivingCycle02.dat','ECUdata\\drivingCycle02\\03_drivingCycle02.dat','ECUdata\\drivingCycle02\\04_drivingCycle02.dat','ECUdata\\drivingCycle02\\05_drivingCycle02.dat','ECUdata\\drivingCycle02\\06_drivingCycle02.dat','ECUdata\\drivingCycle02\\07_drivingCycle02.dat','ECUdata\\drivingCycle02\\08_drivingCycle02.dat','ECUdata\\drivingCycle02\\09_drivingCycle02.dat','ECUdata\\drivingCycle02\\10_drivingCycle02.dat','ECUdata\\drivingCycle02\\11_drivingCycle02.dat','ECUdata\\drivingCycle02\\12_drivingCycle02.dat','ECUdata\\drivingCycle02\\13_drivingCycle02.dat','ECUdata\\drivingCycle02\\14_drivingCycle02.dat','ECUdata\\drivingCycle02\\15_drivingCycle02.dat','ECUdata\\drivingCycle02\\16_drivingCycle02.dat','ECUdata\\drivingCycle02\\17_drivingCycle02.dat','ECUdata\\drivingCycle02\\18_drivingCycle02.dat']
fileNames3 = ['ECUdata\\drivingCycle03\\00_drivingCycle03.dat','ECUdata\\drivingCycle03\\01_drivingCycle03.dat','ECUdata\\drivingCycle03\\02_drivingCycle03.dat','ECUdata\\drivingCycle03\\03_drivingCycle03.dat','ECUdata\\drivingCycle03\\04_drivingCycle03.dat','ECUdata\\drivingCycle03\\05_drivingCycle03.dat','ECUdata\\drivingCycle03\\06_drivingCycle03.dat','ECUdata\\drivingCycle03\\07_drivingCycle03.dat','ECUdata\\drivingCycle03\\08_drivingCycle03.dat','ECUdata\\drivingCycle03\\09_drivingCycle03.dat','ECUdata\\drivingCycle03\\10_drivingCycle03.dat','ECUdata\\drivingCycle03\\11_drivingCycle03.dat','ECUdata\\drivingCycle03\\12_drivingCycle03.dat','ECUdata\\drivingCycle03\\13_drivingCycle03.dat','ECUdata\\drivingCycle03\\14_drivingCycle03.dat','ECUdata\\drivingCycle03\\15_drivingCycle03.dat','ECUdata\\drivingCycle03\\16_drivingCycle03.dat','ECUdata\\drivingCycle03\\17_drivingCycle03.dat','ECUdata\\drivingCycle03\\18_drivingCycle03.dat','ECUdata\\drivingCycle03\\19_drivingCycle03.dat','ECUdata\\drivingCycle03\\20_drivingCycle03.dat','ECUdata\\drivingCycle03\\21_drivingCycle03.dat','ECUdata\\drivingCycle03\\22_drivingCycle03.dat','ECUdata\\drivingCycle03\\23_drivingCycle03.dat','ECUdata\\drivingCycle03\\24_drivingCycle03.dat','ECUdata\\drivingCycle03\\25_drivingCycle03.dat','ECUdata\\drivingCycle03\\26_drivingCycle03.dat','ECUdata\\drivingCycle03\\27_drivingCycle03.dat','ECUdata\\drivingCycle03\\28_drivingCycle03.dat','ECUdata\\drivingCycle03\\29_drivingCycle03.dat','ECUdata\\drivingCycle03\\30_drivingCycle03.dat','ECUdata\\drivingCycle03\\31_drivingCycle03.dat','ECUdata\\drivingCycle03\\32_drivingCycle03.dat','ECUdata\\drivingCycle03\\33_drivingCycle03.dat','ECUdata\\drivingCycle03\\34_drivingCycle03.dat','ECUdata\\drivingCycle03\\35_drivingCycle03.dat','ECUdata\\drivingCycle03\\36_drivingCycle03.dat','ECUdata\\drivingCycle03\\37_drivingCycle03.dat']
fileNames4 = ['ECUdata\\drivingCycle04\\00_drivingCycle04.dat','ECUdata\\drivingCycle04\\01_drivingCycle04.dat','ECUdata\\drivingCycle04\\02_drivingCycle04.dat','ECUdata\\drivingCycle04\\03_drivingCycle04.dat','ECUdata\\drivingCycle04\\04_drivingCycle04.dat','ECUdata\\drivingCycle04\\05_drivingCycle04.dat','ECUdata\\drivingCycle04\\06_drivingCycle04.dat','ECUdata\\drivingCycle04\\07_drivingCycle04.dat','ECUdata\\drivingCycle04\\08_drivingCycle04.dat','ECUdata\\drivingCycle04\\09_drivingCycle04.dat','ECUdata\\drivingCycle04\\10_drivingCycle04.dat','ECUdata\\drivingCycle04\\11_drivingCycle04.dat','ECUdata\\drivingCycle04\\12_drivingCycle04.dat']

fileNames = []
fileNames.append(fileNames0)
fileNames.append(fileNames1)
fileNames.append(fileNames2)
fileNames.append(fileNames3)
fileNames.append(fileNames4)

occ = [findOcc(fileNames0,True,True)]
occ.append(findOcc(fileNames1,True,True))
occ.append(findOcc(fileNames2,True,True))
occ.append(findOcc(fileNames3,True,True))
occ.append(findOcc(fileNames4,True,True))

minOccurences = [11,11,19,38,7]

remainingChannels = [[],[],[],[],[]]
for i in range(0,5) :
    for key in occ[i].keys():
        if occ[i][key]>=minOccurences[i]:
            remainingChannels[i].append(key)
    print('occ ' + str(i) + ' finished')
            
#TODO : Other selections
    

for i in range(0,5) : 
    for file in fileNames[i]:
        dic = mdfreader.Mdf(file)
        ndic = selectChannels(dic,remainingChannels[i])
        target = 'ECUdata\\csvCycle' + str(i) + '\\'
        df,dfbis = sum_up_time(ndic)
        filetab = file.split('\\')
        df.to_csv(target + filetab[len(filetab)-1] + '.csv')
        dfbis.to_csv(target + filetab[len(filetab)-1] + 'desc.csv')
        print (file + ' finished')
        