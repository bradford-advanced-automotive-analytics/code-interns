# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:42:51 2019

@author: sdybella
"""

import mdfreader
import pandas as pd
import numpy as np


#Give the mdfreader.Mdf of the file associate to filename
#input : string, filename
#output : mdfreader.Mdf, raw_dict'''  
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

#Convert the dict to a pandas dataframe which keep only captors associate to time + an other dataframe with description and unit
#input : mdfreader.Mdf, dic
#        string, time
#output : pandas.Dataframe, df1, df2'''
def toDataframe(dic,time):
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    for key in dic.keys():
        if (dic[key]['master'] == time):
            df1[key] = dic[key]['data']
            df2[key] = [dic[key]['description'],dic[key]['unit']]
    return df1, df2


#Find all key which correspond to time values and give their number of values
#input : mdfreader.Mdf, dic
#output : list, time_list'''
def find_time_s(dic) :
    time_list = []
    for key in dic.keys():
         if key[0:4] == 'time' :
             time_list.append([key,dic[key]['data'].size])
    return time_list

#Find the nearest value of value in array
#input : numpy.array, array
#        scalar, value
#output : the nearest value'''
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

#Return true if a and b are similar
#input : float , a, n, rel_tol, abs_tol
#output : boolean'''
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

#Select only the channels from channelList in dic
#input : mdfreader, dic
#        list, hannelList
#output : mdfreader, ndic (contains a mdfreader which is linked to a dictionnary which contains only the selected attribute)'''
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

#Make an aproximation where each captor not associated with the more precise clock are associated to the nearest value of time for each value
#input : mdfreader.Mdf, dic
#output : pandas.Dataframe, df, dfbis'''
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

#Make an aproximation where each captor not associated with the more precise clock are associated to the nearest value of time for each value
#input : mdfreader.Mdf, dic
#output : pandas.Dataframe, df, dfbis'''
def sum_up_time2(dic,timeRate) :
    #Fist, I choose the time that I will keep
    df = pd.DataFrame()
    dfbis = pd.DataFrame()
    maxTime = 0
    for key in dic.keys() :
        if (key[0:4] == 'time') and dic[key]['data'].max() > maxTime  :
            maxTime = dic[key]['data'].max()
    time_values = np.arange(0.1,maxTime,timeRate)
    df['time'] = time_values
    #Secondly, we need to built the two final dataframe
    for key in dic.keys():
        if key[0:4] == 'time' or key[0] == '$' : # We ignored other time
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
    return df, dfbis

#sample the data at a set frequency
#input:  dic: dictionary
#        df: dataframe appending to
#        frequency: sampling frequency
#output: sampled dataframe
def sampleData(dic, df, frequency):
    times = []
    maxTime = 0
    for key in dic.keys():
        if key[0:4] == 'time':
            times.append(key)
            if dic[key]['data'].max() > maxTime:
                maxTime = dic[key]['data'].max()
    for time in times:
        df1,df2 = toDataframe(dic, time)
        for t in np.arange(0, maxTime, 1/frequency):
            subset = df1.loc[(df1[time]>=(t-((1/frequency)/2))) & (df1[time]<(t+((1/frequency)/2)))]
            subsetMean = subset.mean()
            subsetMean[time] = t
            for key in subsetMean.keys():
                if key not in times[1:]:
                    df.loc[t*frequency, key] = subsetMean[key]
    correctName = {}
    for c in df.columns :
        if c[0:4]=='time' :
            correctName[c] = 'time'
    df = df.rename(columns=correctName)
    return df

#Delete column with more than ratio NaN
#input : mdfreader, dic
#        float, ratio
#output : Dataframe, df simplified'''
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

#Same that above for dataframe'''
def dropNaN(data,ratio) :
    listViable = []
    for c in data.columns :
        nbTotal = data.shape[0]
        nbNaN = nbTotal - data.loc[:,c].dropna().size
        if nbNaN/nbTotal < ratio :
            listViable.append(c)
    dataN = data.loc[:,listViable]
    return dataN

#Delete column with more than ratio NaN
#input : mdfreader, dic
#        float, ratio
#output : Dataframe, df simplified'''
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

# Same that above for dataframe and for all constant values'''
def dropConstant(data,ratio) :
    listViable = []
    for c in data.columns :
        nbTotal = data.shape[0]
        nbNConstant = nbTotal - data.loc[:,c].drop_duplicates().size
        if nbNConstant/nbTotal < ratio :
            listViable.append(c)
    dataN = data.loc[:,listViable]
    return dataN

#Compute a dictionnary which tell us how many times a sensor appear in a family of data. nan and zero are boolean which can tell if we igmored or not nan and zero
#input : mdfreader, dic
#        boolean, nan, zero
#output : dictionnay, occ'''
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

#Compute a dictionnary which tell us how many times a sensor appear in a family of data. nan and zero are boolean which can tell if we igmored or not nan and zero
#input : mdfreader, dic
#        boolean, nan, zero
#output : dictionnay, occ'''
def findOcc2(cycle,nan,zero) :
    occ = {}
    for folder in cycle :
        for file in folder:
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
fileNames4 = ['ECUdata\\drivingCycle04\\11_drivingCycle04.dat','ECUdata\\drivingCycle04\\12_drivingCycle04.dat']

fileNames = []
fileNames.append(fileNames0)
fileNames.append(fileNames1)
fileNames.append(fileNames2)
fileNames.append(fileNames3)
fileNames.append(fileNames4)

occ = [findOcc(fileNames0,True,True)]
occ.append(findOcc(fileNames1,True,True))
'''
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
'''
conserve = ['PFlt_tInrDes', 'ETCtl_tOutrDes', 'PFltPOp_stRng_[9]', 'ETCtl_tDesFie_[4]', 'Exh_tAdapTTrbnUs', 'SCRT_tUCatDsT', 'ASMod_tCatOut_[33]', 'ASMod_tCatOut_[30]', 'PFltLd_facTempSimRgn_mp', 'PFltLd_dmO2Aprx', 'CACPmp_rAct', 'Exh_pTrbnUs', 'PFltLd_facSotSimRgnNO2_mp', 'ETCtl_rPInr_mp', 'NSCDs_rLam', 'EnvT_t', 'ASMod_tCatOut_[32]', 'NSCRgn_etaEstFltNscScr_mp', 'InjCrv_qPoI1Splt_[0]', 'ASMod_dmSotBasNrm_mp', 'Oil_tSwmp', 'PFltPOp_vFlt', 'ETCtl_rIInr_mp', 'FuelT_t', 'ASMod_tCatOut_[31]', 'DFES_numDFC_[0]', 'CoEOM_numOpModeActTSync', 'UEGO_ratIPmpAdpnS1B1', 'PFltRgn_tiRgnMax_mp', 'ASMod_dmSotEngNrm1_mp', 'PFltLd_mSot', 'NSCRgn_stCondRlsTNSCDSOx', 'SCRCat_mfNoxScrUsSnsr', 'NSCLd_mSOxFld_[0]', 'PFltLd_pDiffSot_mp', 'ETCtl_tOutrDvt_mp', 'ETCtl_tDesFie_[1]', 'PFltRgn_numRgnSel_mp', 'LSU_rO2Adap_[0]', 'ETCtl_stInrLop', 'Exh_pAdapPPFltDiff', 'LSU_rO2Raw_[0]', 'AirCtl_mGovDvt', 'ETCtl_tDesFie_[7]', 'PFltLd_resFlwRaw_mp', 'PFltRgn_numTot', 'Lub_qPoI1Des_mp', 'ETCtl_tDesFie_[3]', 'SCRCat_mfExhDpfDs', 'PFltLd_resFlwFlt', 'SCRCat_mfNoxScrDsSnsr', 'NSCRgn_stStM', 'Exh_tSensTTrbnUs', 'EnvP_p', 'PFltLd_tiRgnAct_mp', 'PFlt_pPFltDiffFlt_mp', 'PFltLd_dmSotSim', 'AirCtl_qDesVal', 'ASMod_tCatOut_[27]', 'PFltRgn_stOpModeReq', 'ETCtl_tDesFie_[2]', 'PFltLd_rLam_mp', 'PFltPOp_tEngFlt', 'PFltLd_dmSotSimRgnThr_mp', 'PFltLd_dmNO2Tot_mp', 'PFltRgn_stOpMode', 'ASMod_tCatOut_[26]', 'NSCLd_dmNOxDs', 'ASMod_tCatOut_[29]', 'Lub_lToGo_mp', 'Exh_dmNOxNoCat2Ds', 'ETCtl_tAdpnCatUs', 'PFlt_tOutrDes', 'Exh_rNOxNoCat2Ds', 'Exh_tTrbnUs', 'NSCRgn_tNSCSource_mp', 'ETCtl_tAdpnCatDs', 'ETCtl_tCatUsSnsrMdl_mp', 'PFltRgn_tPFltUsMaxHi_mp', 'ETCtl_tCatDsExh_mp', 'Exh_tSensTPFltUs', 'Exh_tAdapTPFltUs', 'AirCtl_rDesVal', 'Exh_tAdapTOxiCatUs', 'ASMod_tPFltSurfSim', 'Exh_tPFltDs', 'Exh_tPFltUs', 'ETCtl_tCatUsExh_mp', 'ASMod_rEGR', 'NSC_tOutrDesVal', 'Exh_rNOxNSCDs', 'ETCtl_tInrDes', 'Exh_dmNOxNSCDs', 'NSCRgn_tNSC_mp', 'AirCtl_rDesValEOMQnt3_mp', 'Air_tCACDsDiff', 'PFltLd_mSotSim', 'ETCtl_rTCatUsMdlDT1_mp', 'SCRAd_rNH3', 'CEngDsT_t', 'PFltLd_dmNOxEG_mp', 'ETCtl_tDesFie_[6]', 'SCRCat_mfNoxSnsrCtl', 'PFltRgn_facSot', 'FlSys_volFlPlausChkd', 'Exh_pPFltDiff', 'ETCtl_tDesFie_[5]', 'Lub_rFuelInOil', 'Exh_pPFltUs', 'InjSys_tECU_mp', 'PFltLd_rLamAprx_mp', 'Exh_tOxiCatUsB2', 'Exh_tAdapTPFltDs', 'PFltLd_mSotSimCont', 'PFltLd_facO2SimRgn_mp', 'PFltRgn_ctRgnSuc', 'SCRCat_mfExhScrUs', 'NSCLd_mSOxFld_[1]', 'PFltLd_mSotSimNoCont', 'DFC_st.DFC_OxiCatMonPasMin', 'SmkLim_mAirPerCyl', 'PFltPOp_stCR', 'ETCtl_tBrickLstMdl_mp', 'ASMod_tCatOut_[28]', 'PFltRgn_rTimeRgnMax', 'ETCtl_tDesFie_[0]', 'Exh_tOxiCatUsB1', 'Exh_pTrbnDs_mp', 'ETCtl_st', 'Exh_tSensTPFltDs', 'PthLead_trqInrCurr', 'Lub_qPoI2Des_mp', 'Gbx_tTCMOil', 'DStgy_stOpMode', 'PFltLd_resFlw', 'PFltLd_dmSotSimRgn', 'SCRLdG_mNH3LoadNom', 'BattU_u', 'Exh_pSensPPFltDiff', 'SmkLim_qLimSmk', 'NSCRgn_stIntrDSOxWd', 'ASMod_dmSotEngNrm3_mp', 'CHdT_tClntMod', 'Lub_rFuelUnFlt_mp', 'DStgy_stMetStgy', 'DStgy_stDosOn', 'Exh_stNOxNSCDs', 'Exh_stNOxNoCat2Ds', 'PCR_pDesVal', 'Exh_tSens1TOxiCatUs_mp', 'SCRMod_mEstNH3Ld', 'Exh_tSens1TPFltUs_mp', 'Air_pCACDs', 'ThrVlv_rAct', 'AirCtl_mDesVal', 'SCRMod_dmEstNOxDs', 'TrbCh_rAct', 'EGRVlvLP_rAct', 'DStgy_dmRdcAgDes', 'Exh_tSensTOxiCatUsB1', 'Air_pIntkVUs', 'Exh_tSensTOxiCatUs', 'Exh_tOxiCatUs', 'AFS_mAirPerCylFlt', 'ASMod_pIndVol', 'ETCtl_qPoI2', 'CoEOM_stOpModeAct', 'InjCrv_qPoI1Min_mp', 'EngPrt_trqLim', 'AirCtl_stAirCtlBits', 'UDosVlv_rPs', 'PFltPOp_stEngPOp', 'ExhMod_ratNOXEGSys', 'InjSys_facPoI1CmbMode3_mp', 'Tra_numGear', 'CoETS_trqInrLim', 'SCRMod_etaEstSel', 'Rail_pSetPoint', 'InjSys_qTot', 'APP_r', 'InjCrv_phiMI1Des', 'EEM_iAltExct', 'Exh_rO2LinNoCat2Ds', 'EGRVlv_rAct', 'ExhMod_ratNoxRefBas', 'CoEOM_stOpModeReq', 'InjCrv_phiPiI1Des', 'ASMod_dmExhMnfDs_r32', 'SCRT_tUCatUsT', 'ETCtl_facT1Inr_mp', 'ASMod_dmNOxEG', 'SCRLdG_mNH3LdNom_mp', 'InjCrv_qPoI1Des_mp', 'ExhMod_ratNOX_mp', 'PFltRgn_tiTmrStOvrRun_mp', 'CoEOM_numOpModeActTSync_mp', 'Air_tNormCalc_mp', 'InjCrv_qPiI1Des_mp', 'EEM_trqAlt', 'ActMod_trqClth', 'PFltRgn_stDem', 'ASMod_rO2ExhMnfDs', 'EEM_stAlt', 'PFltRgn_numSot_mp', 'InjCrv_phiPoI1Des', 'Air_tCACDs', 'InjSys_facPoI1Cmb', 'LSU_rLam_[0]', 'RailP_pFlt', 'CHdT_t', 'ASMod_dmSotEG', 'AirCtl_mAirPerCylDesDyn_r32', 'SCRT_tAvrg', 'InjCrv_qPoI3Des_mp', 'InjCrv_qMI1Des', 'AFS_dm', 'InjCrv_qPoI2Des_mp', 'ASMod_dmIntMnfUs_r32', 'InjCrv_phiPoI3Des', 'Rail_pDvt', 'InjSys_facPoI1CmbMode3Cor_mp', 'ASMod_pPFltUs', 'AirCtl_dmTVADes_r32', 'PFltRgn_numRgn', 'ASMod_dmSotLamCor', 'SCRMod_rNO2NOx', 'ASMod_dmEG_mp', 'InjCrv_phiPoI2Des', 'EngReq_trqInrLimSmk', 'ASMod_qBrn', 'UDC_mRdcAgDosQnt', 'Air_tCACDsNorm_mp', 'PCR_pGovDvt', 'PEGRLPDiff_p', 'PCR_pMaxDvt_mp', 'ASMod_rLamRecEG', 'VehV_v', 'InjCrv_numPoI1Splt', 'PFltRgn_stNumEngPOpDelOn', 'PthSet_stOvrRun', 'PFltLd_stSimOn', 'PFltRgn_stRgnNoCmpln', 'PCR_swtGov', 'AirCtl_swtGovEna', 'Epm_nEng', 'InjCtl_qSetUnBal', 'PthSet_trqInrSet', 'PCR_qCtlVal', 'InjCrv_phiPoI1Dur_mp', 'ASMod_tEGFld_[33]', 'InjVlv_tiET_[2]', 'ASMod_tEGFld_[2]', 'SCRFFC_dmEGFlt_mp', 'InjSys_facPoI3Cmb', 'SCRMod_rEstNOxDs', 'PFltPOp_stRng_[7]', 'PCR_rCtlValHP', 'NSCRgn_stRlsLogicBas_mp', 'NSCUs_rLam', 'NSCRgn_dtDSOx_mp', 'NSCRgn_tScndMaxPrdcIntr_mp', 'InjVlv_tiET_[5]', 'InjVlv_tiET_[7]', 'PFltPOp_stRng_[6]', 'PFltPOp_stRng_[8]', 'OxiCat_tModRefDs_mp', 'GlbDa_lTotDstKm', 'InjSys_facPoI2Cmb', 'NSCRgn_stIntrShrtDNOxWd_mp', 'PFltPOp_stRng_[4]', 'PFltPOp_stRng_[5]', 'SCR_tUCatUsT', 'InjSys_facMI1Cmb', 'OxiCat_tModDs_mp', 'InjVlv_tiET_[6]', 'NSCRgn_tUsTPrim_mp', 'NSCRgn_tPrimMaxPrdcIntr_mp', 'ASMod_tEGFld_[17]', 'InjSys_facMI1CmbCor_mp', 'ExhMod_mfNoxEngOutp', 'PFltPOp_stRng_[1]', 'NSCRgn_stRlsLogic', 'SCRFFC_dmNH3Cnv', 'ETCTl_facPoI1Cor_mp', 'ASMod_tEGFld_[4]', 'InjVlv_tiET_[4]', 'Hegn_mFlowNoxB1S2', 'PFltPOp_stRng_[3]', 'Exh_tExhTMon1_mp', 'NSCRgn_tDeOSC_mp', 'NSCRgn_stRlsDSOxWd', 'InjVlv_tiET_[3]', 'NSCRgn_mSOx_mp', 'PFltPOp_stRng_[2]', 'Exh_tExhTMon3_mp', 'NSCRgn_tPrim_mp', 'ASMod_tEGFld_[5]', 'SCR_tUCatDsT', 'InjSys_facMI1CmbBas_mp', 'CoEng_st', 'NSCRgn_stRlsHtgWd_mp', 'EngPrt_trqNLim', 'NSCRgn_tScnd_mp', 'SCRFFC_dmNOxUs', 'Exh_tExhTMon6_mp', 'NSCRgn_stRlsDNOxWd_mp', 'Exh_tExhTMon4_mp', 'OxiCat_stStrtCondPas_mp', 'OxiCat_dmFlCmb_mp', 'ASMod_tEGFld_[16]', 'Hegn_mFlowNoxB1S1', 'SCR_tSensUCatUsT', 'NSCRgn_stDem', 'CALC_phi_Clash', 'CALC_Lambda', 'CALC_Trq_Derate', 'CALC_LNT_Exo', 'CALC_Delta_tTrb', 'CALC_Delta_pThr', '$ActiveCalibrationPage', 'ASMod_dmExhMnfDs', '$EVENT_COMMENTS'] 

for i in [1] : 
    dff = pd.DataFrame()
    for file in fileNames[i]:
        dic = mdfreader.Mdf(file)
        ndic = selectChannels(dic,conserve)
        target = 'ECUdata\\csvCycle' + str(i) + '\\'
        df = sampleData(ndic,pd.DataFrame(),2)
        dff = pd.concat([dff, df])
        filetab = file.split('\\')
        print(filetab[len(filetab)-1] + 'concatened')
    #Let's write it
    dff.to_csv(target + 'ALL.csv')
    print ('finished')
        