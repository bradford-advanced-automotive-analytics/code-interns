# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:50:32 2019

@author: ssquilla
"""

from libs.data_manager.data_manager import *
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from time import time
import datetime
import random as rd
import copy
"""
Top level processing class. Abstract model.
"""
class processingModel :
    def __init__(self,dm):
        pass
    def perform(self,dm):
        pass
    def checkConsistency(self,dm):
        return True
    def measureTime(self,dm):
        pass

"""
Process all the dates. Continuous dates (timedelta, datetimetz) will be converted
  in 
Param :
   columns : column to process
   strategy : strategy for non-continuous dates (non timedelta, non datetimetz)
       year - month - monthday - weekday - complete (float)
"""
class dateProcessor(processingModel):
    """
    @TODO : remove non continuous dates
    """
    def __init__(self,dataManager,columns,strategy="year",epoch=datetime.datetime(2000,1,1)):
        super().__init__(dataManager)
        self.strategy = strategy
        self.columns = columns
        self.cons = self.checkConsistency(dataManager)
        self.epoch = epoch
        if(not self.cons):
            print("Please correct the database and create another model.")
    def checkConsistency(self,dm):
        t = dm.typesSummary()[0]
        for col in self.columns:
            if(not col in t["datetime"] and not col in t["timedelta"] and not col in t["datetimetz"]):
                print("Consistency : some columns doesn't exist or are not date columns : "+col)
                return False
        return True
    def center(self,dm,column_name):
        print(" verifications (overhead)... ")
        t = dm.typesSummary()[0]
        for col in self.columns: # check if all columns are ok
            if(not col in t["timedelta"] and not col in t["datetimetz"] and not t["datetime"]):
                print(col+" not a date anymore. Dataframe seems to be modified.")
                assert(False)
        for col in self.columns:
            other = dm.dataset[column_name].apply(lambda d : d.replace(tzinfo=None))
            dm.dataset[col] = (dm.dataset[col].apply(lambda d : d.replace(tzinfo=None)) - other).apply(lambda d : d.total_seconds())
            dm.columnManagers[col].isDate = False
            dm.columnManagers[col].isNumerous = True
            dm.columnManagers[col].fromCategorical = False
            dm.dataTypes[col] = DataInfo(dataType.NUMERICAL,dataNature.CONTINOUS)
    """ Guess the time spend to perform the transformation """
    def measureTime(self,dm,test_values = np.linspace(0.1,0.3,num=5,endpoint=True)):
        t = dm.typesSummary()[0]
        lengths = np.array([len(t["datetime"]), len(t["timedelta"]), len(t["datetimetz"])])
        T = []
        dmC = copy.deepcopy(dm)
        dmC.select(dmC.getDatesColumns())
        for val in test_values:    
            dmCop = DataManager(dmC.dataset.sample(frac=val,replace=True,random_state=1))
            if(lengths[0]>0): # measure time for a random datetime column
                lis = rd.sample(list(t["datetime"]),1)[0]
                start = time()
                dmCop.extractDate(lis,self.strategy)
                elapsedDT = time()
                elapsedDT = elapsedDT - start
            else :
                elapsedDT = 0
            if(lengths[1]>0): # measure time for a random timedelta column
                lis = rd.sample(list(t["timedelta"]),1)[0]
                start = time()
                dmCop.continuousDateToFloat(lis)
                elapsedTD = time()
                elapsedTD = elapsedTD - start
            else :
                elapsedTD = 0
            if(lengths[2]>0): # measure time for a random datetimetz column
                lis = rd.sample(list(t["datetimetz"]),1)[0]
                start = time()
                dmCop.continuousDateToFloat(lis)
                elapsedTZ = time()
                elapsedTZ = elapsedTZ - start
            else :
                elapsedTZ = 0
            T += [ np.inner(lengths,np.array([elapsedDT,elapsedTD,elapsedTZ]))]

        return (test_values,T)
    """
    Local epoch, not UTC
    """
    def perform(self,dm):
        print(" verifications (overhead)... ")
        t = dm.typesSummary()[0]
        for col in self.columns: # check if all columns are ok
            if(not col in t["timedelta"] and not col in t["datetimetz"] and not t["datetime"]):
                print(col+" not a date anymore. Dataframe seems to be modified.")
                assert(False)
        print(" performing ... ")
        start = time()
        for col in self.columns:
            if (col in t["timedelta"] or col in t["datetimetz"] or col in t["datetime"]):
                dm.continuousDateToFloat(col,self.epoch)
                print(col+" continuous dates converted.")
            elif(col in t["datetime"]): # STUPID ?
                dm.extractDate(col,self.strategy) 
                print(col+" converted according to the strategy.")
        elapsed = time()
        elapsed -= start
        print("Date processing done. Time elapsed = "+str(elapsed))
            
"""
@unimplemented
Only preprocessing, no apriori algorithm called.
"""
class aPrioriPreprocessor(processingModel):
    """ 
    MUST NOT CONTAINS NaNs! Reducing the size before creating a model might be necessary.
    If the model is not consistent create another one after correcting the database.
    Param :
        column_bins : how many bins to keep for each continous column
    """
    def __init__(self,dataManager,column_bins):
        super().__init__(dataManager)
        self.colBins = column_bins
        self.cons = self.checkConsistency(dataManager)
        if(not self.cons):
            print("Please correct the database and create another model.")
        self.bins_dict = {}
    def measureTime(self,dm,test_values = np.linspace(10**3,7*10**6,20),test_bins=[2,3,4,10,15]):
        t = dm.typesSummary()[0]
        times = {"float":{},"int64":[],"object":[],"date":[]} # merge all date types
        for typ,elm in t.items():
            if(len(elm>0)):
                if(typ=="int64"):
                    pass
                elif(typ=="float" and len(times["float"])==0):
                    for n_b in test_bins:
                        x = []
                        y = []
                        for v in test_values:
                            val = dm.dataset.loc[:v,elm[0]]
                            val = val.fillna(0)
                            start = time() # mauvaise fonction pour le moment
                            est = KBinsDiscretizer(n_bins = n_b, encode="ordinal", strategy="quantile")
                            val = val.values
                            est.fit(np.reshape(val,(-1,1)))
                            elapsed = time()
                            elapsed = elapsed - start
                            x.append(v); y.append(elapsed)
                        times["float"][n_b] = (x,y)
                elif(typ=="object"):
                        pass
                elif((typ in ["datetime","timedelta","datetimetz"]) and len(times["date"])==0):
                    pass
        return times     
    def checkConsistency(self,dm):
        t = dm.typesSummary()[0]
        if(len(t["datetime"])>0 or len(t["timedelta"])>0 or len(t["datetimetz"])>0):
            print("Consistency : this model doesn't support dates")
            return False
        for col,val in self.column_bins.items() :
            if(val<=0):
                print("Consistency : negative bin")
                return False
            if(not col in dm.dataset):
                print("Consistency : unknown column "+col)
                return False
        for col in dm.dataset:
            if(not(col) in self.colBins and dm.dataTypes[col]._asdict()['nature'].value==dataNature.CONTINOUS.value):
                print("Consistency : unspecified bins for column "+col)
                return False
        return True
    def perform(self,dm):
        if(self.cons):
            self.bins_dict = dm.cut2Discretized(self.colBins,overwrite=True)
            print("\n Columns splitted")