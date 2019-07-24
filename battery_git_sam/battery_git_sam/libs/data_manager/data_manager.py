
from __future__ import division
import numpy as np
from math import floor
from sklearn.decomposition import KernelPCA
#from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from  sklearn .manifold import TSNE # t-sne
import pandas as pd
from enum import Enum
from collections import namedtuple
import scipy.sparse as sp
import datetime

""" 
Don't change the values at any cost! Because of of cache problems 
enum.type.STRING == enum.type.STRING return False.
So I used enum.type.STRING.value == enum.type.STRING.value. Never add field
with an value already used.
 """
class dataType(Enum):
    NUMERICAL = 0
    CATEGORICAL = 1
    UNKNOWN = 2
    STRING = 3
    DATE = 4
class dataNature(Enum):
    CONTINOUS = 0
    DISCRETE = 1
    UNKNOWN = 2
DataInfo = namedtuple("DataInfo", "type nature")

""" 
Used to transform data from categorical to numerous in the dataset directly.
NaN should not exist on this column.
"""
class _ColumnManager:
    def __init__(self,column_name,isNumerous,fromCategorical,isDate=False):
        assert(len(column_name)>0)
        self.isDate = isDate
        self.column_name = column_name
        self.isNumerous = isNumerous
        self.fromCategorical = fromCategorical #bool
        self.dictionnary = {}
        self.columns = {}
    """ for datetime only, not datetimetz nor timedelta"""
    def _extractDate(self,dataset,attribute="year"):
        assert(self.isDate)
        assert(not self.isNumerous)
        if(attribute=="year"):
            dataset[self.column_name] = dataset[self.column_name].apply(lambda d : int(d.year))        
        elif(attribute=="month"):
            dataset[self.column_name] = dataset[self.column_name].apply(lambda d : int(d.month))         
        elif(attribute=="monthday"):
            dataset[self.column_name] = dataset[self.column_name].apply(lambda d : int(d.day))        
        elif(attribute=="weekday"):
            dataset[self.column_name] = dataset[self.column_name].apply(lambda d : int(d.weekday()))         
        elif(attribute=="complete"):
            dataset[self.column_name] = dataset[self.column_name].apply(lambda d : d.total_seconds())         
        else:
            print(attribute+" doesn't exist.")
            assert(False)
        return
    def _yearStringToDate(self,dataset):
        assert(not self.isNumerous)
        assert(self.fromCategorical)
        assert(not self.isDate)
        dataset[self.column_name] = dataset[self.column_name].apply(lambda d : datetime.datetime(year=int(d),month=1,day=1,tzinfo=None))
        self.isDate = True
    def _continuousDateToFloat(self,dataset,epoch=datetime.datetime(2000,1,1)):
        assert(self.isDate)
        assert(not self.isNumerous)
        assert(not self.fromCategorical)
        dataset[self.column_name] = dataset[self.column_name].apply(
                lambda d : (d - epoch.replace(tzinfo=d.tzinfo)).total_seconds())
        self.isDate = False
        self.isNumerous = True
    def _prefix(self,dataset):
        dataset[self.column_name].apply(lambda x : self.column_name+"_bin"+str(x))
    def _stringToContinous(self,dataset,saveInSparse):
        assert(not(self.isNumerous))
        assert(self.fromCategorical)
        self.isNumerous = True
        dico = {}
        for elmt in dataset[self.column_name]:
            dico[elmt] = float(elmt)
            elmt = float(elmt)
        return dico
    def _splitToNumerous(self,dataset):
        """ peut etre ;ieux
                results = np.zeros((len(sequences), dimension))
                for i, word_indices in enumerate(sequences):
                    results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
                return results
        """
        assert(not(self.isNumerous))
        assert(self.fromCategorical)
        self.isNumerous = True
        old = dataset.pop(self.column_name)
        dico = old.unique()
        for value in dico:
            self.dictionnary[self.column_name+"_"+str(value)]=self.column_name
            dataset[self.column_name+"_"+str(value)] = (old == value)*1.0
            self.columns[self.column_name+"_"+str(value)] = value
        return dico
    def _findValue(self,i,dataset):
        for key_col in self.columns:
            if(dataset[key_col][i]==1.0):
                return self.columns.get(key_col)
        return None
    def _mergeToCategorical(self,dataset):
        assert(self.isNumerous)
        assert(self.fromCategorical)
        nb_row = np.shape(dataset)[0]
        dataset[self.column_name]=[self._findValue(i,dataset) for i in range(nb_row)]
        [dataset.pop(col) for col in self.columns]
        return
    """
    Non invertible transformation. Cut into boolean bins.
    Not the smartest way to di this but it works. (maybe use pandas.cut?)
    """
    def _cutIntoBins(self,nbins,isNormalized,dataset,overwrite=False):
        old = dataset[self.column_name]
        if(isNormalized):
            ma = 1
            mi = 0
        else :
            ma = float(old.max())
            mi = float(old.min())
        bins = np.linspace(mi,ma,nbins+1)
        if(not(overwrite)):
            for i in range(len(bins)-1):
                dataset[self.column_name+'_bin'+str(i)] = [0.0]*(np.shape(dataset)[0])
            for i,row in dataset.iterrows():
                bin_num = dichotomySearch(row[self.column_name],bins)
                dataset.loc[i,self.column_name+'_bin'+str(bin_num)]=1.0
            dataset.pop(self.column_name)
        else:
            dataset[self.column_name] = dataset[self.column_name].apply(lambda x : self.column_name+"_bin"+str(dichotomySearch(float(x),bins)))
        return
        
"""
Raise if a call is made on a copy. The original DM should be the only owner.
"""
class AccessExceptionError(Exception):
    def __init__(self, message, errors):
        super(ValidationError, self).__init__(message)
        self.errors = errors

class DataManager:
    def __init__(self,dataset):
        self.dataset = dataset[:]
        self.dataTypes = {}
        self.columnManagers = {}
        self.isNormalized = False
        self.setInfos()
        self.columnsSave = []
        self.owner = id(self)
    def _checkOrigin(self):
        if(not self.owner == id(self)):
            raise AccessExceptionError("Calls from a copy isn't allowed. Please use the 'real' manager.")
    """
    Transform a column of string representing a year into a date. (00:00 pm, 1st of january are added.)
    """
    def yearStringToDate(self,col):
        _checkOrigin(self)
        self.columnManagers[col]._yearStringToDate(self.dataset)
        self.dataTypes[col] = DataInfo(dataType.DATE,dataNature.CONTINOUS)
    """
    Return a dictionnary of all infer types. If types come from a different source than
    pandas (Tableau for example), types may be converted into 'O' (object) which is not
    good at all. Check the source of your data and check the the return before apply functions.
    If the result isn't correct for dates, then ???? @TODO repair method for a column list
    """
    def typesSummary(self):
        dico = {"float":[],'int64':[],"object":[],"timedelta":[],"datetimetz":[]}
        for k in dico:
            dico[k] = self.dataset.select_dtypes(include=[k]).columns.values
        return dico,self.dataset.dtypes.value_counts()
    """
    Exctract an attribute from the date : 
        year|month|weekday|monthday => int
         complete => float 
    """
    def extractDate(self,column_name,attribute="year"):
        self._checkOrigin()
        self.columnManagers[column_name]._extractDate(self.dataset,attribute)
        self.dataTypes[column_name] = DataInfo(dataType.Date,dataNature.DISCRETE)
    """
    Datetime to float. Epoch = reference. @TODO : define None reference?
    """
    def continuousDateToFloat(self,column_name,epoch=datetime.datetime(2000,1,1)):
        self._checkOrigin()
        self.columnManagers[column_name]._continuousDateToFloat(self.dataset,epoch) 
        self.dataTypes[column_name] = DataInfo(dataType.NUMERICAL,dataNature.CONTINOUS)
    """
    Delete all colulmns which not appears in the list.
    @TODO : verify if each column in the list exists in the dataframe.
    """
    def select(self,columns):
        listDel = []
        for col in self.dataset:
            if(not(col in columns)):
                listDel.append(col)
        self.delColumns(listDel)
    """
    Delete a list of columns
    """
    def delColumns(self,columns):
        self._checkOrigin()
        for col in columns:
            if(col in self.dataTypes):
                self.dataTypes.pop(col)
            if(col in self.columnManagers):
                self.columnManagers.pop(col)
            if(col in self.dataset):
                self.dataset.pop(col)
    """
    Del all Nan containing at least one NaN
    """
    def delNaNRows(self):
        self._checkOrigin()
        self.dataset = self.dataset.dropna()
    """ 
    Collect all dates types columns.
    """
    def getDatesColumns(self):
        self._checkOrigin()
        t = self.typesSummary()[0]
        return t["timedelta"].tolist()+t["datetimetz"].tolist()
    """
    Infers types and store it for each columns. Necessary! Automatically called
    by __init__.
    """
    def setInfos(self):
        """
        Infer types of all columns in order to verify before each process.
        """
        # "float":[],"int":[],"object":[],"datetime":[],"timedelta":[],"datetimetz":[]
        ty = self.typesSummary()[0]
        for column_name in ty["float"]:
            dataInfo = DataInfo(dataType.NUMERICAL,dataNature.CONTINOUS)
            self.columnManagers[column_name]=_ColumnManager(column_name,True,False)
            self.dataTypes[column_name] = dataInfo
        for column_name in ty["int64"]:
            dataInfo = DataInfo(dataType.NUMERICAL,dataNature.DISCRETE)
            self.columnManagers[column_name]=_ColumnManager(column_name,False,True)
            self.dataTypes[column_name] = dataInfo
        for column_name in ty["object"]: # STRING : CHECK IF INT OR FLOAT BY TRYING A CAST
            try:
                copy = self.dataset[column_name].sample(frac=0.2,replace=False,random_state=1)
                if(pd.to_numeric(copy, downcast='int')):
                    dataInfo = DataInfo(dataType.STRING,dataNature.DISCRETE)
                    self.columnManagers[column_name]=_ColumnManager(column_name,False,True)
            except ValueError:
                try:
                    if(pd.to_numeric(copy, downcast="float")):
                        dataInfo = DataInfo(dataType.STRING,dataNature.CONTINOUS)
                        self.columnManagers[column_name]=_ColumnManager(column_name,False,False)
                except ValueError:
                    dataInfo = DataInfo(dataType.STRING,dataNature.DISCRETE)
                    self.columnManagers[column_name]=_ColumnManager(column_name,False,True)
            self.dataTypes[column_name] = dataInfo
        """
        for column_name in ty["datetime"]:
            dataInfo = DataInfo(dataType.DATE,dataNature.DISCRETE)
            self.columnManagers[column_name]=_ColumnManager(column_name,False,True,isDate=True)
            self.dataTypes[column_name] = dataInfo
        """
        for column_name in ty["timedelta"]:
            dataInfo = DataInfo(dataType.DATE,dataNature.CONTINOUS)
            self.columnManagers[column_name]=_ColumnManager(column_name,False,False,isDate=True)
            self.dataTypes[column_name] = dataInfo
        for column_name in ty["datetimetz"]:
            dataInfo = DataInfo(dataType.DATE,dataNature.CONTINOUS)
            self.columnManagers[column_name]=_ColumnManager(column_name,False,False,isDate=True)
            self.dataTypes[column_name] = dataInfo                     
            
        return
    """
    
    """
    def splitToNumerous(self,column_name):
        # indiquer que l'on passe de categorique discret a numerique discret      
        res = self.columnManagers[column_name]._splitToNumerous(self.dataset)
        self.dataTypes[column_name]._asdict()['type']=dataType.NUMERICAL
        return res
    """ 
    Apply splitToNumerous to a list of columns.
    """
    def splitSeveralToNumerous(self,column_list):
        col_dicts = {}
        for col in column_list:
            col_dicts[col] = self.splitToNumerous(col)
            print(col+" discretized.")
        return col_dicts
    """
    
    """
    def cut2Discretized(self,dico_nbins,overwrite=False):
        for col in dico_nbins :
            self.cut2Bins(col,dico_nbins[col],overwrite)
    def cut2Bins(self,column_name,nb_bin,overwrite=False):
        self.columnManagers[column_name]._cutIntoBins(nb_bin,self.isNormalized,self.dataset,overwrite)
        # indiquer que l'on passe de numerique continue a numerique discret
        self.dataTypes[column_name]._asdict()['nature']=dataNature.DISCRETE
        return   

    """ 
    Transforms datas into [0,1]-values 
    """
    def normalize_dataframe(self):
        self.isNormalized = True
        self.dataset[:] = (self.dataset-self.dataset.min())/(self.dataset.max()-self.dataset.min())
        return
    def string2Numerous(self,col_name):
        info_col = self.dataTypes[col_name]
        assert(info_col._asdict()['type'].value==dataType.STRING.value)
        man_col = self.columnManagers[col_name]
        if(info_col._asdict()['nature'].value==dataNature.DISCRETE.value):
            return man_col._splitToNumerous(self.dataset)
        elif(info_col._asdict()['nature'].value==dataNature.CONTINOUS.value):
            return man_col._stringToContinous(self.dataset)
        else :
            assert(False) # cant convert an unknown type
        return
    """
    =========== truncated PCA ==============
    Applies a kernel PCA and truncates to keep the most effective components.
    exp_var : percentage of variance explained by the components.
    max_cmp: maximum number of components. can not reach the max_var if max_cmp is reached.
    pl : True to plot (slow)
    """
    def truncated_PCA(self,max_cmp,exp_var=0.8,pl=False):
        model = KernelPCA(kernel='linear')
        fitted = model.fit_transform(self.dataset) # train the PCA model
        explained_variance = np.var(fitted, axis=0) # var explained for each component
        explained_variance_ratio = explained_variance / np.sum(explained_variance) # ratio of these vars
        varsum = np.cumsum(explained_variance_ratio)
        #print(varsum)
        if(explained_variance_ratio[0]>=exp_var):
            truncated_fitted = fitted[:,0]
        else:
            truncated_var = [x for x in varsum if x<=exp_var]
            print("PCA components var : ",truncated_var)
            truncated_fitted = fitted[:,0:len(truncated_var)]
            if(np.shape(truncated_fitted)[1]>max_cmp):
                truncated_fitted = truncated_fitted[:,0:max_cmp]
        if(pl):
            plt.figure()
            plt.plot(np.linspace(1,len(varsum),len(varsum)),varsum)
            plt.show()
        return truncated_fitted

    """ 
    Better use this after dimension reduction. (very slow)
    """
    def TSNE_plot_3D(self):
        embedded_data = TSNE(n_components=3).fit_transform(self.dataset.T)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(embedded_data[:,0],embedded_data[:,1],embedded_data[:,2])
        plt.show()

    """
    Return the percentage of NaN cells.
    """
    def NaNCount(self):
        taille = self.dataset.shape
        return self.dataset.isnull().sum().sum()/(taille[0]*taille[1])
    """
    Very stupid... The dataset should be sparsify while a transformation
    which require 'sparsness'is done. So it is not optimal at all.
    """
    def sparsify(self):
        self.columnsSave = []
        for col in self.dataset:
            self.columnsSave.append(col)
        self.dataset = sp.csc_matrix(self.dataset)
        return self.dataset
    
"""
Test if a value belong to a given bin.
"""
def belongBin(elmt,bins,index):
        return elmt>=bins[index] and elmt<bins[index+1]
"""
bins : list of upper bounds.
index : index of the bin
elmt : elmt to compare

Compare a value to a bin.
elmt<bins[index] => return -1
(elmt>bins[index+1]) or (elmt==bins[index+1] and index!=len(bins)-2) => 1
else => 0
"""
def compareBins(elmt,bins,index):
    assert(index<len(bins)-1)
    if(elmt<bins[index]):
        return -1
    elif( (elmt>bins[index+1]) or (elmt==bins[index+1] and index!=len(bins)-2)):
        return 1
    else :
        return 0
"""
Search the index of the bin which contains elmt.
"""
def dichotomySearch(elmt,bins):
    inf=0
    sup=len(bins)-1
    index = floor((inf+sup)/2)
    cmpr = compareBins(elmt,bins,index)
    while(cmpr!=0):
        if(cmpr>0):
            inf=index+1
        elif(cmpr<0):
            sup=index
        index=floor((inf+sup)/2)
        cmpr = compareBins(elmt,bins,index)
    return index

#=======================================================================================================================


""" 
#Example of code
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

#read the data with pandas lib
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataManager = DataManager(dataset)
dataManager.delNaNRows() # remove NaNs
assert(dataManager.NaNCount()==0)

# convert categorical to numerical, splitting into multiple rows
dataManager.splitSeveralToNumerous(['Origin'])
dataManager.normalize_dataframe()
print(dataManager.dataset.head())
dataManager.cut2Discretized({'MPG':3,'Cylinders':3,'Displacement':3,'Horsepower':3,'Weight':3,
                'Acceleration':3,'Model Year':3})
print(dataManager.dataset.head())
# remove NaNs
M = dataManager.sparsify()


fitted = truncated_PCA(dataset,np.shape(dataset)[1],0.95)
print(np.shape(fitted))
TSNE_plot_3D(fitted) # check if components are separated enough
"""