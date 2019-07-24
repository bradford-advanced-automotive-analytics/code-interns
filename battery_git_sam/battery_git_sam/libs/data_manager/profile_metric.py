# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 00:31:23 2019

@author: samue
"""
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import pandas as pd
import copy
from sklearn.cluster import KMeans
import numpy as np
"""
Inheritance of node class. Store data in nodes.
"""
class NodeAtt(Node):
    def __init__(self,name,content,parent=None):
        super().__init__(name,parent)
        self.content = content
    def __mul__(self,other):
        return self.content * other.content
    def __str__(self):
        return self.content.__str__()
class AttributeInstance:
    def __init__(self,events,proba):
        self.events = events
        assert(proba>0 and proba <=1)
        self.proba = proba
    def __str__(self):
        return str(self.events) + " p = " + str(self.proba)
    def __mul__(self,ai):
        d = {}
        for tmp in (self.events,ai.events) : d.update(tmp)
        return AttributeInstance(d,self.proba*ai.proba)
    def compare(self,row):
        cmp = 0
        for k,v in self.events.items():
            if(row[k]!=v):
                cmp += 1
        return cmp
    def __lt__(self,other):
        return  _infDico(self.events,other.events)
    def __le__(self,other):
        return _infEqualDico(self.events,other.events)
    def __eq__(self, other) :
        return _equalDico(self.events,other.events)
    def __ne__(self, other):
        return _difDico(self.events,other.events)
    def __ge__(self, other) :
        return _supEqualDico(self.events,other.events)
    def __gt__(self, other) :
        return _supDico(self.events,other.events)
    def belong(self,other):
        for k,v in self.events.items():
            if not set(v).issubset(other.events[k]):
                return False
        return True
class Profile(AttributeInstance):
    def __init__(self,events,proba,vecteur_freq_dtc=np.array([])):
        super().__init__(events,proba)
        self.dtc_freq = vecteur_freq_dtc
    def __add__(self,other):
        p = copy.deepcopy(self)
        if(other is None):
            return p
        p.dtc_freq += other.dtc_freq
        p.proba += other.proba
        for k,l in p.events.items():
            p.events[k]+=other.events[k].copy()
            del other.events[k]
        assert(len(list(other.events.keys()))==0)
        return p
    def __str__(self):
        return str(super())
    def __eq__(row,self):
        return self.compare(row)==0
    def merge(self,other):
        return self + other
    def setFreq(self,vect):
        self.dtc_freq = vect
        return self
    
_neutral = AttributeInstance({},1)    
_neutralProfile = Profile({},1,None)

class ProfileTree:
    def __init__(self,df,metricClass=None,metricCols=None):
        if(metricClass!=None and metricCols!=None):
            metric = metricClass(df,metricCols)
        else :
            metric = None
        self.tree = _mineProfile(df,metric)
    def __str__(self):
        return str(self.tree)
    #$ dot tree.dot -T png -o tree.png
    def export(self,dot_name):
        DotExporter(self.tree).to_dotfile(dot_name)
    def disp(self):
        for pre, fill, node in RenderTree(self.tree):
            print("%s%s" % (pre, node.name))
    def getProfiles(self):
        return get_attI(self.tree)

""" top level class : abstract """
class Metric(object):
    def __init__(self,df,cols):
        self.cols = copy.deepcopy(cols)
        self.elmt = df[cols].copy()        
        df.drop(cols,axis=1,inplace=True)
        print("metric set shape : "+str(self.elmt.shape)+" ; dataframe\\metric set : "+str(df.shape))
    def measure(self,df):
        return 0
    def majIndex(self,idx):
        self.elmt = self.elmt[idx]
    
def _mineProfile(df,metric=None):
    if(metric==None):
        root = NodeAtt("Profiles",_neutral)
    else :
        root = NodeAtt("Profiles",_neutralProfile)
    feuilles = [(df,root,metric)]
    n_col = len(df.columns)
    flatten = lambda l: [item for sublist in l for item in sublist]
    while(n_col>0):
        sliced = [_sliceCol(f) for f in feuilles]
        feuilles = flatten(sliced)
        n_col -= 1
    return root
""" val : value_counts """
def _cutOnColumnValues(df,col,val,metric):
    d = []
    for v,count in val.items():
        idx = df[col] == v
        dfi = df[df[col] == v].drop(columns=[col])
        met = copy.deepcopy(metric)
        if(met!=None):
            met.majIndex(idx)
            d.append( (dfi,Profile({col:[v]},count/sum(val.values)),met) )
        else :
            d.append( (dfi,AttributeInstance({col:[v]},count/sum(val.values)),met) )
    return d

def pause():
    input("Press enter to continue")
    
def _sliceCol(f,nbClustersMax = 10):
    df = f[0]
    par = f[1]
    metric = f[2]
    val_counts = {x:df[x].value_counts()/df.shape[0] for x in df.columns}
    attribute_length = {col:el.size for col,el in val_counts.items()}
    shortest_column = min(attribute_length, key=attribute_length.get)
    feuilles = _cutOnColumnValues(df,shortest_column,val_counts[shortest_column],metric)
    #print(shortest_column,[f[0].shape[0] for f in feuilles],"elements")
    if(metric!=None):
        list_fils = [(x[0],NodeAtt(str(x[1].proba),x[1].setFreq(x[2].measure(x[2].elmt)),parent=par),x[2]) for x in feuilles]
        assert(nbClustersMax>1)
        if(len(list_fils)>nbClustersMax*3):
            measures = np.array([x[1].content.dtc_freq for x in list_fils])
            kmeans = KMeans(n_clusters=nbClustersMax, random_state=0).fit(measures)
            lab = kmeans.labels_
            new_fils = [None]*nbClustersMax
            for ind,el in enumerate(list_fils):
                new_fils[lab[ind]] = _mergeTupple(el,new_fils[lab[ind]])
            #print(" : ",len(new_fils),"children")
            assert(len(new_fils)==nbClustersMax)
            assert(len(par.children)<=nbClustersMax)
            new_fils = [x for x in new_fils if x is not None]
            #print("events",[x[1].content.events for x in new_fils if x is not None and x[1].content is not None])
            return new_fils
        else :
             return list_fils
    else :
        return [(x[0],NodeAtt(str(x[1].proba),x[1],parent=par),metric) for x in feuilles]

def get_leaves(node):
    flatten = lambda l: [item for sublist in l for item in sublist]
    if not node.children:
        return [node]
    else :
        return flatten([get_leaves(c) for c in node.children])

def get_attI(node):
    flatten = lambda l: [item for sublist in l for item in sublist]
    con = lambda fils : node.content * fils
    if not node.children:
        return [node.content]
    else :
        c= list(map(con,flatten([get_attI(c) for c in node.children])))
        return c
""" t=(dataset,node,metric). t2 may be None """
def _mergeTupple(t1,t2):
    if(t2==None):
        return t1
    new_metric = t1[2]
    if(t2[2]!=None):
        new_metric.elmt = pd.concat([new_metric.elmt,t2[2].elmt])
    t = (pd.concat([t1[0],t2[0]]),_mergeNodes(t1[1],t2[1]),new_metric)
    return t
""" MERGE 2 NODES, 2nd may be None """
def _mergeNodes(node1,node2):
    assert(node1.parent==node2.parent)
    if(node2==None):
        return node1
    node1.content = node1.content.merge(node2.content)
    flatten = lambda l: "".join(l)
    splitted_name1 = node1.name.split("/")
    splitted_name2 = node2.name.split("/")
    node1.name = flatten( splitted_name1[0:-1] )+str(float(splitted_name1[-1])+float(splitted_name2[-1]))
    if(node2 is not None):
        tmp = list(node2.parent.children)
        tmp.remove(node2)
        node2.parent.children = tuple(tmp)
    node2 = None
    return node1
       
def find(row,tree):
    current_node = tree
    while(current_node.children != None and len(current_node.children)>0):
        # only one matches so I can take [0]
        m = [_match(child,row) for child in current_node.children]
        current_node = [x for x in m if x!=None][0]
    return current_node.content
def _eq_list(l1,l2):
    l1.sort()
    l2.sort() 
    return l1==l2
def _ne_list(l1,l2):
    return not _eq_list(l1,l2)
def _lt_list(l1,l2):
    l1.sort()
    l2.sort() 
    # act like there is an infinite list of -1 at the end of each list
    for i in range(min(len(l1),len(l2))):
        if(l1[i]>=l2[i]):
            return False
    return len(l1)<=len(l2)
def _gt_list(l1,l2):
    l1.sort()
    l2.sort()
    for i in range(min(len(l1),len(l2))):
        if(l1[i]<=l2[i]):
            return False
    return len(l1)>=len(l2)

def _supEqualDico(dico1,dico2):
    #assert(len(dico1.keys())==len(dico2.keys()))
    for k in sorted(dico1.keys()):
        v = dico1[k]
        if(_gt_list(v,dico2[k])):
            return True
        elif(_lt_list(v,dico2[k])):
            return False
    return True
def _supDico(dico1,dico2):
    #assert(len(dico1.keys())==len(dico2.keys()))
    for k in sorted(dico1.keys()):
        v = dico1[k]
        if(_gt_list(v,dico2[k])):
            return True
        elif(_lt_list(v,dico2[k])):
            return False
    return False

def _equalDico(dico1,dico2):
    #assert(len(dico1.keys())==len(dico2.keys()))
    if(_ne_list(list(dico1.keys()),list(dico2.keys()))):
        return False
    for k in dico1.keys():
        if(_ne_list(dico1[k],dico2[k])):
            return False
    return True
def _infDico(dico1,dico2):
    return _supEqualDico(dico2,dico1)
def _infEqualDico(dico1,dico2):
    return _supDico(dico2,dico1)
def _difDico(dico1,dico2):
    return not _equalDico(dico1,dico2)
def dichotomySearch(liste,x):
    inf=0
    sup=len(liste)-1
    index = (inf+sup)//2
    l_elmt = liste[index]
    print(x)
    while(x!=l_elmt):
        if(l_elmt.__lt__(x)):
            inf=index+1
        elif(l_elmt.__gt__(x)):
            sup=index-1
        index=(inf+sup)//2
        l_elmt = liste[index]
        if(inf==sup and l_elmt!=x):
            raise ValueError
    return index
def apply_profiles(dataset,profileTree,metric_cols):
    return (dataset.drop(columns=metric_cols)).apply(lambda r : find(r,profileTree.tree), axis=1)

def _match(node,profile_dico):
    if(node.parent == None):
        return node # root
    for att,value in profile_dico.items():
        if(att in list(node.content.events.keys()) and not value in node.content.events[att]):
            return None
    return node