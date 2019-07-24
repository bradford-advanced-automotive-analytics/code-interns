# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 00:31:23 2019

@author: samue
"""

__version__ = 1.0
__autor__ = "Squillaci Samuel"

from anytree import Node, RenderTree
from anytree.exporter import DotExporter

"""
Inheritance of node class. Store data in nodes.
"""
class NodeAtt(Node):
    def __init__(self,name,content,parent=None):
        super().__init__(name,parent)
        self.content = content
    def __add__(self,other):
        return self.content + other.content
    def __str__(self):
        return self.content.__str__()
class AttributeInstance:
    def __init__(self,events,proba):
        self.events = events
        assert(proba>0 and proba <=1)
        self.proba = proba
    def __str__(self):
        return str(self.events) + " p = " + str(self.proba)
    def __add__(self,ai):
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
    
_neutral = AttributeInstance({},1)    

class ProfileTree:
    def __init__(self,df,col_interet=None):
        self.tree = _mineProfile(df,col_interet)
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

def _mineProfile(df,col_interet=None):
    if(col_interet!=None):
        col_dtc = df[col_interet].pop()
    root = NodeAtt("Profiles",_neutral)
    feuilles = [(df,root)]
    n_col = len(df.columns)
    flatten = lambda l: [item for sublist in l for item in sublist]
    while(n_col>0):
        if(n_col==1 and col_interet!=None):
            sliced = [_sliceCol(f,col=col_dtc) for f in feuilles]
        else:
            sliced = [_sliceCol(f) for f in feuilles]
        feuilles = flatten(sliced)
        n_col -= 1
    return root
""" val : value_counts """
def _cutOnColumnValues(df,col,val):
    d = []
    for v,count in val.items():
        d.append( (df[df[col] == v].drop(columns=[col]),AttributeInstance({col:v},count/sum(val.values))) )
    return d

def _sliceCol(f,col=None):
    df = f[0]
    par = f[1]
    val_counts = {x:df[x].value_counts()/df.shape[0] for x in df.columns}
    attribute_length = {col:el.size for col,el in val_counts.items()}
    shortest_column = min(attribute_length, key=attribute_length.get)
    feuilles = _cutOnColumnValues(df,shortest_column,val_counts[shortest_column])
    if(col!=None):
        idx = df.index
        freq = col.loc[idx].value_counts()
        list_fils = [(x[0],Profile(str(x[1].proba),x[1],freq,parent=par)) for x in feuilles]
    else :
        list_fils = [(x[0],NodeAtt(str(x[1].proba),x[1],parent=par)) for x in feuilles]
    return list_fils

def get_leaves(node):
    flatten = lambda l: [item for sublist in l for item in sublist]
    if not node.children:
        return [node]
    else :
        return flatten([get_leaves(c) for c in node.children])

def get_attI(node):
    flatten = lambda l: [item for sublist in l for item in sublist]
    con = lambda fils : node.content + fils
    if not node.children:
        return [node.content]
    else :
        c= list(map(con,flatten([get_attI(c) for c in node.children])))
        return c
    
class Profile(AttributeInstance):
    def __init__(self,events,proba,vecteur_freq_dtc):
        super().__init__(events,proba)
        self.dtc_freq = vecteur_freq_dtc
    def __add__(self,other):
        return self.dtc_freq + other.dtc_freq
    def __sub__(self,other):
        return self.dtc_freq - other.dtc_freq
    def __str__(self):
        return str(super())
    def __eq__(row,self):
        return self.compare(row)==0
def find(row,profiles,isSorted=False):
    if(not isSorted):
        profiles.sort()
    return dichotomySearch(profiles,AttributeInstance(row.to_dict(),1))
    
def _supEqualDico(dico1,dico2):
    assert(len(dico1)==len(dico2))
    for k in sorted(dico1.keys()):
        v = dico1[k]
        if(v>dico2[k]):
            return True
        elif(v<dico2[k]):
            return False
    return True
def _supDico(dico1,dico2):
    assert(len(dico1)==len(dico2))
    for k in sorted(dico1.keys()):
        v = dico1[k]
        if(v>dico2[k]):
            return True
        elif(v<dico2[k]):
            return False
    return False
def _equalDico(dico1,dico2):
    assert(len(dico1)==len(dico2))
    for k in sorted(dico1.keys()):
        v = dico1[k]
        if(v!=dico2[k]):
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
    while(l_elmt!=x):
        if(l_elmt<x):
            inf=index+1
        elif(l_elmt>x):
            sup=index-1
        index=(inf+sup)//2
        l_elmt = liste[index]
        if(inf==sup and l_elmt!=x):
            raise ValueError
    return index
