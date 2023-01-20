import seaborn as sns
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd

class Node():
    def __init__(self,name,value,probs,observed=False):
        self.name=name
        self.probs=probs
        self.parents=()
        self.children=()
        self.val=value
        self.observed=observed
        
    def add_parent(self,a):
        self.parents=(*self.parents,a)  
        
    def add_child(self,a):
        self.children=(*self.children,a)
        
    def __repr__(self):
        return self.name
    
    def __call__(self):
        if len(self.parents)==0:
            return 0
        values=list(map(lambda x:x.val,self.parents))
        prob=self.probs
        for val in values:
            prob=prob[val]
        if self.val:
            return np.log(prob)
        else:
            return np.log(1-prob)

class BN():
    def __init__(self):
        self.varibles=()
        self.names=()
    def add(self,node):
        node.id=len(self.names)
        self.varibles=(*self.varibles,node)
        self.names=(*self.names,node.name)
        return self
    def add_edge(self,a,b):
        object_1=self.varibles[self.names.index(a)]
        object_2=self.varibles[self.names.index(b)]
        object_1.add_child(object_2)
        object_2.add_parent(object_1)
        return self
    def __repr__(self):
        string="0"
        return string
    
    def markov_blancket(self,id):
        object_id=self.varibles[id]
        return (object_id,*object_id.parents,*object_id.children)
    def propability(self,idd):
        return sum(tuple(map(lambda x:x(),self.markov_blancket(idd))))
        
    def gather(self):
        return np.array(list(map(lambda x:x.val,self.varibles)))
    
    def scatter(self,values):
        for node,val in zip(self.varibles,values):
            node.val=val
            
    def gibbs_sampling(self,num,num_burn=0,thin_out=1):
        values_to_choose=[]
        for i,node in enumerate(self.varibles):
            if not node.observed:
                values_to_choose.append(i)
        for _ in tqdm(range(num_burn),"burn in"):
            id=np.random.choice(values_to_choose)
            self.varibles[id].val=0
            prob_1=np.exp(self.propability(id))
            self.varibles[id].val=1
            prob_2=np.exp(self.propability(id))
            self.varibles[id].val=np.random.choice([0,1],p=[prob_1/(prob_1+prob_2),prob_2/(prob_1+prob_2)])
        values=[]
        for i in tqdm(range(num),"sampling"):
            id=np.random.choice(values_to_choose)
            self.varibles[id].val=0
            prob_1=np.exp(self.propability(id))
            self.varibles[id].val=1
            prob_2=np.exp(self.propability(id))
            self.varibles[id].val=np.random.choice([0,1],p=[prob_1/(prob_1+prob_2),prob_2/(prob_1+prob_2)])
            if i%thin_out==0:
                values.append(self.gather()[values_to_choose])
        out=dict()
        values=np.stack(values,axis=0)
        for i,id,value in zip(range(len(values_to_choose)),values_to_choose,values):
            out[self.names[id]]=values[:,i]
        return out


def Rubin_Gelman(dictionary,estimate_d=False):
    odp={}
    for name in dictionary:
        arr=dictionary[name]
        N=arr.shape[1]
        M=arr.shape[0]
        means=np.mean(arr,axis=1)
        vars=np.var(arr,axis=1)
        mean_of_means=np.mean(arr)
        B=N*np.var(means,ddof=1)
        W=np.mean(vars)
        V=(N-1)/N*W+(M+1)/(M*N)*B
        if estimate_d:
            var_W=np.var(vars)
            var_B=2*B**2/(M-1)
            cov_s2x2=np.cov(vars,means**2)[0,1]
            cov_s2x=np.cov(vars,means)[0,1]
            VarV=(N-1)**2/N**2*var_W/M+(M+1)**2/(M*N)**2*var_B+2*(M+1)*(N-1)/(M*N**2)*N/M*(cov_s2x2-2*mean_of_means*cov_s2x)
            d=2*V/(VarV)
            odp[name]=np.sqrt((d+3)*V/(W*(d+1)))
        else:
            odp[name]=np.sqrt(V/W)
    return odp