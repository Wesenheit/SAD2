import seaborn as sns
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import pandas as pd
from Gibbs_sampler import Node, BN,Rubin_Gelman
import matplotlib.pyplot as plt
np.random.seed(42)

def main():
    net=BN()
    rain=Node("rain",0,[0.2,0.8],False)
    sprinkler=Node("sprinkler", 1,[0.5,0.1],True)
    cloud=Node("cloud",0,None,False)
    grass=Node("wet grass",1,np.array([[0.01,0.9],[0.9,0.99]]),True)
    net.add(rain).add(sprinkler).add(cloud).add(grass).add_edge("cloud","sprinkler").add_edge("cloud","rain").add_edge("rain","wet grass").add_edge("sprinkler","wet grass")
    print(net.varibles)

    samples=net.gibbs_sampling(100,0,1)
    print("initaial estimation: ",np.mean(samples["rain"]))

    num=50000
    samples=net.gibbs_sampling(num,0,1)
    fig,[ax1,ax2]=plt.subplots(2,1,figsize=(6,4))
    freq_1=np.cumsum(samples["rain"])/(np.arange(0,num)+1)
    freq_2=np.cumsum(samples["cloud"])/(np.arange(0,num)+1)
    ax1.plot(freq_1,c="black")
    ax1.grid()
    ax1.set_xlabel(r"$t$")
    ax1.set_title(r"Rain")
    ax2.plot(freq_2,c="black")
    ax2.grid()
    ax2.set_xlabel(r"$t$")
    ax2.set_title(r"Cloud")
    plt.tight_layout()
    plt.savefig("plot_5.png",dpi=500)
    num_lags=100
    correlation_rain=np.zeros(num_lags)
    correlation_cloud=np.zeros(num_lags)
    for i in range(num_lags):
        lag_rain=np.roll(samples["rain"],-i)
        lag_cloud=np.roll(samples["cloud"],-i)
        correlation_cloud[i]=np.corrcoef(lag_cloud[:num-i],samples["cloud"][:num-i])[0,1]
        correlation_rain[i]=np.corrcoef(lag_rain[:num-i],samples["rain"][:num-i])[0,1]
    
    fig,[ax1,ax2]=plt.subplots(2,1,figsize=(6,4))
    ax1.plot(correlation_rain,c="black")
    ax1.grid()
    ax1.set_xlabel(r"$k$")
    ax1.set_title(r"Rain")
    ax2.plot(correlation_cloud,c="black")
    ax2.grid()
    ax2.set_xlabel(r"$k$")
    ax2.set_title(r"Cloud")
    plt.tight_layout()
    plt.savefig("plot_6.png",dpi=500)
    thin=50
    burn=10000
    num=thin*100

    samples=net.gibbs_sampling(num,burn,thin)
    print("final estimation: ",np.mean(samples["rain"]))

    M=40
    values={}
    for node in net.varibles:
        if not node.observed:
            values[node.name]=[]
    for _ in range(M):
        samples=net.gibbs_sampling(num,burn,thin)
        for name in samples:
            values[name].append(samples[name])
    for name in values:
        values[name]=np.stack(values[name],axis=0)
        print(values[name].shape)
    
    test_rubin=Rubin_Gelman(values)
    print("RG estimate for final estimate: ",test_rubin)


    values={}
    for node in net.varibles:
        if not node.observed:
            values[node.name]=[]
    for _ in range(M):
        samples=net.gibbs_sampling(100,0,1)
        for name in samples:
            values[name].append(samples[name])
    for name in values:
        values[name]=np.stack(values[name],axis=0)
        print(values[name].shape)
    
    
    test_rubin=Rubin_Gelman(values)
    print("RG value for first estimate: ",test_rubin)
if __name__=="__main__":
    main()
