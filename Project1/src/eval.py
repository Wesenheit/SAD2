import seaborn as sns
import numpy as np 
import torch
import anndata as an
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["text.usetex"]=True
plt.rcParams["test.font"]="Serif"
sns.set_theme()

data_dir="/home/mitra/Files/Math/SAD2/Project1/data"


def plot_first():
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    fig,axes=plt.subplots(2,1,figsize=(20,10))
    ax=axes.flatten()[0]
    sns.histplot(ax=ax,data=x_train.X.flatten())
    plt.show()