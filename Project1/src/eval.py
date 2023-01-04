import seaborn as sns
import numpy as np 
import torch
import anndata as an
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from anndata.experimental.pytorch import AnnLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from numpy.random import default_rng


plt.rcParams["text.usetex"]=True
plt.rcParams['font.family']="Serif"
sns.set_theme()

data_dir="../data"


def plot_first():
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    fig,axes=plt.subplots(2,2,figsize=(8.3,6))
    ax=axes.flatten()[0]
    sns.histplot(ax=ax,data=x_train.X.toarray().flatten(),label="processed data",bins=10,binrange=(0,3),color="red",alpha=0.5)
    ax.legend()
    ax=axes.flatten()[1]
    sns.histplot(ax=ax,data=x_train.layers['counts'].toarray().flatten(),label="raw data",bins=10,binrange=(0,20),color="orange",alpha=0.5)
    ax.legend()
    print(np.sum(x_train.X.toarray()==0)/(x_train.X.shape[0]*x_train.X.shape[1]),x_train.X.shape)


    ax=axes.flatten()[2]
    data=x_train.X.copy().toarray()
    data=data[data>0].flatten()
    print(data.shape,np.mean(data),np.std(data))
    sns.histplot(ax=ax,data=data,label="processed data",bins=100,color="red",alpha=0.5,binrange=(np.quantile(data,0.16),np.quantile(data,0.84)))
    ax.legend()
    ax.set_xlabel("Histogram without zero values")


    ax=axes.flatten()[3]
    data=x_train.layers["counts"].copy().toarray()
    data=data[data>0].flatten()
    print(data.shape,np.mean(data),np.std(data))
    sns.histplot(ax=ax,data=data,label="raw data",bins=100,color="orange",alpha=0.5,binrange=(0,10))
    ax.legend()
    ax.set_xlabel("Histogram without zero values")
    plt.tight_layout()
    plt.savefig("task1_histogram.png",dpi=500)

def plot_pca(name_model,name_to_save,subsample_number=None):
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    x_test=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_test.h5ad")
    model=torch.load(name_model)
    model.eval()
    sample_arr=np.zeros([0,model.latent_size])
    sample_arr_test=np.zeros([0,model.latent_size])
    data_loader_train=AnnLoader(x_train,1000,True,use_cuda=True if model.device=="cuda" else False)
    data_loader_test=AnnLoader(x_test,1000,True,use_cuda=True if model.device=="cuda" else False)
    for data_batch in data_loader_train:
        samples=model.embed(data_batch.X).cpu().detach().numpy()
        sample_arr=np.concatenate([sample_arr,samples],axis=0)
    
    for data_batch in data_loader_test:
        samples=model.embed(data_batch.X).cpu().detach().numpy()
        sample_arr_test=np.concatenate([sample_arr_test,samples],axis=0)
    pca=PCA().fit(sample_arr)
    print("data for model: {}".format(name_model))
    print("explained variance ratio: ",pca.explained_variance_ratio_)
    print("95% of variance after: {}".format(1+np.argmax(np.cumsum(pca.explained_variance_ratio_)>0.95)))
    values=pca.transform(sample_arr_test)[:,:2]
    obs=x_test.obs.cell_type.values
    plt.close()
    fig=plt.figure(figsize=(6,4))
    ax=plt.gca()
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Mapping of test set")
    if subsample_number is None:
        sns.scatterplot(ax=ax,x=values[:,0],y=values[:,1],hue=obs,legend=False)
    else:
        rng = default_rng()
        set_ind = rng.choice(x_test.X.shape[0], size=subsample_number, replace=False)
        sns.scatterplot(ax=ax,x=values[set_ind,0],y=values[set_ind,1],hue=obs[set_ind],legend=False)
    #lower=np.quantile(values,0.0001,axis=0)
    #uper=np.quantile(values,0.9999,axis=0)
    #ax.set_xlim(lower[0],uper[0])
    #ax.set_ylim(lower[1],uper[1])
    plt.tight_layout()
    plt.savefig(name_to_save,dpi=500)

def plot_pca_3d(name_model,name_to_save,subsample_number=None):
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    model=torch.load(name_model)
    model.eval()
    sample_arr=np.zeros([0,model.latent_size])
    data_loader_train=AnnLoader(x_train,1000,True,use_cuda=True if model.device=="cuda" else False)
    for data_batch in data_loader_train:
        samples=model.embed(data_batch.X).cpu().detach().numpy()
        sample_arr=np.concatenate([sample_arr,samples],axis=0)
    pca=PCA().fit(sample_arr)
    print("data for model: {}".format(name_model))
    print("explained variance ratio: ",pca.explained_variance_ratio_)
    print("95% of variance after: {}".format(1+np.argmax(np.cumsum(pca.explained_variance_ratio_)>0.95)))
    values=pca.transform(sample_arr)[:,:3]
    obs=x_train.obs.cell_type.cat.codes
    plt.close()
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    ax.set_title("PCA of train set")
    if subsample_number is None:
        sns.scatter3D(ax=ax,x=values[:,0],y=values[:,1],z=values[:,2],hue=obs,legend=False)
    else:
        set_ind=np.random.randint(0,len(values),subsample_number)
        ax.scatter(xs=values[set_ind,0],ys=values[set_ind,1],zs=values[set_ind,2],c=obs[set_ind])
    plt.tight_layout()
    plt.savefig(name_to_save)


def plot_tsne(name_model,name_to_save,subsample_number):
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    model=torch.load(name_model)
    model.eval()
    sample_arr=np.zeros([0,model.latent_size])
    data_loader_train=AnnLoader(x_train,1000,True,use_cuda=True if model.device=="cuda" else False)
    for data_batch in data_loader_train:
        samples=samples=model.embed(data_batch.X).cpu().detach().numpy()
        sample_arr=np.concatenate([sample_arr,samples],axis=0)
    

    values=TSNE(n_components=2, learning_rate='auto',init='random',perplexity=100).fit_transform(sample_arr)
    obs=x_train.obs.cell_type.values

    fig=plt.figure()
    ax=plt.gca()
    if subsample_number is None:
        sns.scatterplot(x=values[:,0],y=values[:,1],hue=obs,legend=False)
    else:
        set_ind=np.random.randint(0,len(values),subsample_number)
        sns.scatterplot(ax=ax,x=values[set_ind,0],y=values[set_ind,1],hue=obs[set_ind],legend=False)
    #plt.show()
    plt.savefig(name_to_save)

def compare_pca(name1,name2,name_to_save,names,subsample_number):
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    x_test=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_test.h5ad")
    fig,axis=plt.subplots(1,2,figsize=(7,4))
    name_list=[name1,name2]
    rng = default_rng()
    set_ind = rng.choice(x_test.X.shape[0], size=subsample_number, replace=False)
    for name,i,name_title in zip(name_list,range(2),names):
        model=torch.load(name)
        model.eval()
        sample_arr=np.zeros([0,model.latent_size])
        sample_arr_test=np.zeros([0,model.latent_size])
        data_loader_train=AnnLoader(x_train,10000,True,use_cuda=True if model.device=="cuda" else False)
        data_loader_test=AnnLoader(x_test,10000,True,use_cuda=True if model.device=="cuda" else False)
        for data_batch in data_loader_train:
            samples=model.embed(data_batch.X).cpu().detach().numpy()
            sample_arr=np.concatenate([sample_arr,samples],axis=0)
    
        for data_batch in data_loader_test:
            samples=model.embed(data_batch.X).cpu().detach().numpy()
            sample_arr_test=np.concatenate([sample_arr_test,samples],axis=0)
        pca=PCA().fit(sample_arr)
        print("data for model: {}".format(name))
        print("explained variance ratio: ",pca.explained_variance_ratio_)
        print("95% of variance after: {}".format(1+np.argmax(np.cumsum(pca.explained_variance_ratio_)>0.95)))
        values=pca.transform(sample_arr_test)[:,:2]
        obs=x_test.obs.cell_type.values
        ax=axis.flatten()[i]
        #ax.set_xlabel("PCA 1")
        #ax.set_ylabel("PCA 2")
        ax.set_title(name_title)
        sns.scatterplot(ax=ax,x=values[set_ind,0],y=values[set_ind,1],hue=obs[set_ind],legend=False)
    plt.tight_layout()
    plt.savefig(name_to_save,dpi=500)

def compare_pca_ext(name1,name2,name_to_save,names,subsample_number):
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    x_test=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_test.h5ad")
    fig,axis=plt.subplots(3,2,figsize=(7,10))
    name_list=[name1,name2]
    rng = default_rng()
    set_ind = rng.choice(x_test.X.shape[0], size=subsample_number, replace=False)
    for name,i,name_title in zip(name_list,range(2),names):
        model=torch.load(name)
        model.eval()
        sample_arr=np.zeros([0,model.latent_size])
        sample_arr_test=np.zeros([0,model.latent_size])
        data_loader_train=AnnLoader(x_train,10000,True,use_cuda=True if model.device=="cuda" else False)
        data_loader_test=AnnLoader(x_test,10000,True,use_cuda=True if model.device=="cuda" else False)
        for data_batch in data_loader_train:
            samples=model.embed(data_batch.X).cpu().detach().numpy()
            sample_arr=np.concatenate([sample_arr,samples],axis=0)
    
        for data_batch in data_loader_test:
            samples=model.embed(data_batch.X).cpu().detach().numpy()
            sample_arr_test=np.concatenate([sample_arr_test,samples],axis=0)
        pca=PCA().fit(sample_arr)
        print("data for model: {}".format(name))
        print("explained variance ratio: ",pca.explained_variance_ratio_)
        print("95% of variance after: {}".format(1+np.argmax(np.cumsum(pca.explained_variance_ratio_)>0.95)))
        values=pca.transform(sample_arr_test)[:,:2]
        obs_batch=x_test.obs.batch.values
        ax=axis.flatten()[i]
        if i==0:
            ax.set_ylabel("Color by batch")
        ax.set_title(name_title)
        sns.scatterplot(ax=ax,x=values[set_ind,0],y=values[set_ind,1],hue=obs_batch[set_ind],legend=False)
        ax=axis.flatten()[i+2]
        if i==0:
            ax.set_ylabel("Color by donor ID")
        obs_id=x_test.obs.DonorID.values
        sns.scatterplot(ax=ax,x=values[set_ind,0],y=values[set_ind,1],hue=obs_id[set_ind],legend=False)
        ax=axis.flatten()[i+4]
        if i==0:
            ax.set_ylabel("Color by site")
        obs_site=x_test.obs.Site.values
        sns.scatterplot(ax=ax,x=values[set_ind,0],y=values[set_ind,1],hue=obs_site[set_ind],legend=False)
    plt.tight_layout()
    plt.savefig(name_to_save,dpi=500)


def plot_sum(name_model,name_to_save,subsample_number=None):
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    x_test=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_test.h5ad")
    model=torch.load(name_model)
    model.eval()
    sample_arr=np.zeros([0,model.latent_size])
    sample_arr_test=np.zeros([0,model.latent_size])
    data_loader_train=AnnLoader(x_train,1000,True,use_cuda=True if model.device=="cuda" else False)
    data_loader_test=AnnLoader(x_test,1000,True,use_cuda=True if model.device=="cuda" else False)
    for data_batch in data_loader_train:
        samples=model.embed(data_batch.X).cpu().detach().numpy()
        sample_arr=np.concatenate([sample_arr,samples],axis=0)
    
    for data_batch in data_loader_test:
        samples=model.embed(data_batch.X).cpu().detach().numpy()
        sample_arr_test=np.concatenate([sample_arr_test,samples],axis=0)
    pca=PCA().fit(sample_arr)
    print("data for model: {}".format(name_model))
    print("explained variance ratio: ",pca.explained_variance_ratio_)
    print("95% of variance after: {}".format(1+np.argmax(np.cumsum(pca.explained_variance_ratio_)>0.95)))
    values=pca.transform(sample_arr_test)[:,:2]
    obs=np.sum(x_test.layers["counts"].toarray(),axis=1)
    plt.close()
    fig=plt.figure(figsize=(6,4))
    ax=plt.gca()
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Mapping of test set")
    if subsample_number is None:
        sns.scatterplot(ax=ax,x=values[:,0],y=values[:,1],hue=obs,legend=False)
    else:
        rng = default_rng()
        set_ind = rng.choice(x_test.X.shape[0], size=subsample_number, replace=False)
        sns.scatterplot(ax=ax,x=values[set_ind,0],y=values[set_ind,1],hue=obs[set_ind],legend=False)
    #lower=np.quantile(values,0.0001,axis=0)
    #uper=np.quantile(values,0.9999,axis=0)
    #ax.set_xlim(lower[0],uper[0])
    #ax.set_ylim(lower[1],uper[1])
    plt.tight_layout()
    plt.savefig(name_to_save,dpi=500)
if __name__=="__main__":
    plot_first()
    plot_sum("batch_VAE_100.tc","test.png",500)
    compare_pca("vanila_VAE_100.tc","custom_VAE_100.tc","compare.png",["PCA of latent space - Gaussian VAE","PCA of latent space - Negative Binomial VAE"],5000//2)
    compare_pca_ext("vanila_VAE_100.tc","custom_VAE_100.tc","compare_ext.png",["PCA of latent space - Gaussian VAE","PCA of latent space - Negative Binomial VAE"],5000//2)
    #plot_pca_3d("vanila_VAE_120.tc","gaussian_3dpca_120.png",5000)
    plot_pca("vanila_VAE_50.tc","gaussian_pca_50.png",2500)
    plot_pca("vanila_VAE_70.tc","gaussian_pca_70.png",5000//2)
    plot_pca("vanila_VAE_100.tc","gaussian_pca_100.png",5000//2)
    plot_pca("vanila_VAE_120.tc","gaussian_pca_120.png",5000//2)
    plot_pca("vanila_VAE_150.tc","gaussian_pca_150.png",5000//2)
    plot_pca("batch_VAE_70.tc","negbinom_batch_pca_70.png",5000//2)
    plot_pca("batch_VAE_100.tc","negbinom_batch_pca_100.png",5000//2)
    plot_pca("batch_VAE_120.tc","negbinom_batch_pca_120.png",5000//2)
    plot_pca("custom_VAE_70.tc","negbinom_pca_70.png",5000//2)
    plot_pca("custom_VAE_100.tc","negbinom_pca_100.png",5000//2)
    plot_pca("custom_VAE_120.tc","negbinom_pca_120.png",5000//2)
    plot_tsne("batch_VAE_100.tc","batch_tsne_100.png",2000)