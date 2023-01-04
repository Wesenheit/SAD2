import torch
from pyro.optim import StepLR
import numpy as np
import seaborn as sns
from VAE import NegBinomialVAE,ZeroPoissVAE,log1p_layer,norm_layer
import anndata as an
from anndata.experimental.pytorch import AnnLoader
from matplotlib import pyplot as plt
import pyro
from pyro.infer import SVI, TraceMeanField_ELBO,JitTraceMeanField_ELBO,Trace_ELBO
from pyro.poutine import block
import warnings
import argparse
sns.set_theme()

warnings.filterwarnings("ignore")
plt.rcParams["text.usetex"]=True
plt.rcParams['font.family']="Serif"




data_dir="../data"

def train_custom_VAE(args):
    print("parameters: ",args)
    x_train=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_train.h5ad")
    x_test=an.read_h5ad(data_dir+"/SAD2022Z_Project1_GEX_test.h5ad")
    n=x_train.X.shape[0]
    n_test=x_test.X.shape[0]
    mean=np.mean(x_train.layers["counts"].toarray())
    std=np.std(x_train.layers["counts"].toarray())
    print("mean: {},std: {}".format(mean,std))
    VAE=NegBinomialVAE([5000,500,200,args.latent_size],[args.latent_size,200,5000],"cuda" if args.cuda else "cpu",preprocess=norm_layer(mean,std),gelu=True) # model parameters
    optim=StepLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': args.learning_rate}, 'step_size': args.step_size,'gamma':args.gamma}) 
    data_loader_train=AnnLoader(x_train,args.batch_size,True,use_cuda=args.cuda)
    data_loader_test=AnnLoader(x_test,args.batch_size,True,use_cuda=args.cuda)
    svi=SVI(VAE.model,VAE.guide,optim=optim,loss=TraceMeanField_ELBO())
    err_list=[]
    err_list_kl=[]
    err_list_test=[]
    err_list_test_kl=[]
    for i in range(args.num_epoche):
        err=0
        err_kl=0
        for data_batch in data_loader_train:
            err+=svi.step(data_batch.layers["counts"])
            err_kl+=VAE.kl_loss(data_batch.layers["counts"]).item()
        optim.step()
        if i%10==0:
            err_list_kl.append(err_kl/len(data_loader_train))
            err_list.append(err/n)
            err_test=0
            err_kl_test=0
            for data_batch in data_loader_test:
                err_test+=svi.evaluate_loss(data_batch.layers["counts"])
                err_kl_test+=VAE.kl_loss(data_batch.layers["counts"]).item()
            print("epoche: {}, train loss: {}, test loss: {}".format(i,err/n,err_test/n_test))
            err_list_test.append(err_test/n_test)
            err_list_test_kl.append(err_kl_test/len(data_loader_test))

        if i%args.checkpoint==0 and i>0:
            print("Saving model!")
            VAE.save(name="custom_VAE_{}.tc".format(args.latent_size))
    fig,axes=plt.subplots(3,1,figsize=(10,10))
    ax=axes.flatten()[0]
    sns.scatterplot(ax=ax,x=np.arange(0,len(err_list))*10,y=err_list,label="train error")
    sns.scatterplot(ax=ax,x=np.arange(0,len(err_list_test))*10,y=err_list_test,label="test error")
    ax.set_xlabel("-ELBO loss")

    ax=axes.flatten()[1]
    sns.scatterplot(ax=ax,x=np.arange(0,len(err_list))*10,y=err_list_kl,label="train error")
    sns.scatterplot(ax=ax,x=np.arange(0,len(err_list_test))*10,y=err_list_test_kl,label="test error")
    ax.set_xlabel("KL divergence loss")

    ax=axes.flatten()[2]
    sns.scatterplot(ax=ax,x=np.arange(0,len(err_list))*10,y=np.array(err_list)-np.array(err_list_kl),label="train error")
    sns.scatterplot(ax=ax,x=np.arange(0,len(err_list_test))*10,y=np.array(err_list_test)-np.array(err_list_test_kl),label="test error")
    ax.set_xlabel("reconstruction loss")

    plt.tight_layout()
    plt.savefig("custom_VAE_training_{}.png".format(args.latent_size))
    VAE.save(name="custom_VAE_{}.tc".format(args.latent_size))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="SAD 2, Project nr 1, training script for NegBinom VAE")
    parser.add_argument("-b","--batch-size",default=128,type=int)
    parser.add_argument("-n","--num-epoche",default=600,type=int)
    parser.add_argument("-lr","--learning-rate",default=1e-3,type=float)
    parser.add_argument("-c","--cuda",default=True,type=bool)
    parser.add_argument("-lt","--latent-size",default=100,type=int)
    parser.add_argument("-ss","--step-size",default=100,type=int)
    parser.add_argument("-ch","--checkpoint",default=200,type=int)
    parser.add_argument("-g","--gamma",default=0.6,type=float)
    train_custom_VAE(parser.parse_args())
