import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image
import pyro
from pyro import distributions as dst


class log1p_layer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return torch.log1p(x)


class norm_layer(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.mean=mean
        self.std=std
    def forward(self,x):
        return (x-self.mean)/self.std

class Encoder(nn.Module):
    def __init__(self,sizes,preproces=None,gelu=False):
        super().__init__()
        self.latent_size=sizes[-1]
        self.network=[]
        if preproces is not None:
            self.network.append(preproces)
        for i in range(len(sizes)-2):
            self.network.append(nn.Linear(sizes[i],sizes[i+1]))
            self.network.append(nn.BatchNorm1d(sizes[i+1]))
            if gelu:
                self.network.append(nn.GELU())
            else:
                self.network.append(nn.ReLU())
            self.network.append(nn.Dropout(p=0.05))
        self.encode_mu=nn.Linear(sizes[-2],sizes[-1])
        self.encode_sigma=nn.Linear(sizes[-2],sizes[-1])
        self.network=nn.Sequential(*self.network)

    def forward(self,x):
        x=self.network(x)
        mu=self.encode_mu(x)
        sigma=nn.functional.relu(self.encode_sigma(x))+1e-5
        return mu,sigma



class Decoder(nn.Module):
    def __init__(self,sizes,gelu=False):
        super().__init__()
        self.network=[]
        for i in range(len(sizes)-2):
            self.network.append(nn.Linear(sizes[i],sizes[i+1]))
            self.network.append(nn.BatchNorm1d(sizes[i+1]))
            if gelu:
                self.network.append(nn.GELU())
            else:
                self.network.append(nn.ReLU())
            self.network.append(nn.Dropout(p=0.05))
        self.network.append(nn.Linear(sizes[-2],sizes[-1]))
        self.network=nn.Sequential(*self.network)

    def forward(self,x):
        return self.network(x)

class Decoder_Ext(nn.Module):
    def __init__(self,sizes,gelu=False):
        super().__init__()
        self.network=[]
        for i in range(len(sizes)-2):
            self.network.append(nn.Linear(sizes[i],sizes[i+1]))
            self.network.append(nn.BatchNorm1d(sizes[i+1]))
            if gelu:
                self.network.append(nn.GELU())
            else:
                self.network.append(nn.ReLU())
            self.network.append(nn.Dropout(p=0.05))
        self.network=nn.Sequential(*self.network)
        self.output=nn.Linear(sizes[-2],sizes[-1])
        self.param=nn.Linear(sizes[-2],sizes[-1])

    def forward(self,x):
        x=self.network(x)
        return self.output(x),self.param(x)

class TemplateVAE(nn.Module):
    def __init__(self,sizes1,sizes2,device="cpu",ext=False,preprocess=None,gelu=False):
        super().__init__()
        self.encoder=Encoder(sizes1,preproces=preprocess).to(device)
        if ext:
            self.decoder=Decoder_Ext(sizes2,gelu=gelu).to(device)
        else:
            self.decoder=Decoder(sizes2,gelu=gelu).to(device)
        self.latent_size=sizes1[-1]
        self.device=device

    def model(self,data):
        pass

    def guide(self,data):
        pyro.module("encoder",self.encoder)
        with pyro.plate("data",data.shape[0]):
            mu,sigma=self.encoder(data)
            #print(torch.max(sigma))
            pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))

    def sample(self,data,n_sample):
        samples=torch.zeros(data.shape[0],n_sample,self.latent_size,device=self.device)
        with torch.no_grad():
            mu,sigma=self.encoder(data)
            for i in range(n_sample):
                samples[:,i,:]=mu+sigma*torch.normal(0,1,mu.shape,device=self.device)
        return samples.view(data.shape[0]*n_sample,self.latent_size)
    
    def embed(self,data):
        with torch.no_grad():
            mu,sigma=self.encoder(data)
            return mu
        
    def kl_loss(self,data):
        with torch.no_grad():
            mu,sigma=self.encoder(data)
            return 0.5*torch.sum(mu.pow(2)+sigma.pow(2)-torch.log(sigma.pow(2))-1)/data.shape[0]   

    def save(self,name="template_VAE.tc"):
        torch.save(self, name)


class GaussianVAE(TemplateVAE):
    def __init__(self,sizes1,sizes2,device="cpu",scale=1,preprocess=None,gelu=False):
        super().__init__(sizes1,sizes2,device,preprocess=preprocess,gelu=gelu)
        self.scale=scale
    def model(self,data):
        pyro.module("decoder",self.decoder)
        with pyro.plate("data",data.shape[0]):
            mu = torch.zeros(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            sigma = torch.ones(torch.Size((data.shape[0], self.latent_size)),device=self.device)

            z_latent=pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
            mean=nn.functional.relu(self.decoder(z_latent))
            pyro.sample("observation",dst.Normal(loc=mean,scale=self.scale).to_event(1),obs=data)
    def save(self,name="vanila_VAE.tc"):
        torch.save(self, name)

class NegBinomialVAE(TemplateVAE):
    def __init__(self,sizes1,sizes2,device="cpu",preprocess=None,gelu=False):
        super().__init__(sizes1,sizes2,device,ext=True,preprocess=preprocess,gelu=gelu)
    def model(self,data):
        pyro.module("decoder",self.decoder)
        with pyro.plate("data",data.shape[0]):
            mu = torch.zeros(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            sigma = torch.ones(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            z_latent=pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
            total_count,logits=self.decoder(z_latent)
            
            pyro.sample("observation",dst.NegativeBinomial(total_count=nn.functional.relu(total_count)+1e-5,logits=logits).to_event(1),obs=data.int())
    def save(self,name="NegBinom_VAE.tc"):
        torch.save(self, name)


class ZeroPoissVAE(TemplateVAE):
    def __init__(self,sizes1,sizes2,device="cpu",preprocess=None):
        super().__init__(sizes1,sizes2,device,preprocess=preprocess)
        self.tot_count=nn.Linear(self.latent_size,sizes2[-1]).to(self.device)
    def model(self,data):
        pyro.module("decoder",self.decoder)
        pyro.module("decoder",self.tot_count)
        with pyro.plate("data",data.shape[0]):
            mu = torch.zeros(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            sigma = torch.ones(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            z_latent=pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
            logs=self.decoder(z_latent)
            total_count=self.tot_count(z_latent)
            pyro.sample("observation",dst.ZeroInflatedPoisson(gate_logits=total_count,rate=nn.functional.relu(logs)+1e-6).to_event(1),obs=data.int())
    def save(self,name="NegBinom_VAE.tc"):
        torch.save(self, name)


class NegBinomBatchVAE(TemplateVAE):
    def __init__(self,sizes1,sizes2,batch,device="cpu",preprocess=None,gelu=False):
        temp=sizes2
        temp[0]+=batch
        super().__init__(sizes1,temp,device,ext=True,preprocess=preprocess,gelu=gelu)
    def model(self,data,batch):
        pyro.module("decoder",self.decoder)
        with pyro.plate("data",data.shape[0]):
            mu = torch.zeros(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            sigma = torch.ones(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            z_latent=pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
            new=torch.cat((z_latent,batch),dim=1)
            total_count,logits=self.decoder(new)
            
            pyro.sample("observation",dst.NegativeBinomial(total_count=nn.functional.relu(total_count)+1e-5,logits=logits).to_event(1),obs=data.int())

    def guide(self,data,batch):
        pyro.module("encoder",self.encoder)
        with pyro.plate("data",data.shape[0]):
            mu,sigma=self.encoder(data)
            pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
    def save(self,name="NegBinomBatch_VAE.tc"):
        torch.save(self, name)

class ZeroPoissBatchVAE(TemplateVAE):
    def __init__(self,sizes1,sizes2,batch,device="cpu",preprocess=None):
        temp=sizes2
        temp[0]+=batch
        super().__init__(sizes1,temp,device,ext=True,preprocess=preprocess)
    def model(self,data,batch):
        pyro.module("decoder",self.decoder)
        with pyro.plate("data",data.shape[0]):
            mu = torch.zeros(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            sigma = torch.ones(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            z_latent=pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
            new=torch.cat((z_latent,batch),dim=1)
            total_count,logits=self.decoder(new)
            
            pyro.sample("observation",dst.ZeroInflatedPoisson(rate=nn.functional.relu(total_count)+1e-5,gate_logits=logits).to_event(1),obs=data.int())

    def guide(self,data,batch):
        pyro.module("encoder",self.encoder)
        with pyro.plate("data",data.shape[0]):
            mu,sigma=self.encoder(data)
            pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
    def save(self,name="ZeroPoissBatch_VAE.tc"):
        torch.save(self, name)


class GaussianBatchVAE(TemplateVAE):
    def __init__(self,sizes1,sizes2,batch,device="cpu"):
        temp=sizes2
        temp[0]+=batch
        super().__init__(sizes1,temp,device)
    def model(self,data,batch):
        pyro.module("decoder",self.decoder)
        with pyro.plate("data",data.shape[0]):
            mu = torch.zeros(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            sigma = torch.ones(torch.Size((data.shape[0], self.latent_size)),device=self.device)
            z_latent=pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
            new=torch.cat((z_latent,batch),dim=1)
            mean=self.decoder(new)
            
            pyro.sample("observation",dst.Normal(loc=mean,scale=1).to_event(1),obs=data)

    def guide(self,data,batch):
        pyro.module("encoder",self.encoder)
        with pyro.plate("data",data.shape[0]):
            mu,sigma=self.encoder(data)
            pyro.sample("latent",dst.Normal(mu,sigma).to_event(1))
    def save(self,name="GaussianBatch_VAE.tc"):
        torch.save(self, name)