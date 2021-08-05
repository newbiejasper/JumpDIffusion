import sys,os
import numpy as np
import scipy.stats as stats
from scipy import linalg
from numpy import dot
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import grad
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.modules import Linear
from torch.autograd.functional import jacobian,hessian,vjp,vhp,hvp

import random
import math

# random seed
def setup_seed(seed):
     torch.cuda.manual_seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark=False
     torch.backends.cudnn.deterministic = True
setup_seed(40)

## sample
def PopulationSample(path,time,samples_n,samples_total,in_features): 
    indices = list(range(samples_total)) 
    random.shuffle(indices) 
    train = torch.empty((samples_n,in_features),device="cuda")
    j = 0
    for i in indices[:samples_n]:
        train[j] = path[i][time]
        j += 1

    test = torch.empty((samples_total-samples_n,in_features),device="cuda")
    j = 0
    for i in indices[samples_n:]:
        test[j] = path[i][time]
        j += 1

    return train,test
    
def compute_gradient_penalty(D, real_sample, fake_sample,k,p):
    real_samples = real_sample.requires_grad_(True)
    fake_samples = fake_sample.requires_grad_(True)

    real_validity = D(real_samples)
    fake_validity = D(fake_samples)

    real_grad_out = torch.ones((real_samples.shape[0],1),dtype=torch.float32,requires_grad=False,device="cuda")
    real_grad = grad(
        real_validity, real_samples, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

    fake_grad_out = torch.ones((fake_samples.shape[0],1),dtype=torch.float32,requires_grad=False,device="cuda")
    fake_grad = grad(
        fake_validity, fake_samples, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

    return torch.mean(real_grad_norm + fake_grad_norm) * k / 2


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=64, num_hidden=0, activation=nn.Tanh(),addLast=False):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList() 
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        for m in self.linears:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias,0) 
 
        self.activation = activation 
        self.addLast = addLast

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))

        if self.addLast:
            return nn.Softplus()(self.linears[-1](x))

        return self.linears[-1](x)

# generator
class JumpEulerForwardCuda(nn.Module):

    def __init__(self,in_features,num_hidden,dim_hidden,step_size):
        super(JumpEulerForwardCuda,self).__init__()

        self.drift = MLP(in_features,in_features,dim_hidden,num_hidden,addLast=False)
        self.intensity = torch.tensor(40.0,device="cuda")
        self.rate = torch.tensor([10.0,1.0],device="cuda")
        
        self.diffusion = nn.Parameter(torch.tensor(1.0))
        self.in_features = in_features
        self.step_size = step_size

    def forward(self,z0,Nsim,steps):

        PopulationPath = torch.empty(size = (Nsim,steps+1,self.in_features),device="cuda")
        PopulationPath[:,0,:] = z0
        state = z0

        for i in range(1,steps+1):
            DP = D.poisson.Poisson(self.intensity*self.step_size) 
            pois = DP.sample((Nsim,1)).cuda()
            state = state + self.drift(state)*self.step_size + self.diffusion*math.sqrt(self.step_size)*torch.normal(0,1,size=(Nsim,self.in_features),device="cuda")+\
                torch.distributions.Gamma(pois,self.rate).sample().cuda()
            PopulationPath[:,i,:] = state
        return PopulationPath


# Simulated data
class Synthetic2D(nn.Module):

    def __init__(self,in_features,diffusion,intensity,step_size,mu1,mu2,S,index):
        super(Synthetic2D,self).__init__()

        self.diffusion = diffusion
        self.intensity = intensity
     
        self.in_features = in_features
        self.step_size = step_size
        self.index = index
        self.mu1 = mu1
        self.mu2 = mu2
        self.S = S

    def drift(self,x):
        if self.index == 1:
            xp = torch.tensor([x[0],x[0]],device="cuda")
            p = 1/(1+self.S[0]/self.S[1]*torch.exp((x[0]-self.mu1[0])**2/(2*self.S[0]**2)+(x[0]-self.mu1[1])**2/(2*self.S[0]**2)-(x[1]-self.mu2[0])**2/(2*self.S[1]**2)-(x[1]-self.mu2[1])**2/(2*self.S[1]**2)))
            return -(1/self.S[0]*p*(x-self.mu1)+1/self.S[1]*(1-p)*(xp-self.mu2))

    def forward(self,z0,steps):
        
        path = torch.empty(size = (steps+1,self.in_features),device="cuda")
        path[0] = z0
        state = z0
        ta = 0
        tb = 0
        tb += torch.distributions.Exponential(self.intensity).sample()
        i = 1
        while i<= steps:

            if tb <  i*self.step_size:
                state = state + self.drift(state)*(tb-ta) + math.sqrt(tb-ta)*self.diffusion*torch.normal(0,1,size=(self.in_features,),device="cuda")+\
                        torch.distributions.Exponential(torch.tensor([10.0,1.0])).sample().cuda()
                ta = tb
                tb = tb + torch.distributions.Exponential(self.intensity).sample()

            elif tb > i*self.step_size:
                state = state + self.drift(state)*(i*self.step_size-ta) + math.sqrt(i*self.step_size-ta)*self.diffusion*torch.normal(0,1,size=(self.in_features,),device="cuda")
                path[i] = state
                ta = i*self.step_size
                i += 1
            else:
                state = state + self.drift(state)*(tb-ta) + math.sqrt(tb-ta)*self.diffusion*torch.normal(0,1,size=(self.in_features,),device="cuda")+\
                        torch.distributions.Exponential(torch.tensor([10.0,1.0])).sample()
                path[i]  = state
                ta = i*self.step_size
                tb = tb + torch.distributions.Exponential(self.intensity).sample().cuda()
                i += 1

        return path

Synthetic = Synthetic2D(2,1,40,0.02,torch.tensor([16,12],device="cuda"),\
                       torch.tensor([-18,-10],device="cuda"),torch.tensor([1,0.95],device="cuda"),1).cuda()

init_state = D.MultivariateNormal(torch.zeros(2,device="cuda"),torch.eye(2,device="cuda")).sample((2000,))

PopulationPath = list(map(Synthetic,init_state,[10]*2000))

train0,test0 = PopulationSample(PopulationPath,0,1200,2000,2)
train3,test3 = PopulationSample(PopulationPath,3,1200,2000,2)
train6,test6 = PopulationSample(PopulationPath,6,1200,2000,2)
train10,test10 = PopulationSample(PopulationPath,10,1200,2000,2)


### training parameter
n_sims = 1200
n_steps = [3,6]
in_features = 2
step_size = 0.02

lrD3 = 0.0001
lrD6 = 0.0001
lrG = 0.00005

k = 2
p = 6

# generator and critic
netG = JumpEulerForwardCuda(2,2,32,0.02).cuda()
netD3 = MLP(2,1,dim_hidden=32,num_hidden=3).cuda()
netD6 = MLP(2,1,dim_hidden=32,num_hidden=3).cuda()

optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999))
optimizerSD3 = optim.Adam(netD3.parameters(), lr=lrD3, betas=(0.5, 0.999))
optimizerSD6 = optim.Adam(netD6.parameters(), lr=lrD6, betas=(0.5, 0.999))

n_epochs = 40000

w = [[],[]]
for epoch in range(n_epochs):
    
    start = time.perf_counter()
    
    # -------------------
    # critic
    # -------------------
    

    for _ in range(5):
        fake_data = netG(train0,n_sims,n_steps[1])
        fake3 = fake_data[:,3,:]
        fake6 = fake_data[:,6,:]

        optimizerSD3.zero_grad()

        div_gp3 = compute_gradient_penalty(netD3,train3,fake3,k,p)
        d3_loss = -torch.mean(netD3(train3))+torch.mean(netD3(fake3))+div_gp3
        d3_loss.backward(retain_graph=True) # retain_graph=True

        optimizerSD3.step()


        optimizerSD6.zero_grad()
        
        div_gp6 = compute_gradient_penalty(netD6,train6,fake6,k,p)
        d6_loss = -torch.mean(netD6(train6))+torch.mean(netD6(fake6))+div_gp6
        d6_loss.backward(retain_graph=True)

        optimizerSD6.step()


    # ----------------
    # generator
    # ----------------

    
    for _ in range(1):
        optimizerG.zero_grad()
        
        g_loss = -torch.mean(netD6(fake6))-torch.mean(netD3(fake3))
        g_loss.backward()

        optimizerG.step()
    
    elapsed = time.perf_counter()-start

    w[0].append((-d3_loss+div_gp3).item())
    w[1].append((-d6_loss+div_gp6).item())

    if epoch %10==0:
        print("epoch:",epoch,";", "d3_loss:",-d3_loss+div_gp3,";","d6_loss:",-d6_loss+div_gp6,";","g_loss:",g_loss,";","Time used:",elapsed)

file1 = open('Swj1', 'w')  
for ip in w[0]:  
    file1.write(str(ip))  
    file1.write('\n')  
file1.close()  

file2 = open('Swj2', 'w')  
for ip in w[1]:  
    file2.write(str(ip))  
    file2.write('\n')  
file2.close() 

torch.save(netG.state_dict(),"./netGSwj.pt")