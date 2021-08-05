# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import linalg
from numpy import dot
import geomloss as gs

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

FilePath = '../../'

file_list = ['GSM1599494_ES_d0_main.csv', 'GSM1599497_ES_d2_LIFminus.csv', 'GSM1599498_ES_d4_LIFminus.csv', 'GSM1599499_ES_d7_LIFminus.csv']

table_list = []
for filein in file_list:
    table_list.append(pd.read_csv(FilePath+filein, header=None))

matrix_list = []
gene_names = table_list[0].values[:,0]
for table in table_list:
    matrix_list.append(table.values[:,1:].astype('float32'))

cell_counts = [matrix.shape[1] for matrix in matrix_list]

def normalize_run(mat):
    rpm = np.sum(mat,0)/1e6
    detect_pr = np.sum(mat==0,0)/float(mat.shape[0])
    return np.log(mat*(np.median(detect_pr)/detect_pr)*1.0/rpm + 1.0)

norm_mat = [normalize_run(matrix) for matrix in matrix_list]

qt_mat = [np.percentile(norm_in,q=np.linspace(0,100,50),axis=1) for norm_in in norm_mat] 
wdiv=np.sum((qt_mat[0]-qt_mat[3])**2,0)
w_order = np.argsort(-wdiv)

wsub = w_order[0:100]


def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6, print_iter=200):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    #X = X.toarray()   I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features) 
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % print_iter == 0 or i == 1 or i == max_iter:
            print('Iteration {}:'.format(i),)
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print('fit residual', np.round(fit_residual, 4),)
            print('total residual', np.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break
    return A, Y, dot(A,Y)

np.random.seed(0)
norm_imputed = [nmf(normin[wsub,:], latent_features = len(wsub)*4, max_iter=500)[2] for normin in norm_mat]

norm_adj = np.mean(norm_imputed[3],1)[:,np.newaxis]
subvec = np.array([0,1,2,3,4,5,6,7,8,9]) ## the first 10 genes

gnvec = gene_names[w_order[subvec]]

cov_mat = np.cov(norm_imputed[3][subvec,:])
whiten = np.diag(np.diag(cov_mat)**(-0.5))
unwhiten = np.diag(np.diag(cov_mat)**(0.5))

norm_imputed2 = [np.dot(whiten,(normin - norm_adj)[subvec,:]) for normin in norm_imputed]


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=64, num_hidden=0, activation=nn.LeakyReLU()):
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
            #nn.init.xavier_uniform_(m.weight)
            nn.init.xavier_normal_(m.weight)
            nn.init.uniform_(m.bias,a=-0.1,b=0.1)
            #nn.init.constant_(m.bias,0) 
 
        self.activation = activation 

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))
            #x = F.dropout(x,p=0.5)

        return self.linears[-1](x)


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

    return (torch.sum(real_grad_norm) + torch.sum(fake_grad_norm)) * k / (real_sample.shape[0]+fake_sample.shape[0])

# EM scheme
class JumpEulerForwardCuda(nn.Module):
    def __init__(self,in_features,num_hidden,dim_hidden,step_size):
        super(JumpEulerForwardCuda,self).__init__()

        self.drift = MLP(in_features,in_features,dim_hidden,num_hidden)
        self.intensity = torch.tensor(intensity,device="cuda")
        self.mean = nn.Parameter(0.01*torch.ones(in_features))
        self.covHalf = nn.Parameter(0.08*torch.eye(in_features))
        self.diffusion = nn.Parameter(torch.ones(bd,10))
        self.in_features = in_features
        self.jump = MLP(in_features,in_features,dim_hidden,num_hidden)
        self.step_size = step_size

    def forward(self,z0,Nsim,steps):

        PopulationPath = torch.empty(size = (Nsim,steps+1,self.in_features),device="cuda")
        PopulationPath[:,0,:] = z0
        state = z0

        for i in range(1,steps+1):
            DP = D.poisson.Poisson(self.intensity*self.step_size) 
            pois = DP.sample((Nsim,1)).cuda()
            state = state + self.drift(state)*self.step_size + math.sqrt(self.step_size)*torch.normal(0,1,size=(Nsim,bd),device="cuda")@self.diffusion+\
                (pois*self.mean + pois**(0.5)*torch.normal(0,1,size=(Nsim,self.in_features),device="cuda")@self.covHalf)*self.jump(state)
            PopulationPath[:,i,:] = state
        return PopulationPath


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

a=gs.SamplesLoss(loss='sinkhorn',p=2,blur=0.01)


train_data = norm_imputed2

train0 = torch.tensor(train_data[0],dtype=torch.float32,requires_grad = True,device="cuda").t()
train2 = torch.tensor(train_data[1],dtype=torch.float32,requires_grad = True,device="cuda").t()
train4 = torch.tensor(train_data[2],dtype=torch.float32,requires_grad = True,device="cuda").t()
train7 = torch.tensor(train_data[3],dtype=torch.float32,requires_grad = True,device="cuda").t()


#{'intensity': 10, 'learning_rate': 0.0003, 'step_size': 0.03, 'hidden_size': 256, 'dim_hidden': 4, 'beishu': 10, 'bd': 2, 'sed': 200, 'n_critic': 3}
sed = 200
intensity = 10
lr = 0.0003
step_size = 0.03
kuan = 256
ceng = 4
bd = 2
n_critic = 3
k = 2
p = 6

n_sims = train0.shape[0]
in_features = train0.shape[1]
n_steps = [10,20,35]

setup_seed(sed)

netG = JumpEulerForwardCuda(10,ceng,kuan,step_size).cuda()
netD1 = MLP(10,1,dim_hidden=kuan,num_hidden=ceng).cuda()
netD2 = MLP(10,1,dim_hidden=kuan,num_hidden=ceng).cuda()
netD3 = MLP(10,1,dim_hidden=kuan,num_hidden=ceng).cuda()


optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerSD1 = optim.Adam(netD1.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerSD2 = optim.Adam(netD2.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerSD3 = optim.Adam(netD3.parameters(), lr=lr, betas=(0.5, 0.999))

n_epochs =  20000

wd = []
for epoch in range(n_epochs):
  
 
    

    for _ in range(n_critic):
        fake_data = netG(train0,n_sims,n_steps[2])
        fake1 = fake_data[:,n_steps[0],:]
        fake2 = fake_data[:,n_steps[1],:]
        fake3 = fake_data[:,n_steps[2],:]

        optimizerSD1.zero_grad()

        div_gp1 = compute_gradient_penalty(netD1,train2,fake1,k,p)
        d1_loss = -torch.mean(netD1(train2))+torch.mean(netD1(fake1))+div_gp1
        d1_loss.backward(retain_graph=True) # retain_graph=True

        optimizerSD1.step()



        optimizerSD2.zero_grad()
        
        div_gp2 = compute_gradient_penalty(netD2,train4,fake2,k,p)
        d2_loss = -torch.mean(netD2(train4))+torch.mean(netD2(fake2))+div_gp2
        d2_loss.backward(retain_graph=True)

        optimizerSD2.step()
        
        
        optimizerSD3.zero_grad()
        
        div_gp3 = compute_gradient_penalty(netD3,train7,fake3,k,p)
        d3_loss = -torch.mean(netD3(train7))+torch.mean(netD3(fake3))+div_gp3
        d3_loss.backward(retain_graph=True)

        optimizerSD3.step()
        

    
    for _ in range(1):
        optimizerG.zero_grad()
        fake_data = netG(train0,n_sims,n_steps[2])
        fake1 = fake_data[:,n_steps[0],:]
        fake2 = fake_data[:,n_steps[1],:]
        fake3 = fake_data[:,n_steps[2],:]
        g_loss = -torch.mean(netD1(fake1))-torch.mean(netD2(fake2))-torch.mean(netD3(fake3))
        g_loss.backward() 

        optimizerG.step()

    if epoch %10==0:
        #wd.append(a(fake_data[:,2*n_steps[0],:],train4).item())
        x1 = a(fake_data[:,n_steps[0],:],train2).item()
        x2 = a(fake_data[:,n_steps[1],:],train4).item()
        x3 = a(fake_data[:,n_steps[2],:],train7).item()
        
        wd.append(x1+x2+x3)
        
        print("training error: ",x1," and ",x2," and ",x3)
        print("total error: ",wd[-1])
    
file1 = open('TrainingError10Genes', 'w')  
for ip in wd:  
    file1.write(str(ip))  
    file1.write('\n')  
file1.close() 

torch.save(netG.state_dict(),"./TrainedModel/10GenesnetG.pt")


