import torch
from torch import nn
from torch import optim
import numpy as np

# class that computes the activations of the ESN
class ESNActivations:
    def __init__(self,device,N,N_in,N_out,N_tr,N_te,T_max,alpha=0.9,rho=0.99,gamma=0.1):
        self.device = device
        ## hyperparams
        self.alpha=alpha
        self.rho=rho
        self.gamma=gamma
        ## number of neurons, input & output sizes
        self.N=N
        self.N_in=N_in
        self.N_out=N_out
        ## number of training & testing samples, and maximum length of a movements
        self.N_tr = N_tr
        self.N_te = N_te
        self.T_max = T_max
        ## initializing input & output weights
        self.W_in = self.gamma*torch.normal(0,1,size=(N_in,N)).to(self.device)
        self.W_out = torch.normal(0,1,size=(N,N_out)).to(self.device)
        ## initializing reservoir weight
        # create sparsity matrix -- 
        sparsity = [(1 if torch.rand(1)<(10/N) else 0) for ind in range(N*N)]
        sparsity = torch.reshape(torch.tensor(sparsity), (N,N))
        # create weight matrix, sampling from a uniform distrib, and apply sparsity
        W = torch.zeros(N,N).uniform_(-1,1) * sparsity
        # normalize it by dividing with the largest eigenvalue
        lambdas = torch.linalg.eigvals(W)
        lambdas_abs = torch.abs(lambdas)
        lambda_max = lambdas_abs[torch.argmax(lambdas_abs)]
        
        W_norm = W / lambda_max
        self.W = torch.transpose(self.rho*W_norm,0,1).to(self.device)
                
    def Compute(self,X):
        ## compute reservoir activations
        # convert X to a torch tensor if it is a numpy array
        if isinstance(X, np.ndarray):
            X = torch.tensor(X,dtype=torch.float)

        # send input to cuda
        X = X.to(self.device)
        # initialize variables for time length (ie. movement length) & number of movements
        T = X.size(2)
        N_s = X.size(0)
        # initialize ESN activations vector & activation function (tanh in this case)
        Z = torch.zeros(N_s, self.N, T).to(self.device)
        tnh = nn.Tanh()
        # calculate activations
        # Z(t) = (1-alpha)*Z(t-1) + alpha*[ Z(t-1)*W + X(t)*W_in ]
        for t in range(1,T):
            Z[:,:,t]=torch.add( (1-self.alpha)*Z[:,:,t-1], self.alpha*tnh( torch.add( torch.matmul(Z[:,:,t-1],self.W),torch.matmul(X[:,:,t],self.W_in) ) ) )
        
        return Z.to('cpu')

# class for training the readout weights of the ESN
class ESNReadout(nn.Module):
    ## readout weights for ESN are a simple linear layer 
    # which takes the ESN activations as input
    def __init__(self, in_features, out_features):
        super(ESNReadout, self).__init__()
        
        # nn.Linear(in_features, out_features, bias)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, S):
        return self.linear(S)