import numpy as np
from numpy.random import default_rng
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.io import loadmat
rng = default_rng()

class RidgeRegression:
    def __init__(self, N_te, N_tr, T_max, N_out, beta=0.1):
        self.beta = beta
        self.W_out = np.zeros(2)

        self.N_te = N_te 
        self.N_tr = N_tr 
        self.T_max = T_max 
        self.N_out = N_out

    def Ridge_Regression(self, X, Y):
        N = X.shape[1]
        X=np.transpose(X)
        Y=np.transpose(Y)
            
        self.W_out=np.matmul( np.matmul(Y,X.T),np.linalg.inv(np.matmul(X,X.T)+self.beta*np.eye(N)) ).T

    def Evaluate(self, X_tr, Y_tr, X_te, Y_te, Y_tr_t, Y_te_t):
        y_tr=np.matmul(X_tr,self.W_out)
        y_te=np.matmul(X_te,self.W_out)

        lab_tr=np.argmax(y_tr,1)
        lab_te=np.argmax(y_te,1)

        Acc_tr=np.mean(np.equal(np.argmax(Y_tr,1),lab_tr))
        Acc_te=np.mean(np.equal(np.argmax(Y_te,1),lab_te))

        print('Accuracy: ', Acc_tr, Acc_te)
        
        # calculate accuracy using majority voting from each samples to find label for entire movement
        y_tr_t = np.reshape(y_tr, (self.N_tr, self.T_max, self.N_out))
        lab_tr_t = np.argmax(np.sum(y_tr_t[:,:,:], axis=1),axis=1)
        
        y_te_t = np.reshape(y_te, (self.N_te, self.T_max, self.N_out))
        lab_te_t = np.argmax(np.sum(y_te_t[:,:,:], axis=1),axis=1)
        
        Acc_tr_t = np.mean(np.equal(Y_tr_t, lab_tr_t))
        Acc_te_t = np.mean(np.equal(Y_te_t, lab_te_t))
        
        print('Accuracy (after majority voting): ', Acc_tr_t, Acc_te_t)

    def Run(self, X_tr, Y_tr, X_te, Y_te, Y_tr_t, Y_te_t, X_tr_nopad=None, Y_tr_nopad=None, use_nopad=False):
        if use_nopad:
            self.Ridge_Regression(X_tr_nopad, Y_tr_nopad)
        else:
            self.Ridge_Regression(X_tr, Y_tr)
        self.Evaluate(X_tr, Y_tr, X_te, Y_te, Y_tr_t, Y_te_t)