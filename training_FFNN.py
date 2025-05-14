## IMPORTS ###############################################################################
import numpy as np
from numpy.random import default_rng
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import io
import math

import torch
torch.cuda.is_available()
import torch.jit as jit
from torch import nn
from torch import optim
import sys
import os

import FFNN
from LoadNinaPro import load_pytorch_files
import train_test_FFNN as train_test
import utils
rng = default_rng()

import pickle

# compute on gpu or cpu
use_gpu = True

# number of trials for each combination of params
trials = 10 #4

# learning rates to try
etas = [0.01, 0.005, 0.001, 0.0005, 0.0001]

# write accs to pt files?
write_accs_to_files = True
file_name = 'FFNN_accs_fulldata'
file_dir = os.path.join('./accs_text', file_name)

###########################################################################################

## SELECT DEVICE ##########################################################################
if torch.cuda.is_available() and use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

## DATA LOADING INFO ######################################################################
parent_dir = './'
directory_info = {
    'db_dir':'NinaproDB1',
    'exercise_dir':'E2'} #_ws37_resamp50
file_names = {
    'movements':'movements.pt',
    'labels':'labels.pt'
} #training_ws42_resamp50.pt labels_ws42_resamp50.pt
percentage_split = {
    'training': 60, 
    'validation': 20
}

# tensors to store mean accuracies
accs_tr_mean = torch.tensor([])
accs_te_mean = torch.tensor([])
accs_va_mean = torch.tensor([])
ste_vector = torch.tensor([])

## TRAINING AND DATA COLLECTION ###############################################################
for eta in etas:
    ## TRAINING & TESTING #####
    # parameters
    N_train=100000 #100000
    batch_size=300 
    #eta=0.001

    # training
    print("Initialising train class...\n")
    training_class = train_test.Train(device, N_train, batch_size, eta, prnt=True)
    print("Training...\n")
    accs_tr_cat, accs_te_cat, accs_va_cat = training_class.training(trials, directory_info, file_names, percentage_split)
    print(accs_tr_cat.shape)
    print("Storing accuracies in vectors...\n")
    accs_tr_mean, accs_te_mean, accs_va_mean = utils.calc_accs_mean((accs_tr_cat,accs_tr_mean), (accs_te_cat,accs_te_mean), (accs_va_cat,accs_va_mean))
    ste = torch.std(accs_te_cat[1::2,-1])
    print("standard dev = ", ste)
    ste = 2 * ste / math.sqrt(trials)
    print("standard error = ", ste, "\n")
    ste = torch.tensor([ste])
    # length equals number of hyp tested, one Standard Error for each hyp, representing the SE between the mjr vting accuracies for each trial
    ste_vector = torch.cat((ste_vector,ste)) 
    del training_class
    del accs_te_cat
    del accs_tr_cat
    del accs_va_cat
    del ste
    torch.cuda.empty_cache()

## CHANGE HYPERPARAMETER IN ESN1/2_HYP ############################################################
highest_va_mean = 0
place_of_highest_mean = 0
testing_esn1 = True

## WRITE ACC RECORDS IN PT FILES ##################################################################
if write_accs_to_files:
    file_name = os.path.join('pt_acc_files', file_name)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    etas_tested = torch.tensor(etas)
    torch.save(accs_tr_mean, '{}/accuracies_training_FFNN.pt'.format(file_name))
    torch.save(accs_va_mean, '{}/accuracies_validation_FFNN.pt'.format(file_name))
    torch.save(accs_te_mean, '{}/accuracies_testing_FFNN.pt'.format(file_name))
    torch.save(etas_tested, '{}/etas_tested.pt'.format(file_name))

    print("#### FINAL ACCURACIES SAVED IN FILES ###################\n")

## PRINT RESULTS OF TESTING HYPERPAR ##############################################################
with open(file_dir, 'a', encoding="utf-8") as f:
    f.write("#### FINAL RESULTS ###################\n")
    f.write("DATA DIRECTORY: {}/{}\n".format(directory_info['db_dir'], directory_info['exercise_dir']))
    f.write("TYPE: FFNN\n")
    f.write("NR. TRIALS: {} \n".format(trials))
    f.write("N_train = {} & batch_size = {}\n".format(N_train, batch_size))
    f.write('{:16} {:16} {:16} {:16} {:16}\n'.format(\
        "eta", "avg mjr acc tr", "avg mjr acc va", "avg mjr acc te", "SE te"))
    #f.write('{:15} {:15} {:15}\n'.format(param_name, "avg acc tr", "avg mjr acc tr"))
    
    for place in range(len(etas)):
        place_accs = place*2+1
        #place_accs = 1
        f.write('{:<16.3f} {:<16.6f} {:<16.6f} {:<16.6f} {:<16.6f}\n'.format(\
            etas[place], accs_tr_mean[place_accs, -1], accs_va_mean[place_accs, -1], accs_te_mean[place_accs, -1], ste_vector[0]))

print("#### FINAL RESULTS SAVED IN FILE ###################\n")


