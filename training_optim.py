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

import ESN 
from LoadNinaPro import load_pytorch_files
import train_test
import utils
rng = default_rng()

import pickle

## CHANGE THESE VARIABLES BASED ON WHAT YOU WANT TO DO ####################################
# command line argument
inp = sys.argv[1]
inp2 = sys.argv[2] # if not testing two hyps, put second argument as 'none' or some other word that's not a hyp in esn2
#data_to_try = [(37,30),(37,40),(37,50),(45,30),(45,40),(45,50),(42,30),(48,30)] #0-7
#nr_nodes_to_try = [300, 400, 600, 700, 800, 900, 1000] # 6
# whether to use gpu or cpu
use_gpu = True

# standard hyperparameters - values to be used when not testing that particular hyperparameter
with open('hyp_values/esn1_hyp.pkl', 'rb') as f:
    esn1_hyp = pickle.load(f)

with open('hyp_values/esn2_hyp.pkl', 'rb') as f:
    esn2_hyp = pickle.load(f)

# parameter to be tested - can choose one from either esn, or a combination of one from esn1 and one from esn2
# if writing a combination - ALWAYS put first the param from ESN1, and second the param from ESN2
# or it won't work!
testing_two_hyps = False # why tf did I define this?? it's not used ??
test_param = [inp] 
for k,v in esn2_hyp.items():
    if k == inp2:
        test_param = [inp, inp2] 
    
print("testing: {}".format(test_param))

# number of trials for each combination of params
trials = 10 #4

# this will dictate which values will be tested
number_hyp_values = 10
if inp=='alpha1':
    initial_value=0.15
else:
    initial_value = 0.99
modifier = 0.7

# Number of ESN nodes
N=500 #500

#esn1_hyp = {'gamma1':0.99,'alpha1':0.04,'rho1':0.99} # alpha1 = 0.34 and 0.082 work well for 1 ESN
#esn2_hyp = {'gamma2':0.116,'alpha2':0.116,'rho2':0.99}

# doulbe or single ESN 
Double_ESN = True

if Double_ESN:
    file_name = 'Test1_v2'#.format(N)
else:
    file_name = 'Test2_v2'#.format(N)

file_dir = os.path.join('accs_text', file_name)

# write accs to pt files?
write_accs_to_files = True

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
    'exercise_dir':'E1'} #_ws37_resamp50
file_names = {
    'movements':'movements.pt',
    'labels':'labels.pt'
} #training_ws42_resamp50.pt labels_ws42_resamp50.pt
percentage_split = {
    'training': 60, 
    'validation': 15
}

## TEST DIFFERENT HYPERPARAMETERS ##########################################################
# hyp values to test
if len(test_param)==1:
    change_esn = [initial_value]
    for length in range(number_hyp_values):
        change_esn.append(change_esn[-1]*modifier)
elif len(test_param)==2:
    a_esn1 = initial_value
    change_esn = []
    for length in range(number_hyp_values):
        a_esn2 = initial_value
        for length2 in range(number_hyp_values):
            change_esn.append((a_esn1, a_esn2))
            a_esn2 *= modifier
        a_esn1 *= modifier
        a_esn2 = initial_value
else:
    print("Error: test_param needs to be of length 1 or 2")
print(len(change_esn))

# tensors to store mean accuracies
accs_tr_mean = torch.tensor([])
accs_te_mean = torch.tensor([])
accs_va_mean = torch.tensor([])
ste_vector = torch.tensor([])

## TRAINING AND DATA COLLECTION ###############################################################
for hyp in change_esn:
    # replace values in dicts with the ones we want to test
    print("changing hyperparameter lists...\n")
    if len(test_param)==1:
        # if testing only one hyperparameter
        if test_param[0] in esn1_hyp: # check which esn it belongs to
            esn1_hyp[test_param[0]] = hyp # and then change that parameter's value
        elif test_param[0] in esn2_hyp:
            esn2_hyp[test_param[0]] = hyp 
        else:
            print('Error: did not find test_param in neither esn1_hyp or esn2_hyp')
        print("TESTING {}={}\n".format(test_param[0], hyp))
    else:
        # if testing 2 hyperparameters
        hyp1,hyp2 = hyp # unpack
        # check where the first hyperparameter is (esn1 or esn2)
        # and change its value in the dict
        if test_param[0] in esn1_hyp:
            esn1_hyp[test_param[0]] = hyp1 
        elif test_param[0] in esn2_hyp:
            esn2_hyp[test_param[0]] = hyp1
        # then do the same for the second hyp
        if test_param[1] in esn1_hyp:
            esn1_hyp[test_param[0]] = hyp2 
        elif test_param[1] in esn2_hyp:
            esn2_hyp[test_param[0]] = hyp2
        # print what is currently being tested
        print("TESTING {}={} AND {}={}\n".format(test_param[0], hyp1, test_param[1], hyp2))

    ## TRAINING & TESTING #####
    # parameters
    N_train=100000 #100000
    batch_size=300 #15 #300 (when using train_loop_samples, but that train function is hard to adapt to new train regime)
    eta=0.001

    # training
    print("Initialising train class...\n")
    training_class = train_test.Train(device, Double_ESN, N_train, batch_size, N, eta, prnt=True)
    print("Training...\n")
    accs_tr_cat, accs_te_cat, accs_va_cat = training_class.training(trials, directory_info, file_names, percentage_split, esn1_hyp, esn2_hyp)
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

## CHANGE HYPERPARAMETER IN ESN1/2_HYP ############################################################
highest_va_mean = 0
place_of_highest_mean = 0
testing_esn1 = True

## PRINT RESULTS OF TESTING HYPERPAR ##############################################################
param_fixed=''
if not Double_ESN:
    for k,v in esn1_hyp.items():
        if k!=test_param[0]:
            param_fixed = param_fixed+'{}={} '.format(k,v)
    # write to file
    with open(file_dir, 'a', encoding="utf-8") as f:
        f.write("#### FINAL RESULTS ###################\n")
        f.write("DATA DIRECTORY: {}/{}\n".format(directory_info['db_dir'], directory_info['exercise_dir']))
        f.write("TYPE: SINGLE ESN| {}\n".format(param_fixed))
        f.write("NR ESN NODES: {}\n".format(N))
        f.write("NR. TRIALS FOR EACH VALUE: {} \n".format(trials))
        f.write("N_train = {} & batch_size = {}\n".format(N_train, batch_size))
        f.write('{:16} {:16} {:16} {:16} {:16}\n'.format(\
            test_param[0], "avg mjr acc tr", "avg mjr acc va", "avg mjr acc te", "SE te"))
        #f.write('{:15} {:15} {:15}\n'.format(param_name, "avg acc tr", "avg mjr acc tr"))
        
        for place in range(len(change_esn)):
                place_accs = place*2+1
                f.write('{:<16.3f} {:<16.6f} {:<16.6f} {:<16.6f} {:<16.6f}\n'.format(\
                    change_esn[place], accs_tr_mean[place_accs, -1], accs_va_mean[place_accs, -1], accs_te_mean[place_accs, -1], ste_vector[place]))
                if accs_va_mean[place_accs, -1] > highest_va_mean:
                    highest_va_mean = accs_va_mean[place_accs, -1]
                    place_of_highest_mean = place
        
    print("Highest validation test mean is for {} = {}\n".format(test_param[0], change_esn[place_of_highest_mean]))
    esn1_hyp[test_param[0]] = change_esn[place_of_highest_mean]
    print("esn1_hyp[{}] changed to {}".format(test_param[0], change_esn[place_of_highest_mean]))
else:
    if len(test_param)==1:
        for k,v in esn1_hyp.items():
            if k!=test_param[0]:
                param_fixed = param_fixed+'{}={} '.format(k,v)
            if k==test_param[0]:
                testing_esn1 = True
        for k,v in esn2_hyp.items():
            if k!=test_param[0]:
                param_fixed = param_fixed+'{}={} '.format(k,v)
            if k==test_param[0]:
                testing_esn1 = False
        # write to file
        with open(file_dir, 'a', encoding="utf-8") as f:
            f.write("#### FINAL RESULTS ###################\n")
            f.write("DATA DIRECTORY: {}/{}\n".format(directory_info['db_dir'], directory_info['exercise_dir']))
            f.write("TYPE: DOUBLE ESN | {}\n".format(param_fixed))
            f.write("NR ESN NODES: {}\n".format(N))
            f.write("NR. TRIALS FOR EACH VALUE: {} \n".format(trials))
            f.write("N_train = {} & batch_size = {}\n".format(N_train, batch_size))
            f.write('{:16} {:16} {:16} {:16} {:16}\n'.format(\
                test_param[0], "avg mjr acc tr", "avg mjr acc va", "avg mjr acc te", "SE te"))
            #f.write('{:15} {:15} {:15}\n'.format(param_name, "avg acc tr", "avg mjr acc tr"))
            
            for place in range(len(change_esn)):
                    place_accs = place*2+1
                    f.write('{:<16.3f} {:<16.6f} {:<16.6f} {:<16.6f} {:<16.6f}\n'.format(\
                        change_esn[place], accs_tr_mean[place_accs, -1], accs_va_mean[place_accs, -1], accs_te_mean[place_accs, -1], ste_vector[place]))

                    if accs_va_mean[place_accs, -1] > highest_va_mean:
                        highest_va_mean = accs_va_mean[place_accs, -1]
                        place_of_highest_mean = place
        if testing_esn1:
            print("Highest validation test mean is for {} = {}\n".format(test_param[0], change_esn[place_of_highest_mean]))
            esn1_hyp[test_param[0]] = change_esn[place_of_highest_mean]
            print("esn1_hyp[{}] changed to {}".format(test_param[0], change_esn[place_of_highest_mean]))
        else:
            print("Highest validation test mean is for {} = {}\n".format(test_param[0], change_esn[place_of_highest_mean]))
            esn2_hyp[test_param[0]] = change_esn[place_of_highest_mean]
            print("esn2_hyp[{}] changed to {}".format(test_param[0], change_esn[place_of_highest_mean]))   
        if test_param[0] == 'rho1':
            esn2_hyp['rho2'] = change_esn[place_of_highest_mean][0]
    else:
        for k,v in esn1_hyp.items():
            if k!=test_param[0] and k!=test_param[1]:
                param_fixed = param_fixed+'{}={} '.format(k,v)
        for k,v in esn2_hyp.items():
            if k!=test_param[0] and k!=test_param[1]:
                param_fixed = param_fixed+'{}={} '.format(k,v) 
        # write to file
        with open(file_dir, 'a', encoding="utf-8") as f:
            f.write("#### FINAL RESULTS ###################\n")
            f.write("DATA DIRECTORY: {}/{}\n".format(directory_info['db_dir'], directory_info['exercise_dir']))
            f.write("TYPE: DOUBLE ESN | {}\n".format(param_fixed))
            f.write("NR ESN NODES: {}\n".format(N))
            f.write("NR. TRIALS FOR EACH VALUE: {} \n".format(trials))
            f.write("N_train = {} & batch_size = {}\n".format(N_train, batch_size))
            f.write('{:16} {:16} {:16} {:16} {:16} {:16}\n'.format(\
                test_param[0], test_param[1], "avg mjr acc tr", "avg mjr acc va", "avg mjr acc te", "SE te"))
            #f.write('{:15} {:15} {:15}\n'.format(param_name, "avg acc tr", "avg mjr acc tr"))
            
            for place in range(len(change_esn)):
                    place_accs = place*2+1
                    f.write('{:<16.3f} {:<16.3f} {:<16.6f} {:<16.6f} {:<16.6f} {:<16.6f}\n'.format(\
                        change_esn[place][0], change_esn[place][1], accs_tr_mean[place_accs, -1], accs_va_mean[place_accs, -1], accs_te_mean[place_accs, -1], ste_vector[place]))
                    
                    if accs_va_mean[place_accs, -1] > highest_va_mean:
                        highest_va_mean = accs_va_mean[place_accs, -1]
                        place_of_highest_mean = place
        print("Highest validation test mean is for {} = {} and {}={}\n".format(test_param[0], change_esn[place_of_highest_mean][0], test_param[1], change_esn[place_of_highest_mean][1]))
        esn1_hyp[test_param[0]] = change_esn[place_of_highest_mean][0]
        esn2_hyp[test_param[1]] = change_esn[place_of_highest_mean][1]
        print("esn1_hyp[{}] changed to {} and esn2_hyp[{}] change to {}".format(test_param[0], change_esn[place_of_highest_mean][0], test_param[1], change_esn[place_of_highest_mean][1]))   

with open('hyp_values/esn1_hyp.pkl', 'wb') as f:
    pickle.dump(esn1_hyp, f)

with open('hyp_values/esn2_hyp.pkl', 'wb') as f:
    pickle.dump(esn2_hyp, f)

print("#### FINAL RESULTS SAVED IN FILE ###################\n")

## TODO: SELECT BEST HYPS, WRITE TO FILE, SO CAN USE THEM IN NEXT ITER
if write_accs_to_files:
    file_name = os.path.join('pt_acc_files', file_name)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    # WRITE ACCS_?_MEAN TO FILES
    hyp_combinations = torch.tensor(change_esn)
    torch.save(accs_tr_mean, '{}/accuracies_training_{}.pt'.format(file_name,test_param[0]))
    torch.save(accs_va_mean, '{}/accuracies_validation_{}.pt'.format(file_name,test_param[0]))
    torch.save(accs_te_mean, '{}/accuracies_testing_{}.pt'.format(file_name,test_param[0]))
    torch.save(hyp_combinations, '{}/hyperparameter_combinations_{}.pt'.format(file_name,test_param[0]))

    print("#### FINAL ACCURACIES SAVED IN FILES ###################\n")
