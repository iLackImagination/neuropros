import torch
import utils
from torch import nn
from torch import optim
import FFNN
import numpy as np
from LoadNinaPro import load_pytorch_files

class Train():
    def __init__(self, device, N_train, batch_size, eta, prnt = True):
        # (self, device, double_esn, N_train, batch_size, N, N_out, N_tr, N_te, N_va, eta, Z_tr, Y_tr, Z_te, Y_te, Z_va, Y_va, prnt = True):
        # put all data in here -- this will remain the same until new hyperparams will be tested

        ## print to terminal?
        self.prnt = prnt
        ## device to be used 
        self.device = device
        self.N_train = N_train # training iterations in inner loop
        self.batch_size = batch_size # batch size in inner loop
        # learning rate for Adam
        self.eta = eta 

    def initialize_model(self):
        # initialize model separately -- this will need re-init at every trial
        self.model = FFNN.FFNN().to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss().to(self.device) #mse # BCEWITHLOGITLOSS
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)

    def load_data(self, directory_info, file_names, percentage_split):
        if self.prnt:
            print("Loading data...\n")
        N_mvmnts, self.N_in, self.T_max, self.N_class, self.N_tr, self.N_te, self.N_va, self.Z_tr, self.Z_te, self.Z_va, self.Y_tr_og, self.Y_te_og, self.Y_va_og = load_pytorch_files(directory_info, file_names=file_names, percentage_split=percentage_split)
        self.N_out = self.N_class

    def process_data(self):
        if self.prnt:
            print("Reshaping data...\n")
        self.Z_tr, self.Y_tr, self.Z_te, self.Y_te, self.Z_va, self.Y_va = \
            utils.reshape_NdataxNsamples(self.Z_tr, self.Y_tr_og, self.Z_te, self.Y_te_og, self.Z_va, self.Y_va_og)

    def test_during_training(self, dataset):
        # as below, but adjusted for class variables
        # TODO: CHECK THAT VARIABLES MATCH
        with torch.no_grad():
            if dataset == 'training':
                Y = self.Y_tr 
                Z = self.Z_tr 
                Y_mjrmask = self.Y_tr_og 
                N = self.N_tr
            elif dataset == 'testing':
                Y = self.Y_te 
                Z = self.Z_te 
                Y_mjrmask = self.Y_te_og 
                N = self.N_te
            elif dataset == 'validation':
                Y = self.Y_va 
                Z = self.Z_va 
                Y_mjrmask = self.Y_va_og 
                N = self.N_va

            # removal of padding
            Y = Y.to(self.device)
            Z = Z.to(self.device)
            summed = torch.sum(Y, 1)
            mask = summed!=0
            Z_m = Z[mask, :]
            Y_m = Y[mask, :]

            pr = self.model(Z_m)
            #test_loss = loss_fn(pr, Y_m).item()
            acc = torch.mean( torch.eq(torch.argmax(pr,1),torch.argmax(Y_m,1)).float() ).item()
            del Z_m
            del Y_m

            # majority voting + removal of padding
            Y_mjrmask = Y_mjrmask.to(self.device)
            Y_mjrvtn = torch.zeros(N,self.N_out)
            Y_mjrvtn[torch.arange(N), torch.argmax(torch.sum(Y_mjrmask, dim=2),dim=1)] = 1
            Y_mjrvtn = torch.argmax(Y_mjrvtn,dim=1)

            pr2 = self.model(Z)
            test_nr = N
            pr_labels = torch.argmax(pr2,1)
            pr_labels = torch.reshape(pr_labels, (test_nr, -1))
            
            pr_mjrvtn = torch.zeros(test_nr)
            for mvmnt in range(test_nr):
                corr_labels_mask = torch.sum(Y_mjrmask[mvmnt, :, :], 0)
                corr_labels_mask = corr_labels_mask!=0
                pr_labels_masked = pr_labels[mvmnt, corr_labels_mask]
                pr_mjrvtn[mvmnt] =  torch.argmax(torch.bincount(pr_labels_masked))
                
            acc_mjr = torch.mean(torch.eq(pr_mjrvtn, Y_mjrvtn).float()).item()

        return acc, acc_mjr #, test_loss

    def inner_train_loop(self):
        # as train_loop, but again adjusted
        #train_loop_samples(Z_tr_torch, Y_tr_torch, model, loss_fn, optimizer, N_train, batch_size, Z_te_both, Y_te_torch, Z_va_both, Y_va_torch, N_tr, N_te, N_va, N_out):
        # TODO: CHECK IF VAR NAMES MATCH
        test_interval = int(self.N_train/20) #5000
        length_test = int(self.N_train/test_interval)
        
        # row 0 for acc, row 1 for acc mjr vtn
        accs_tr_all = torch.zeros(2,self.N_train)
        accs_tr = torch.zeros(2,length_test)
        accs_te = torch.zeros(2,length_test)
        accs_va = torch.zeros(2,length_test)

        for n in range(self.N_train):
            # get image & its label
            rand_ind=np.random.randint(0,self.Z_tr.shape[0],self.batch_size)
            smpl = torch.clone(self.Z_tr[rand_ind,:])
            label = torch.clone(self.Y_tr[rand_ind,:])
            
            # removal of padding
            summed = torch.sum(label, 1)
            mask = summed!=0
            smpl = smpl[mask, :].to(self.device)
            label = label[mask, :].to(self.device)
            
            # prediction
            pr = self.model(smpl)
            loss = self.loss_fn(pr, label)
            
            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            smpl = smpl.to('cpu')
            label = label.to('cpu')
            
            # TEST
            if n%test_interval==0 and n!=0:
                step = int(n/test_interval)
                # training accuracies 
                accs_tr[0,step], accs_tr[1,step] = self.test_during_training('training')

                # validation accuracies
                accs_va[0,step], accs_va[1,step] = self.test_during_training('validation')

                # testing accuracies
                accs_te[0,step], accs_te[1,step] = self.test_during_training('testing')
        return accs_tr, accs_va, accs_te #,acc_tr_all

    def training(self, trials, directory_info, file_names, percentage_split):
        # as below/as in training.py, but again adjusted for class variables etc
        # tensors to store results
        accs_tr_cat = torch.tensor([])
        accs_te_cat = torch.tensor([])
        accs_va_cat = torch.tensor([])
        for trial in range(trials):
            self.load_data(directory_info, file_names, percentage_split)
            self.process_data()
            self.initialize_model()
            print(self.Z_tr.shape)
            print(self.Y_tr.shape)
            print(self.Y_tr_og.shape)
            accs_tr, accs_va, accs_te = self.inner_train_loop()
            
            # accs_?_cat (2*step, step) --> 0 = acc, 1 = mjr vtn acc
            accs_tr_cat = torch.cat((accs_tr_cat, accs_tr),0)
            accs_te_cat = torch.cat((accs_te_cat, accs_te),0)
            accs_va_cat = torch.cat((accs_va_cat, accs_va),0)
            if self.prnt:
                print("#### END OF TRIAL ", trial+1, " ###################\n")
        print(accs_tr_cat.shape)
        return accs_tr_cat, accs_te_cat, accs_va_cat