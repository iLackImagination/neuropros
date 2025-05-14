import numpy as np
from scipy.io import loadmat
import torch
import os

## TODO: MAKE LOADNINAPRO USE TORCH INSTEAD OF NUMPY
class LoadNinaPro:
    def __init__(self,fldr='Ninapro',file_ending='A1_E1',N_class=12,N_in=10,n_subjects=10,training_percentage=60, testing_p=20):
        """
        fldr = name of folder where dataset is held, class assumes datasets are in a folder contained in the same
            folder as this file, './datasets/', and that each dataset is contained in another subfolder, which this
            variable hold, ie. './datasets/fldr/'
        file_ending = how the file ends, e.g. 'A1_E1', this will depend on the dataset/exercise chosen
            this class assumes that the files start with 'S{}_', representing the subject number,
            and will not work if the files don't follow this convention
        N_class = how many classes/number of movements the data loaded contains
        N_in = the size of the input, ie. the number of electrodes
        n_subjects = how many subjects to load
        training_percentage = percentage of data to be used for training, eg. 70 means ~70% will be used
            for training
        testing_p = percentage of data to be used for testing, eg: 20 means ~20% will be used for testing
        --> if testing_p+training_percentage < 100, the rest of the data will be used for validation
        """
        # variables for dataset info
        self.n_subjects=n_subjects
        self.training_percentage = training_percentage
        self.testing_p = testing_p
        # variables for file path info
        self.folder = fldr
        self.file_ending = file_ending
        
        # variables for splitting the dataset
        self.begin=0
        self.X1=[]
        self.Y1=[]
        self.TS_begin=[]
        self.TS_end=[]
        
        # total number of movement samples, number of output classes, reservoir input length, & max length of sample
        self.N_seq = 0
        self.N_class = N_class # output classes == number of different movements performed
        self.N_in = N_in # input length == number of electrodes on the participants' arm
        self.T_max = None
        
        # the numpy arrays (placeholders)
        self.X = np.zeros(1)
        self.Y = np.zeros(1)

    def load_data(self):
        for subj in range(1,self.n_subjects+1):
            # load data
            path = './datasets/{}/S{}_{}.mat'.format(self.folder,subj,self.file_ending)
            current_subj = loadmat(path)
            
            # relevant variables
            T=np.shape(current_subj['restimulus'])[0] # number of total samples across all movements & rest periods
            S=current_subj['emg'] # input data
            Label=current_subj['restimulus']  # labels
            
            # create array that stores ideal values for labels (ie: if there are 10 labels, the 
            # label array should contain values in the range [0,10])
            unique_labels = np.unique(Label)
            ideal_labels = np.arange(self.N_class+1)
            # if the labels aren't that format, make them
            equal_labels = np.mean(np.unique(Label) == ideal_labels)
            if equal_labels != 1:
                Label = np.where(Label == 0, Label, Label - unique_labels[1]+1)
            # split data into different movements
            # X1 will be a list where each element is a matrix containing the samples for a movement
            # a movement is of size n_samples x 10
            for t in range(1,T): 
                if Label[t]>0 and Label[t-1]==0: 
                    self.begin=t
                    self.TS_begin.append(t)

                if Label[t-1]>0 and Label[t]==0:  
                    self.X1.append(S[self.begin:t,:])
                    self.Y1.append(Label[self.begin:t,:])
                    self.TS_end.append(t)

    def load_specific_subject(self, subj):
        # reset vars
        self.begin=0
        self.X1=[]
        self.Y1=[]
        self.TS_begin=[]
        self.TS_end=[]

        # load data
        path = './datasets/{}/S{}_{}.mat'.format(self.folder,subj,self.file_ending)
        current_subj = loadmat(path)
        
        # relevant variables
        T=np.shape(current_subj['restimulus'])[0] # number of total samples across all movements & rest periods
        S=current_subj['emg'] # input data
        Label=current_subj['restimulus']  # labels
        
        # create array that stores ideal values for labels (ie: if there are 10 labels, the 
        # label array should contain values in the range [0,10])
        unique_labels = np.unique(Label)
        ideal_labels = np.arange(self.N_class+1)
        # if the labels aren't that format, make them
        equal_labels = np.mean(np.unique(Label) == ideal_labels)
        if equal_labels != 1:
            Label = np.where(Label == 0, Label, Label - unique_labels[1]+1)
        # split data into different movements
        # X1 will be a list where each element is a matrix containing the samples for a movement
        # a movement is of size n_samples x 10
        for t in range(1,T): 
            if Label[t]>0 and Label[t-1]==0: 
                self.begin=t
                self.TS_begin.append(t)

            if Label[t-1]>0 and Label[t]==0:  
                self.X1.append(S[self.begin:t,:])
                self.Y1.append(Label[self.begin:t,:])
                self.TS_end.append(t)

    def create_numpy_arrays(self):
        # calculate how many samples each movement has
        I=np.array(self.TS_end)-np.array(self.TS_begin)
        if self.T_max == None:
            # store the max length of a sample (will be used to pad the smaller ones)
            self.T_max=np.max(I)

        # total number of movements & number of input nodes to network
        self.N_seq=len(self.X1)

        # np arrays to store X1 & Y1
        self.X=np.zeros([self.N_seq,self.N_in,self.T_max])
        self.Y=np.zeros([self.N_seq,self.N_class,self.T_max])

        # add X1 & Y1 to their np arrays, padding the entries that are shorter than T_max
        for n in range(self.N_seq):
            self.X[n,:,0:I[n]]=np.copy(np.transpose(self.X1[n])) 
            self.Y[n,self.Y1[n][:,0]-1,np.arange(0,I[n])]=1

    def get_data(self):
        self.load_data()
        self.create_numpy_arrays()

        # select random data to be used for training & testing
        rand=np.random.permutation(self.N_seq)

        mov_per_subject = int(self.N_seq / self.n_subjects) # how many movements each subjects does
        N_tr = int(self.training_percentage*self.n_subjects/100*mov_per_subject) # movements used for training
        N_te = int(self.testing_p*self.n_subjects/100*mov_per_subject) # movements used for testing
        N_va = self.N_seq - N_tr - N_te # movemnets used for validation
        rand_tr=rand[0:N_tr]

        # training set 
        X_tr=np.copy(self.X[rand_tr,:,:])
        Y_tr=np.copy(self.Y[rand_tr,:,:])

        if N_te!=0:
            end_te = N_tr+N_te
            rand_te=rand[N_tr:end_te]
            X_te=np.copy(self.X[rand_te,:,:])
            Y_te=np.copy(self.Y[rand_te,:,:])
        else:
            X_te=np.zeros((1,))
            Y_te=np.zeros((1,))

        if N_va!=0:
            rand_va=rand[end_te:]
            X_va=np.copy(self.X[rand_va,:,:])
            Y_va=np.copy(self.Y[rand_va,:,:])
        else:
            X_va=np.zeros((1,))
            Y_va=np.zeros((1,))

        return self.N_class, self.N_in, self.T_max, N_tr, N_te, N_va, X_tr, Y_tr, X_te, Y_te, X_va, Y_va 

    def get_nonrandom_data(self):
        self.load_data()
        self.create_numpy_arrays()

        X_np = np.copy(self.X[:,:,:])
        Y_np = np.copy(self.Y[:,:,:])

        return self.N_class, self.N_in, self.T_max, X_np, Y_np

    def get_maximum_length(self):
        self.load_data()
        # calculate how many samples each movement has
        I=np.array(self.TS_end)-np.array(self.TS_begin)

        # store the max length of a sample (will be used to pad the smaller ones)
        self.T_max=np.max(I)

    def get_specific_subject_data(self,subj):
        self.load_specific_subject(subj)
        self.create_numpy_arrays()

        X_np = np.copy(self.X[:,:,:])
        Y_np = np.copy(self.Y[:,:,:])

        return self.N_class, self.N_in, self.T_max, X_np, Y_np



def load_pytorch_files(directory_info: dict, parent_dir = './',file_names: dict = {'movements':'movements.pt','labels':'labels.pt'}, percentage_split: dict = {'training': 60, 'validation':20}, prnt = True):
    """
    Input details:
    - directory_info: should be dict of the form {'db_dir': 'something', 'exercise_dir':'something_else'}
    -- db_dir should be the folder containing the database files (eg. NinaproDB2) and exercise_dir the folder containing the tensors for that exercise (eg. E1)
    -- if all the files are in the same directory (not split into separate exercise folders), only include db_dir, ie: {'db_dir': 'something'}
    -- if all the files are in the parent directory (parent_dir), and not in any subfolder, directory_info should be an empty dict
    - parent_dir: the parent directory of db_dir, if the folder is not stored in the same directory as this file
    - file_names: file names for the movement and label tensors, if different from those already written (ie. movements.pt & labels.pt)
    - percentage split: how you want the arrays to be split for training, validation, & testing --> the default is 60/20/20 for training/validation/testing
    -- only provide percentage for training and validation, testing will be inferred from these
    -- validation can be '0'
    """
    # create variable to hold parent directory
    if len(directory_info)==0:
        files_dir = parent_dir
    if len(directory_info)==1:
        files_dir = os.path.join(parent_dir, directory_info['db_dir'])
    elif len(directory_info)==2:
        files_dir = os.path.join(parent_dir, directory_info['db_dir'], directory_info['exercise_dir'])
    else:
        if prnt:
            print("Error: expected first argument to be a dictionary of length 0, 1, or 2.")
        return 0, 0
    # variables that hold the directory for the movements and labels respectively
    mov_dir = os.path.join(files_dir, file_names['movements'])
    lab_dir = os.path.join(files_dir, file_names['labels'])

    # load movements & labels into variables
    X_n = torch.load(mov_dir)
    Y_n = torch.load(lab_dir)


    # get information from tensors
    N_mvmnts = X_n.shape[0] # number of movements
    N_in = X_n.shape[1] # input length
    T_max = X_n.shape[2] # maximum length of movements
    N_class = Y_n.shape[1] # number of classes, ie. how many movements there are

    # randomize
    rand=np.random.permutation(N_mvmnts)
    X = torch.clone(X_n[rand])
    Y = torch.clone(Y_n[rand])

    # split into training, validation, testing
    te_split = 100 - percentage_split['training'] - percentage_split['validation']

    N_tr = int(percentage_split['training']*N_mvmnts/100)
    N_te = int(te_split*N_mvmnts/100)
    N_va = N_mvmnts - N_tr - N_te

    end_te = N_tr+N_te
    X_te = torch.clone(X[N_tr:end_te])
    Y_te = torch.clone(Y[N_tr:end_te])
    X_va = torch.clone(X[end_te:])
    Y_va = torch.clone(Y[end_te:])
    X = torch.clone(X[:N_tr])
    Y = torch.clone(Y[:N_tr])

    return N_mvmnts, N_in, T_max, N_class, N_tr, N_te, N_va, X, X_te, X_va, Y, Y_te, Y_va