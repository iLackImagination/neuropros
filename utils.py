import torch 

def reshape_NdataxNsamples(*args):
    # args should be a sequence of the torch tensors to reshape
    ret = []
    for arg in args:
        arg = torch.transpose(arg, 1, 2)
        arg = torch.reshape(arg, (-1, arg.size(2)))
        ret.append(arg)
    return ret

def calc_accs_mean(*args):
    # arg should be tuples ()
    # tensors to store mean accuracies
    # arg = accs_?_cat, arg2 = accs_?_mean
    ret = []
    for arg,arg2 in args:
        mn = torch.reshape(torch.mean(arg[0::2, :], 0), (1,-1))
        arg2 = torch.cat((arg2, mn),0)
        mn = torch.reshape(torch.mean(arg[1::2, :], 0), (1,-1))
        arg2 = torch.cat((arg2, mn),0)
        ret.append(arg2)
    return ret
    # this should hopefully reshape & store without needing to return stuff