

import torch
import os
import os.path
import numpy as np



def pt_var_to_numpy(var):
    return (var.data).cpu().numpy()


def last_checkpoint(path):
    list_elts = os.listdir(path)

    list_chkpt = []
    for elt in list_elts:
        if elt[-3:] == ".pt":
            list_chkpt.append(int(elt[:-3]))

    return str(max(list_chkpt)).zfill(6)+".pt"# list_elts[list_elts.index(str(max(list_chkpt))+".pt")]


def get_reg_loss(model, targeted_name, reg_type):

    reg_loss = 0.0

    if reg_type=='l2':
        for name, p in model.named_parameters():
            if targeted_name in name:
                reg_loss += (p.pow(2).sum())
    elif reg_type=='l1':
        for name, p in model.named_parameters():
            if targeted_name in name:
                reg_loss += (p.abs().sum())

    return reg_loss


def get_param_crf(model):

    for name, p in model.named_parameters():
        if 'psi_0' in name:
            psi_0 = p
        elif 'psi_1' in name:
            psi_1 = p

    return psi_0, psi_1


class Cumulator():

    def __init__(self, num_classes):
        self.accumuled = np.array([[0]*num_classes])
    
    def append(self, np_vector):
        np_vector = np.transpose(np_vector, [0,2,1]).reshape([-1,np_vector.shape[1]])
        self.accumuled = np.concatenate((self.accumuled, np_vector))
    
    def clear(self):
        self.accumuled = []