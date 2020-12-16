
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io

from cca_functions    import *


# ORIG -> REV -> PHASE -> MEAS
def stim_resp(resp_tr_a, resp_tr_b, resp_val_a, resp_val_b, resp_te_a, resp_te_b, stim_tr_3d, stim_val_3d, stim_te_3d, count):
    resp_tr = None
    if resp_tr_a is not None and resp_tr_b is not None:
        resp_tr = np.concatenate([resp_tr_a[0], resp_tr_b[0]],1)
        # print(resp_tr.shape)
        for i in range(1,count):
            temp    = np.concatenate([resp_tr_a[i], resp_tr_b[i]],1)
            # print(temp.shape)
            resp_tr = np.concatenate([resp_tr, temp],1)
        # print(resp_tr.shape)
        resp_tr = resp_tr.T

    resp_val = None
    if resp_val_a is not None and resp_val_b is not None:
        resp_val = np.concatenate([resp_val_a[0], resp_val_b[0]],1)
        # print(resp_val.shape)
        for i in range(1,count):
            temp    = np.concatenate([resp_val_a[i], resp_val_b[i]],1)
            # print(temp.shape)
            resp_val = np.concatenate([resp_val, temp],1)
        # print(resp_val.shape)
        resp_val = resp_val.T

    resp_te = None
    if resp_te_a is not None and resp_te_b is not None:
        resp_te = np.concatenate([resp_te_a[0], resp_te_b[0]],1)
        # print(resp_te.shape)
        for i in range(1,count):
            temp    = np.concatenate([resp_te_a[i], resp_te_b[i]],1)
            # print(temp.shape)
            resp_te = np.concatenate([resp_te, temp],1)
        # print(resp_te.shape)
        resp_te = resp_te.T

    stim_tr = None
    if stim_tr_3d is not None:
        stim_tr = np.concatenate([stim_tr_3d[0], stim_tr_3d[0]],1)
        # print(stim_tr.shape)
        for i in range(1,count):
            temp    = np.concatenate([stim_tr_3d[i], stim_tr_3d[i]],1)
            # print(temp.shape)
            stim_tr = np.concatenate([stim_tr, temp],1)
        # print(stim_tr.shape)
        stim_tr = stim_tr.T

    stim_val = None
    if stim_val_3d is not None:
        stim_val = np.concatenate([stim_val_3d[0], stim_val_3d[0]],1)
        # print(stim_val.shape)
        for i in range(1,count):
            temp    = np.concatenate([stim_val_3d[i], stim_val_3d[i]],1)
            # print(temp.shape)
            stim_val = np.concatenate([stim_val, temp],1)
        # print(stim_val.shape)
        stim_val = stim_val.T

    stim_te = None
    if stim_te_3d is not None:
        stim_te = np.concatenate([stim_te_3d[0], stim_te_3d[0]],1)
        # print(stim_te.shape)
        for i in range(1,count):
            temp    = np.concatenate([stim_te_3d[i], stim_te_3d[i]],1)
            # print(temp.shape)
            stim_te = np.concatenate([stim_te, temp],1)
        # print(stim_te.shape)
        stim_te = stim_te.T

    return [stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te]
