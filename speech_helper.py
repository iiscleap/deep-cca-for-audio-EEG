import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io

from cca_functions    import *

num_blocks = 20


# NUMBER OF CHANNELS IN THE PROCESSED STIMULI AFTER FILTERBANK
stim_chans = 21
# NUMBER OF CHANNELS IN prePREPROCESSED STIMULI (1D)
stim_chans_pre = 1


# HELPER FUNCTION TO LOAD DATA FOR SPEECH
def load_data(subs, block=0, filtered=True):
    """
    THIS IS VALID ONLY FOR THE LIBERTO ET AL. AUDIOBOOK SPEECH DATA. OR DATA SIMILAR TO THAT (WHERE ALL THE SUBJECTS LISTEN TO THE SAME STIMULI).
    ARGUMENTS:
        subs:  IDS OF THE SUBJECTS FOR WHICH LCCA IS TO BE PERFORMED.
        block: OUT OF THE 20 CROSS-FOLD VALIDATION BLOCKS, WHICH BLOCK IS TO BE CHOSEN FOR TESTING. 
               OUT OF THE REMAINING, 18 BLOCKS ARE FOR TRAINING AND 1 FOR VALIDATING.
        filtered: TO SPECIFY WHETHER THE DATA IS ALREADY PROCESSED AS IN LCCA METHOD OR NOT.
                IF YES, THE DATA MUST HAD BEEN FILTERED (AND PCA TO 139D, FOR THE STIMULUS) AND ARE READY TO BE SENT TO CCA MODELS.
    
    RETURNS: 
        data_subs: AN (N+1) ELEMENT LIST WITH FIRST N ELEMENTS FOR THE SUBJECTS' DATA AND THE LAST ELEMENT FOR THE COMMON STIMULUS DATA.
    """
    # IF THE DATA IS ALREADY PROCESSED THROUGH THE FILTERBANK AND PCA
    print('block: ' + str(block))
    if block == num_blocks - 1:
        val_idx = 0
    else:
        val_idx = block + 1

    if filtered:
        # data_path = '/data2/data/dmcca/data/data_'+str(block)+'.pkl'
        data_path = 'data_'+str(block)+'.pkl' # REPLACE THE DATA PATH HERE
        fp = open(data_path, 'rb')
        data1 = pkl.load(fp)
        fp.close()
        print("Loaded FILTERED Data.")

        print('Data INITIALIZED for block : {}'.format(str(block)))
        data_subs = []
        for sub in subs:
            data_subs.append([data1[0][:,:,sub], data1[1][:,:,sub], data1[2][:,:,sub]])
        data_subs.append([data1[0][:,:stim_chans,-1], data1[1][:,:stim_chans,-1], data1[2][:,:stim_chans,-1]])
        del data1
    else:
        # IF THE DATA IS NOT PROCESSED
        data_path = 'data_pre_'+str(block)+'.pkl'  # REPLACE THE DATA PATH HERE
        fp = open(data_path, 'rb')
        pre_data = pkl.load(fp)
        fp.close()
        print("Loaded DEMEANED Data.")

        print('Data INITIALIZED for block : {}'.format(str(block)))
        data_subs_pre = []
        for sub in subs:
            data_subs_pre.append([pre_data[0][:,:,sub], pre_data[1][:,:,sub], pre_data[2][:,:,sub]])

        data_stim_pre = [pre_data[0][:,:stim_chans_pre,-1], pre_data[1][:,:stim_chans_pre,-1], pre_data[2][:,:stim_chans_pre,-1]]
        # LOAD THE pre PREPROCESSED DATA HERE 
        # AND THEN PROCESS IT
        # data_subs_pre HAS N SUBJECTS' RESPONSES SUCH THAT
        # data_subs_pre[n] = [TRAINING_DATA, VALIDATION_DATA, TESTING_DATA]
        # WHERE
        # data_subs_pre[n][j].shape = [Number_of_samples, dimensions]
        # data_stim_pre IS ALSO PRESENT IN THE SAME WAY AS data_subs_pre[n].
        # ASSUMPTION: THE STIMULI DATA IS OF 1 DIMENSION.
        # IF NOT:
        # WE CAN either DO PCA ONTO 1D AND THEN DO FILTERBANK.
        # or FILTERBANK AND THEN, PCA.

        processed_data_subs = []
        for data_sub in data_subs_pre:
            processed_data_subs.append(pca_filt_pca_resp(data_sub))
        
        processed_data_subs.append(filtone(data_stim_pre[0], data_stim_pre[1], data_stim_pre[2]))

        data_subs = list(processed_data_subs)
        del processed_data_subs

    return data_subs


# HELPER FUNCTION TO LOAD DATA FOR SPEECH
def load_mcca_data(subs, block=0):
    """
    THIS IS VALID ONLY FOR THE LIBERTO ET AL. AUDIOBOOK SPEECH DATA. OR DATA SIMILAR TO THAT (WHERE ALL THE SUBJECTS LISTEN TO THE SAME STIMULI).
    ARGUMENTS:
        subs:  IDS OF THE SUBJECTS FOR WHICH LCCA IS TO BE PERFORMED.
        block: OUT OF THE 20 CROSS-FOLD VALIDATION BLOCKS, WHICH BLOCK IS TO BE CHOSEN FOR TESTING. 
               OUT OF THE REMAINING, 18 BLOCKS ARE FOR TRAINING AND 1 FOR VALIDATING.
    
    RETURNS: 
        data_subs: AN (N+1) ELEMENT LIST WITH FIRST N ELEMENTS FOR THE SUBJECTS' DATA AND THE LAST ELEMENT FOR THE COMMON STIMULUS DATA.
    """
    # IF THE DATA IS ALREADY PROCESSED THROUGH THE FILTERBANK AND PCA
    print('block: ' + str(block))
    if block == num_blocks - 1:
        val_idx = 0
    else:
        val_idx = block + 1

    # IF THE DATA IS NOT PROCESSED
    data_path = '/data2/jaswanthr/data/dmcca/data/data1_raw_'+str(block)+'.pkl'
    fp = open(data_path, 'rb')
    pre_data = pkl.load(fp)
    fp.close()
    print("Loaded DEMEANED Data.")

    print('Data INITIALIZED for block : {}'.format(str(block)))
    data_subs_pre = []
    for sub in subs:
        data_subs_pre.append([pre_data[0][:,:,sub], pre_data[1][:,:,sub], pre_data[2][:,:,sub]])
    data_subs_pre.append([pre_data[0][:,:stim_chans_pre,-1], pre_data[1][:,:stim_chans_pre,-1], pre_data[2][:,:stim_chans_pre,-1]])

    return data_subs_pre


# HELPER FUNCTION TO LOAD DATA FOR SPEECH
def load_dmcca_data(subs, mid_shape, block=0):
    """
    THIS IS VALID ONLY FOR THE LIBERTO ET AL. AUDIOBOOK SPEECH DATA. OR DATA SIMILAR TO THAT (WHERE ALL THE SUBJECTS LISTEN TO THE SAME STIMULI).
    ARGUMENTS:
        subs:  IDS OF THE SUBJECTS FOR WHICH LCCA IS TO BE PERFORMED.
        mid_shape : TIME-LAGS APPLIED TO THE STIMLUS
        block: OUT OF THE 20 CROSS-FOLD VALIDATION BLOCKS, WHICH BLOCK IS TO BE CHOSEN FOR TESTING. 
               OUT OF THE REMAINING, 18 BLOCKS ARE FOR TRAINING AND 1 FOR VALIDATING.
    
    RETURNS: 
        data_subs: AN (N+1) ELEMENT LIST WITH FIRST N ELEMENTS FOR THE SUBJECTS' DATA AND THE LAST ELEMENT FOR THE COMMON STIMULUS DATA.
    """
    # IF THE DATA IS ALREADY PROCESSED THROUGH THE FILTERBANK AND PCA
    print('block: ' + str(block))
    if block == num_blocks - 1:
        val_idx = 0
    else:
        val_idx = block + 1

    # IF THE DATA IS NOT PROCESSED
    data_path = '/data2/jaswanthr/data/dmcca/data/data1_raw_'+str(block)+'.pkl'
    fp = open(data_path, 'rb')
    pre_data = pkl.load(fp)
    fp.close()
    print("Loaded DEMEANED Data.")

    print('Data INITIALIZED for block : {}'.format(str(block)))
    data_subs_pre = []
    for sub in subs:
        data_subs_pre.append([pre_data[0][:,:,sub], pre_data[1][:,:,sub], pre_data[2][:,:,sub]])
    data_subs_pre.append([pre_data[0][:,:stim_chans_pre,-1], pre_data[1][:,:stim_chans_pre,-1], pre_data[2][:,:stim_chans_pre,-1]])


    stim_lagged_midshape = [None, None, None]
    for i in range(3):
        stim_lagged_midshape[i] = lagGen(data_subs_pre[-1][i], np.arange(mid_shape))

    data_subs_pre[-1] = stim_lagged_midshape

    return data_subs_pre
