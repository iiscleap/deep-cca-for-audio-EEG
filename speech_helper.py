import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io
# from pdb import set_trace as bp  #################added break point accessor####################

from cca_functions    import *

num_blocks = 20


# NUMBER OF CHANNELS IN THE PROCESSED STIMULI AFTER FILTERBANK
stim_chans = 21
# NUMBER OF CHANNELS IN prePREPROCESSED STIMULI (1D)
stim_chans_pre = 1


# HELPER FUNCTION TO LOAD DATA FOR SPEECH for LCCA and DCCA
def load_data(subs, block=0):
    """
    THIS IS VALID FOR THE LIBERTO ET AL. AUDIOBOOK SPEECH DATA. OR DATA SIMILAR TO THAT (WHERE ALL THE SUBJECTS LISTEN TO THE SAME STIMULI).
    ARGUMENTS:
        subs:  IDS OF THE SUBJECTS FOR WHICH LCCA IS TO BE PERFORMED.
        block: OUT OF THE 20 CROSS-FOLD VALIDATION BLOCKS, WHICH BLOCK IS TO BE CHOSEN FOR TESTING. 
               OUT OF THE REMAINING, 18 BLOCKS ARE FOR TRAINING AND 1 FOR VALIDATING.
    
    RETURNS: 
        data_subs: AN (N+1) ELEMENT LIST WITH FIRST N ELEMENTS FOR THE SUBJECTS' DATA AND THE LAST ELEMENT FOR THE COMMON STIMULUS DATA.
    """

    print('block: ' + str(block))
    if block == num_blocks - 1: val_idx = 0
    else:                       val_idx = block + 1

    # LOAD THE prePREPROCESSED DATA HERE 
    # AND THEN PROCESS IT
    # data_subs_pre will have N SUBJECTS' RESPONSES SUCH THAT
    # data_subs_pre[n] = [TRAINING_DATA, VALIDATION_DATA, TESTING_DATA]
    # WHERE
    # data_subs_pre[n][j].shape = [Number_of_samples, dimensions]
    # data_stim_pre IS ALSO PRESENT IN THE SAME WAY AS data_subs_pre[n].
    # ASSUMPTION: THE STIMULI DATA IS OF 1 DIMENSION.
    # IF NOT:
    #    WE CAN either DO PCA ONTO 1D AND THEN DO FILTERBANK.
    #    or FILTERBANK AND THEN, PCA to 21D.

    # Each subject's stimulus and response are preprocessed are saved as "Subject{sub}_Preprocessed_ENV_EEG.mat"
    # The data are processed using "preProcessEEG_Audio_Chevigne_2018.m"

    folder_path = "/speech_data/" # Path to the data folder here

    print('Data INITIALIZING for block : {}'.format(str(block)))
    data_subs_pre = []
    for sub in subs:
        resp_data = scipy.io.loadmat(f"{folder_path}/Subject{sub}_Preprocessed_ENV_EEG.mat")["resp"][0]

        # Loading the response data, dividing them into training, validation and test data
        
        resp_train = np.concatenate([resp_data[x] for x in range(len(resp_data)) if x not in [block, val_idx]], 0)
        resp_val   = resp_data[val_idx]
        resp_test  = resp_data[block]

        data_subs_pre.append([resp_train, resp_val, resp_test])
        
    # Loading the stimulus data, dividing them into training, validation and test data
    stim_data = scipy.io.loadmat(f"{folder_path}/Subject{sub}_Preprocessed_ENV_EEG.mat")["stim"][0]
    stim_train = np.concatenate([stim_data[x] for x in range(len(stim_data)) if x not in [block, val_idx]], 0)
    stim_val   = stim_data[val_idx]
    stim_test  = stim_data[block]

    data_stim_pre = [stim_train, stim_val, stim_test]

    # processing the response by
    # PCA to 60D ------> filterbank (21 filters) to 1260D ------> PCA to 139D
    # processing the stimulus by 
    # stimulus to filterbank => 21D 
    #
    # USED "pca_filt_pca_resp" from cca_functions to perform this 
    processed_data_subs = []
    for data_sub in data_subs_pre:
        processed_data_subs.append(pca_filt_pca_resp(data_sub))
    
    processed_data_subs.append(filtone(data_stim_pre[0], data_stim_pre[1], data_stim_pre[2]))

    data_subs = list(processed_data_subs)
    del processed_data_subs

    return data_subs


# HELPER FUNCTION TO LOAD DATA FOR SPEECH for LMCCA
def load_mcca_data(subs, block=0):
    """
    THIS IS VALID FOR THE LIBERTO ET AL. AUDIOBOOK SPEECH DATA. OR DATA SIMILAR TO THAT (WHERE ALL THE SUBJECTS LISTEN TO THE SAME STIMULI).
    ARGUMENTS:
        subs:  IDS OF THE SUBJECTS FOR WHICH LCCA IS TO BE PERFORMED.
        block: OUT OF THE 20 CROSS-FOLD VALIDATION BLOCKS, WHICH BLOCK IS TO BE CHOSEN FOR TESTING. 
               OUT OF THE REMAINING, 18 BLOCKS ARE FOR TRAINING AND 1 FOR VALIDATING.
    
    RETURNS: 
        data_subs: AN (N+1) ELEMENT LIST WITH FIRST N ELEMENTS FOR THE SUBJECTS' DATA AND THE LAST ELEMENT FOR THE COMMON STIMULUS DATA.
    """

    print('block: ' + str(block))
    if block == num_blocks - 1: val_idx = 0
    else:                       val_idx = block + 1

    # Each subject's stimulus and response are preprocessed are saved as "Subject{sub}_Preprocessed_ENV_EEG.mat"
    # The data are processed using "preProcessEEG_Audio_Chevigne_2018.m"

    folder_path = "/speech_data/" # Path to the data folder here

    print('Data INITIALIZING for block : {}'.format(str(block)))
    data_subs_pre = []
    for sub in subs:
        resp_data = scipy.io.loadmat(f"{folder_path}/Subject{sub}_Preprocessed_ENV_EEG.mat")["resp"][0]
        
        # Loading the response data, dividing them into training, validation and test data

        resp_train = np.concatenate([resp_data[x] for x in range(len(resp_data)) if x not in [block, val_idx]], 0)
        resp_val   = resp_data[val_idx]
        resp_test  = resp_data[block]

        data_subs_pre.append([resp_train, resp_val, resp_test])

    # Loading the stimulus data, dividing them into training, validation and test data
    stim_data = scipy.io.loadmat(f"/data2/jaswanthr/data/Subject{sub}_Preprocessed_ENV_EEG.mat")["stim"][0]
    stim_train = np.concatenate([stim_data[x] for x in range(len(stim_data)) if x not in [block, val_idx]], 0)
    stim_val   = stim_data[val_idx]
    stim_test  = stim_data[block]

    data_stim_pre = [stim_train, stim_val, stim_test]
    data_subs_pre.append(data_stim_pre)

    return data_subs_pre


# HELPER FUNCTION TO LOAD DATA FOR SPEECH for DMCCA
def load_dmcca_data(subs, mid_shape, block=0):
    """
    THIS IS VALID FOR THE LIBERTO ET AL. AUDIOBOOK SPEECH DATA. OR DATA SIMILAR TO THAT (WHERE ALL THE SUBJECTS LISTEN TO THE SAME STIMULI).
    ARGUMENTS:
        subs:  IDS OF THE SUBJECTS FOR WHICH LCCA IS TO BE PERFORMED.
        mid_shape : TIME-LAGS APPLIED TO THE STIMLUS
        block: OUT OF THE 20 CROSS-FOLD VALIDATION BLOCKS, WHICH BLOCK IS TO BE CHOSEN FOR TESTING. 
               OUT OF THE REMAINING, 18 BLOCKS ARE FOR TRAINING AND 1 FOR VALIDATING.
    
    RETURNS: 
        data_subs: AN (N+1) ELEMENT LIST WITH FIRST N ELEMENTS FOR THE SUBJECTS' DATA AND THE LAST ELEMENT FOR THE COMMON STIMULUS DATA.
    """
    
    print('block: ' + str(block))
    if block == num_blocks - 1: val_idx = 0
    else:                       val_idx = block + 1

    # Each subject's stimulus and response are preprocessed are saved as "Subject{sub}_Preprocessed_ENV_EEG.mat"
    # The data are processed using "preProcessEEG_Audio_Chevigne_2018.m"

    folder_path = "/speech_data/" # Path to the data folder here

    print('Data INITIALIZING for block : {}'.format(str(block)))
    data_subs_pre = []
    for sub in subs:
        resp_data = scipy.io.loadmat(f"{folder_path}/Subject{sub}_Preprocessed_ENV_EEG.mat")["resp"][0]
        # Loading the response data, dividing them into training, validation and test data
        resp_train = np.concatenate([resp_data[x] for x in range(len(resp_data)) if x not in [block, val_idx]], 0)
        resp_val   = resp_data[val_idx]
        resp_test  = resp_data[block]

        data_subs_pre.append([resp_train, resp_val, resp_test])

    # Loading the stimulus data, dividing them into training, validation and test data
    stim_data = scipy.io.loadmat(f"{folder_path}/Subject{sub}_Preprocessed_ENV_EEG.mat")["stim"][0]
    stim_train = np.concatenate([stim_data[x] for x in range(len(stim_data)) if x not in [block, val_idx]], 0)
    stim_val   = stim_data[val_idx]
    stim_test  = stim_data[block]

    data_stim_pre = [stim_train, stim_val, stim_test]
    data_subs_pre.append(data_stim_pre)

    # Applying stimulus lag (d_S)
    stim_lagged_midshape = [None, None, None]
    for i in range(3):
        stim_lagged_midshape[i] = lagGen(data_subs_pre[-1][i], np.arange(mid_shape))

    data_subs_pre[-1] = stim_lagged_midshape

    return data_subs_pre
