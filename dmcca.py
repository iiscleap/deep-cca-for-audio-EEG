import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io
import random
# from pdb import set_trace as bp  #################added break point accessor####################

import torch 

from cca_functions  import *
from speech_helper  import load_dmcca_data
from music_helper   import stim_resp
from deep_models    import dcca_model, dmcca_model

def plot_data(x, y,s):
    plt.clf()
    x = x[0]
    y = y[0]
    plt.plot(x, color='orange')
    plt.plot(y, color='blue')
    plt.legend(['stim', 'resp'])
    plt.savefig(f'{s}.eps', format="eps")

def plot_losses_tr_val_te(losses, s, marker="o"):
    plt.clf()
    plt.plot(losses[:, 0], marker=marker, color='red')
    plt.plot(losses[:, 1], marker=marker, color='blue')
    plt.plot(losses[:, 2], marker=marker, color='green')
    plt.legend(['training', 'valid', 'test'])
    # plt.savefig(s+'.png', format="png")
    plt.savefig(s+'.eps', format="eps")

name_of_the_script = sys.argv[0].split('.')[0]
a = sys.argv[1:]
eyedee = str(a[0])  # ID OF THE EXPERIMENT.
# o_dim = int(a[1])   # THE INTERESTED OUTPUTS DIMENSIONALITY
num_blocks_start = int(a[1])
num_blocks_end   = int(a[2])
lambda_          = float(a[3])
mid_shape        = int(a[4])
D                = [float(x) for x in a[5:]]

# dropout    = 0.05
learning_rate = 1e-3
epoch_num  = 20
batch_size = 800
reg_par    = 1e-4
o_dim      = 1
use_all_singular_values = False
best_only  = True

print(f"eyedee    : {eyedee}")
print(f"best_only : {best_only}")
print(f"epoch_num : {epoch_num}")
# print(f"dropout   : {dropout}")

device = torch.device('cuda')
torch.cuda.empty_cache()

# CREATING A FOLDER TO STORE THE RESULTS
path_name = f"dmcca_{eyedee}_{num_blocks_start}_{num_blocks_end}_{lambda_}_{mid_shape}_{D[0]}/"

i = 1
while path.exists(path_name):
    path_name = f"dmcca_{eyedee}_{num_blocks_start}_{num_blocks_end}_{lambda_}_{mid_shape}_{D[0]}_{i}/"
    i = i + 1

del i
os.mkdir(path_name)
print(path_name)


##################### SEED #####################
# seed = np.ceil(np.random.rand(10)*100)
seed = np.ceil(np.random.rand(1)*100) * np.ones(1)
print(seed)
###############################################

# D = [0, 0.05, 0.1, 0.2]
# D = [0.05, 0.2]
# CAN REPLACE D WITH A SINGLE ELEMENT LIST WHOSE VALUE IS EQUAL TO THE DESIRED DROPOUT.

# COEFFICIENT TO THE MSE REGULARIZATION LOSS OF THE DECODER
# lambda_       = 0.1

# MIDDLE LAYER UNITS IN THE DMCCA ARCHITECTURE
# IS ALSO THE TIME-LAGS APPLIED TO THE STIMULUS
# mid_shape  = 60

# HELPER FUNCTION FOR PERFORMING DCCA
def dcca_method(stim_data, resp_data, dropout, dataset, saving_name_root):
    """
    CUSTOM DCCA METHOD
    """
    print(f"DCCA for {saving_name_root}")

    new_data_d, correlations, model_d = dcca_model(stim_data, resp_data, o_dim, learning_rate, use_all_singular_values, epoch_num, batch_size, reg_par, dropout, best_only, path_name, seed)

    x1 = new_data_d[2][0]
    x2 = new_data_d[2][1]
    x3 = new_data_d[1][0]
    x4 = new_data_d[1][1]
    corr_d     = np.squeeze(my_corr(x1, x2, o_dim))
    corr_d_val = np.squeeze(my_corr(x1, x2, o_dim))
    print(f'DCCA is : {[corr_d, corr_d_val]}')

    # PLOTTING THE NEW DATA
    plot_data_name = f"{path_name}/{dataset}_plot_dmdc_data_{saving_name_root}"
    plot_data(x1, x2, plot_data_name)

    # # PLOTTING THE TRAINING LOSSES
    # s = f"{path_name}/{dataset}_plot_losses_{saving_name_root}"
    # plot_losses_tr_val_te(correlations, s)

    # SAVING THE NEW DATA
    save_data_name = f"{path_name}/{dataset}_dmdc_data_{saving_name_root}.pkl"
    fp = open(save_data_name, 'wb')
    pkl.dump(new_data_d, fp)
    fp.close()

    # SAVING THE DCCA MODEL
    save_model_name = f"{path_name}/{dataset}_dmdc_model_{saving_name_root}.path.tar"
    torch.save(model_d, save_model_name)
    # save_dict_name = f"{path_name}/{dataset}_dmdc_model_dict_{saving_name_root}.pth.tar"
    # torch.save({'state_dict': model_d.state_dict()}, save_dict_name)
    del model_d

    return [corr_d, corr_d_val]


# HELPER FUNCTION FOR PERFORMING LCCA
def lcca_method(stim_data, resp_data, dataset, saving_name_root):
    """
    CUSTOM LCCA METHOD
    """
    print(f"LCCA for {saving_name_root}")

    _, new_data_l = cca_model(stim_data, resp_data, o_dim)
    x1 = new_data_l[2][0] ; x3 = new_data_l[1][0]
    x2 = new_data_l[2][1] ; x4 = new_data_l[1][1]
    corr_l = [np.squeeze(my_corr(x1, x2, o_dim)), np.squeeze(my_corr(x3, x4, o_dim))]
    print(f'LCCA is : {corr_l}')

    s = f"{path_name}/{dataset}_plot_dmlc_data_{saving_name_root}"
    plot_data(my_standardize(x1), my_standardize(x2), s)

    fp = open(f'{path_name}/{dataset}_dmlc_data_{saving_name_root}.pkl', 'wb')
    pkl.dump(new_data_l, fp)
    fp.close()
    del new_data_l

    return corr_l[0], corr_l[1]


def dmcca_method(all_data, dataset, dropout, saving_name_root):
    o_dim = 10
    # providing the data to DMCCA model
    dmcca_data, training_losses, dmcca_model_ = dmcca_model(all_data, o_dim, learning_rate, use_all_singular_values, epoch_num, batch_size, reg_par, dropout, best_only, lambda_, path_name, mid_shape, seed)

    # SAVING THE DMCCA MODEL
    save_model_name = f"{path_name}/{dataset}_dmcca_model_{saving_name_root}.path.tar"
    torch.save(dmcca_model_, save_model_name)
    # save_dict_name = f"{path_name}/{dataset}_dmcca_dict_{saving_name_root}.pth.tar"
    # torch.save({'state_dict': dmcca_model.state_dict()}, save_dict_name)
    del dmcca_model_

    # TO MAKE SURE EVERYTHING IS in CPU and NUMPY
    for gg in range(3):
        for hh,_ in enumerate(dmcca_data[gg]):
            for ii,_ in enumerate(dmcca_data[gg][hh]):
                if torch.is_tensor(dmcca_data[gg][hh][ii]):
                    dmcca_data[gg][hh][ii] = dmcca_data[gg][hh][ii].cpu().numpy()

    new_dmcca_data = dmcca_data
    del dmcca_data

    # SAVING THE DMCCA DATA
    fp = open(f'{path_name}/{dataset}_dmcca_data_{saving_name_root}.pkl', 'wb')
    pkl.dump(new_dmcca_data, fp)
    fp.close()
    del new_dmcca_data

    n_subs = len(all_data) - 1

    dmdc_corrs     = np.zeros((n_subs))
    dmdc_corrs_val = np.zeros((n_subs))

    dmlc_corrs     = np.zeros((n_subs))
    dmlc_corrs_val = np.zeros((n_subs))

    for sub in range(6, n_subs):
        print(f"Sub: {subs[sub]}")

        data_subs = pkl.load(open(f'{path_name}/{dataset}_dmcca_data_{saving_name_root}.pkl', 'rb'))
        data_stim = [data_subs[0][0][-1],  data_subs[1][0][-1],  data_subs[2][0][-1]]
        data_sub  = [data_subs[0][0][sub], data_subs[1][0][sub], data_subs[2][0][sub]]
        del data_subs

        new_stim_data, new_resp_data, _, _ = pca_stim_filtem_pca_resp(data_sub, data_stim)

        # DMCCA + LCCA
        print(f"DMCCA + LCCA : {saving_name_root}")
        dmlc_corrs[sub], dmlc_corrs_val[sub] = lcca_method(new_stim_data, new_resp_data, dataset, f"{saving_name_root}_sub_{sub}")

        # DMCCA + DCCA METHOD.
        print(f"DMCCA + DCCA : {saving_name_root}")
        dmdc_corrs[sub], dmdc_corrs_val[sub] = dcca_method(data_stim, data_sub, dropout, dataset, f"{saving_name_root}_sub_{sub}")

        print(f'DMDC corrs are : {dmdc_corrs}')
    
    os.remove(f'{path_name}/{dataset}_dmcca_data_{saving_name_root}.pkl')

    print(f'DONE {dataset} - {saving_name_root}.')
    return [dmlc_corrs, dmlc_corrs_val], [dmdc_corrs, dmdc_corrs_val]





speech_dmcca = True
if speech_dmcca:
    num_blocks = 20        # IF SPEECH DATA BY LIBERTO ET AL.

    # subs ARE THE SUBJECTS IDS TO WORK WITH
    subs = [1, 2]       # REPLACE WITH THE REQUIRED SUBJECTS' IDS.
    subs = sorted(subs) # TO KEEP THEIR IDS SORTED
    n_subs = len(subs)

    str_subs = str(subs[0])
    for each_sub in subs[1:]: 
        str_subs += f"_{each_sub}"

    # num_blocks_start = 0
    # num_blocks_end   = 1
    # CAN CHANGE BOTH VALUES ACCORDING TO THE INTERESTED CROSS-VALIDATION EXPERIMENTS.
    # CAN SUBMIT THESE TWO AS THE ARGUMENTS AND PARSE OVER THERE, FOR BULK EXPERIMENTS.

    tst_corrs = np.zeros((2, num_blocks, len(D), n_subs))
    val_corrs = np.zeros((2, num_blocks, len(D), n_subs))
    tst_corrs_name =  f'{path_name}/speech_corrs_{str_subs}.npy'
    val_corrs_name =  f'{path_name}/speech_corrs_val_{str_subs}.npy'

    print(f"n_subs     : {n_subs}")
    print(f"subs       : {subs}")
    print(f"D          : {D}")
    print(f"num_blocks : {num_blocks}")
    print(f"num_blocks_start: {num_blocks_start}")
    print(f"num_blocks_end  : {num_blocks_end}")
    print(f"num_blocks_net  : {num_blocks_end - num_blocks_start}")

    for d_cnt, dropout in enumerate(D):
        for block in range(num_blocks_start, num_blocks_end):
            # THE DATA data_subs_pre IS LOADED SUCH THAT 
            # ALL THE N EEG RESPONSES ARE LOADED IN THE FIRST N LISTS
            # AND THE LAST LIST HAS STIMULUS
            # data_subs_pre IS A list OF SIZE N+1
            # EACH ELEMENT IS A list OF SIZE 3
            # SUCH THAT
            # data_subs_pre[n] = [TRAINING_DATA, VALIDATION_DATA, TESTING_DATA]
            # AND
            # data_subs_pre[n][j].shape = [Number_of_samples, dimensions]
            data_subs_pre = load_dmcca_data(subs, mid_shape, block)

            ## DEEP MCCA
            print("DEEP MCCA + LCCA")

            dmlcs, dmdcs = dmcca_method(data_subs_pre, "speech", dropout, f"block_{block}_drpt_{dropout}")

            tst_corrs[0, block, d_cnt] = dmlcs[0]
            tst_corrs[1, block, d_cnt] = dmdcs[0]

            val_corrs[0, block, d_cnt] = dmlcs[1]
            val_corrs[1, block, d_cnt] = dmdcs[1]

            np.save(tst_corrs_name, tst_corrs)
            np.save(val_corrs_name, val_corrs)

            print('saved SPEECH')


nmedh_dmcca = False
if nmedh_dmcca:
    # subs ARE THE SUBJECTS IDS TO WORK WITH
    
    # THE 4 STIMULI FEATURES ARE ORDERED AS:
    # ENV -> PCA1 -> FLUX -> RMS
    tst_corrs = np.zeros((2, 4, len(D), 16, 12))
    val_corrs = np.zeros((2, 4, len(D), 16, 12))
    tst_corrs_name = f'{path_name}/nmedh_corrs.npy'
    val_corrs_name = f'{path_name}/nmedh_corrs_val.npy'

    stims = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    stims_types = ["ENV", "PC1", "FLX", "RMS"] 

    for stim_id, stim_str in enumerate(stims_types):
        for stim_num, stim__ in enumerate(stims):

            print(f"Stimulus Feature: {stim_str}, Stimulus Number : {stim__}")
            lmc_corrs = np.zeros(tst_corrs.shape[1])
            # data_path = '/data2/data/nmed-h/stim_data2/'
            data_path = None # LOAD YOUR DATA PATH HERE

            # LOAD DATA

            # "mcca_{stim__}.pkl" IS ARRANGED SUCH THAT IT CONTAINS A LIST OF TWO ITEMS:
            # 0: PREPROCESSED 125D CLEAN EEG DATA
            # 1: PREPROCESSED 1D COMMON STIMULUS DATA 
            # BOTH AS [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
            #
            # 0 IS A LIST OF 12 SUBJECTS' PREPROCESSED EEG DATA
            # ARRANGED AS [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
            # EACH ONE IS IN THE SHAPE T x 125 
            # 1, [0, 1, 2], 0: PREPROCESSED 1D ENVELOPE      DATA
            # 1, [0, 1, 2], 1: PREPROCESSED 1D PCA1          DATA
            # 1, [0, 1, 2], 2: PREPROCESSED 1D SPECTRAL FLUX DATA
            # 1, [0, 1, 2], 3: PREPROCESSED 1D RMS           DATA
            # ALL AS [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
            # EACH STIMULI FEATURE IS IN THE SHAPE T x 1

            mcca_data = pkl.load(open(f"{data_path}/mcca_{stim__}.pkl", "rb"))
            all_data = mcca_data[0]
            stim_data = [mcca_data[1][0][:,stim_id].reshape(-1,1), mcca_data[1][1][:,stim_id].reshape(-1,1), mcca_data[1][2][:,stim_id].reshape(-1,1)]

            all_data.append(stim_data)

            for d_cnt, dropout in enumerate(D):

                dmlcs, dmdcs = dmcca_method(data_subs_pre, "music", f"{stim_str}_{stim__}_drpt_{dropout}")

                tst_corrs[0, stim_id, stim_num] = dmlcs[0]
                tst_corrs[1, stim_id, stim_num] = dmdcs[0]

                val_corrs[0, stim_id, stim_num] = dmlcs[1]
                val_corrs[1, stim_id, stim_num] = dmdcs[1]

                np.save(tst_corrs_name, tst_corrs)
                np.save(val_corrs_name, val_corrs)

    print(f'saved music.')




# FOR CUSTOM,
# ONE CAN REPLACE THE all_data LIST WITH THE INTERESTED DATASET.
# all_data IS A LIST OF N+1 ITEMS
# FIRST N ITEMS BELONG TO EEG RECORDINGS OF N SUBJECTS RESPECTIVELY.
# LAST  1 ITEM BELONGS TO THE COMMON STIMULI PROVIDED TO ALL THE SUBJECTS
# EACH ITEM OF THE (N+1) LENGTH LIST IS ARRANGED AS 
# [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
# EACH OF THESE DATA ARE IN THE SHAPE : NUMBER OF SAMPLES X VECTOR DIMENSION OF EACH SAMPLE
#
# AFTER LOADING THE DATA INTO all_data,
# ONE CAN CALL THE dmcca_method FUNCTION ON IT
# Then process the data through PCA and filterbank
# Then provide the data to LCCA or DCCA models to obtain final representations


