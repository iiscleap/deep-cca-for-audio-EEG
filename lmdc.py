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
from speech_helper  import load_mcca_data
from music_helper   import stim_resp
from deep_models    import dcca_model

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
o_dim = int(a[1])   # THE INTERESTED OUTPUTS DIMENSIONALITY

dropout    = 0.05
learning_rate = 1e-3
epoch_num  = 12
batch_size = 1600
reg_par    = 1e-4
o_dim      = 1
use_all_singular_values = False
best_only  = True


print(f"eyedee    : {eyedee}")
print(f"best_only : {best_only}")
print(f"epoch_num : {epoch_num}")
print(f"dropout   : {dropout}")

device = torch.device('cuda')
torch.cuda.empty_cache()

# CREATING A FOLDER TO STORE THE RESULTS
path_name = f"{eyedee}_lmdc/"

i = 1
while path.exists(path_name):
    path_name = f"{eyedee}_lmdc_{i}/"
    i = i + 1

del i
os.mkdir(path_name)
# print(path_name)


##################### SEED #####################
# seed = np.ceil(np.random.rand(10)*100)
seed = np.ceil(np.random.rand(1)*100) * np.ones(1)
print(seed)
###############################################


# NUMBER OF CHANNELS IN THE PROCESSED STIMULI AFTER FILTERBANK
stim_chans = 21
# NUMBER OF CHANNELS IN prePREPROCESSED STIMULI (1D)
stim_chans_pre = 1

pca_chans = 40

D = [0, 0.05, 0.1, 0.2]
# CAN REPLACE D WITH A SINGLE ELEMENT LIST WHOSE VALUE IS EQUAL TO THE DESIRED DROPOUT.


# HELPER FUNCTION FOR PERFORMING DCCA
def dcca_method(stim_data, resp_data, dropout, saving_name_root):
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
    plot_data_name = f"{path_name}/plot_data_{saving_name_root}"
    plot_data(x1, x2, plot_data_name)

    # # PLOTTING THE TRAINING LOSSES
    # s = f"{path_name}/plot_losses_{saving_name_root}"
    # plot_losses_tr_val_te(correlations, s)

    # SAVING THE NEW DATA
    save_data_name = f"{path_name}/new_deep_data_{saving_name_root}.pkl"
    fp = open(save_data_name, 'wb')
    pkl.dump(new_data_d, fp)
    fp.close()

    # SAVING THE DCCA MODEL

    save_model_name = f"{path_name}/dcca_model_{saving_name_root}.path.tar"
    torch.save(model_d, save_model_name)
    # save_dict_name = f"{path_name}/dcca_model_dict_{saving_name_root}.pth.tar"
    # torch.save({'state_dict': model_d.state_dict()}, save_dict_name)
    del model_d

    return [corr_d, corr_d_val]


speech_lmdc = True
if speech_lmdc:
    num_blocks = 20        # IF SPEECH DATA BY LIBERTO ET AL.

    # subs ARE THE SUBJECTS IDS TO WORK WITH
    subs = [1, 2]       # REPLACE WITH THE REQUIRED SUBJECTS' IDS.
    subs = sorted(subs) # TO KEEP THEIR IDS SORTED
    n_subs = len(subs)

    str_subs = str(subs[0])
    for each_sub in subs[1:]: 
        str_subs += f"_{each_sub}"

    num_blocks_start = 0
    num_blocks_end   = 1
    # CAN CHANGE BOTH VALUES ACCORDING TO THE INTERESTED CROSS-VALIDATION EXPERIMENTS.
    # CAN SUBMIT THESE TWO AS THE ARGUMENTS AND PARSE OVER THERE, FOR BULK EXPERIMENTS.

    all_corrs = np.zeros((num_blocks, 1 + len(D), n_subs))
    all_corrs_name =  f'{path_name}/speech_corrs_{str_subs}.npy'
    val_corrs = np.zeros((num_blocks, 1 + len(D), n_subs))
    val_corrs_name =  f'{path_name}/speech_corrs_val_{str_subs}.npy'

    print(f"n_subs     : {n_subs}")
    print(f"subs       : {subs}")
    print(f"D          : {D}")
    print(f"num_blocks : {num_blocks}")
    print(f"num_blocks_start: {num_blocks_start}")
    print(f"num_blocks_end  : {num_blocks_end}")
    print(f"num_blocks_net  : {num_blocks_end - num_blocks_start}")

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
        data_subs_pre = load_mcca_data(subs, pca_chans, block)

        ## LINEAR MCCA
        print("LINEAR MCCA + LCCA")
        lmc_corrs = np.zeros(all_corrs.shape[1])

        lmcca_data, lmlc_data, lmc_corrs, pre_lmdc_data = linear_mcca_with_stim(data_subs_pre, pca_chans, o_dim)

        print(f'LMCCA + LCCA corrs are : {lmc_corrs}')
        all_corrs[block,0]  = lmc_corrs
        np.save(all_corrs_name, all_corrs)


        # PLOTTING LMLC OUTPUT 
        for sub_num in range(n_subs):
            x1 = lmlc_data[sub_num][2][0]
            x2 = lmlc_data[sub_num][2][1]
            s = f"{path_name}/speech_plot_data_lmlc_sub_{sub_num}"
            plot_data(my_standardize(x1), my_standardize(x2), s)

        # SAVING THE LMLC OUTPUT
        fp = open(f'{path_name}/speech_lmlc_data_block_{block}_{str_subs}.pkl', 'wb')
        pkl.dump(lmlc_data, fp)
        fp.close()
        del lmlc_data

        # SAVING LMCCA + PROCESSED DATA SO THAT WE CAN DIRECTLY LOAD THEM TO DCCA METHOD FOR LMDC
        fp = open(f'{path_name}/speech_pre_lmdc_data_block_{block}_{str_subs}.pkl', 'wb')
        pkl.dump(pre_lmdc_data, fp)
        fp.close()
        del pre_lmdc_data

        # PERFORMING DCCA METHOD HERE.
        for d_cnt, dropout in enumerate(D):
            print(f"block: {block}, subjects: {subs}, dropout : {dropout}")

            # DEEP CCA METHOD.
            print("LMCCA + DCCA SPEECH")
            dcca_corrs     = np.zeros((n_subs))
            dcca_corrs_val = np.zeros((n_subs))

            for sub in range(n_subs):
                print(f"Sub: {subs[sub]}")
                # LOADING THE LMCCA + PROCESSED DATA
                data_subs = pkl.load(open(f'{path_name}/speech_pre_lmdc_data_block_{block}_{str_subs}.pkl', 'rb'))
                data_stim = data_subs[sub][0]
                data_sub  = data_subs[sub][1]
                del data_subs

                saving_name_root = f"speech_lmdc_block_{block}_sub_{subs[sub]}_{dropout}"
                dcca_corrs[sub], dcca_corrs_val[sub] = dcca_method(data_stim, data_sub, dropout, saving_name_root)

                print(f'LMDC corrs are : {dcca_corrs}')

                all_corrs[block, d_cnt+1] = dcca_corrs
                val_corrs[block, d_cnt+1] = dcca_corrs_val

                np.save(all_corrs_name, all_corrs)
                np.save(val_corrs_name, val_corrs)

            print(f'LMDC corrs for {block}, {dropout} are : {all_corrs[block, 1+d_cnt]}')
            print(f'saved speech.')

        print('saved')


nmedh_lmdc = True
if nmedh_lmdc:
    # subs ARE THE SUBJECTS IDS TO WORK WITH
    # FOR THE LMCCA DENOISING STEP.
    pca_chans = 40
    
    # THE 4 STIMULI FEATURES ARE ORDERED AS:
    # ENV -> PCA1 -> FLUX -> RMS
    all_corrs = np.zeros((4, 1 + len(D), 16, 12))
    all_corrs_name = f'{path_name}/nmedh_corrs.npy'
    val_corrs = np.zeros((4, 1 + len(D), 16, 12))
    val_corrs_name = f'{path_name}/nmedh_corrs_val.npy'

    stims = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    stims_types = ["ENV", "PC1", "FLX", "RMS"] 

    for stim_id, stim_str in enumerate(stims_types):
        for stim_num, stim__ in enumerate(stims):

            print(f"Stimulus Feature: {stim_str}, Stimulus Number : {stim__}")
            lmc_corrs = np.zeros(all_corrs.shape[1])
            # data_path = '/data2/data/nmed-h/stim_data2/'
            data_path = # LOAD YOUR DATA PATH HERE

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
            datas = mcca_data[0]
            stim_data = [mcca_data[1][0][:,stim_id].reshape(-1,1), mcca_data[1][1][:,stim_id].reshape(-1,1), mcca_data[1][2][:,stim_id].reshape(-1,1)]

            datas.append(stim_data)

            n_subs = 12

            ## LINEAR MCCA
            print("LINEAR MCCA + LCCA")
            lmc_corrs = np.zeros(all_corrs.shape[1])

            # PERFORMING LMCCA, LMCCA + LCCA, LMCCA + PROCESSING FOR DCCA
            lmcca_data, lmlc_data, lmc_corrs, pre_lmdc_data = linear_mcca_with_stim(datas, pca_chans, o_dim)

            print('LMCCA + LCCA corrs are : ' + str(lmc_corrs))
            all_corrs[stim_id, stim_num, 0]  = lmc_corrs
            np.save(all_corrs_name, all_corrs)


            # PLOTTING LMLC OUTPUT 
            for sub_num in range(n_subs):
                x1 = lmlc_data[sub_num][2][0]
                x2 = lmlc_data[sub_num][2][1]
                s = f"{path_name}/music_plot_data_{stim_str}_{stim__}_sub_{sub_num}"
                plot_data(my_standardize(x1), my_standardize(x2), s)


            # SAVING LMLC OUTPUT 
            fp = open(f'{path_name}/music_lmlc_data_{stim_str}_{stim__}.pkl', 'wb')
            pkl.dump(lmlc_data, fp)
            fp.close()
            del lmlc_data


            #  SAVING LMCCA + PROCESSED DATA SO THAT WE CAN DIRECTLY LOAD THEM TO DCCA METHOD FOR LMDC
            fp = open(f'{path_name}/music_pre_lmdc_data_{stim_str}_{stim__}.pkl', 'wb')
            pkl.dump(pre_lmdc_data, fp)
            fp.close()
            del pre_lmdc_data

                
            # PERFORMING DCCA METHOD HERE.

            for d_cnt, dropout in enumerate(D):
                print(f'STIM_ID : {stim_str}, STIMULUS : {stim__}, DROPOUT: {dropout}')

                # DEEP CCA METHOD.
                print("LMCCA + DCCA MUSIC")
                dcca_corrs     = np.zeros((n_subs))
                dcca_corrs_val = np.zeros((n_subs))

                for sub in range(n_subs):
                    print(f"Sub: {subs[sub]}")
                    # LOADING THE LMCCA + PROCESSED DATA
                    data_subs = pkl.load(open(f'{path_name}/music_pre_lmdc_data_{stim_str}_{stim__}.pkl', 'rb'))
                    data_stim = data_subs[sub][0]
                    data_sub  = data_subs[sub][1]
                    del data_subs

                    saving_name_root = f"music_lmdc_block_{stim_str}_{stim__}_sub_{sub}_{dropout}"
                    dcca_corrs[sub], dcca_corrs_val[sub] = dcca_method(data_stim, data_sub, dropout, saving_name_root)

                    print(f'LMDC corrs are : {dcca_corrs}')

                    all_corrs[stim_id, stim_num, d_cnt+1] = dcca_corrs
                    val_corrs[stim_id, stim_num, d_cnt+1] = dcca_corrs_val

                    np.save(all_corrs_name, all_corrs)
                    np.save(val_corrs_name, val_corrs)

                print(f'LMDC corrs for {stim_id}, {stim_num}, {dropout} are : {all_corrs[stim_id, stim_num, d_cnt+1]}')
                
            print(f'saved music.')

            print('saved')




# FOR CUSTOM,
# ONE CAN REPLACE THE datas LIST WITH THE INTERESTED DATASET.
# DATAS IN A LIST OF N+1 ITEMS
# FIRST N ITEMS BELONG TO EEG RECORDINGS OF N SUBJECTS RESPECTIVELY.
# LAST  1 ITEM BELONGS TO THE COMMON STIMULI PROVIDED TO ALL THE SUBJECTS
# EACH ITEM OF THE (N+1) LENGTH LIST IS ARRANGED AS 
# [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
# EACH OF THESE DATA ARE IN THE SHAPE : NUMBER OF SAMPLES X VECTOR DIMENSION OF EACH SAMPLE
#
# AFTER LOADING THE DATA INTO datas,
# ONE CAN CALL THE linear_mcca_with_stim FUNCTION ON IT
# IT RETURNS 
# THE DENOISED EEG RECORDINGS, 
# DENOISED STIMULI,
# LMLC DATA,
# DATA FOR PERFORMING LMDC.  