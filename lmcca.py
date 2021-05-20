import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io

from cca_functions  import *
from speech_helper  import load_mcca_data
from music_helper   import stim_resp


def plot_data(x, y,s):
    plt.clf()
    x = x[0]
    y = y[0]
    plt.plot(x, color='orange')
    plt.plot(y, color='blue')
    plt.legend(['stim', 'resp'])
    plt.savefig(f'{s}.eps', format="eps")

name_of_the_script = sys.argv[0].split('.')[0]
a = sys.argv[1:]
eyedee = str(a[0])  # ID OF THE EXPERIMENT.
o_dim = int(a[1])   # THE INTERESTED OUTPUTS DIMENSIONALITY

print(f"eyedee : {eyedee}")

# CREATING A FOLDER TO STORE THE RESULTS
path_name = f"lmcca_{eyedee}/"

i = 1
while path.exists(path_name):
    path_name = f"lmcca_{eyedee}_{i}/"
    i = i + 1

del i
os.mkdir(path_name)
print(path_name)


# For stimuli lags (d_s)
pca_chans = 40


speech_lmlc = True
if speech_lmlc:
    num_blocks = 20        # Number of Sessins in the SPEECH DATA BY LIBERTO ET AL.

    # subs ARE THE SUBJECTS IDS TO WORK WITH
    subs = [1, 2]       # REPLACE WITH THE REQUIRED SUBJECTS' IDS.
    subs = sorted(subs) # TO KEEP THEIR IDS SORTED
    n_subs = len(subs)

    str_subs = str(subs[0])
    for each_sub in subs[1:]: 
        str_subs += f"_{each_sub}"

    num_blocks_start = 0
    num_blocks_end   = 20
    # CAN CHANGE BOTH VALUES ACCORDING TO THE INTERESTED CROSS-VALIDATION EXPERIMENTS.
    # CAN SUBMIT THESE TWO AS THE ARGUMENTS AND PARSE OVER THERE, FOR BULK EXPERIMENTS.

    tst_corrs = np.zeros((num_blocks, n_subs))
    val_corrs = np.zeros((num_blocks, n_subs))
    tst_corrs_name =  f'{path_name}/speech_corrs_{str_subs}.npy'
    val_corrs_name =  f'{path_name}/speech_corrs_val_{str_subs}.npy'

    print(f"n_subs     : {n_subs}")
    print(f"subs       : {subs}")
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
        data_subs_pre = load_mcca_data(subs, block)

        ## LINEAR MCCA
        print("LINEAR MCCA + LCCA")
        lmc_corrs = np.zeros(tst_corrs.shape[1])

        lmcca_data, lmlc_data, lmc_corrs, pre_lmdc_data = linear_mcca_with_stim(data_subs_pre, pca_chans, o_dim)

        print(f'LMCCA + LCCA corrs are : {lmc_corrs}')
        tst_corrs[block,:]  = lmc_corrs
        np.save(tst_corrs_name, tst_corrs)


        # # PLOTTING LMLC OUTPUT 
        # for sub_num in range(n_subs):
        #     x1 = lmlc_data[sub_num][2][0]
        #     x2 = lmlc_data[sub_num][2][1]
        #     s = f"{path_name}/speech_plot_data_lmlc_sub_{sub_num}"
        #     plot_data(my_standardize(x1), my_standardize(x2), s)

        # SAVING THE LMLC OUTPUT
        fp = open(f'{path_name}/speech_lmlc_data_block_{block}_{str_subs}.pkl', 'wb')
        pkl.dump(lmlc_data, fp)
        fp.close()
        del lmlc_data


        # EXTRAS:

        # SAVING LMCCA + PROCESSED DATA SO THAT WE CAN DIRECTLY LOAD THEM TO DCCA METHOD FOR LMDC
        fp = open(f'{path_name}/speech_pre_lmdc_data_block_{block}_{str_subs}.pkl', 'wb')
        pkl.dump(pre_lmdc_data, fp)
        fp.close()
        del pre_lmdc_data

        # CAN PERFORM DCCA METHOD HERE.
        print('saved')


nmedh_lmlc = False
if nmedh_lmlc:
    # subs ARE THE SUBJECTS IDS TO WORK WITH
    # FOR THE LMCCA DENOISING STEP.
    pca_chans = 40
    
    # THE 4 STIMULI FEATURES ARE ORDERED AS:
    # ENV -> PCA1 -> FLUX -> RMS
    tst_corrs = np.zeros((4, 16, 12))
    tst_corrs_name = f'{path_name}/nmedh_corrs.npy'
    val_corrs = np.zeros((4, 16, 12))
    val_corrs_name = f'{path_name}/nmedh_corrs_val.npy'

    stims = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    stims_types = ["ENV", "PC1", "FLX", "RMS"] 

    for stim_id, stim_str in enumerate(stims_types):
        for stim_num, stim__ in enumerate(stims):

            print(f"Stimulus Feature: {stim_str}, Stimulus Number : {stim__}")
            lmc_corrs = np.zeros(tst_corrs.shape[1])
            # data_path = '/data2/data/nmed-h/stim_data2/'
            data_path = '/data2/jaswanthr/data/nmed-h/stim_data2/'
            # data_path = # LOAD YOUR DATA PATH HERE

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

            n_subs = 12

            ## LINEAR MCCA
            print("LINEAR MCCA + LCCA")
            lmc_corrs = np.zeros(tst_corrs.shape[1])

            # PERFORMING LMCCA, LMCCA + LCCA, LMCCA + PROCESSING FOR DCCA
            lmcca_data, lmlc_data, lmc_corrs, pre_lmdc_data = linear_mcca_with_stim(all_data, pca_chans, o_dim)

            print('LMCCA + LCCA corrs are : ' + str(lmc_corrs))
            tst_corrs[stim_id, stim_num]  = lmc_corrs
            np.save(tst_corrs_name, tst_corrs)


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


            # EXTRAS:

            #  SAVING LMCCA + PROCESSED DATA SO THAT WE CAN DIRECTLY LOAD THEM TO DCCA METHOD FOR LMDC
            fp = open(f'{path_name}/music_pre_lmdc_data_{stim_str}_{stim__}.pkl', 'wb')
            pkl.dump(pre_lmdc_data, fp)
            fp.close()
            del pre_lmdc_data

            # CAN PERFORM DCCA METHOD HERE.

            print('saved')



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
# ONE CAN CALL THE linear_mcca_with_stim FUNCTION ON IT
#
# IT RETURNS 
    # THE DENOISED EEG RECORDINGS, 
    # DENOISED STIMULI,
    # LMLC DATA,
    # DATA FOR PERFORMING LMDC.  


