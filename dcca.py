import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io
import random
# from pdb import set_trace as bp  #################added break point accessor####################
# from scipy.signal import lfilter

import torch

from cca_functions import *
from speech_helper import load_data
from music_helper  import stim_resp
from deep_models   import dcca_model

def plot_data(x, y,s):
    plt.clf()
    x = x[0]
    y = y[0]
    plt.plot(x, color='orange')
    plt.plot(y, color='blue')
    plt.legend(['stim', 'resp'])
    plt.savefig(s+'.eps', format="eps")

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
path_name = f"{eyedee}_dcca/"

i = 1
while path.exists(path_name):
    path_name = f"{eyedee}_dcca_{i}/"
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



speech_dcca = True
if speech_dcca:
    num_blocks = 20        # IF SPEECH DATA BY LIBERTO ET AL.

    # subs ARE THE SUBJECTS IDS TO WORK WITH
    subs = [None]       # REPLACE WITH THE REQUIRED SUBJECTS' IDS.
    subs = sorted(subs) # TO KEEP THEIR IDS SORTED
    n_subs = len(subs)

    str_subs = str(subs[0])
    for each_sub in subs[1:]: 
        str_subs += f"_{each_sub}"

    num_blocks_start = 0
    num_blocks_end   = 1
    # CAN CHANGE BOTH VALUES ACCORDING TO THE INTERESTED CROSS-VALIDATION EXPERIMENTS.
    # CAN SUBMIT THESE TWO AS THE ARGUMENTS AND PARSE OVER THERE, FOR BULK EXPERIMENTS.

    all_corrs = np.zeros((num_blocks, len(D), n_subs))
    all_corrs_name =  f'{path_name}/speech_corrs_{str_subs}.npy'
    val_corrs = np.zeros((num_blocks, len(D), n_subs))
    val_corrs_name =  f'{path_name}/speech_corrs_val_{str_subs}.npy'

    print(f"n_subs     : {n_subs}")
    print(f"subs       : {subs}")
    print(f"D          : {D}")
    print(f"num_blocks : {num_blocks}")
    print(f"num_blocks_start: {num_blocks_start}")
    print(f"num_blocks_end  : {num_blocks_end}")
    print(f"num_blocks_net  : {num_blocks_end - num_blocks_start}")

    for block in range(num_blocks_start, num_blocks_end):
        for d_cnt, dropout in enumerate(D):
            print(f"block: {block}, subjects: {subs}, dropout : {dropout}")

            # data_subs IS A LIST OF N SUBJECTS DATA AND 1 COMMON STIMULUS DATA (AS THE LAST ELEMENT.)
            # ALL THE DATA ARE PROCESSED USING PCA AND THE FILTERBANK
            data_subs = load_data(subs, block)
            data_stim = data_subs[-1]

            # saving the data_subs so that we can decrease the load on RAM.
            fp = open(f'{path_name}/data_subs.pkl', 'wb')
            pkl.dump(data_subs, fp)
            fp.close()
            del data_subs

            # DEEP CCA METHOD.
            print("DCCA SPEECH")
            dcca_corrs     = np.zeros((n_subs))
            dcca_corrs_val = np.zeros((n_subs))

            for sub in range(n_subs):
                print(f"Sub: {subs[sub]}")
                data_subs = pkl.load(open(f"{path_name}/data_subs.pkl", "rb"))
                data_sub = data_subs[sub]
                del data_subs

                saving_name_root = f"speech_block_{block}_sub_{subs[sub]}_{dropout}"
                dcca_corrs[sub], dcca_corrs_val[sub] = dcca_method(data_stim, data_sub, dropout, saving_name_root)

                print(f'DCCA corrs are : {dcca_corrs}')

                all_corrs[block, d_cnt] = dcca_corrs
                val_corrs[block, d_cnt] = dcca_corrs_val

                np.save(all_corrs_name, all_corrs)
                np.save(val_corrs_name, val_corrs)

            print(f'DCCA corrs for {block}, {dropout} are : {all_corrs[block, d_cnt]}')
            print(f'saved speech.')






nmedh_dcca = True
if nmedh_dcca:
    fs = 80
    N = 125
    subs = 58
    all_corrs = np.zeros((subs, 4, len(D)))
    all_corrs_name = f'{path_name}/nmedh_corrs.npy'
    val_corrs = np.zeros((subs, 4, len(D)))
    val_corrs_name = f'{path_name}/nmedh_corrs_val.npy'
    rm_list = [0, 8, 20, 23, 24, 34, 37, 40, 45, 46, 53]

    # data_path = '/data/nmed-h/sub_data/'
    data_path = "# ADD YOUR DATA PATH HERE."

    for sub_num1 in range(3):
        if sub_num1 not in rm_list:
            sub_num = int(sub_num1)
            print(f'SUBJECT NUM : {sub_num}')
            sub_data_path = f'{data_path}/Sub_{sub_num}_data.mat'
            print(sub_data_path)
            if path.exists(sub_data_path):
                mat1 = scipy.io.loadmat(sub_data_path)

                resp_data_a = mat1['resp_data_a'][0] 
                resp_data_b = mat1['resp_data_b'][0] 
                resp_tr_a   = mat1['resp_tr_a'][0]   
                resp_tr_b   = mat1['resp_tr_b'][0]   
                resp_val_a  = mat1['resp_val_a'][0]  
                resp_val_b  = mat1['resp_val_b'][0]  
                resp_te_a   = mat1['resp_te_a'][0]   
                resp_te_b   = mat1['resp_te_b'][0]   

                stim_data   = mat1['stim_data'][0]   
                # stim_tr     = mat1['stim_tr'][0]    
                stim_tr_3d  = mat1['stim_tr_3d'][0]  
                # stim_val    = mat1['stim_val'][0]   
                stim_val_3d = mat1['stim_val_3d'][0] 
                # stim_te     = mat1['stim_te'][0]    
                stim_te_3d  = mat1['stim_te_3d'][0]  

                # stimulus     = mat1['stimulus'][0]      
                stimulus_tr  = mat1['stimulus_tr' ][0]   
                stimulus_val = mat1['stimulus_val'][0]   
                stimulus_te  = mat1['stimulus_te' ][0]   

                del mat1
                count = len(resp_data_a)

                # AGGREGATING ALL STIMULUS AND corresponding RESPONSES OF THIS SUBJECT TO A [TRAIN, VAL, TEST] SET
                # HERE STIMULUS IS IN 3D. AS PROPOSED BY ALURI ET AL.
                stim1, resp = stim_resp(resp_tr_a, resp_tr_b, resp_val_a, resp_val_b, resp_te_a, resp_te_b, stim_tr_3d, stim_val_3d, stim_te_3d, count)
                print('Loaded Data!')

                # HERE STIMULUS IS IN 1D. ENVELOPE OF STIMULUS.
                stim, _ = stim_resp(None,None,None,None,None,None, stimulus_tr, stimulus_val, stimulus_te, count)
                print('Loaded Data!')

                del resp_tr_a, resp_tr_b, resp_val_a, resp_val_b, resp_te_a, resp_te_b, stim_tr_3d, stim_val_3d, stim_te_3d, stimulus_tr, stimulus_val, stimulus_te, resp_data_a, resp_data_b

                # THE SUBJECT'S EEG DATA AND CORRESPONDING STIMULUS ARE AGGREGATED TO FORM T x d MATRICES. 
                # NEXT STEP:
                # FILTER AND PCA THE DATA TO OBTAIN 139D EEG DATA AND 21D STIMULI DATA.

                print(f'SUBJECT : {sub_num}, STIM_ID: ENVELOPE')
                # 125D TO 60D
                pca_num = 60
                [meanp, W, resptr_60] = my_PCA(resp[0], pca_num)
                respval_60 = apply_PCA(resp[1], meanp, W)
                respte_60  = apply_PCA(resp[2], meanp, W)
                # 60D TO 1260D
                resp_tr, resp_val, resp_te = filtone(resptr_60, respval_60, respte_60)
                del resptr_60, respval_60, respte_60
                # 1260D TO 139D
                pca_num1 = 139
                [meanp, W, resp_tr] = my_PCA(resp_tr, pca_num1)
                resp_val = apply_PCA(resp_val, meanp, W)
                resp_te  = apply_PCA(resp_te, meanp, W)

                # MAKING SURE THE STIMULUS IN 2D MATRIX FORM.
                stimtr  = np.reshape(stim[0], (-1, 1))
                stimval = np.reshape(stim[1], (-1, 1))
                stimte  = np.reshape(stim[2], (-1, 1))

                # STIM ENVELOPE
                stim_id = 0
                stim_str = "ENV"
                # 1D ENVELOPE TO 21D
                stim_tr, stim_val, stim_te = filtone(stimtr, stimval, stimte)
                del stimtr, stimval, stimte

                for d_cnt, dropout in enumerate(D):
                    all_corrs[sub_num-1, stim_id, d_cnt], val_corrs[sub_num-1, stim_id, d_cnt] = dcca_method([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], dropout,  f"nmedh_sub_{sub_num}_{stim_str}_{dropout}")
                    np.save(all_corrs_name, all_corrs)
                    np.save(val_corrs_name, val_corrs)

                # PC1 -> SPECTRAL FLUX -> RMS
                stim_id__s = ["PC1", "FLX", "RMS"]
                for stim_id, stim_str in enumerate(stim_id__s):
                    print(f'SUBJECT : {sub_num}, STIM_ID : {stim_str}')

                    # CONSIDERING NTH DIMENSION OF STIMULUS 3D FEATURES
                    stimtr  = np.reshape(stim1[0][:,stim_id], (-1, 1))
                    stimval = np.reshape(stim1[1][:,stim_id], (-1, 1))
                    stimte  = np.reshape(stim1[2][:,stim_id], (-1, 1))
                    # 1D TO 21D
                    stim_tr, stim_val, stim_te = filtone(stimtr, stimval, stimte)

                    for d_cnt, dropout in enumerate(D):
                        all_corrs[sub_num-1, stim_id+1, d_cnt], val_corrs[sub_num-1, stim_id+1, d_cnt] = dcca_method([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], dropout,  f"nmedh_sub_{sub_num}_{stim_str}_{dropout}")
                        np.save(all_corrs_name, all_corrs)
                        np.save(val_corrs_name, val_corrs)







custom_data = True
if custom_data:
    # TO PERFORM THE LINEAR CCA METHOD ON A CUSTON AUDIO-EEG DATA.
    
    # FIRST LOAD THE DATA.
    # stim_data and resp_data MUST BE A THREE ELEMENTS LIST.
    # SUCH AS [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
    # REPLACE THE None WITH THE DATA.
    stim_data = None 
    resp_data = None

    # FIRST, PROCESS THE EEG.
    # 1. PCA TO 60D.
    # 2. FILTER USING FILTERBANK.
    # 3. PCA TO 139D.

    # d D TO 60D.
    pca_num = 60
    [meanp, W, resptr_60] = my_PCA(resp_data[0], pca_num)
    respval_60 = apply_PCA(resp_data[1], meanp, W)
    respte_60  = apply_PCA(resp_data[2], meanp, W)

    # 60D TO 1260D.
    resp_tr, resp_val, resp_te = filtone(resptr_60, respval_60, respte_60)
    del resptr_60, respval_60, respte_60

    # 1260D TO 139D.
    pca_num1 = 139
    [meanp, W, resp_tr] = my_PCA(resp_tr, pca_num1)
    resp_val = apply_PCA(resp_val, meanp, W)
    resp_te  = apply_PCA(resp_te, meanp, W)

    # SECOND, PROCESS THE STIMULUS.
    # 1. IF NOT 1D, CAN PCA TO 1D. (OPTIONAL. CAN LEAVE IT TOO.)
    # 2. FILTERBANK
    pca_num = 1
    [meanp, W, stim_1] = my_PCA(stim_data[0], pca_num)
    stimval_1 = apply_PCA(stim_data[1], meanp, W)
    stimte_1  = apply_PCA(stim_data[2], meanp, W)

    stim_tr, stim_val, stim_te = filtone(stimtr, stimval, stimte)
    del stimtr, stimval, stimte


    all_corrs = np.zeros(len(D))
    val_corrs = np.zeros(len(D))
    all_corrs_name = f'{path_name}/corrs.npy'
    val_corrs_name = f'{path_name}/corrs_val.npy'

    for d_cnt, dropout in enumerate(D):
        save_name_root = f"custom_{dropout}"
        corrs[d_cnt], corrs_val[d_cnt] = dcca_method([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], dropout, save_name_root)

        np.save(all_corrs_name, all_corrs)
        np.save(val_corrs_name, val_corrs)

    print("SAVED.")
















