import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io

from cca_functions  import *
from speech_helper  import load_data
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
path_name = f"{eyedee}_lcca/"

i = 1
while path.exists(path_name):
    path_name = f"{eyedee}_lcca_{i}/"
    i = i + 1

del i
os.mkdir(path_name)
# print(path_name)


# NUMBER OF CHANNELS IN THE PROCESSED STIMULI AFTER FILTERBANK
stim_chans = 21
# NUMBER OF CHANNELS IN prePREPROCESSED STIMULI (1D)
stim_chans_pre = 1



# HELPER FUNCTION FOR PERFORMING LCCA TO NMED-H DATASET
def lcca(stim_tr, stim_val, stim_te, resp_tr, resp_val, resp_te, sub_num, stim_str):
    print(f'SUBJECT : {sub_num}, STIM_ID: {stim_str}, LCCA')

    _, new_data_l = cca_model([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], o_dim)
    x1 = new_data_l[2][0] ; x3 = new_data_l[1][0]
    x2 = new_data_l[2][1] ; x4 = new_data_l[1][1]
    corr_l = [np.squeeze(my_corr(x1, x2, o_dim)), np.squeeze(my_corr(x3, x4, o_dim))]
    print(f'LCCA is : {corr_l}')

    fp = open(f'{path_name}/music_data_sub_{stim_str}_{sub_num}.pkl', 'wb')
    pkl.dump(new_data_l, fp)
    fp.close()
    del new_data_l

    return corr_l[0]



speech_lcca = False
if speech_lcca:
    num_blocks = 20        # IF SPEECH DATA BY LIBERTO ET AL.

    # subs ARE THE SUBJECTS IDS TO WORK WITH
    subs = [1,2]         # REPLACE THEM WITH THE INTERESTED SUBJECTS.
    subs = sorted(subs)  # TO KEEP THEIR IDS SORTED
    n_subs = len(subs)

    str_subs = str(subs[0])
    for each_sub in subs[1:]: 
        str_subs += f"_{each_sub}"

    all_corrs = np.zeros((num_blocks, n_subs))
    all_corrs_name = f'{path_name}/speech_corrs_{str_subs}.npy'

    print(f"n_subs    : {n_subs}")
    print(f"subs      : {subs}")
    print(f"num_blocks: {num_blocks}")

    for block in range(num_blocks):
        data_subs = load_data(subs, block)
        # data_subs IS A LIST OF N SUBJECTS DATA AND 1 COMMON STIMULUS DATA (AS THE LAST ELEMENT.)
        # ALL THE DATA ARE PROCESSED USING PCA AND THE FILTERBANK

        # LINEAR CCA METHOD.
        print("LCCA")
        lcca_corrs = np.zeros((n_subs))
        new_data_lcca = []
        for sub in range(n_subs):
            lcca_corrs[sub], sub_data = cca_model(data_subs[-1], data_subs[sub], o_dim)
            new_data_lcca.append(sub_data)
            x1 = sub_data[2][0]
            x2 = sub_data[2][1]
            s = f"{path_name}/speech_plot_data_lcca_sub_{block}_{sub}"
            plot_data(my_standardize(x1), my_standardize(x2), s)

        del new_data_lcca
        # CAN SAVE THEM IF REQUIRED.

        print(f'LCCA corrs for {block} are : {lcca_corrs}')
        all_corrs[block,:]  = lcca_corrs
        np.save(all_corrs_name, all_corrs)
        print(f'saved BLOCK:{block}')






nmedh_lcca = True
if nmedh_lcca:
    fs = 80
    N = 125
    subs = 58
    all_corrs = np.zeros((subs, 4))
    all_corrs_name = f'{path_name}/nmedh_corrs.npy'
    rm_list = [0, 8, 20, 23, 24, 34, 37, 40, 45, 46, 53]

    data_path = '/data2/jaswanthr/data/nmed-h/sub_data'
    # data_path = "# ADD YOUR DATA PATH HERE."

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

                del resp_tr_a, resp_tr_b, resp_val_a, resp_val_b, resp_te_a, resp_te_b, stim_tr_3d, stim_val_3d, stim_te_3d, stimulus_tr, stimulus_val, stimulus_te

                # THE SUBJECT'S EEG DATA AND CORRESPONDING STIMULUS ARE AGGREGATED TO FORM T x d MATRICES. 
                # NEXT STEP:
                # FILTER AND PCA THE DATA TO OBTAIN 139D EEG DATA AND 21D STIMULI DATA.

                print(f'SUBJECT : {sub_num}, STIM_ID: ENV')
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

                # print(all_corrs[sub_num-1, stim_id, :])
                all_corrs[sub_num-1, stim_id] = lcca(stim_tr, stim_val, stim_te, resp_tr, resp_val, resp_te, sub_num, stim_str)
                # print(all_corrs[sub_num-1, stim_id, :])
                np.save(all_corrs_name, all_corrs)

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
                    
                    all_corrs[sub_num-1, stim_id+1] = lcca(stim_tr, stim_val, stim_te, resp_tr, resp_val, resp_te, sub_num, stim_str)
                    # print(all_corrs[sub_num-1, stim_id+1, :])
                    np.save(all_corrs_name, all_corrs)






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
    resp_tr, resp_val, resp_te = filtone(resptr_60, respval_60, respte_60, pca_num)
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

    stim_tr, stim_val, stim_te = filtone(stimtr, stimval, stimte, 1)
    del stimtr, stimval, stimte

    print(f'SUBJECT : {sub_num}, STIM_ID: {stim_str}, LCCA')

    _, new_data_l = cca_model([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], o_dim)
    x1 = new_data_l[2][0]
    x2 = new_data_l[2][1]
    corr_l = np.squeeze(my_corr(x1, x2, o_dim))
    print(f'LCCA is : {corr_l}')

    fp = open(f'{path_name}/custom_lcca_data.pkl', 'wb')
    pkl.dump(new_data_l, fp)
    fp.close()
    del new_data_l

















