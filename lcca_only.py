import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io

from cca_functions    import *
from all_linear_stuff import *

def plot_data(x, y,s):
    plt.clf()
    x = x[0]
    y = y[0]
    plt.plot(x, color='orange')
    plt.plot(y, color='blue')
    plt.legend(['stim', 'resp'])
    plt.savefig(s+'.eps', format="eps")

name_of_the_script = sys.argv[0].split('.')[0]
a = sys.argv[1:]
eyedee = str(a[0])
o_dim = int(a[1])   # THE INTERESTED FINAL DIMENSION

# subs ARE THE SUBJECTS IDS TO WORK WITH
subs = [None]
subs = sorted(subs) # TO KEEP THEIR IDS SORTED
n_subs = len(subs)

str_subs = str(subs[0])
for each_sub in subs[1:]: 
    str_subs += "_{}".format(each_sub)

print("eyedee : {}".format(eyedee))

# CREATING A FOLDER TO STORE THE RESULTS
crrnt_dir = os.getcwd()
strings = crrnt_dir.split('/')
strings = strings[:-1]
strings[-1] = 'results'
crrnt_dir = '/'.join(strings)

if not(path.exists(crrnt_dir + "/" + name_of_the_script + "/")):
    os.mkdir(crrnt_dir + "/" + name_of_the_script + "/")
crrnt_dir = crrnt_dir + "/" + name_of_the_script
path_name = crrnt_dir + "/" + eyedee + f"lcca_only" + "/"

i = 1
while path.exists(path_name):
    path_name = crrnt_dir + "/" + eyedee + f"lcca_only_" + str(i) + "/"
    i = i + 1

del i
os.mkdir(path_name)
# print(path_name)


# NUMBER OF CHANNELS FOR THE PREPROCESSED STIMULI AFTER FILTERBANK
stim_chans = 21
# NUMBER OF CHANNELS FOR pre PREPROCESSED STIMULI (1D)
stim_chans_pre = 1


# HELPER FUNCTION TO LOAD DATA FOR SPEECH
def load_data(blocks=0, filtered=True):
    # IF THE DATA IS ALREADY PROCESSED THROUGH THE FILTERBANK AND PCA
    print('blocks = ' + str(blocks))
    if blocks == num_blocks - 1:
        val_idx = 0
    else:
        val_idx = blocks + 1

    if filtered:
        data_path = 'data1_'+str(blocks)+'.pkl'
        fp = open(data_path, 'rb')
        data1 = pkl.load(fp)
        fp.close()
        print("Loaded FILTERED Data.")

        print('Data INITIALIZED for block : {}'.format(str(blocks)))
        data_subs = []
        for sub in subs:
            data_subs.append([data1[0][:,:,sub], data1[1][:,:,sub], data1[2][:,:,sub]])
        data_subs.append([data1[0][:,:stim_chans,-1], data1[1][:,:stim_chans,-1], data1[2][:,:stim_chans,-1]])
        del data1
    else:
        # IF THE DATA IS NOT PROCESSED
        data_path = 'data1_pre_'+str(blocks)+'.pkl'
        fp = open(data_path, 'rb')
        pre_data = pkl.load(fp)
        fp.close()
        print("Loaded DEMEANED Data.")

        print('Data INITIALIZED for block : {}'.format(str(blocks)))
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


# HELPER FUNCTION FOR PERFORMING LCCA TO NMED-H DATASET
def lcca(stim5_tr, stim5_val, stim5_te, resp5_tr, resp5_val, resp5_te, sub_num, stim_id, stim_str):
    corrs = np.zeros(len(D) + 1)
    print('SUBJECT : {}, STIM_ID: {}, LCCA3'.format(sub_num, stim_str))

    _, new_data_l = cca3_model([stim5_tr, stim5_val, stim5_te], [resp5_tr, resp5_val, resp5_te], o_dim)
    x1 = new_data_l[2][0]
    x2 = new_data_l[2][1]
    corr_l = np.squeeze(my_corr(x1, x2, o_dim))
    print('LCCA3 is : ' + str(corr_l))

    fp = open(path_name + f'lcca_data_sub_{sub_num}_{stim_id}_lin.pkl', 'wb')
    pkl.dump(new_data_l[1:], fp)
    fp.close()
    del new_data_l

    return corr_l



speech_lcca = True
if speech_lcca:
    n_blocks = 20        # IF SPEECH DATA BY LIBERTO ET AL.
    all_corrs = np.zeros((n_blocks, n_subs))
    all_corrs_name = path_name + 'speech_corrs_{}.npy'.format(str_subs)

    print("n_subs    : {}".format(n_subs))
    print("subs      : {}".format(subs))
    print("num_blocks: {}".format(num_blocks))

    for block in range(num_blocks):
        data_subs = load_data(block)
        # data_subs IS A LIST OF N SUBJECTS DATA AND 1 COMMON STIMULUS DATA (AS THE LAST ELEMENT.)
        # ALL THE DATA ARE PROCESSED USING PCA AND THE FILTERBANK

        # LINEAR CCA3
        print("LCCA3.")
        lcca_corrs = np.zeros((n_subs))
        new_data_lcca3 = []
        for sub in range(n_subs):
            lcca_corrs[sub], sub_data = cca3_model_new(data_subs[-1], data_subs[sub], 5)
            new_data_lcca3.append(sub_data)
            x1 = sub_data[2][0]
            x2 = sub_data[2][1]
            s = path_name + "/plot_data_lcca3_sub_"+str(sub)
            plot_data(my_standardize(x1), my_standardize(x2), s)

        del new_data_lcca3
        # CAN SAVE IT IF REQUIRED.

        print(f'LCCA3 corrs for {block} are : {lcca_corrs}')
        all_corrs[block,:]  = lcca_corrs
        np.save(all_corrs_name, all_corrs)
        print(f'saved BLOCK:{block}')






nmedh_lcca = True
if nmedh_lcca:
    fs = 80
    N = 125
    subs = 58
    all_corrs = np.zeros((subs, 4, len(D) + 1))
    all_corrs_name = path_name + 'nmedh_corrs.npy'
    rm_list = [0, 8, 20, 23, 24, 34, 37, 40, 45, 46, 53]

    # data_path = '/data/nmed-h/sub_data/'
    data_path = "# ADD YOUR DATA PATH HERE."

    for sub_num1 in range(59):
        if sub_num1 not in rm_list:
            sub_num = int(sub_num1)
            print('SUBJECT NUM : ' + str(sub_num))
            if path.exists(data_path + 'Sub_'+str(sub_num)+'_data.mat'):
                mat1 = scipy.io.loadmat(data_path + 'Sub_'+str(sub_num)+'_data.mat')
                print(data_path + 'Sub_'+str(sub_num)+'_data.mat')

                resp_data_a = mat1['resp_data_a'][0] ;
                resp_data_b = mat1['resp_data_b'][0] ;
                resp_tr_a   = mat1['resp_tr_a'][0]   ;
                resp_tr_b   = mat1['resp_tr_b'][0]   ;
                resp_val_a  = mat1['resp_val_a'][0]  ;
                resp_val_b  = mat1['resp_val_b'][0]  ;
                resp_te_a   = mat1['resp_te_a'][0]   ;
                resp_te_b   = mat1['resp_te_b'][0]   ;

                stim_data   = mat1['stim_data'][0]   ;
                stim_tr     = mat1['stim_tr'][0]     ;
                stim_tr_3d  = mat1['stim_tr_3d'][0]  ;
                stim_val    = mat1['stim_val'][0]    ;
                stim_val_3d = mat1['stim_val_3d'][0] ;
                stim_te     = mat1['stim_te'][0]     ;
                stim_te_3d  = mat1['stim_te_3d'][0]  ;

                # stimulus     = mat1['stimulus'][0]   ;
                stimulus_tr  = mat1['stimulus_tr'][0]     ;
                stimulus_val = mat1['stimulus_val'][0]    ;
                stimulus_te  = mat1['stimulus_te'][0]     ;

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

                print('SUBJECT : {}, STIM_ID: ENVELOPE'.format(sub_num))
                # 125D TO 60D
                pca_num = 60
                [meanp, W, resptr_60] = my_PCA(resp[0], pca_num)
                respval_60 = apply_PCA(resp[1], meanp, W)
                respte_60  = apply_PCA(resp[2], meanp, W)
                # 60D TO 1260D
                resp5_tr, resp5_val, resp5_te = filtone(resptr_60, respval_60, respte_60, pca_num)
                del resptr_60, respval_60, respte_60
                # 1260D TO 139D
                pca_num1 = 139
                [meanp, W, resp5_tr] = my_PCA(resp5_tr, pca_num1)
                resp5_val = apply_PCA(resp5_val, meanp, W)
                resp5_te  = apply_PCA(resp5_te, meanp, W)

                # JUST MAKING SURE THE STIMULUS IN 2D MATRIX FORM.
                stimtr  = np.reshape(stim[0], (-1, 1))
                stimval = np.reshape(stim[1], (-1, 1))
                stimte  = np.reshape(stim[2], (-1, 1))
                # STIM ENVELOPE
                stim_id = 0
                stim_str = "ENVELOPE"
                # 1D ENVELOPE TO 21D
                stim5_tr, stim5_val, stim5_te = filtone(stimtr, stimval, stimte, 1)
                del stimtr, stimval, stimte

                # print(all_corrs[sub_num-1, stim_id, :])
                all_corrs[sub_num-1, stim_id] = lcca(stim5_tr, stim5_val, stim5_te, resp5_tr, resp5_val, resp5_te, sub_num, stim_id, stim_str)
                # print(all_corrs[sub_num-1, stim_id, :])
                np.save(all_corrs_name, all_corrs)

                for stim_type in range(3):
                    stim_id = stim_type+1
                    stim_str = str(stim_id)+"th DIMENSION"
                    print('SUBJECT : {}, STIM_ID : {}th DIMENSION'.format(sub_num, stim_id))

                    # CONSIDERING NTH DIMENSION OF STIMULUS 3D FEATURES
                    stimtr  = np.reshape(stim1[0][:,stim_type], (-1, 1))
                    stimval = np.reshape(stim1[1][:,stim_type], (-1, 1))
                    stimte  = np.reshape(stim1[2][:,stim_type], (-1, 1))
                    # 1D TO 21D
                    stim5_tr, stim5_val, stim5_te = filtone(stimtr, stimval, stimte, 1)
                    
                    all_corrs[sub_num-1, stim_id] = lcca(stim5_tr, stim5_val, stim5_te, resp5_tr, resp5_val, resp5_te, sub_num, stim_id, stim_str)
                    # print(all_corrs[sub_num-1, stim_id, :])
                    np.save(all_corrs_name, all_corrs)























