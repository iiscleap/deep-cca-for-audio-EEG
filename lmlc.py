import numpy as np
import pickle
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path

from cca_functions import *
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
blocks = int(a[2])
# BLOCKS IS USED FOR SPEECH DATASET. (OUT OF 20 CROSS-VALIDATION EXPERIMENTS)

pca_chans = 40
o_dim = 1

crrnt_dir = os.getcwd()
strings = crrnt_dir.split('/')
strings = strings[:-1]
strings[-1] = 'results'
crrnt_dir = '/'.join(strings)


# HELPER FUNCTION TO LOAD DATA.
def load_data(blocks=0, dataset="speech"):
    if dataset == "speech":
        print('blocks = ' + str(blocks))
        if blocks == num_blocks - 1:
            val_idx = 0
        else:
            val_idx = blocks + 1

        data_path = 'data1_raw_'+str(blocks)+'.pkl'
        fp = open(data_path, 'rb')
        raw_data = pickle.load(fp)
        fp.close()
        print("Loaded DEMEANED Data.")

        print('Data INITIALIZED for block : {}'.format(str(blocks)))
        data_subs_raw = []

        for sub in subs:
            data_subs_raw.append([raw_data[0][:,:,sub], raw_data[1][:,:,sub], raw_data[2][:,:,sub]])
        data_subs_raw.append([raw_data[0][:,:stim_chans_raw,-1], raw_data[1][:,:stim_chans_raw,-1], raw_data[2][:,:stim_chans_raw,-1]])
    
    else:
        pass

    return data_subs_raw

# subs_set = [[0,1,2,3,4,5], [0,1], [0,2], [0,3], [0,4], [0,5], [1,2], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [3,5], [4, 5]]

# CAN SPECIFY THE SUBJECTS-PAIRS SET HERE.
# FOR THE LMCCA DENOISING STEP.

for subs in subs_set:
    n_subs = len(subs)

    str_subs = str(subs[0])
    for each_sub in subs[1:]:
        str_subs += "_{}".format(each_sub)

    print("eyedee    : {}".format(eyedee))
    print("n_subs    : {}".format(n_subs))
    print("subs      : {}".format(subs))

    # SAVES THE RESULTS IN THE path_name FOLDER.
    if not(path.exists(crrnt_dir + "/" + name_of_the_script + "/")):
        os.mkdir(crrnt_dir + "/" + name_of_the_script + "/")
    crrnt_dir = crrnt_dir + "/" + name_of_the_script
    path_name = crrnt_dir + "/" + eyedee + f"_subs_{str_subs}" + "/"

    i = 1
    while path.exists(path_name):
        path_name = crrnt_dir + "/" + eyedee + f"_subs_{str_subs}_" + str(i) + "/"
        i = i + 1

    del i
    os.mkdir(path_name)
    print(path_name)


    # NUMBER OF CHANNELS FOR THE PREPROCESSED STIMULI AFTER FILTERBANK
    stim_chans = 21
    # NUMBER OF CHANNELS FOR RAW PREPROCESSED STIMULI (1D)
    stim_chans_raw = 1


    # SAVED THE CORRELATIONS IN all_corrs MATRIX
    all_corrs = np.zeros(n_subs)
    all_corrs_name = path_name + 'corrs_{}.npy'.format(str_subs)


    data_subs_raw = load_data(blocks)
    # THE DATA data_subs_raw IS LOADED SUCH THAT 
    # ALL THE N EEG RESPONSES ARE LOADED IN THE FIRST N LISTS
    # AND THE LAST LIST HAS STIMULUS
    # data_subs_raw IS A list OF SIZE N+1
    # EACH ELEMENT IS A list OF SIZE 3
    # SUCH THAT
    # data_subs_raw[n] = [TRAINING_DATA, VALIDATION_DATA, TESTING_DATA]
    # AND
    # data_subs_raw[n][j].shape = [Number_of_samples, dimensions]

    ## LINEAR MCCA
    print("LINEAR MCCA + LCCA3")
    lmc_corrs = np.zeros(all_corrs.shape[1])
    pca_chans = 40

    resp = data_subs_raw[:-1]
    stim = data_subs_raw[-1]
    del data_subs_raw

    new_chans = int(pca_chans)
    nchans2 = int(new_chans); 
    new_data = [[],[],[]] ; W = [None]*n_subs ; meanp = [None]*n_subs 

    for num in range(n_subs):
        [meanp[num], W[num], resptr_PCA] = my_PCA(resp[num][0],nchans2);
        respval_PCA = apply_PCA(resp[num][1],meanp[num],W[num]);
        respte_PCA  = apply_PCA(resp[num][2],meanp[num],W[num]);
        new_data[0].append(resptr_PCA)
        new_data[1].append(respval_PCA)
        new_data[2].append(respte_PCA)

    resptr_x  = np.concatenate(new_data[0], 1)
    respval_x = np.concatenate(new_data[1], 1)
    respte_x  = np.concatenate(new_data[2], 1)
    del new_data

    stimtr  = stim[0]
    stimval = stim[1]
    stimte  = stim[2]
    del stim
    stimtr_x = lagGen(stimtr, np.arange(nchans2))


    data_tr = np.concatenate([resptr_x, stimtr_x], 1)
    C = np.matmul(data_tr.T, data_tr)
    del data_tr, resptr_x, stimtr_x

    # PROVIDING THE CORRELATION MATRIX TO THE MCCA BLOCK
    [V, rho, AA] = my_mcca(C, nchans2)
    
    [stim5_tr, stim5_val, stim5_te,_,_,_] = filtem(stimtr, stimval, stimte, None, None, None)

    dout = 110
    new_lmlc_data = []
    data_subs_mc = []
    lmc_corrs = np.zeros(n_subs)
    for isub in range(n_subs):
        print("Subject number: "+str(isub))
        Vn = np.matmul(W[isub], AA[isub])
        Vo = Vn[:, :dout]
        Vninv = np.linalg.pinv(Vn)
        Vninv = Vninv[:dout, :]
        Vn = np.matmul(Vo, Vninv)

        sub_tr  = np.matmul(resp[isub][0] - meanp[isub], Vn)
        sub_val = np.matmul(resp[isub][1] - meanp[isub], Vn)
        sub_te  = np.matmul(resp[isub][2] - meanp[isub], Vn)
        pre_sub_data = [sub_tr, sub_val, sub_te]

        [meanp1, W1, resptr] = my_PCA(sub_tr,60)
        respval = apply_PCA(sub_val,meanp1,W1)
        respte  = apply_PCA(sub_te, meanp1,W1)

        # print(resptr, respval, respte)
        # print(stimtr, stimval, stimte)

        [_, _, _, resp5_tr, resp5_val, resp5_te] = filtem(None, None, None, resptr, respval, respte)

        lmc_corrs[isub], temp = cca3_model_new([stim5_tr, stim5_val, stim5_te], [resp5_tr, resp5_val, resp5_te], 1)
        new_lmlc_data.append(temp)
        data_subs_mc.append(pre_sub_data)
        print(lmc_corrs[isub])

    print('Linear MCCA corrs are : {}'.format(lmc_corrs))
    # return new_lmlc_data, lmc_corrs, pre_subs_data

    del meanp, meanp1, W, W1


    print('LMCCA + LCCA3 corrs are : ' + str(lmc_corrs))
    all_corrs[1,:]  = lmc_corrs
    np.save(all_corrs_name, all_corrs)


    # SAVING LMLC OUTPUT 

    # new_lmlc_data, lmc_corrs, data_subs_mc = linear_mcca_custom(data_subs_raw, pca_chans)
    for sub_num in range(n_subs):
        # x1 = new_lmlc_data[sub_num][2][0]
        # x2 = new_lmlc_data[sub_num][2][1]
        s = path_name + "/plot_data_lmlc_sub_"+str(sub)
        plot_data(my_standardize(new_lmlc_data[sub_num][2][0]), my_standardize(new_lmlc_data[sub_num][2][1]), s)
    fp = open(path_name + f'/lmlc_data_block_{blocks}_{str_subs}.pkl', 'wb')
    pickle.dump(new_lmlc_data, fp)
    fp.close()
    del new_lmlc_data

    new_data_lmcca = []
    for sub_num in range(n_subs):
        fp = open(path_name + f'/lmdc_data_block_{blocks}_{sub_num}.pkl', 'wb')
        pickle.dump(pca_filt_pca_resp(data_subs_mc[sub_num]), fp)
        fp.close()
        # sub_resp = [sub_data_mc[0][1], sub_data_mc[1][1], sub_data_mc[2][1]]
        # new_data_lmcca.append(pca_filt_pca_resp(sub_data_mc))
        # del sub_data_mc
    del data_subs_mc
    
    # new_data_lmcca.append(data_subs[-1])
    # data_path = 'data1_'+str(blocks)+'.pkl'
    # fp = open(data_path, 'rb')
    # data1 = pickle.load(fp)
    # fp.close()
    # print("Loaded FILTERED Data.")

    # data_subs = []
    # for sub in subs:
    #     data_subs.append([       data1[0][:,:,sub],    data1[1][:,:,sub],    data1[2][:,:,sub]])
    # data_subs.append([data1[0][:,:stim_chans,-1],  data1[1][:,:stim_chans,-1], data1[2][:,:stim_chans,-1]])

    # fp = open(path_name + f'/lmdc_data_block_{blocks}_stim.pkl', 'wb')
    # pickle.dump(pca_filt_pca_resp(data_subs[-1]), fp)
    # fp.close()
    # del data_subs

    print('saved')


