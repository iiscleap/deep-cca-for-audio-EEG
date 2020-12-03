import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import os
from os import path
import scipy.io
import random
from pdb import set_trace as bp  #################added break point accessor####################
from scipy.signal import lfilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data import DataLoader

from cca_functions import *
from all_i_need    import final_dcca3
from all_linear_stuff import *

def plot_data(x, y,s):
    plt.clf()
    x = x[0]
    y = y[0]
    plt.plot(x, color='orange')
    plt.plot(y, color='blue')
    plt.legend(['stim', 'resp'])
    plt.savefig(s+'.eps', format="eps")

def plot_losses_3_columns_tr_val_te(losses, s, marker="o"):
    plt.clf()
    plt.plot(losses[:, 0], marker=marker, color='red')
    plt.plot(losses[:, 1], marker=marker, color='blue')
    plt.plot(losses[:, 2], marker=marker, color='green')
    plt.legend(['training', 'valid', 'test'])
    # plt.savefig(s+'.png', format="png")
    plt.savefig(s+'.eps', format="eps")

name_of_the_script = sys.argv[0].split('.')[0]
a = sys.argv[1:]
eyedee = str(a[0])
o_dim = int(a[1])
if a[2] == 'all_drps':
    D = [0.05, 0.1, 0.2]
else:
    D = [float(x) for x in a[2:]]


best_only  = True
learning_rate = 1e-3
epoch_num  = 20
batch_size = 1600
reg_par    = 1e-4
o_dim      = 1
use_all_singular_values = False

print("eyedee    : {}".format(eyedee))
print("best_only : {}".format(best_only))
print("epoch_num : {}".format(epoch_num))
print("D         : {}".format(D))


device = torch.device('cuda')
torch.cuda.empty_cache()

crrnt_dir = os.getcwd()
strings = crrnt_dir.split('/')
strings = strings[:-1]
strings[-1] = 'results'
crrnt_dir = '/'.join(strings)

str_subs = str(subs[0])
for each_sub in subs[1:]:
    str_subs += "_{}".format(each_sub)

##################### SEED #####################
# seed = np.ceil(np.random.rand(10)*100)
seed = np.ceil(np.random.rand(1)*100) * np.ones(1)
print(seed)
###############################################


str_D = "all_drps"
if len(D) == 1:
    str_D = str(D[0])

# CREATING A FOLDER TO STORE THE RESULTS
if not(path.exists(crrnt_dir + "/" + name_of_the_script + "/")):
    os.mkdir(crrnt_dir + "/" + name_of_the_script + "/")
crrnt_dir = crrnt_dir + "/" + name_of_the_script
path_name = crrnt_dir + "/" + eyedee + f"_dcca_only_{str_D}" + "/"

i = 1
while path.exists(path_name):
    path_name = crrnt_dir + "/" + eyedee + f"_dcca_only_{str_D}_" + str(i) + "/"
    i = i + 1

del i
os.mkdir(path_name)
print(path_name)


# NUMBER OF CHANNELS FOR THE PREPROCESSED STIMULI AFTER FILTERBANK
stim_chans = 21
# NUMBER OF CHANNELS FOR pre PREPROCESSED STIMULI (1D)
stim_chans_pre = 1









# HELPER FUNCTION TO LOAD DATA FOR SPEECH
def load_data(subs, block=0, filtered=True):
    # IF THE DATA IS ALREADY PROCESSED THROUGH THE FILTERBANK AND PCA
    print('block: ' + str(block))
    if block == num_blocks - 1:
        val_idx = 0
    else:
        val_idx = block + 1

    if filtered:
        data_path = 'data1_'+str(block)+'.pkl'
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
        data_path = 'data1_pre_'+str(block)+'.pkl'
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


    fp = open(path_name + '/data_subs.pkl', 'wb')
    pickle.dump(data_subs, fp)
    fp.close()

    return data_subs


speech_dcca = True
if speech_dcca:
    num_blocks = 20     # SPECIFY THE CROSS-VALIDATION NUMBER HERE
    subs = [None]       # SPECIFY THE SUBJECTS INTERESTED HERE
    subs = sorted(subs)
    n_subs = len(subs)

    all_corrs = np.zeros((len(D), n_subs))
    val_corrs = np.zeros((len(D), n_subs))
    all_corrs_name = path_name + 'corrs_{}_{}.npy'.format(    blocks, str_subs)
    val_corrs_name = path_name + 'corrs_val_{}_{}.npy'.format(blocks, str_subs)

    for block in range(num_blocks):

        load_data(subs, block) 

        for d_cnt, dropout in enumerate(D) : 
            print(' {} individual subject DCCA for block : {} , for dropout : {}'.format(str_subs, blocks, dropout))
            file_name = 'run_{}_subs_{}_with_drpt_{}'.format(str(blocks), str(str_subs), str(dropout))    
            
            ## DEEP CCA METHOD
            print("DCCA")
            data_subs = pkl.load(open(path_name + "/data_subs.pkl", "rb"))
            data_stim = data_subs[-1]
            del data_subs

            dcca_corrs     = np.zeros((n_subs))
            dcca_corrs_val = np.zeros((n_subs))
            for sub_num in range(n_subs):
                print("Sub: "+str(subs[sub_num]))
                data_subs = pkl.load(open(path_name + "/data_subs.pkl", "rb"))
                data_sub = data_subs[sub_num]
                del data_subs
                new_datax, lossesx1, model_d = dcca_model(data_stim, data_sub, o_dim, learning_rate, use_all_singular_values, epoch_num, batch_size, reg_par, dropout, path_name, seed)
                print(lossesx1)
                x1 = new_datax[2][0]
                x2 = new_datax[2][1]
                dcca_corrs[sub_num] = np.squeeze(my_corr(x1, x2, o_dim))

                s = path_name + "/plot_data_dcca3_5_blocks_"+str(blocks)+"_sub_"+str(subs[sub_num])+"_dropout_"+str(dropout)
                plot_data(my_standardize(x1), my_standardize(x2), s)
                s = path_name + "/plot_losses_dcca3_5_blocks_"+str(blocks)+"_sub_"+str(subs[sub_num])+"_dropout_"+str(dropout)
                plot_losses_3_columns_tr_val_te(lossesx1, s)
                print(dcca_corrs[sub_num])

                x3 = new_datax[1][0] ; x4 = new_datax[1][1]
                dcca_corrs_val[sub_num] = np.squeeze(my_corr(x3, x4, o_dim))

                fp = open(path_name + f'/dcca_5_data_block_{blocks}_{subs[sub_num]}_for_dropout_{dropout}.pkl', 'wb')
                pkl.dump(new_datax, fp)
                fp.close()
                del new_datax

                model_name = path_name + f'/dcca_5_model_{subs[sub_num]}_complete_{dropout}.pth.tar'
                torch.save(model_d, model_name)
                # torch.save({'state_dict': model_d.state_dict()}, path_name + f'/dcca_5_model_{subs[sub_num]}_dict_{dropout}.pth.tar')
                del model_d

                print('DCCA3 corrs are : ' + str(dcca_corrs))

                all_corrs[1 + d_cnt] = dcca_corrs
                val_corrs[1 + d_cnt] = dcca_corrs_val

                np.save(all_corrs_name, all_corrs)
                np.save(val_corrs_name, val_corrs)


            print('So..\n\n\n')
            print('DCCA3 corrs for dim are : ' + str(dcca_corrs))

            np.save(all_corrs_name, all_corrs)
            np.save(val_corrs_name, val_corrs)

            print('saved')


        os.remove(path_name + '/best_model.pth')
        os.remove(path_name + '/data_subs.pkl')


nmedh_dcca = True
if nmedh_dcca:
    fs = 80
    N = 125
    subs = 58
    all_corrs = np.zeros((subs, 4, len(D)))
    all_corrs_name = path_name + 'nmedh_dcca_corrs.npy'
    rm_list = [0, 8, 20, 23, 24, 34, 37, 40, 45, 46, 53]

    # data_path = '/data/nmed-h/sub_data/'
    data_path = "# ADD YOUR DATA PATH HERE."


    for sub_num1 in range(59):
        if sub_num1 not in rm_list:
            sub_num = int(sub_num1)
            datas = []
            resp = pkl.load(open(data_path + f'all_sub_{sub_num}_data.pkl','rb'))[1]

            for stim_id in range(4):
                stim = pkl.load(open(data_path + 'all_stim_data.pkl','rb'))[1][stim_id]

                for d_cnt, drpt in enumerate(D):
                    print('SUBJECT: {}, STIM_ID: {}, DROPOUT: {}'.format(sub_num, stim_id, drpt))
                    
                    new_data_d, losses = dcca_model([stim5_tr, stim5_val, stim5_te], [resp5_tr, resp5_val, resp5_te], o_dim, l_rate, use_all_singular_values, epoch_num, batch_size, reg_par, drpt, path_name, seed)
                    # plots(path_name + 'sub_{}_{}_{}'.format(sub_num, stim_id, drpt), losses)

                    x1 = new_data_d[2][0].cpu().numpy()
                    x2 = new_data_d[2][1].cpu().numpy()
                    x3 = new_data_d[1][0].cpu().numpy()
                    x4 = new_data_d[1][1].cpu().numpy()
                    plot_data(x1, x2, path_name + f'plot_sub_{sub_num}_{stim_id}_{drpt}')
                    corr_d = np.squeeze(my_corr(x1, x2, o_dim))
                    print('DCCA3 is : ' + str(corr_d))

                    fp = open(path_name + f'new_data_sub_{sub_num}_{stim_id}_{drpt}.pkl', 'wb')
                    pickle.dump(new_data_d[1:], fp)
                    fp.close()

                    all_corrs[sub_num, stim_id, d_cnt] = corr_d
                    val_corrs[sub_num, stim_id, d_cnt] = np.squeeze(my_corr(x3, x4, o_dim))

                    del new_data_d

                    np.save(all_corrs_name, all_corrs)
                    np.save(val_corrs_name, val_corrs)
                
                print(all_corrs[sub_num, stim_id])

