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

from pdb import set_trace as bp  #################added break point accessor####################
from scipy.signal import lfilter
try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp  # noqa 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.nn import Parameter
from   torch.utils.data import DataLoader

from deep_nets   import *
from deep_losses import *

from cca_functions import *
from speech_helper import load_new_data
# from music_helper  import stim_resp
# from deep_models   import dcca_model

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
# o_dim = int(a[1])   # THE INTERESTED OUTPUTS DIMENSIONALITY
num_blocks_start = int(a[1])
num_blocks_end = int(a[2])

# dropout    = 0.05
learning_rate = 1e-3
epoch_num  = 18
batch_size = 800
reg_par    = 1e-4
o_dim      = 5
use_all_singular_values = False
best_only  = True


print(f"eyedee    : {eyedee}")
print(f"best_only : {best_only}")
print(f"epoch_num : {epoch_num}")
# print(f"dropout   : {dropout}")

device = torch.device('cuda')
# device = torch.device('cpu')
torch.cuda.empty_cache()

# CREATING A FOLDER TO STORE THE RESULTS
path_name = f"dcca_{eyedee}_{num_blocks_start}_{num_blocks_end}/"

i = 1
while path.exists(path_name):
    path_name = f"dcca_{eyedee}_{num_blocks_start}_{num_blocks_end}_{i}/"
    i = i + 1

del i
os.mkdir(path_name)
# print(path_name)


##################### SEED #####################
# seed = np.ceil(np.random.rand(10)*100)
seed = np.ceil(np.random.rand(1)*100) * np.ones(1)
print(seed)
###############################################

D = [0.05, 0.1, 0.2]
# CAN REPLACE D WITH A SINGLE ELEMENT LIST WHOSE VALUE IS EQUAL TO THE DESIRED DROPOUT.


# HELPER FUNCTION FOR PERFORMING DCCA
def dcca_method(stim_data, resp_data, dropout, saving_name_root):
    """
    CUSTOM DCCA METHOD
    """
    print(f"DCCA for {saving_name_root}")

    # USING dcca_model for DCCA
    new_data_d, correlations, model_d = dcca_model(stim_data, resp_data, o_dim, learning_rate, use_all_singular_values, epoch_num, batch_size, reg_par, dropout, best_only, path_name, seed)

    x1 = new_data_d[2][0]
    x2 = new_data_d[2][1]
    x3 = new_data_d[1][0]
    x4 = new_data_d[1][1]
    corr_d     = np.squeeze(my_corr(x1, x2, o_dim))
    corr_d_val = np.squeeze(my_corr(x3, x4, o_dim))
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
    subs = [11, 13]       # REPLACE WITH THE REQUIRED SUBJECTS' IDS.
    subs = sorted(subs) # TO KEEP THEIR IDS SORTED
    n_subs = len(subs)

    str_subs = str(subs[0])
    for each_sub in subs[1:]: 
        str_subs += f"_{each_sub}"

    # num_blocks_start = 0
    # num_blocks_end   = 20
    # CAN CHANGE BOTH VALUES ACCORDING TO THE INTERESTED CROSS-VALIDATION EXPERIMENTS.
    # CAN SUBMIT THESE TWO AS THE ARGUMENTS AND PARSE OVER THERE, FOR BULK EXPERIMENTS.

    tst_corrs = np.zeros((num_blocks, len(D), n_subs))
    val_corrs = np.zeros((num_blocks, len(D), n_subs))
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
                # data_sub = pkl.load(open(f"/data2/jaswanthr/data/mcca/all_subs_data/data{subs[sub]}_{block}.pkl", "rb"))["resp"]
                # data_stim = pkl.load(open(f"/data2/jaswanthr/data/mcca/all_subs_data/data{subs[sub]}_{block}.pkl", "rb"))["stim"]

                data_sub = data_subs[sub]

                saving_name_root = f"speech_block_{block}_sub_{subs[sub]}_{dropout}"
                dcca_corrs[sub], dcca_corrs_val[sub] = dcca_method(data_stim, data_sub, dropout, saving_name_root)

                print(f'DCCA corrs are : {dcca_corrs}')

                tst_corrs[block, d_cnt] = dcca_corrs
                val_corrs[block, d_cnt] = dcca_corrs_val

                np.save(tst_corrs_name, tst_corrs)
                np.save(val_corrs_name, val_corrs)

            print(f'DCCA corrs for {block}, {dropout} are : {tst_corrs[block, d_cnt]}')
            print(f'saved speech.')






nmedh_dcca = False
if nmedh_dcca:
    fs = 80
    N = 125
    subs = 58
    tst_corrs = np.zeros((subs, 4, len(D)))
    tst_corrs_name = f'{path_name}/nmedh_corrs.npy'
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
                    tst_corrs[sub_num-1, stim_id, d_cnt], val_corrs[sub_num-1, stim_id, d_cnt] = dcca_method([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], dropout,  f"nmedh_sub_{sub_num}_{stim_str}_{dropout}")
                    np.save(tst_corrs_name, tst_corrs)
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
                        tst_corrs[sub_num-1, stim_id+1, d_cnt], val_corrs[sub_num-1, stim_id+1, d_cnt] = dcca_method([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], dropout,  f"nmedh_sub_{sub_num}_{stim_str}_{dropout}")
                        np.save(tst_corrs_name, tst_corrs)
                        np.save(val_corrs_name, val_corrs)







custom_data = False
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


    tst_corrs = np.zeros(len(D))
    val_corrs = np.zeros(len(D))
    tst_corrs_name = f'{path_name}/corrs.npy'
    val_corrs_name = f'{path_name}/corrs_val.npy'

    for d_cnt, dropout in enumerate(D):
        save_name_root = f"custom_{dropout}"
        corrs[d_cnt], corrs_val[d_cnt] = dcca_method([stim_tr, stim_val, stim_te], [resp_tr, resp_val, resp_te], dropout, save_name_root)

        np.save(tst_corrs_name, tst_corrs)
        np.save(val_corrs_name, val_corrs)

    print("SAVED.")























# IF GIVEN 10 SEEDS, ALL THE MODELS GET ONE FORWARD PASS AND SEED WITH BEST VALIDATION IS SELECTED
# IF ONLY ONE SEED, THE WEIGHTS ARE INITIALIZED ACCORDINGLY
# TRAIN AND RETURN THE MODEL
# MODEL : model2_13
# LOSS  : cca_loss
def dcca_model(stim_data, resp_data, o_dim, learning_rate=1e-3, use_all_singular_values=False, epoch_num=12, batch_size=2048, reg_par=1e-4, dropout=0.05, best_only=True, path_name="", seeds=np.ceil(np.random.rand(10)*100)):
    """
    ARGUMENTS: 
        stim_data  : A THREE ELEMENT LIST OF STIMULI  DATA ARRANGED AS: [STIM_TRAINING, STIM_VALIDATION, STIM_TEST]
        resp_data  : A THREE ELEMENT LIST OF RESPONSE DATA ARRANGED AS: [RESP_TRAINING, RESP_VALIDATION, RESP_TEST]
        learning_rate : LEARNING RATE OF THE MODEL (DEFAULT: 1e-3)
        use_all_singular_values : WHETHER THE MODEL SHOULD USE ALL THE SINGULAR VALUES IN THE CCA LOSS (DEFAULT: False)
        epoch_num  : NUMBER OF EPOCHS OF TRAINING (DEFAULT: 12)
        batch_size : MINIBATCH SIZES FOR TRAINING THE MODEL (DEFAULT: 2048)
        reg_par    : REGULARIZATION PARAMETER FOR WEIGHT DECAY (DEFAULT: 1e-4)
        dropout    : DROPOUTS PERCENTAGE IN THE MODEL (DEFAULT: 0.05)
        best_only  : SAVE THE MODEL ONLY WITH THE BEST VALIDATION LOSS (DEFAULT: True)
        path_name  : WHERE THE MODEL IS TO BE SAVED. (DEFAULT: "")
        seeds      : SEED FOR THE DEEP MODEL. If given one seed, the model will be initialized with that seed.  
                     IF given more than one seed, the seed with best val loss is selected.
    
    RETURNS:
        new_data      : NEW REPRESENTATIONS AFTER PERFORMING DEEP CCA
        correlations  : THE TRAINING, VALIDATION AND TEST SET LOSSES WHILE TRAINING THE MODEL - TO TRACK THE MODEL AS TRAINING PROGRESSED.
        model         : THE TRAINED MODEL.
    """

    stimtr  = stim_data[0]
    stimval = stim_data[1]
    stimte  = stim_data[2]
    resptr  = resp_data[0]
    respval = resp_data[1]
    respte  = resp_data[2]

    # stimtr, mean1, std1 = my_standardize(stimtr)
    # resptr, mean2, std2 = my_standardize(resptr)

    # stimval = (stimval - mean1) / std1
    # stimte  = (stimte  - mean1) / std1
    # respval = (respval - mean2) / std2
    # respte  = (respte  - mean2) / std2

    resp_tr  = torch.from_numpy(resptr ).float()
    resp_val = torch.from_numpy(respval).float()
    resp_te  = torch.from_numpy(respte ).float()

    stim_tr  = torch.from_numpy(stimtr ).float();        
    stim_val = torch.from_numpy(stimval).float();      
    stim_te  = torch.from_numpy(stimte ).float();        

    data_tr  = torch.cat([resp_tr,  stim_tr ], 1)
    data_val = torch.cat([resp_val, stim_val], 1)
    data_te  = torch.cat([resp_te,  stim_te ], 1)

    i_shape1 = resp_tr.shape[1]
    i_shape2 = stim_tr.shape[1]

    # best_only = True
    act = "sigmoid"
    o_act = 'leaky_relu'

    if (isinstance(seeds, int)): seed = seeds
    elif not(isinstance(seeds, int)) and len(seeds) == 1: seed = seeds[0]
    else:
        torch.backends.cudnn.deterministic = True
        first_and_last = np.zeros((len(seeds),3))
        models = [None] * len(seeds)
        print('seeds: ', seeds)

        for seed_num, seed in enumerate(seeds) : 
            torch.manual_seed(seed)
            if torch.cuda.is_available() : torch.cuda.manual_seed_all(seed)
            
            num_layers = 2
            h_size     = 512

            model = LSTM_13(num_layers, i_shape1, i_shape2, h_size, o_dim)
            # model = model2_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            model = model.to(device)
            model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_par)

            print('MODEL : {}'.format(seed_num))

            model.eval()
            torch.cuda.empty_cache()

            tr_loss = 0 ; count = 0
            # dataloader = DataLoader(data_tr, batch_size, shuffle=True)
            dataloader = DataLoader(data_tr, batch_size, shuffle=False)
            with torch.no_grad():
                for trs in dataloader : 
                    trs = trs.to(device)
                    outputs = model(trs)
                    loss = cca_loss(outputs, o_dim, use_all_singular_values)
                    tr_loss = tr_loss + loss
                    count = count + 1
                    del trs
            tr_loss = tr_loss / count
            
            data_val = data_val.to(device)
            val_ops = model(data_val)
            val_loss = cca_loss(val_ops, o_dim, use_all_singular_values)
            data_val = data_val.cpu()
            torch.cuda.empty_cache()
            
            data_te = data_te.to(device)
            test_ops = model(data_te)
            test_loss = cca_loss(test_ops, o_dim, use_all_singular_values)
            data_te = data_te.cpu()
            torch.cuda.empty_cache()

            models[seed_num] = model
            first_and_last[seed_num] = [-tr_loss, -val_loss, -test_loss]
            print('{:0.4f} {:0.4f} {:0.4f}'.format(-tr_loss, -val_loss, -test_loss))

        np.set_printoptions(precision=4)
        idx = np.argsort(-first_and_last[:,1])
        print(first_and_last[idx,1:])
        print(seeds[idx])
        seed = seeds[idx[0]]

    print("seed:    ", seed   )

    torch.manual_seed(seed)
    if torch.cuda.is_available() : torch.cuda.manual_seed_all(seed)
    model = model2_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    model = model.to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_par)

    model_state_dict = []
    min_loss = 0.00 ; min_loss2 = 0.00
    correlations = np.zeros((epoch_num, 3))
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        model.train()
        # dataloader = DataLoader(data_tr, batch_size, shuffle=True)
        dataloader = DataLoader(data_tr, batch_size, shuffle=False)
        for trs in dataloader : 
            model_optimizer.zero_grad()
            trs = trs.to(device)
            outputs = model(trs)
            loss = cca_loss(outputs, o_dim, use_all_singular_values)
            loss.backward()
            model_optimizer.step()
            del trs
        
        model.eval()
        torch.cuda.empty_cache()
        tr_loss = 0
        count = 0
        # dataloader = DataLoader(data_tr, batch_size, shuffle=True)
        dataloader = DataLoader(data_tr, batch_size, shuffle=False)
        with torch.no_grad():
            for trs in dataloader :
                trs = trs.to(device)
                outputs = model(trs)
                loss = cca_loss(outputs, o_dim, use_all_singular_values)
                loss = loss.item()
                tr_loss = tr_loss + loss
                count = count + 1
                del trs
        correlations[epoch, 0] = -tr_loss / (count)
        torch.cuda.empty_cache()

        print('EPOCH : {}'.format(epoch))
        print('  Training CORRELATION   : {:0.4f}'.format(correlations[epoch, 0]))

        data_val = data_val.to(device)
        val_ops = model(data_val)
        val_loss = cca_loss(val_ops, o_dim, use_all_singular_values)
        correlations[epoch, 1] = -val_loss
        data_val = data_val.cpu()
        torch.cuda.empty_cache()
        print('  Validation CORRELATION : {:0.4f}'.format(-val_loss))
        
        data_te = data_te.to(device)
        test_ops = model(data_te)
        test_loss = cca_loss(test_ops, o_dim, use_all_singular_values)
        correlations[epoch, 2] = -test_loss
        data_te = data_te.cpu()
        torch.cuda.empty_cache()
        print('  Test CORRELATION       : {:0.4f}'.format(-test_loss))

        print("  val. loss is : {:0.4f} & the min. loss is : {:0.4f}".format(val_loss, min_loss))
        print("  AND since, val_loss < min_loss is {}".format(val_loss < min_loss))

        if val_loss < min_loss2:
            min_loss2 = val_loss

        model_file_name = path_name + '/best_model.pth'

        if best_only == True:
            if val_loss < min_loss or epoch == 0:
                torch.save({
                            'epoch' : epoch,
                            'model_state_dict' : model.state_dict(),
                            'optimizer_state_dict': model_optimizer.state_dict(),
                            'loss': loss}, model_file_name)
                print('  Saved the model at epoch : {}\n'.format(epoch))
                min_loss = val_loss
            else:
                if epoch != 0:
                    checkpoint = torch.load(model_file_name)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    best_epoch = checkpoint['epoch']
                    # loss = checkpoint['loss']
                    print('  Loaded the model from epoch : {}.\n'.format(best_epoch))
                    model.train()

    model.eval()
    data2 = [data_tr, data_val, data_te]
    with torch.no_grad():
        new_data = []
        for k in range(3):
            temp = data2[k].to(device)
            pred_out = model(temp)
            new_data.append([pred_out[0].cpu().numpy(), pred_out[1].cpu().numpy()])

    # x1 = new_data[2][0]
    # x2 = new_data[2][1]
    # result = np.squeeze(my_corr(x1, x2, o_dim))
    # print(result)

    return new_data, correlations, model
