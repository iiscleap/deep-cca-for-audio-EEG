import numpy as np
from os import path
import scipy.io
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

from cca_functions import my_standardize, my_corr

from deep_nets   import *
from deep_losses import *

device = torch.device('cuda')
torch.cuda.empty_cache()




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

    stimtr, mean1, std1 = my_standardize(stimtr)
    resptr, mean2, std2 = my_standardize(resptr)

    stimval = (stimval - mean1) / std1
    stimte  = (stimte  - mean1) / std1
    respval = (respval - mean2) / std2
    respte  = (respte  - mean2) / std2

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

    if (isinstance(seeds, int)):
        seed = seeds
    elif not(isinstance(seeds, int)) and len(seeds) == 1:
        seed = seeds[0]
    else:
        torch.backends.cudnn.deterministic = True
        first_and_last = np.zeros((len(seeds),3))
        models = [None] * len(seeds)
        print('seeds: ', seeds)

        for seed_num, seed in enumerate(seeds) : 
            torch.manual_seed(seed)
            if torch.cuda.is_available() : torch.cuda.manual_seed_all(seed)

            model = model2_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            model = model.to(device)
            model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_par)

            print('MODEL : {}'.format(seed_num))

            model.eval()
            torch.cuda.empty_cache()

            tr_loss = 0 ; count = 0
            dataloader = DataLoader(data_tr, batch_size, shuffle=True)
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
        dataloader = DataLoader(data_tr, batch_size, shuffle=True)
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
        dataloader = DataLoader(data_tr, batch_size, shuffle=True)
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



# DMCCA MODEL WITH N RESPS AND 1 STIM
# IF GIVEN 10 SEEDS, ALL THE MODELS GET ONE FORWARD PASS AND SEED WITH BEST VALIDATION IS SELECTED
# IF ONLY ONE SEED, THE WEIGHTS ARE INITIALIZED ACCORDINGLY
# THE MODEL GETS TRAINED AND 
# MODEL : dmcca_model_n_resp_1_stim
# LOSS  : dmcca_model_loss
# RETURNS : NEW DATA, TRAINING LOSSES, AND THE TRAINED MODEL
def dmcca_model(all_data, o_dim, learning_rate=1e-3, use_all_singular_values=False, epoch_num=12, batch_size=2048, reg_par=1e-4, dropout=0.05, best_only=True, lambda_=0.1, path_name="", mid_shape=60, seeds=np.ceil(np.random.rand(10)*100)):
    """
    ARGUMENTS: 
        all_data      : AN (N) ELEMENT LIST OF DATA WITH EACH ELEMENT AS: [DATA_i_TRAINING, DATA_i_VALIDATION, DATA_i_TEST]
        ASSUMPTION : 
            THE FIRST (N-1) ELEMENTS ARE THE (N-1) EEG RESPONSES FOR A COMMON STIMULUS.
            THE LAST    1    ELEMENT IS  THE COMMON AUDITORY STIMULUS.

        learning_rate : LEARNING RATE OF THE MODEL (DEFAULT: 1e-3)
        use_all_singular_values : WHETHER THE MODEL SHOULD USE ALL THE SINGULAR VALUES IN THE CCA LOSS (DEFAULT: False)
        epoch_num  : NUMBER OF EPOCHS OF TRAINING (DEFAULT: 12)
        batch_size : MINIBATCH SIZES FOR TRAINING THE MODEL (DEFAULT: 2048)
        reg_par    : REGULARIZATION PARAMETER FOR WEIGHT DECAY (DEFAULT: 1e-4)
        dropout    : DROPOUTS PERCENTAGE IN THE MODEL (DEFAULT: 0.05)
        best_only  : SAVE THE MODEL ONLY WITH THE BEST VALIDATION LOSS (DEFAULT: True)
        lambda_       : MSE REGULARIZATION PARAMETER
        path_name  : WHERE THE MODEL IS TO BE SAVED. (DEFAULT: "")
        seeds      : SEED FOR THE DEEP MODEL. (DEFAULT: 10 RANDOM SEEDS)
    
    RETURNS:
        new_data         : NEW REPRESENTATIONS AFTER PERFORMING DEEP CCA
        training_losses  : THE TRAINING, VALIDATION AND TEST SET LOSSES WHILE TRAINING THE MODEL - TO TRACK THE MODEL AS TRAINING PROGRESSED.
        model            : THE TRAINED MODEL.
    """
    print('Started multiway DCCA.')

    # data = [resp1, resp2, ..., respn, stim]
    N = len(all_data)
    
    torch.cuda.empty_cache()

    data_tr  = np.concatenate([i[0] for i in all_data], 1)
    data_val = np.concatenate([i[1] for i in all_data], 1)
    data_te  = np.concatenate([i[2] for i in all_data], 1)

    data = [data_tr, data_val, data_te]

    i_shape1 =  all_data[0][0].shape[1]
    i_shape2 = all_data[-1][0].shape[1]

    print(i_shape1)
    print(i_shape2)

    # EACH ONE : T x (R1 + R2 + STIM)
    train_set = torch.from_numpy(data_tr).float()
    val_set   = torch.from_numpy(data_val).float()
    te_set    = torch.from_numpy(data_te).float()

    [data_tr, data_val, data_te] = [train_set, val_set, te_set]

    best_only = True
    act = "sigmoid"
    o_act = 'leaky_relu'

    if (isinstance(seeds, int)):
        seed = seeds
    elif not(isinstance(seeds, int)) and len(seeds) == 1:
        seed = seeds[0]
    else:
        torch.backends.cudnn.deterministic = True
        first_and_last = np.zeros((len(seeds),3))
        to_append = np.zeros((len(seeds), 3, int(comb(N,2))+1))
        models=[None]*len(seeds)
        print('seeds: ', seeds)
        for seed_num, seed in enumerate(seeds) : 
            torch.manual_seed(seed)
            if torch.cuda.is_available() : torch.cuda.manual_seed_all(seed)

            model = dmcca_model_n_resp_1_stim(N-1, i_shape1, i_shape2, mid_shape, o_dim, dropout)
            model = model.to(device)

            model_optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=reg_par)
            print('MODEL : {} for seed : {}'.format(seed_num, seed))

            model.eval()
            torch.cuda.empty_cache()

            tr_corr_loss = 0
            count   = 0
            dataloader = DataLoader(data_tr, batch_size, shuffle=True)
            with torch.no_grad():
                for trs in dataloader :
                    trs = trs.to(device)
                    outputs = model(trs)
                    _, corr_loss, _,neg_corrs,_ = dmcca_model_loss(trs, outputs, i_shape1, o_dim, lambda_, use_all_singular_values)
                    trs = trs.cpu()
                    tr_corr_loss  = tr_corr_loss + corr_loss
                    count = count + 1
                    del trs
            tr_corr_loss = tr_corr_loss  / (count)
            to_append[seed_num, 0, :] =  np.concatenate([[-tr_corr_loss.detach().numpy()], -neg_corrs.detach().numpy()])

            data_val = data_val.to(device)
            val_ops = model(data_val)
            _, val_corr_loss, _,neg_corrs,_ = dmcca_model_loss(data_val, val_ops, i_shape1, o_dim, lambda_, use_all_singular_values)
            data_val = data_val.cpu()
            torch.cuda.empty_cache()
            to_append[seed_num, 1, :] =  np.concatenate([[-val_corr_loss.detach().numpy()], -neg_corrs.detach().numpy()])
            
            data_te = data_te.to(device)
            test_ops = model(data_te)
            _, test_corr_loss, _,neg_corrs,_ = dmcca_model_loss(data_te, test_ops, i_shape1, o_dim, lambda_, use_all_singular_values)
            data_te = data_te.cpu()
            torch.cuda.empty_cache()
            to_append[seed_num, 2, :] =  np.concatenate([[-test_corr_loss.detach().numpy()], -neg_corrs.detach().numpy()])

            models[seed_num] = model
            first_and_last[seed_num] = [-tr_corr_loss, -val_corr_loss, -test_corr_loss]
            print('{:0.4f} {:0.4f} {:0.4f}'.format(-tr_corr_loss, -val_corr_loss, -test_corr_loss))

        nums = 1
        results = np.zeros(nums)
        idx = np.argsort(-first_and_last[:,1])
        # print(first_and_last[idx,1:])
        # print(idx)
        # print(np.array(seeds)[idx])
        seed =  seeds[idx[0]]

    print("seed:    ", seed   )

    training_lossses = []
    new_data = []

    torch.manual_seed(seed)
    if torch.cuda.is_available() : 
        torch.cuda.manual_seed_all(seed)
    model = dmcca_model_n_resp_1_stim(N-1, i_shape1, i_shape2, mid_shape, o_dim, dropout)
    model = model.to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_par)

    model, training_losses = train_the_dmcca_model(model, model_optimizer, train_set, val_set, te_set, N, epoch_num, batch_size, o_dim, i_shape1, lambda_, use_all_singular_values, path_name)
    
    model.eval()
    data = [train_set, val_set, te_set]

    with torch.no_grad():
        new_data = []
        for k in range(3):
            temp = data[k].to(device)
            pred_out = model(temp)
            del temp
            new_data.append(pred_out)

    return new_data, training_losses, model



# TRAINS THE MODEL IN dmcca_model
def train_the_dmcca_model(model, model_optimizer, data_tr, data_val, data_te, N, epoch_num, batch_size, o_dim, i_shape1, lambda_, use_all_singular_values, path_name, best_only=True):
    """
    ARGUMENTS: 
        THE DMCCA MODEL TO BE TRAINED, THE MODEL'S OPTIMIZER, THE DATA FOR TRAINING, VALIDATING AND TESTING THE MODEL; AND ALL OTHER HYPERPARAMETERS REQUIRED TO TRAIN THE MODEL.
    
    RETURNS:
        THE TRAINED MODEL AND THE LOSSES WHILE TRAINING THE MODEL.
    """
    print("Started training.")
    best_epoch = 0
    min_loss = 0.00
    loss_epochs = np.zeros((epoch_num, 3))
    corr_epochs = np.zeros((epoch_num, 3, int(comb(N,2)) + 1))
    mses_epochs = np.zeros((epoch_num, 3, N+1))

    model.to(device)
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        model.train()
        dataloader = DataLoader(data_tr, batch_size, shuffle=True)
        for trs in dataloader : 
            model_optimizer.zero_grad()
            trs = trs.to(device)
            outputs = model(trs)
            loss, _, _, _, _ = dmcca_model_loss(trs, outputs, i_shape1, o_dim, lambda_, use_all_singular_values)
            loss.backward()
            model_optimizer.step()
            del trs
        
        model.eval()

        torch.cuda.empty_cache()
        tr_loss = 0 ; tr_corrs = np.zeros(int(comb(N, 2))+1) ; tr_mses = np.zeros(N+1)
        count = 0
        dataloader = DataLoader(data_tr, batch_size, shuffle=True)
        with torch.no_grad():
            for trs in dataloader :
                trs = trs.to(device)
                outputs = model(trs)
                loss, corr, mse, neg_corrs, mses = dmcca_model_loss(trs, outputs, i_shape1, o_dim, lambda_, use_all_singular_values)
                trs = trs.cpu()
                tr_loss  = tr_loss + loss
                tr_corrs = tr_corrs + np.concatenate([[-corr], -neg_corrs.detach().numpy()])
                tr_mses  = tr_mses  + np.concatenate([[mse], mses.detach().numpy()])

                count = count + 1
                del trs

        loss_epochs[epoch, 0]    = tr_loss  / (count)
        corr_epochs[epoch, 0, :] = tr_corrs / (count)
        mses_epochs[epoch, 0, :] = tr_mses  / (count)
        torch.cuda.empty_cache()
        print('EPOCH : {}'.format(epoch))
        print('  Training corr LOSS   : {:0.4f}'.format(corr_epochs[epoch, 0, 0]))
        # print("{} - {} = {}       {}".format(corr_epochs[epoch, 0, 0], mses_epochs[epoch, 0, 0], -loss_epochs[epoch,0], corr_epochs[epoch, 0, 1:]))
        print("{} - {} = {}".format(corr_epochs[epoch, 0, 0], mses_epochs[epoch, 0, 0], -loss_epochs[epoch,0]))

        data_val = data_val.to(device)
        val_ops = model(data_val)

        val_loss, corr, mse, neg_corrs, mses = dmcca_model_loss(data_val, val_ops, i_shape1, o_dim, lambda_, use_all_singular_values)
        loss_epochs[epoch, 1]    = val_loss
        corr_epochs[epoch, 1, :] = np.concatenate([[-corr], -neg_corrs.detach().numpy()])
        mses_epochs[epoch, 1, :] = np.concatenate([[mse], mses.detach().numpy()])
        data_val = data_val.cpu()
        torch.cuda.empty_cache()
        print('  Validation corr LOSS : {:0.4f}'.format(-corr))
        # print("{} - {} = {}      {}".format(-corr, mse, -val_loss, -neg_corrs))
        print("{} - {} = {}".format(-corr, mse, -val_loss))
        
        data_te = data_te.to(device)
        print(data_te.shape)
        test_ops = model(data_te)
        test_loss, corr, mse, neg_corrs, mses = dmcca_model_loss(data_te, test_ops, i_shape1, o_dim, lambda_, use_all_singular_values)
        loss_epochs[epoch, 2]    = test_loss
        corr_epochs[epoch, 2, :] = np.concatenate([[-corr], -neg_corrs.detach().numpy()])
        mses_epochs[epoch, 2, :] = np.concatenate([[mse], mses.detach().numpy()])
        data_te = data_te.cpu()
        torch.cuda.empty_cache()
        print('  Test corr LOSS       : {:0.4f}'.format(-corr))
        # print("{} - {} = {}       {}".format(-corr, mse, -test_loss, -neg_corrs))
        print("{} - {} = {}".format(-corr, mse, -test_loss))

        print("  val. loss is : {:0.4f} & the min. loss is : {:0.4f}".format(val_loss, min_loss))
        print("  AND since, val_loss < min_loss is {}".format(val_loss < min_loss))

        model_file_name = path_name + 'best_model.pth'

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

    return model, [loss_epochs, corr_epochs, mses_epochs]



# DCCA MODELS WITH DIFFERENT ARCHITECTURES
# 13_2,  13_3,  15_4,  13_5,  100_2,  100_3,  100_4,  100_5,  10000_2,  10000_3,  10000_4,  
def generic_dcca3(stim_data, resp_data, type, o_dim, learning_rate=1e-3, use_all_singular_values=False, epoch_num=12, batch_size=2048, reg_par=1e-4, dropout=0.05, path_name="", seeds=np.ceil(np.random.rand(10)*100)):
    """
    CAN BE USED TO ACCESS DIFFERENT DCCA MODELS FROM THE deep_nets.py.
    THESE ARE THE MODELS EXPLORED AND REPORTED IN THE PAPER.
    OTHER THAN SETTING THE DCCA MODEL, EVERYTHING ELSE IS SAME AS THE "dcca_model".
    """

    stimtr  = stim_data[0]
    stimval = stim_data[1]
    stimte  = stim_data[2]
    resptr  = resp_data[0]
    respval = resp_data[1]
    respte  = resp_data[2]

    stimtr, mean1, std1 = my_standardize(stimtr)
    resptr, mean2, std2 = my_standardize(resptr)

    stimval = (stimval - mean1) / std1
    stimte  = (stimte  - mean1) / std1
    respval = (respval - mean2) / std2
    respte  = (respte  - mean2) / std2

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

    best_only = True
    act = "sigmoid"
    o_act = 'leaky_relu'

    if (isinstance(seeds, int)):
        seed = seeds
    elif not(isinstance(seeds, int)) and len(seeds) == 1:
        seed = seeds[0]
    else:
        torch.backends.cudnn.deterministic = True
        first_and_last = np.zeros((len(seeds),3))
        models = [None] * len(seeds)
        print('seeds: ', seeds)

        for seed_num, seed in enumerate(seeds) : 
            torch.manual_seed(seed)
            if torch.cuda.is_available() : torch.cuda.manual_seed_all(seed)
            model = None
            if type == "13_2": model = model2_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "13_3": model = model3_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "15_4": model = model2_15(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "13_5": model = model5_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "100_2": model = model_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "100_3": model = model_3_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "100_4": model = model_4_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "100_5": model = model_5_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            if type == "10000_2": model = model_10000s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
            model = model.to(device)
            model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_par)

            print('MODEL : {}'.format(seed_num))

            model.eval()
            torch.cuda.empty_cache()

            tr_loss = 0 ; count = 0
            dataloader = DataLoader(data_tr, batch_size, shuffle=True)
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
    model = None
    if type == "13_2": model = model2_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "13_3": model = model3_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "15_4": model = model2_15(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "13_5": model = model5_13(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "100_2": model = model_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "100_3": model = model_3_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "100_4": model = model_4_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "100_5": model = model_5_100s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    if type == "10000_2": model = model_10000s(i_shape1, i_shape2, act, o_act, o_dim, dropout)
    model = model.to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_par)

    model_state_dict = []
    min_loss = 0.00 ; min_loss2 = 0.00
    correlations = np.zeros((epoch_num, 3))
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        model.train()
        dataloader = DataLoader(data_tr, batch_size, shuffle=True)
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
        dataloader = DataLoader(data_tr, batch_size, shuffle=True)
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


