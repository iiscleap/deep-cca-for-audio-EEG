import numpy as np
import scipy.io
from scipy.signal import lfilter

from cca_fns import my_standardize, lagGen, my_PCA, apply_PCA, my_corr, linear_cca, cca3_model_new

# HELPER FUNCTION
def my_mcca(R, din):
    nblocks = int(R.shape[1] / din)
    V1 = np.zeros(R.shape)
    for iBlock in range(nblocks):
        idx = (iBlock)*din + np.arange(din)
        RR = R[idx, :]
        RR = RR[:, idx]

        lambdas, U = np.linalg.eigh(RR)
        U = np.real(U) ; lambdas = np.real(lambdas)
        sorted_lambdas = -np.sort(-lambdas)
        indices = np.argsort(-lambdas)
        Sorted_U = U[:, indices]

        lambd_inv = 1 / sorted_lambdas
        lambd_inv[ lambd_inv <= 0] = 0;
        lambd_sqrt_inv = np.sqrt(lambd_inv)

        V1[idx, (iBlock) * din : (iBlock+1) * din] = np.matmul(Sorted_U, np.diag(lambd_sqrt_inv))

    R_tild = np.matmul(np.matmul(V1.T, R), V1)

    lambdas2, V = np.linalg.eigh(R_tild)
    V = np.real(V) ; lambdas2 = np.real(lambdas2)
    indices2 = np.argsort(-lambdas2)
    Sorted_V = V[:, indices2]

    final_transform = V1 * Sorted_V
    score = np.diag(np.matmul(np.matmul(Sorted_V.T, R_tild), Sorted_V))
    
    each_transform = []
    for iBlock in range(nblocks):
        temp = np.zeros((din, (nblocks-1)*din))
        temp[:din, :din] = final_transform[iBlock*din : (iBlock+1)*din, iBlock*din : (iBlock+1)*din]
        each_transform.append(temp)

    return final_transform, score, each_transform

# FOR APPLYING FILTERBANK TO TWO SETS OF DATA
def filtem(x1, x2, x3, y1, y2, y3, num1=1, num2=60):
    # x1 : stim_train
    # x2 : stim_val
    # x3 : stim_test
    # y1 : resp_train
    # y2 : resp_val
    # y3 : resp_test
    mat2 = scipy.io.loadmat('psi.mat')
    psi = mat2['psi'][0]
    nfilts = 21
    print('Filtering loaded.')
    x1_new = []
    if x1 is not None:
        for k1 in range(num1):
            X = []
            for k2 in range(nfilts):
                X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(x1[:, k1])))
            x1_new.append(X)
        x1_new = np.vstack(x1_new).T
    x2_new = []
    if x2 is not None:
        for k1 in range(num1):
            X = []
            for k2 in range(nfilts):
                X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(x2[:, k1])))
            x2_new.append(X)
        x2_new = np.vstack(x2_new).T
    x3_new = []
    if x3 is not None:
        for k1 in range(num1):
            X = []
            for k2 in range(nfilts):
                X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(x3[:, k1])))
            x3_new.append(X)
        x3_new = np.vstack(x3_new).T
    y1_new = []
    if y1 is not None:
        for k1 in range(num2):
            Y = []
            for k2 in range(nfilts):
                Y.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(y1[:, k1])))
            y1_new.append(Y)
        y1_new = np.vstack(y1_new).T
    y2_new = []
    if y2 is not None:
        for k1 in range(num2):
            Y = []
            for k2 in range(nfilts):
                Y.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(y2[:, k1])))
            y2_new.append(Y)
        y2_new = np.vstack(y2_new).T
    y3_new = []
    if y3 is not None:
        for k1 in range(num2):
            Y = []
            for k2 in range(nfilts):
                Y.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(y3[:, k1])))
            y3_new.append(Y)
        y3_new = np.vstack(y3_new).T
    print('Filtered.')
    return x1_new, x2_new, x3_new, y1_new, y2_new, y3_new

# FOR APPLYING FILTERBANK TO ONE SET OF DATA
def filtone(x1, x2, x3):
    mat2 = scipy.io.loadmat('psi.mat')
    psi = mat2['psi'][0]
    nfilts = 21
    num = np.shape(x1)[1]
    print('Filtering loaded.')
    x1_new = []
    for k1 in range(num):
        X = []
        for k2 in range(nfilts):
            X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(x1[:, k1])))
        x1_new.append(X)
    x1_new = np.vstack(x1_new).T
    x2_new = []
    for k1 in range(num):
        X = []
        for k2 in range(nfilts):
            X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(x2[:, k1])))
        x2_new.append(X)
    x2_new = np.vstack(x2_new).T
    x3_new = []
    for k1 in range(num):
        X = []
        for k2 in range(nfilts):
            X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(x3[:, k1])))
        x3_new.append(X)
    x3_new = np.vstack(x3_new).T
    
    print('Filtered.')
    return x1_new, x2_new, x3_new

# MCCA for LMLC
def linear_mcca_custom(datas, new_chans):
    n_subs = len(datas)-1

    nchans = 128
    nsets  = len(datas)
    resp = datas[:-1]
    stim = datas[-1]
    del datas

    nchans2 = new_chans; 
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

    stimtr  = stim[0]
    stimval = stim[1]
    stimte  = stim[2]
    stimtr_x = lagGen(stimtr, np.arange(nchans2))

    data_tr = np.concatenate([resptr_x, stimtr_x], 1)
    C = np.matmul(data_tr.T, data_tr)

    [V, rho, AA] = my_mcca(C, nchans2)
    
    dout = 110
    new_subs_data = []
    pre_subs_data = []
    corr = np.zeros(n_subs)
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

        [stim5_tr, stim5_val, stim5_te, resp5_tr, resp5_val, resp5_te] = filtem(stimtr, stimval, stimte, resptr, respval, respte, 1, 60)

        corr[isub], temp = cca3_model_new([stim5_tr, stim5_val, stim5_te], [resp5_tr, resp5_val, resp5_te], 1)
        new_subs_data.append(temp)
        pre_subs_data.append(pre_sub_data)
        print(corr[isub])

    print('Linear MCCA corrs are : {}'.format(corr))
    return new_subs_data, corr, pre_subs_data


# AFTER DMCCA, for DMLC or DMDC
def pca_stim_filtem_pca_resp(data_sub, stim_data):
    # print(data_sub[0].shape)
    # print(data_sub[1].shape)
    # print(data_sub[2].shape)
    # print(stim_data[0].shape)
    # print(stim_data[1].shape)
    # print(stim_data[2].shape)
    
    [meanp_s, W_s, x1_1] = my_PCA(stim_data[0], 1)
    x2_1 = apply_PCA(stim_data[1], meanp_s, W_s)
    x3_1 = apply_PCA(stim_data[2], meanp_s, W_s)
    new_stim_data = [x1_1, x2_1, x3_1]
    del x1_1, x2_1, x3_1

    stim_and_new_sub = filtem(new_stim_data[0], new_stim_data[1], new_stim_data[2], data_sub[0], data_sub[1], data_sub[2], 1, data_sub[0].shape[1])

    new_data_sub = stim_and_new_sub[3:]
    [meanp_r, W_r, x1_139] = my_PCA(new_data_sub[0], 139)
    x2_139 = apply_PCA(new_data_sub[1], meanp_r, W_r)
    x3_139 = apply_PCA(new_data_sub[2], meanp_r, W_r)
    new_data_sub = [x1_139, x2_139, x3_139]
    del x1_139, x2_139, x3_139

    return stim_and_new_sub[:3], new_data_sub, [meanp_r, W_r], [meanp_s, W_s]


# TO CONVERT 128D EEG TO 60D TO 1260D TO 139D
def pca_filt_pca_resp(data_sub):
    [meanp, W, x1_60] = my_PCA(data_sub[0], 60)
    x2_60 = apply_PCA(data_sub[1], meanp, W)
    x3_60 = apply_PCA(data_sub[2], meanp, W)
    new_data_sub = [x1_60, x2_60, x3_60]
    del x1_60, x2_60, x3_60

    new_data_sub = filtone(new_data_sub[0], new_data_sub[1], new_data_sub[2])

    [meanp, W, x1_139] = my_PCA(new_data_sub[0], 139)
    x2_139 = apply_PCA(new_data_sub[1], meanp, W)
    x3_139 = apply_PCA(new_data_sub[2], meanp, W)
    new_data_sub = [x1_139, x2_139, x3_139]
    del x1_139, x2_139, x3_139

    return new_data_sub


# MCCA WITH ONLY N RESPS
def linear_mcca_custom_resps(datas, new_chans):
    n_subs = len(datas)-1

    nchans = 128
    nsets  = len(datas)
    resp = datas[:-1]
    stim = datas[-1]
    del datas

    nchans2 = new_chans; 
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

    stimtr  = stim[0]
    stimval = stim[1]
    stimte  = stim[2]
    # stimtr_x = lagGen(stimtr, np.arange(nchans2))

    # data_tr = np.concatenate([resptr_x, stimtr_x], 1)
    data_tr = resptr_x
    C = np.matmul(data_tr.T, data_tr)

    [V, rho, AA] = my_mcca(C, nchans2)
    
    new_subs_data = []
    corr = np.zeros(n_subs)
    for isub in range(n_subs):
        print("Subject number: "+str(isub))
        dout = 110
        Vn = np.matmul(W[isub], AA[isub])
        Vo = Vn[:, :dout]
        Vninv = np.linalg.pinv(Vn)
        Vninv = Vninv[:dout, :]
        Vn = np.matmul(Vo, Vninv)

        sub_tr  = np.matmul(resp[isub][0] - meanp[isub], Vn)
        sub_val = np.matmul(resp[isub][1] - meanp[isub], Vn)
        sub_te  = np.matmul(resp[isub][2] - meanp[isub], Vn)

        [meanp1, W1, resptr] = my_PCA(sub_tr,60);
        respval = apply_PCA(sub_val,meanp1,W1);
        respte  = apply_PCA(sub_te, meanp1,W1);

        # print(resptr, respval, respte)
        # print(stimtr, stimval, stimte)

        [stim5_tr, stim5_val, stim5_te, resp5_tr, resp5_val, resp5_te] = filtem(stimtr, stimval, stimte, resptr, respval, respte, 1, 60) ;

        corr[isub], temp = cca3_model_new([stim5_tr, stim5_val, stim5_te], [resp5_tr, resp5_val, resp5_te], 1) ;
        new_subs_data.append(temp)
        print(corr[isub])

    print('Linear MCCA corrs are : {}'.format(corr))
    return new_subs_data, corr

# MCCA WITH RESPS AND 1 COMMON STIMULUS
def linear_mcca_custom_stim(datas, new_chans):
    n_subs = len(datas)-1

    nchans = 128
    nsets  = len(datas)
    resp = datas[:-1]
    stim = datas[-1]
    del datas

    nchans2 = new_chans; 
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

    stimtr  = stim[0]
    stimval = stim[1]
    stimte  = stim[2]
    stimtr_x = lagGen(stimtr, np.arange(nchans2))

    data_tr = np.concatenate([resptr_x, stimtr_x], 1)
    C = np.matmul(data_tr.T, data_tr)

    [V, rho, AA] = my_mcca(C, nchans2)

    dout = 110
    Vn = AA[-1]
    Vo = Vn[:, :dout]
    Vninv = np.linalg.pinv(Vn)
    Vninv = Vninv[:dout, :]
    Vn = np.matmul(Vo, Vninv)

    stim_tr = np.matmul(stimtr_x, Vn)
    stim_val = np.matmul(lagGen(stimval, np.arange(nchans2)), Vn)
    stim_te = np.matmul(lagGen(stimte, np.arange(nchans2)), Vn)

    [meanp1, W1, stimtr] = my_PCA(stim_tr,1)
    stimval = apply_PCA(stim_val,meanp1,W1)
    stimte  = apply_PCA(stim_te, meanp1,W1)

    new_subs_data = []
    corr = np.zeros(n_subs)
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

        [meanp1, W1, resptr] = my_PCA(sub_tr,60);
        respval = apply_PCA(sub_val,meanp1,W1);
        respte  = apply_PCA(sub_te, meanp1,W1);

        [stim5_tr, stim5_val, stim5_te, resp5_tr, resp5_val, resp5_te] = filtem(stimtr, stimval, stimte, resptr, respval, respte, 1, 60) ;

        corr[isub], temp = cca3_model_new([stim5_tr, stim5_val, stim5_te], [resp5_tr, resp5_val, resp5_te], 1) ;
        new_subs_data.append(temp)
        print(corr[isub])

    print('Linear MCCA corrs are : {}'.format(corr))
    return new_subs_data, corr


