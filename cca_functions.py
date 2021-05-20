import numpy as np
import scipy.io
from scipy.signal import lfilter

def my_standardize(X):
    """
    STANDARDIZING D DIMENSIONAL DATA
    """
    X_mean = np.mean(X, axis = 0)
    X = X - X_mean
    X_std = np.std(X, axis = 0)
    X = X / X_std
    return X, X_mean, X_std

def my_PCA(X, dout):
    ''' 
    FIRST OF ALL: dimensions of X should be (N, din)
    Does PCA on X to dout dimensions and outputs mean and the transform as well
    '''
    X_mean = np.mean(X, axis=0)
    X_norm = X - X_mean
    X_cov = np.dot(X_norm.T, X_norm)
    din = X.shape[1]
    [D, W] = np.linalg.eigh(X_cov)
    # The eigenvalues in ascending order, each repeated according to its multiplicity.
    D = D[-dout:]
    W = W[:, -dout:]
    outs = np.dot(X_norm, W)
    return X_mean, W, outs

def apply_PCA(X, meanp, W):
    ''' Applies PCA to X '''
    X_norm = X - meanp
    outs = np.dot(X_norm, W)
    return outs

def lagGen(x, lags, one=0):
    x = np.array(x)
    if one == 1:
        print('if entered.')
        x = x.reshape(-1, 1)
    lags = np.array(lags)
    xLag = np.zeros((x.shape[0], x.shape[1] * lags.shape[0]))
    i = 0
    for j in range(len(lags)):
        if lags[j] < 0:
            xLag[: xLag.shape[0] + lags[j],
                 i: i + x.shape[1]] = x[-lags[j]:, :]
        elif lags[j] > 0:
            xLag[lags[j]:, i: i + x.shape[1]
                 ] = x[0: xLag.shape[0] - lags[j], :]
        else:
            xLag[:, i: i + x.shape[1]] = x
        i = i + x.shape[1]
    return xLag

def filtem(x1, x2, x3, y1, y2, y3, num1=1, num2=60):
    """
        USED TO FILTER THE STIMULUS AND RESPONSE DATA USING THE FILTERBANK MENTIONED IN THE LINEAR CCA MODEL.
    ARGUMENTS: 
        x1, x2, x3 : TRAINING, VALIDATION AND TEST DATA OF THE STIMULUS
        y1, y2, y3 : TRAINING, VALIDATION AND TEST DATA OF THE RESPONSE
        STIMULUS ARE ASSUMED TO BE OF 1D. AND RESPONSES OF 60D. ( THAT IS WHY num1 and num2 ARE SET TO THOSE VALUES BY DEFAULT.)
        SUPPORTS DIFFERENT SHAPES TOO.
    RETURNS: 
            FILTERED DATA.
    """
    # x1 : stim_train
    # x2 : stim_val
    # x3 : stim_test
    # y1 : resp_train
    # y2 : resp_val
    # y3 : resp_test
    mat2 = scipy.io.loadmat('/data2/jaswanthr/data/new_psi.mat')
    psi = mat2['psi'][0]
    nfilts = 21
    print('Filtering loaded.')
    data_ = [x1, x2, x3, y1, y2, y3]

    new_data = []
    for data_num, xx in enumerate(data_):
        xx_new = []
        if xx is not None:
            for k1 in range(xx.shape[1]):
                X = []
                for k2 in range(nfilts):
                    X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(xx[:, k1])))
                xx_new.append(X)
            xx_new = np.vstack(xx_new).T
        new_data.append(xx_new)
    print('Filtered.')

    [x1_new, x2_new, x3_new, y1_new, y2_new, y3_new] = new_data
    return x1_new, x2_new, x3_new, y1_new, y2_new, y3_new

def filtone(x1, x2, x3):
    """
    USED TO FILTER ONLY ONE SET OF DATA.
    """
    mat2 = scipy.io.loadmat('/data2/jaswanthr/data/new_psi.mat')
    psi = mat2['psi'][0]
    nfilts = 21
    num = x1.shape[1]
    print('Filtering loaded.')
    data_ = [x1, x2, x3]
    new_data = []
    for data_num, xx in enumerate(data_):
        xx_new = []
        for k1 in range(num):
            X = []
            for k2 in range(nfilts):
                X.append(lfilter(np.squeeze(psi[k2]), 1, np.squeeze(xx[:, k1])))
            xx_new.append(X)
        xx_new = np.vstack(xx_new).T
        new_data.append(xx_new)
    print('Filtered.')
    [x1_new, x2_new, x3_new] = new_data
    return x1_new, x2_new, x3_new

def standardize_1(x):
    """
    FOR STANDARDIZING A 1D VECTOR.
    """
    mean_x = np.mean(x)
    x = x - mean_x
    var = np.dot(x.T, x)
    x = x / np.sqrt(var)
    return x

def my_corr(X1, X2, K=None, rcov1=0, rcov2=0):
    """
    FINDS THE CORRELATION OF TWO D DIMENSIONAL X1 AND X2 VECTORS USING THE CCA LOSS.
    ARGUMENTS:
        X1 : T x D1
        X2 : T x D2
        K  : OUTPUT DIMENSION TO BE CONSIDERED
    
    RETURNS:
        corr: CORRELATION OF THE TWO VECTORS.
    """
    if K == 1:
        x1 = standardize_1(X1)
        x2 = standardize_1(X2)
        corr = np.dot(x1.T, x2)
    else:
        N, d1 = X1.shape
        d2 = X2.shape[1]
        if K is None:
            K = min(d1, d2)
        m1 = np.mean(X1, 0)
        X1 = X1 - m1
        m2 = np.mean(X2, 0)
        X2 = X2 - m2
        S11 = np.dot(X1.T, X1)/(N-1) + rcov1*np.eye(d1)
        S22 = np.dot(X2.T, X2)/(N-1) + rcov2*np.eye(d2)
        S12 = np.dot(X1.T, X2)/(N-1)
        D1, V1 = np.linalg.eigh(S11)
        D2, V2 = np.linalg.eigh(S22)
        # D1 = (D1 + abs(D1)) / 2 + 1e-10
        # D2 = (D2 + abs(D2)) / 2 + 1e-10
        idx1 = np.nonzero(D1 > 1e-10)
        D1 = D1[idx1]
        V1 = np.squeeze(V1[:, idx1])
        idx2 = np.nonzero(D2 > 1e-10)
        D2 = D2[idx2]
        V2 = np.squeeze(V2[:, idx2])
        K11 = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        K22 = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)
        T = np.dot(np.dot(K11, S12), K22)
        [U, D, V] = np.linalg.svd(T)
        # U = U[:, 0:K]
        # V = V[:, 0:K]
        D = D[0:K]
        corr = np.sum(D)
    return corr

def linear_cca(H1, H2, outdim_size):
    """
    An implementation of linear CCA
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices 
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o1 = H1.shape[1]

    o2 = H2.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)

    V1 = V1[:, np.nonzero(D1 > 1e-9)]
    D1 = D1[np.nonzero(D1 > 1e-9)]
    V1 = np.squeeze(V1)
    D1 = np.squeeze(D1)

    V2 = V2[:, np.nonzero(D2 > 1e-9)]
    D2 = D2[np.nonzero(D2 > 1e-9)]
    V2 = np.squeeze(V2)
    D2 = np.squeeze(D2)

    SigmaHat11RootInv = np.matmul(np.matmul(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.matmul(np.matmul(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.matmul(np.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)

    V = V.T
    A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B, mean1, mean2, D

def cca_model(stim_data, resp_data, F):
    """
    ARGUMENTS:
        STIM_DATA: DATA1 (ASSUMED TO BE STIMULUS DATA.) [STIM_TRAINING, STIM_VALIDATION, STIM_TEST]
        RESP_DATA: DATA1 (ASSUMED TO BE RESPONSE DATA.) [RESP_TRAINING, RESP_VALIDATION, RESP_TEST]
             THE RESP DATA IS ASSUMED TO BE READY AS PER THE LCCA3 MODEL (DESCRIBED IN THE "CCA ANALYSIS FOR DECODING THE AUDITORY BRAIN" BY CHEVIEGNE ET AL.)
             THAT IS : TO BE IN T x 139 FORMAT. 
             IF NOT: WE ARE GOING TO PCA IT TO 139D.
        F  : OUTPUT DIMENSION.
    
    RETURNS:
        corr:     THE CORRELATION BETWEEN THE TWO NEWLY PROJECTED TEST DATA.
        new_data: A THREE ELEMENT LIST WITH NEWLY PROJECTED STIM AND RESP DATA. 
                ARRANGED AS : [[STIM_TRAINING, RESP_TRAINING], [STIM_VAL, RESP_VAL], [STIM_TE, RESP_TE]]
    """
    print('CCA Model Started...')

    stim5_tr  = stim_data[0]
    stim5_val = stim_data[1]
    stim5_te  = stim_data[2]
    resp5_tr  = resp_data[0]
    resp5_val = resp_data[1]
    resp5_te  = resp_data[2]

    if resp5_te.shape[1] != 139:
        [meanp, W, resptr_139] = my_PCA(resp5_tr, 139)
        respval_139 = apply_PCA(resp5_val, meanp, W)
        respte_139  = apply_PCA(resp5_te,  meanp, W)
    else:
        resptr_139  = resp_data[0]
        respval_139 = resp_data[1]
        respte_139  = resp_data[2]

    # STANDARDIZING THE DATA BEFORE PROVIDING TO THE CCA MODEL.
    stim5_tr,   mean1, std1 = my_standardize(stim5_tr)
    resptr_139, mean2, std2 = my_standardize(resptr_139)
    stim5_te   = (stim5_te   - mean1) / std1
    respte_139 = (respte_139 - mean2) / std2

    A5, B5, m1, m2, _ = linear_cca(stim5_tr, resptr_139, F)

    x = np.dot((stim5_te   - m1), A5)
    y = np.dot((respte_139 - m2), B5)
    corr = np.squeeze(my_corr(x, y, F))

    x = stim5_tr.shape[0]

    new_data1_tr = np.dot((stim5_tr   - m1), A5)
    new_data2_tr = np.dot((resptr_139 - m2), B5)

    new_data1_val = np.dot((stim5_val   - m1), A5)
    new_data2_val = np.dot((respval_139 - m2), B5)

    new_data1_te = np.dot((stim5_te   - m1), A5)
    new_data2_te = np.dot((respte_139 - m2), B5)

    new_data_tr  = [new_data1_tr,  new_data2_tr]
    new_data_val = [new_data1_val, new_data2_val]
    new_data_te  = [new_data1_te,  new_data2_te]

    new_data = [new_data_tr, new_data_val, new_data_te]

    print(corr)

    print('CCA Model Ended.')

    return corr, new_data




# MCCA STUFF

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

def linear_mcca(all_data, new_chans, o_dim):
    """
     PERFORMING MCCA FOR A SET OF N EEG RESPONSES AND 1 COMMON STIMULUS.
     HERE, STIMULUS IS CONSIDERED FOR THE LMCCA STEP AND NOT USED AFTER THAT.
    ARGUMENTS:
        all_data: AN (N+1) ELEMENTS LIST WITH N EEG RESPONSES AND THE ELEMENT AS THE COMMON STIMULI DATA.
               ARRANGED AS [DATA_i_TRAINING, DATA_i_VALIDATION, DATA_i_TEST]
    RETURNS:
        lmcca_data : FINAL DATA AFTER PERFORMING THE LMCCA METHOD.
        lmlc_data  : FINAL DATA AFTER PERFORMING THE LMCCA + LCCA (LMLC) METHOD.
        pre_lmdc_data : PROCESSED DATA WHICH CAN BE PROVIDED DIRECTLY TO DCCA METHOD. (FOR PERFORMING LMDC ANALYSIS.)
        corr          : CORRELATIONS OF THE TEST DATA FROM THE N SUBJECTS AFTER PERFORMING THE LMLC METHOD.
        pre_subs_data : DATA AFTER THE LMCCA MODEL. BEFORE BEING PROCESSED (FILTERED AND PCA) FOR THE LCCA MODEL.
    """
    n_subs = len(all_data)-1

    nchans = 128
    nsets  = len(all_data)
    resp = all_data[:-1]
    stim = all_data[-1]
    del all_data

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
    
    dout = 110          # HYPERPARAMETER; CAN BE CHANGED ACCRODINGLY.
    lmcca_data = []
    lmlc_data  = []
    pre_lmdc_data = []
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
        lmcca_data_sub = [sub_tr, sub_val, sub_te]

        [meanp1, W1, resptr] = my_PCA(sub_tr,60)
        respval = apply_PCA(sub_val,meanp1,W1)
        respte  = apply_PCA(sub_te, meanp1,W1)

        # print(resptr.shape, respval.shape, respte.shape)
        # print(stimtr.shape, stimval.shape, stimte.shape)
        
        # DIMENSIONS OF THE STIM AND RESPONSES AS 1 AND 60, RESPECTIVELY, ARE DECIDED BASED ON THE CCA3 MODEL PROPOSED BY CHEVIEGNE ET AL. IN "CCA ANALYSIS FOR DECODING THE AUDITORY BRAIN."

        [stim_tr, stim_val, stim_te, resp_tr, resp_val, resp_te] = filtem(stimtr, stimval, stimte, resptr, respval, respte, 1, 60)

        [meanp_139, W_139, resptr_139] = my_PCA(resp_tr, 139)
        respval_139 = apply_PCA(resp_val, meanp_139, W_139)
        respte_139  = apply_PCA(resp_te, meanp_139, W_139)
        del meanp_139, W_139

        corr[isub], temp = cca_model([stim_tr, stim_val, stim_te], [resptr_139, respval_139, respte_139], o_dim)
        lmlc_data.append(temp)
        lmcca_data.append(lmcca_data_sub)
        pre_lmdc_data.append([[stim_tr, stim_val, stim_te], [resptr_139, respval_139, respte_139]])
        print(corr[isub])

    print('Linear MCCA corrs are : {}'.format(corr))
    return lmcca_data, lmlc_data, corr, pre_lmdc_data

# AFTER DMCCA, for DMLC or DMDC
def pca_stim_filtem_pca_resp(data_sub, stim_data):
    """
    PROCESSING THE OUTPUTS OF THE DMCCA MODEL.
    GOAL: TO OBTAIN 21D REPRESENTATION FOR THE STIMULUS DATA AND 139D FOR THE RESPONSE DATA.
    ARGUMENTS:
        data_sub : A SUBJECT'S EEG REPRESENTATION AFTER DMCCA.
                   A 3 ELEMENTS LIST ARRANGED AS [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
        stim_data: THE STIMULUS' REPRESENTATION AFTER DMCCA.
                   A 3 ELEMENTS LIST ARRANGED AS [TRAINING_DATA, VALIDATION_DATA, TEST_DATA]
    
    RETURNS:
        stim_and_new_sub[:3] : STIMULI DATA AFTER FILTERED.  3 ELEMENTS LIST. [TR, VAL, TE]. SHAPE: T x 21
        new_data_sub         : RESPONSE DATA AFTER FILTERED. 3 ELEMENTS LIST. [TR, VAL, TE]. SHAPE: T x 139
        meanp_r, W_r         : PCA mean and transform FOR THE RESPONSE DATA.
        meanp_S, W_S         : PCA mean and transform FOR THE STIMULUS DATA.
    """
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

def pca_filt_pca_resp(data_sub):
    """
    TO CONVERT ANY D DIMENSIONAL EEG DATA TO 139D (AS PROCESSED BY THE LINEAR CCA METHOD.)
    ARGUMENT: 
        data_sub: A 3 ELEMENT LIST OF EEG DATA. [TR_DATA, VAL_DATA, TE_DATA]. EACH OF SHAPE: T x D
    RETURNS:
        new_data_sub : A 3 ELEMENT LIST OF TRANSFORMED EEG DATA. [TR_DATA, VAL_DATA, TE_DATA]. EACH OF SHAPE: T x 139
    """
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

def pca_filt_resp(data_sub):
    """
    TO CONVERT ANY D DIMENSIONAL EEG DATA TO 139D (AS PROCESSED BY THE LINEAR CCA METHOD.)
    ARGUMENT: 
        data_sub: A 3 ELEMENT LIST OF EEG DATA. [TR_DATA, VAL_DATA, TE_DATA]. EACH OF SHAPE: T x D
    RETURNS:
        new_data_sub : A 3 ELEMENT LIST OF TRANSFORMED EEG DATA. [TR_DATA, VAL_DATA, TE_DATA]. EACH OF SHAPE: T x 139
    """
    [meanp, W, x1_60] = my_PCA(data_sub[0], 60)
    x2_60 = apply_PCA(data_sub[1], meanp, W)
    x3_60 = apply_PCA(data_sub[2], meanp, W)
    new_data_sub = [x1_60, x2_60, x3_60]
    del x1_60, x2_60, x3_60
    new_data_sub = filtone(new_data_sub[0], new_data_sub[1], new_data_sub[2])
    return new_data_sub


def linear_mcca_resps_only(all_data, new_chans, o_dim):
    """
        LINEAR MCCA WITH ONLY THE N SUBJECTS' EEG RESPONSES.
        THE COMMON STIMULUS IS NOT CONSIDERED FOR THE MCCA STEP.
    ARGUMENTS:
        all_data: AN (N+1) ELEMENTS LIST WITH N EEG RESPONSES AND THE ELEMENT AS THE COMMON STIMULI DATA.
               ARRANGED AS [DATA_i_TRAINING, DATA_i_VALIDATION, DATA_i_TEST]
    RETURNS:
        lmcca_data : FINAL DATA AFTER PERFORMING THE LMCCA METHOD.
        lmlc_data  : FINAL DATA AFTER PERFORMING THE LMCCA + LCCA (LMLC) METHOD.
        pre_lmdc_data : PROCESSED DATA WHICH CAN BE PROVIDED DIRECTLY TO DCCA METHOD. (FOR PERFORMING LMDC ANALYSIS.)
        corr          : CORRELATIONS OF THE TEST DATA FROM THE N SUBJECTS AFTER PERFORMING THE LMLC METHOD.
        pre_subs_data : DATA AFTER THE LMCCA MODEL. BEFORE BEING PROCESSED (FILTERED AND PCA) FOR THE LCCA MODEL.
    """
    n_subs = len(all_data)-1

    nchans = 128
    nsets  = len(all_data)
    resp = all_data[:-1]
    stim = all_data[-1]
    del all_data

    nchans2 = new_chans; 
    new_data = [[],[],[]] ; W = [None]*n_subs ; meanp = [None]*n_subs 

    for num in range(n_subs):
        [meanp[num], W[num], resptr_PCA] = my_PCA(resp[num][0],nchans2)
        respval_PCA = apply_PCA(resp[num][1],meanp[num],W[num])
        respte_PCA  = apply_PCA(resp[num][2],meanp[num],W[num])
        new_data[0].append(resptr_PCA)
        new_data[1].append(respval_PCA)
        new_data[2].append(respte_PCA)

    resptr_x  = np.concatenate(new_data[0], 1)
    respval_x = np.concatenate(new_data[1], 1)
    respte_x  = np.concatenate(new_data[2], 1)

    stimtr  = stim[0]
    stimval = stim[1]
    stimte  = stim[2]

    data_tr = resptr_x
    C = np.matmul(data_tr.T, data_tr)

    [V, rho, AA] = my_mcca(C, nchans2)
    
    lmcca_data = []
    lmlc_data  = []
    pre_lmdc_data = []
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

        [meanp1, W1, resptr] = my_PCA(sub_tr,60)
        respval = apply_PCA(sub_val,meanp1,W1)
        respte  = apply_PCA(sub_te, meanp1,W1)

        # print(resptr.shape, respval.shape, respte.shape)
        # print(stimtr.shape, stimval.shape, stimte.shape)

        # DIMENSIONS OF THE STIM AND RESPONSES AS 1 AND 60, RESPECTIVELY, ARE DECIDED BASED ON THE CCA3 MODEL PROPOSED BY CHEVIEGNE ET AL. IN "CCA ANALYSIS FOR DECODING THE AUDITORY BRAIN."

        [stim_tr, stim_val, stim_te, resp_tr, resp_val, resp_te] = filtem(stimtr, stimval, stimte, resptr, respval, respte, 1, 60)

        [meanp_139, W_139, resptr_139] = my_PCA(resp_tr, 139)
        respval_139 = apply_PCA(resp_val, meanp_139, W_139)
        respte_139  = apply_PCA(resp_te, meanp_139, W_139)
        del meanp_139, W_139

        corr[isub], temp = cca_model([stim_tr, stim_val, stim_te], [resptr_139, respval_139, respte_139], o_dim)
        lmcca_data.append([sub_tr, sub_val, sub_te])
        lmlc_data.append(temp)
        pre_lmdc_data.append([[stim_tr, stim_val, stim_te], [resptr_139, respval_139, respte_139]])
        print(corr[isub])

    print('Linear MCCA corrs are : {}'.format(corr))
    return lmcca_data, lmlc_data, corr, pre_lmdc_data

def linear_mcca_with_stim(all_data, new_chans, o_dim):
    """
    LINEAR MCCA WITH ONLY THE N SUBJECTS' EEG RESPONSES.
    THE COMMON STIMULUS IS CONSIDERED FOR THE MCCA STEP AND THE OBTAINED TRANSFORMED IS USED FOR THE LATER STEPS TOO.

    ARGUMENTS:
        all_data: AN (N+1) ELEMENTS LIST WITH N EEG RESPONSES AND THE ELEMENT AS THE COMMON STIMULI DATA.
               ARRANGED AS [DATA_i_TRAINING, DATA_i_VALIDATION, DATA_i_TEST]
    RETURNS:
        lmcca_data : FINAL DATA AFTER PERFORMING THE LMCCA METHOD.
        lmlc_data  : FINAL DATA AFTER PERFORMING THE LMCCA + LCCA (LMLC) METHOD.
        pre_lmdc_data : PROCESSED DATA WHICH CAN BE PROVIDED DIRECTLY TO DCCA METHOD. (FOR PERFORMING LMDC ANALYSIS.)
        corr          : CORRELATIONS OF THE TEST DATA FROM THE N SUBJECTS AFTER PERFORMING THE LMLC METHOD.
        pre_subs_data : DATA AFTER THE LMCCA MODEL. BEFORE BEING PROCESSED (FILTERED AND PCA) FOR THE LCCA MODEL.
    """
    n_subs = len(all_data)-1

    nchans = 128
    nsets  = len(all_data)
    resp = all_data[:-1]
    stim = all_data[-1]
    del all_data

    nchans2 = new_chans; 
    new_data = [[],[],[]] ; W = [None]*n_subs ; meanp = [None]*n_subs 

    for num in range(n_subs):
        [meanp[num], W[num], resptr_PCA] = my_PCA(resp[num][0],nchans2)
        respval_PCA = apply_PCA(resp[num][1],meanp[num],W[num])
        respte_PCA  = apply_PCA(resp[num][2],meanp[num],W[num])
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
    del data_tr, resptr_x

    # PROVIDING THE MATRIX TO THE MCCA BLOCK
    [V, rho, AA] = my_mcca(C, nchans2)

    dout = 110
    Vn = AA[-1]
    Vo = Vn[:, :dout]
    Vninv = np.linalg.pinv(Vn)
    Vninv = Vninv[:dout, :]
    Vn = np.matmul(Vo, Vninv)

    stimtr  = np.matmul(stimtr_x, Vn)
    stimval = np.matmul(lagGen(stimval, np.arange(nchans2)), Vn)
    stimte  = np.matmul(lagGen(stimte,  np.arange(nchans2)), Vn)

    [meanp1, W1, stimtr] = my_PCA(stimtr,1)
    stimval = apply_PCA(stimval,meanp1,W1)
    stimte  = apply_PCA(stimte, meanp1,W1)

    lmcca_data = []
    lmlc_data  = []
    pre_lmdc_data = []
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
        lmcca_data.append([[stimtr, stimval, stimte], [sub_tr, sub_val, sub_te]])

        [meanp1, W1, resptr] = my_PCA(sub_tr,60)
        respval = apply_PCA(sub_val,meanp1,W1)
        respte  = apply_PCA(sub_te, meanp1,W1)

        # DIMENSIONS OF THE STIM AND RESPONSES AS 1 AND 60, RESPECTIVELY, ARE DECIDED BASED ON THE CCA3 MODEL PROPOSED BY CHEVIEGNE ET AL. IN "CCA ANALYSIS FOR DECODING THE AUDITORY BRAIN."

        [stim_tr, stim_val, stim_te, resp_tr, resp_val, resp_te] = filtem(stimtr, stimval, stimte, resptr, respval, respte, 1, 60)

        [meanp_139, W_139, resptr_139] = my_PCA(resp_tr, 139)
        respval_139 = apply_PCA(resp_val, meanp_139, W_139)
        respte_139  = apply_PCA(resp_te, meanp_139, W_139)
        del meanp_139, W_139

        corr[isub], temp = cca_model([stim_tr, stim_val, stim_te], [resptr_139, respval_139, respte_139], o_dim)
        lmlc_data.append(temp)
        pre_lmdc_data.append([[stim_tr, stim_val, stim_te], [resptr_139, respval_139, respte_139]])
        print(corr[isub])

    print('Linear MCCA + LCCA corrs are : {}'.format(corr))
    return lmcca_data, lmlc_data, corr, pre_lmdc_data


