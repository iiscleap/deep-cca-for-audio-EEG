import numpy as np

def my_standardize(X):
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

def standardize_1(x):
    mean_x = np.mean(x)
    x = x - mean_x
    var = np.dot(x.T, x)
    x = x / np.sqrt(var)
    return x

def my_corr(X1, X2, K=None, rcov1=0, rcov2=0):
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

def cca3_model_new(stim_data, resp_data, F):
    print('CCA3 Model Started...')

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

    stim5_tr, mean1, std1   = my_standardize(stim5_tr)
    resptr_139, mean2, std2 = my_standardize(resptr_139)
    stim5_te   = (stim5_te   - mean1) / std1
    respte_139 = (respte_139 - mean2) / std2

    A5, B5, m1, m2, _ = linear_cca(stim5_tr, resptr_139, F)

    x = np.dot((stim5_te   - m1), A5)
    y = np.dot((respte_139 - m2), B5)
    corr5 = np.squeeze(my_corr(x, y, F))

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

    print(corr5)

    print('CCA3 Model Ended.')

    return corr5, new_data




