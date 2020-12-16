import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device('cuda')
# torch.cuda.empty_cache()

def obj_standardize(x):
    mean_x = torch.mean(x)
    x = x - mean_x
    var = torch.matmul(x.T, x)
    x = x / torch.sqrt(var)
    return x

def get_corr(H1, H2, dims, outdim_size, use_all_singular_values, eps):
    m = dims[0] ; n = dims[1]

    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9
    # print(m, n)
    if n != 1 :
        H1bar = H1 - H1.mean(dim=0).unsqueeze(dim=0)
        H2bar = H2 - H2.mean(dim=0).unsqueeze(dim=0)

        SigmaHat11 = (1 / (m - 1)) * torch.matmul(H1bar.T, H1bar) + r1*torch.eye(n, device=device)
        SigmaHat22 = (1 / (m - 1)) * torch.matmul(H2bar.T, H2bar) + r2*torch.eye(n, device=device)
        SigmaHat12 = (1 / (m - 1)) * torch.matmul(H1bar.T, H2bar)

        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.nonzero(torch.gt(D1, eps), as_tuple=False)[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.nonzero(torch.gt(D2, eps), as_tuple=False)[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.T)

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values == False:
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).float()*eps).to(device))
            U = U.topk(outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))

        else :
            # all singular values are used to calculate the correlation
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            # print(tmp)
            corr = torch.sqrt(tmp)

    if n == 1:
        x1 = obj_standardize(H1)
        x2 = obj_standardize(H2)
        corr = torch.matmul(x1.T, x2)
        corr = torch.squeeze(corr)

    return corr

def cca_loss(y_pred, outdim_size, use_all_singular_values):
    eps = 1e-12
    H = [None, None]
    H[0] = y_pred[0]
    H[1] = y_pred[1]
    dims = H[0].shape
    neg_corr = - get_corr(H[0], H[1], dims, outdim_size, use_all_singular_values, eps)

    # IF WE WANT TO REGULARIZE THE MODEL WITH THE NORMS OF THE INDIVIDUAL OUTPUTS, THEIR L2 NORMS CAN BE ADDED TO THE COST FUNCTION.
    # l2_reg = torch.norm(H[0]) + torch.norm(H[1])
    # neg_corr = l2_reg - neg_corr

    return neg_corr

def dmcca_model_loss(y_act, y_pred, i_shape1, outdim_size, lambda_, use_all_singular_values):
    eps = 1e-5
    N = len(y_pred[0])
    mse_loss = nn.MSELoss()

    G = y_pred[1]
    H = y_pred[0]

    dims = H[0].shape
    
    neg_corrs = torch.zeros(int(comb(N, 2)))
    mses      = torch.zeros(N)
    k = 0
    for i in range(N):
        mses[i] = mse_loss(G[i], y_act[:, i*i_shape1: (i+1)*i_shape1])
        for j in range(i+1, N):
            neg_corrs[k] = -get_corr(H[i], H[j], dims, outdim_size, use_all_singular_values, eps)
            k = k + 1

    neg_corr = torch.sum(neg_corrs)
    mse      = torch.sum(mses)

    total_loss = neg_corr + (lambda_ * mse)
    return total_loss, neg_corr, mse, neg_corrs, mses

# ONE MORE VARIANT OF THE DMCCA LOSS. A MORE REGULARIZED VERSION.
def dmcca_model_loss_regularized(y_act, y_pred, i_shape1, outdim_size, lambda_, use_all_singular_values, model=None, lambda_s=[0,0]):
    """
    TWO MORE VARIANTS OF THE DMCCA LOSS CAN BE TRIED. WE CAN TWO MORE TERMS TO THE REGULARIZATION:
    1. PENALIZING THE L2-NORMS OF THE FINAL REPRES
    2. PENALIZING THE FINAL LAYERS OF THE ENCODERS
    BY DEFAULT, THEIR REGULARIZATION PARAMETERS ARE SET TO ZERO.
    """
    eps = 1e-5
    N = len(y_pred[0])
    mse_loss = nn.MSELoss()

    G = y_pred[1]
    H = y_pred[0]

    dims = H[0].shape
    
    neg_corrs   = torch.zeros(int(comb(N, 2)))
    mses        = torch.zeros(N)
    l2_norms    = torch.zeros(N)
    layer_norms = torch.zeros(N)

    k = 0
    for i in range(N):
        mses[i]     = mse_loss(G[i], y_act[:, i*i_shape1: (i+1)*i_shape1])
        l2_norms[i] = torch.norm(H[i])
        for j in range(i+1, N):
            neg_corrs[k] = -get_corr(H[i], H[j], dims, outdim_size, use_all_singular_values, eps)
            k = k + 1
    
    if lambdas_[1] != 0:
        for i in range(N):
            l2_norms[i] = torch.norm(H[i])
            if i < N-1  : 
                exec(f'layer_norms[{i}] = torch.norm(model.enc_net{i}.thr.weight)')
            if i == N-1 : 
                layer_norms[i] = torch.norm(model.enc_nets.thr.weight)

    neg_corr   = torch.sum(neg_corrs)
    mse        = torch.sum(mses)
    l2_norm    = torch.sum(l2_norms)
    layer_norm = torch.sum(layer_norms)

    total_loss = neg_corr + (lambda_ * mse) + (lambdas_[0] * l2_norm) + (lambdas_[1] * layer_norm)
    
    return total_loss, neg_corr, mse, neg_corrs, mses

