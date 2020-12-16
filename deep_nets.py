import torch
import torch.nn as nn
import torch.nn.functional as F


# DCCA MODEL
class model2_13(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model2_13, self).__init__()
        self.net1 = get_13network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get_13network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get_13network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get_13network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 2038)
        self.fc2 = nn.Linear(2038, 1608)
        self.fc3 = nn.Linear(1608, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc2(x)))
        x = self.lre(self.fc3(x))
        return x





# DMCCA MODEL
class enc_model(nn.Module):
    def __init__(self, i_shape, mid_shape, o_dim, p):
        super(enc_model, self).__init__()
        self.i_shape = i_shape
        self.mid_shape = mid_shape
        self.o_dim = o_dim
        self.act = nn.Sigmoid()
        self.o_act = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)

        self.one = nn.Linear(self.i_shape, self.mid_shape)
        self.sec = nn.Linear(self.mid_shape, self.mid_shape)
        self.thr = nn.Linear(self.mid_shape, self.o_dim)
    def forward(self, x):
        x = self.sec(self.drp(self.act(self.one(x))))
        x = self.o_act(self.thr(self.drp(self.act(x))))
        return x

class dec_model(nn.Module):
    def __init__(self, n_resps, i_shape, mid_shape, mid_shape2, o_dim, p):
        super(dec_model, self).__init__()
        self.i_shape = i_shape
        self.n_resps = n_resps
        self.mid_shape = mid_shape
        self.mid_shape2 = mid_shape2
        self.o_dim = o_dim
        self.act = nn.Sigmoid()
        self.o_act = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)

        self.de1 = nn.Linear((self.n_resps+1)*self.o_dim, self.mid_shape)
        self.de2 = nn.Linear(self.mid_shape, self.mid_shape2)
        self.de3 = nn.Linear(self.mid_shape2, self.i_shape)
    def forward(self, y):
        y = self.drp(self.act(self.de1(y)))
        y = self.de3(self.drp(self.act(self.de2(y))))
        return y

# DMCCA model for N RESPS + 1 STIM.
class dmcca_model_n_resp_1_stim(nn.Module):
    def __init__(self, n_resps, i_shape1, i_shape2, mid_shape, o_dim, p=0):
        super(dmcca_model_n_resp_1_stim, self).__init__()
        self.n_resps = n_resps
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
        self.mid_shape  = mid_shape
        self.mid_shape2 = 110
        self.o_dim = o_dim
        self.p = p
        self.drp = nn.Dropout(p=p)

        for i in range(self.n_resps):
            exec(f'self.enc_net{i} = enc_model(self.i_shape1, self.mid_shape, self.o_dim, self.p)')
            exec(f'self.dec_net{i} = dec_model(self.n_resps, self.i_shape1, self.mid_shape, self.mid_shape2, self.o_dim, self.p)')

        self.enc_nets = enc_model(self.i_shape2, self.mid_shape, self.o_dim, self.p)
        self.dec_nets = dec_model(self.n_resps, self.i_shape2, self.mid_shape, self.mid_shape2, self.o_dim, self.p)
    
    def forward(self, x):
        x1 = []
        for i in range(self.n_resps):
            exec(f'x1.append(self.enc_net{i}(x[:, {i}*self.i_shape1 : ({i+1})*self.i_shape1]))')
        x1.append(self.enc_nets(x[:, -self.i_shape2: ]))
        x = torch.cat(x1, 1)
        
        y = []
        for i in range(self.n_resps):
            exec(f'y.append(self.dec_net{i}(x))')
        y.append(self.dec_nets(x))
        z = [x1, y]
        return z

# CUSTOM DMCCA model 
class dmcca_model(nn.Module):
    def __init__(self, n_data, i_shapes, mid_shape, o_dim, p=0):
        super(dmcca_model_n_resp_1_stim, self).__init__()
        self.n_data     = n_data
        self.i_shapes   = i_shapes
        self.mid_shape  = mid_shape
        self.mid_shape2 = 110
        self.o_dim = o_dim
        self.p   = p
        self.drp = nn.Dropout(p=p)

        for i in range(self.n_data):
            exec(f'self.enc_net{i} = enc_model(self.i_shapes[i], self.mid_shape, self.o_dim, self.p)')
            exec(f'self.dec_net{i} = dec_model(self.n_data, self.i_shapes[i], self.mid_shape, self.mid_shape2, self.o_dim, self.p)')

    def forward(self, x):
        x1 = []
        for i in range(self.n_data):
            exec(f'x1.append(self.enc_net{i}(x[:, {i}*self.i_shapes[{i}] : ({i+1})*self.i_shapes[{i}]]))')
        x = torch.cat(x1, 1)
        
        y = []
        for i in range(self.n_data):
            exec(f'y.append(self.dec_net{i}(x))')

        z = [x1, y]
        return z





# DMCCA with only encoders (Similar to DGCCA or DMCCA with lambda=0)
class dmcca_model_enc_only_n_resp_1_stim(nn.Module):
    """ DMCCA with only encoders (Similar to DGCCA or DMCCA with lambda=0) """
    def __init__(self, n_resps, i_shape1, i_shape2, mid_shape, act, o_act, o_dim, p=0):
        super(dmcca_model_enc_only_n_resp_1_stim, self).__init__()
        self.n_resps = n_resps
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
        self.mid_shape  = mid_shape
        self.mid_shape2 = 110
        self.o_dim = o_dim
        self.p = p
        self.drp = nn.Dropout(p=p)

        for i in range(self.n_resps):
            exec(f'self.enc_net{i} = enc_model(self.i_shape1, self.mid_shape, self.o_dim, self.p)')
        self.enc_nets = enc_model(self.i_shape2, self.mid_shape, self.o_dim, self.p)
    
    def forward(self, x):
        x1 = []
        for i in range(self.n_resps):
            exec(f'x1.append(self.enc_net{i}(x[:, {i}*self.i_shape1 : ({i+1})*self.i_shape1]))')
        x1.append(self.enc_nets(x[:, -self.i_shape2: ]))
        return x1











# DIFFERENT DCCA ARCHITECTURES

# 256;2
class model_100s(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model_100s, self).__init__()
        self.net1 = get_100network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get_100network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get_100network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get_100network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc2(x)))
        x = self.lre(self.fc3(x))
        return x

# 256;3
class model_3_100s(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model_3_100s, self).__init__()
        self.net1 = get_3_100network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get_3_100network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get_3_100network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get_3_100network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 256)
        self.fc21 = nn.Linear(256, 256)
        self.fc22 = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc21(x)))
        x = self.drp(self.sig(self.fc22(x)))
        x = self.lre(self.fc3(x))
        return x

# 256;4
class model_4_100s(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model_4_100s, self).__init__()
        self.net1 = get_4_100network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get_4_100network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get_4_100network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get_4_100network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 256)
        self.fc21 = nn.Linear(256, 256)
        self.fc22 = nn.Linear(256, 256)
        self.fc23 = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc21(x)))
        x = self.drp(self.sig(self.fc22(x)))
        x = self.drp(self.sig(self.fc23(x)))
        x = self.lre(self.fc3(x))
        return x

# 256;5
class model_5_100s(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model_5_100s, self).__init__()
        self.net1 = get_5_100network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get_5_100network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get_5_100network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get_5_100network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 256)
        self.fc21 = nn.Linear(256, 256)
        self.fc22 = nn.Linear(256, 256)
        self.fc23 = nn.Linear(256, 256)
        self.fc24 = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc21(x)))
        x = self.drp(self.sig(self.fc22(x)))
        x = self.drp(self.sig(self.fc23(x)))
        x = self.drp(self.sig(self.fc24(x)))
        x = self.lre(self.fc3(x))
        return x

# 1024;3
class model3_13(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model3_13, self).__init__()
        self.net1 = get3_13network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get3_13network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get3_13network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get3_13network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 1024)
        self.fc21 = nn.Linear(1024, 1024)
        self.fc22 = nn.Linear(1024, 1024)
        self.fc3  = nn.Linear(1024, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc21(x)))
        x = self.drp(self.sig(self.fc22(x)))
        x = self.lre(self.fc3(x))
        return x

# 1024;4 Model
class get_15network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get_15network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc2(x)))
        x = self.drp(self.sig(self.fc3(x)))
        x = self.lre(self.fc4(x))
        return x

class model2_15(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p):
        super(model2_15, self).__init__()
        self.net1 = get_15network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get_15network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

# 1024;5
class model5_13(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model5_13, self).__init__()
        self.net1 = get5_13network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get5_13network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get5_13network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get5_13network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 1024)
        self.fc21 = nn.Linear(1024, 1024)
        self.fc22 = nn.Linear(1024, 1024)
        self.fc23 = nn.Linear(1024, 1024)
        self.fc24 = nn.Linear(1024, 1024)
        self.fc3  = nn.Linear(1024, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc21(x)))
        x = self.drp(self.sig(self.fc22(x)))
        x = self.drp(self.sig(self.fc23(x)))
        x = self.drp(self.sig(self.fc24(x)))
        x = self.lre(self.fc3(x))
        return x

# 10240;2
class model_10000s(nn.Module):
    def __init__(self, i_shape1, i_shape2, act, o_act, o_dim, p=0):
        super(model_10000s, self).__init__()
        self.net1 = get_10000network(i_shape1, act, o_act, o_dim, p)
        self.net2 = get_10000network(i_shape2, act, o_act, o_dim, p)
        self.i_shape1 = i_shape1
        self.i_shape2 = i_shape2
    def forward(self, x):
        # print(x1.shape); print(x2.shape)
        y1 = self.net1(x[:, :self.i_shape1])
        y2 = self.net2(x[:, self.i_shape1:])
        y = [y1, y2]
        return y

class get_10000network(nn.Module):
    def __init__(self, i_shape, act, o_act, o_dim, p):
        super(get_10000network, self).__init__()
        self.fc1 = nn.Linear(i_shape, 10240)
        self.fc2 = nn.Linear(10240, 1024)
        self.fc3 = nn.Linear(1024, o_dim)
        self.sig = nn.Sigmoid()
        self.lre = nn.LeakyReLU(0.1)
        self.drp = nn.Dropout(p=p)
    def forward(self, x):
        # print(self.fc1)
        x = self.drp(self.sig(self.fc1(x)))
        x = self.drp(self.sig(self.fc2(x)))
        x = self.lre(self.fc3(x))
        return x






