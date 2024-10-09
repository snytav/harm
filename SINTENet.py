import torch
import torch.nn as nn


class SINTENet(nn.Module):
    def __init__(self,N):
        super(SINTENet,self).__init__()
        self.N = N
        fc1 = nn.Linear(2,self.N)
        self.fc1 = fc1
        self.fc2 = nn.Linear(self.N,1)
    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

def loss(v_torch,net,fi,al):
    y = 0.0
    for i,f in enumerate(fi):
        for j,a in enumerate(al):
            t = torch.tensor([f,a]).float()
            h = net(t)
            y += torch.abs(v_torch[i][j]-h)

    return y

def get_result(v_torch,net,fi,al,name):
    res = torch.zeros_like(v_torch)

    file1 = open(name+".txt", "w")

    for i,f in enumerate(fi):
        for j,a in enumerate(al):
            t = torch.tensor([f,a]).float()
            h = net(t)
            res[i][j] = h
            file1.write('fi '+'{:15.5f}'.format(f))

            #y += torch.abs(v_torch[i][j]-h)



if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import torch

    crd = np.loadtxt('testgrid.grid')
    N_fi = np.unique(crd[:, 0]).shape[0]
    N_al = np.unique(crd[:, 1]).shape[0]
    df = df_user_key_word_org = pd.read_csv("new",
                                            sep="\s+|;|:",
                                            engine="python")
    # net = GeoNet(10)
    v = df['P_mod'].values
    fi = df['fi'].values
    lb = df['lb'].values
    # t = np.concatenate((fi.reshape(fi.shape[0], 1), lb.reshape(fi.shape[0], 1)), axis=1)
    # t = torch.from_numpy(t)
    # vt = net.forward(t.float())
    # optimizer = torch.optim.Adam(net.parameters(),lr = 0.01)
    # lf = torch.ones(1)*1e9
    v = v.reshape(N_fi,N_al)
    fi = fi.reshape(N_fi,N_al)
    lb = lb.reshape(N_fi,N_al)
    v_torch = torch.from_numpy(v)

    N_train_FI = 10
    N_train_AL = 10

    v_train  = v[:N_train_FI,:N_train_AL]
    fi_train = fi[:N_train_FI, :N_train_AL]
    lb_train = lb[:N_train_FI, :N_train_AL]




    net = SINTENet(10)

    # y = loss(v_torch,net,fi[:,0],lb[0,:])

    optim = torch.optim.Adam(net.parameters(),lr = 0.01)

    for n in range(100):
        optim.zero_grad()
        y = loss(v_train, net,fi_train[:,0],lb_train[0,:])

        y.backward()

        optim.step()
        print(n,y.item())

    qq = 0






