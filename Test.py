# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Missile_Environment for Reinforcement Learning
#    v-1.6
#       LPF available
#       Normalize all state

import vpython as vp

import math as m
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import time
import copy
import CraftDynamics as ENV
import DaseonTypesNtf as Daseon
import matplotlib.pyplot as plt

rad2deg         = 57.29577951
deg2rad         = 0.01745329252
gpu_num         = -1


# %%
device      = ('cuda'+':'+ str(gpu_num)) if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == ('cuda'+ ':' + str(gpu_num)):
    torch.cuda.manual_seed_all(777)
if gpu_num==-1:
    device = 'cpu'


# %%
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        
        self.fc1     = nn.Linear(4, 150, bias = True)    
        self.relu1   = nn.ReLU()
        self.fc2     = nn.Linear(150, 300, bias = True) 
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(300, 500, bias = True)
        self.relu3   = nn.ReLU()
        self.fc4     = nn.Linear(500, 300, bias = True)
        self.relu4   = nn.ReLU()
        self.fc5     = nn.Linear(300, 150, bias = True)
        self.relu5   = nn.ReLU()
        self.fc6     = nn.Linear(150, 1, bias = True)
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        torch.nn.init.xavier_uniform_(self.fc5.weight)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
        
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        
        out = self.fc4(out)
        out = self.relu4(out)
        
        out = self.fc5(out)
        out = self.relu5(out)
        
        out = self.fc6(out)
        
        return out


# %%
def ITCG(Vm, R, lam_psi, N, d_lam, s, k1):
    acp1 = (-Vm*(lam_psi)*m.cos(lam_psi))/2/R
    acp2 = (Vm*(2*N-1)*(1-m.cos(lam_psi)))/R/(lam_psi)
    sterm = k1*lam_psi*s
    a_c = d_lam+acp1+acp2+sterm
    aM = a_c*Vm
    return Daseon.Vector3(0., aM, 0.)

def PPNG(N,Vm,Vj,Omega):
    print(N,Vm,Vj,Omega)
    print(Vj.mag * np.array(Vm.vec))
    return Daseon.Vector3.cast(-N*Vj.mag*np.cross(np.array(Vm.vec)/Vm.mag, Omega.vec))

def get_dlam(Vm,R,look):
    return Vm/R*m.sin(look)

def get_analytic_T2go(look, N, Rmag, Vm):
    return (1+((look)**2)/2/(2*N-1))*Rmag/Vm

def get_neural_T2go(model, mean4norm, std4norm, state):
    stateNorm       = (state - mean4norm[0:-1])/std4norm[0:-1]
    inference_gpu   = model(torch.from_numpy(stateNorm).to(device))
    inference_np    = inference_gpu.to('cpu').numpy()
    inference_val   = copy.deepcopy(inference_np*std4norm[-1] + mean4norm[-1])
    return inference_val

def get_s(t, t_d, t2go):
    return t - t_d + t2go

def get_k():
    return 1


# %%
model = DNN().to(device)
model.load_state_dict(torch.load('./360WEIGHT_0.02194962464272976', map_location=torch.device('cpu')))
model.eval()
model.train(False)


# %%
DATmean     = np.array([ 5.0023467e+03,-1.5984995e+00,1.1228751e+02,1.7544924e+00,2.1465954e+01],dtype='float32')
DATstd      = np.array([ 5.5374233e+03, 1.7694662e+00,1.6407622e+02,1.4172097e+00,2.4340271e+01],dtype='float32')

DATmean     = np.array([1.0429623e+04, -3.8220634e+00,  1.2230603e+02,  1.9110317e+00, 6.3654346e+01],dtype='float32')
DATstd      = np.array([1.2492353e+04, 3.3081369e+00, 1.5931599e+02, 1.3973013e+00, 8.8367859e+01],dtype='float32')
# %%
initpos     = Daseon.Vector3(5000., 3200., 0.)
initpos0    = Daseon.Vector3(0., 0., 0.)
scavel      = 250.
N           = 3.
initatt     = Daseon.Vector3(0., 0., deg2rad*150)
initatt0    = Daseon.Vector3(0., 0., 0.)
dt          = 0.01
t_d         = 60.
Missile1    = ENV.Craft(scavel, initpos, initatt, dt)
TargetShip  = ENV.Craft(0., initpos0, initatt0, dt)
Missile1EYE = ENV.Seeker(Missile1, TargetShip)


# %%

mode = 'p' # stands for pure png
mode = 'a' # stands for analytic ITCG
mode = 'n' # stands for neural

mode = 'n'
with torch.no_grad():
    t = 0
    pos_spots = []
    while True:
        Missile1EYE.seek(t)
        if mode == 'a':
            t2goA       = get_analytic_T2go(Missile1EYE.Look.z, N, Missile1EYE.Rvec.mag, Missile1.scavel)
            acc         = ITCG( Missile1.scavel,\
                            Missile1EYE.Rvec.mag,\
                            Missile1EYE.Look.z,\
                            N,\
                            get_dlam(Missile1.scavel, Missile1EYE.Rvec.mag, Missile1EYE.Look.z),\
                            get_s(t, t_d, t2goA),\
                            1)

        elif mode == 'n':
            state4t2go  = np.array([Missile1EYE.Rvec.mag,Missile1EYE.Look.z,Missile1.scavel,N],dtype='float32')
            t2goA       = get_analytic_T2go(Missile1EYE.Look.z, N, Missile1EYE.Rvec.mag, Missile1.scavel)
            t2goN       = get_neural_T2go(model, DATmean, DATstd, state4t2go)[0]
            
            print('t2gN: ',t2goN, 't2gA: ',t2goA)
            acc         = ITCG( Missile1.scavel,\
                            Missile1EYE.Rvec.mag,\
                            Missile1EYE.Look.z,\
                            N,\
                            get_dlam(Missile1.scavel, Missile1EYE.Rvec.mag, Missile1EYE.Look.z),\
                            get_s(t, t_d, t2goN),\
                            1)
            
        elif mode == 'p':
            acc = PPNG(N,Missile1.dpos,Missile1EYE.Vvec,Missile1EYE.dLOS)
            acc.x = 0
            acc.z = 0
            


        Missile1.simulate_via_acc(acc)
        _, _, OOR, HIT = Missile1EYE.spit_reward(acc)
        pos_spots.append(Missile1.pos.vec)
        print(HIT)
        if mode == 'p':
            if (HIT)or(OOR)or(t_d<=t):break
        else:
            if (HIT)or(t_d<=t):break

        
        t = t+dt
        print('t=', t)

pos_spots = np.array(pos_spots)
plt.scatter(pos_spots[:,0], pos_spots[:,1])
plt.show()

