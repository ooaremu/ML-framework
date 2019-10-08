'''
Created on Aug 31, 2018

@author: aremu
'''
'''
This file is used to run the PDE solver to determin ideal number of clusters

'''
import sys
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import subprocess
import scipy.stats as scy_stats
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, integrate
from root.nested.PDE_KDE_Estimate import PDE_KDE_BWDTH
from scipy.interpolate import UnivariateSpline
from datetime import date
import os
##-----------------------------------------------------------------------#
# Data set input
Raw_data =['CMAPPS_train_FD001','CMAPPS_train_FD002','CMAPPS_train_FD003','CMAPPS_train_FD004']
#select which data you want to work on
selectd_dataset = pd.read_pickle('{0}'.format(Raw_data[2]))
##-----------------------------------------------------------------------#
Data_Full = pd.read_pickle('CMAPPS_train_{0}'.format('FD004'))

X_orig = Data_Full.loc[Data_Full.unit==4].f4.values
X = (X_orig-min(X_orig))/(max(X_orig)-min(X_orig)) #normalized #change for normalized points

def odefunc(u, t):
    print(t)
    k = 0.5#np.size(X)
    X_in = Data_Full.loc[Data_Full.unit==4].f4.values
    X = (X_in-min(X_in))/(max(X_in)-min(X_in)) #change for normalized points
    X_S = np.linspace(X.min(), X.max(), X.shape[0])
    
    dudt = np.zeros((X.shape[0],))
    dudt[0] = 0 # constant at boundary condition
    dudt[-1] = 0

    u = (u-min(u))/(max(u)-min(u))
    dens_func = UnivariateSpline(X_S,u/np.size(X),s=0)
    hessian = dens_func.derivative(n=2)

    hessi_one = hessian(X_S) 
    d2u_dux2 = [k*n for n in hessi_one]
    dudt[1:-1] = d2u_dux2[1:-1]
    print(dudt)
    return dudt

#support_ode = np.linspace(X_ode.min(), X_ode.max(), X_ode.shape[0])
BD_Rand_EST = (1.06 * X.std() * (X.size) ** (-1 / 5.))/10

u0_int = PDE_KDE_BWDTH().Dis_fun(X,t=BD_Rand_EST)
u0_int = (u0_int-min(u0_int))/(max(u0_int)-min(u0_int)) #change for normalized points 
u0_int[0] = 0
u0_int[-1] = 0

# plt.plot(support_ode,u0_int)
tspan = np.linspace(0.0, 40.0, 100)

sol = odeint(odefunc, u0_int, tspan, printmessg=1,mxstep=400) 

#---------------------------------------------------------------------#
support = np.linspace(X.min(), X.max(), X.shape[0])
plt.figure()
for i in range(1, len(tspan), 1):
    if np.max(sol[i]) > 1.01:
        break
    if i==1:
        time_sets= 0.0
    else:
        time_sets= tspan[i]
    plt.plot(support, sol[i], label='t={0:1.2f}'.format(time_sets))
 
# put legend outside the figure
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.ylim(0, 1)
plt.xlim(0, 1)
plt.xlabel('Sensor observation samples (X)')
plt.ylabel('Distribution')
 
# adjust figure edges so the legend is in the figure
plt.subplots_adjust(top=0.89, right=0.77)
plt.savefig('Thesis_pde-transient-heat-1_fixedhessi_{}.png'.format(date.today().__str__()))
 
 
# Make a 3d figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
  
SX, ST = np.meshgrid(support, tspan)
ax.plot_surface(ST,SX,sol, cmap='jet')
ax.set_xlabel('Sensor observation samples (X)')
ax.set_ylabel('time')
ax.set_zlabel('Distribution')
ax.set_zlim3d(0,0.5)
ax.set_ylim3d(0,1)
ax.view_init(elev=15, azim=-124) # adjust view so it is easy to see
plt.savefig('Thesis_pde-transient-heat-hessi_{}.png'.format(date.today().__str__()))
  
# animated solution. We will use imagemagick for this
 
# we save each frame as an image, and use the imagemagick convert command to 
# make an animated gif

for i in range(len(tspan)):
    if np.max(sol[i+1]) > 1.01:
        break
    plt.clf()
    plt.plot(support, sol[i+1])#plt.plot(support, sol[i][1:-1])
    plt.xlabel('Sensor observation samples (X)')
    plt.ylabel('Distribution')
    plt.ylim(0, 0.35)
    plt.xlim(0, 1)
    plt.title('t = {0}'.format(tspan[i+1]))
    plt.savefig('___t{0:03d}.png'.format(i+1))

print (subprocess.getoutput('convert -quality 100 ___t*.png Thesis_transient_heat_fixedhessi_{0}.gif'.format(date.today().__str__())))
print (subprocess.getoutput('rm ___t*.png')) #remove temp files