# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:34:03 2024

@author: ityulnev
"""
import sys
from pathlib import Path

import glob
import numpy as np
from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import numpy, scipy.optimize
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.colors as colors
from matplotlib import cm
import math
import re
from datetime import datetime,time
from scipy.signal import savgol_filter, hilbert, chirp
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.constants import *


#%% Definitions for Lorentz terms
def epsilon_drude(ww,wp,tp):
    return -wp**2/(ww**2+1j*ww/tp)
def epsilon1(ww,s1,w1,t1):
    return s1**2/(w1**2-ww**2-1j*ww/t1)
def epsilon2(ww,s2,w2,t2):
    return s2**2/(w2**2-ww**2-1j*ww/t2)

def epsilonPH(ww,n_p1,w_p1,t_p1,n_p2,w_p2,t_p2,n_p3,w_p3,t_p3,n_p4,w_p4,t_p4,n_p5,w_p5,t_p5): #,n_p5,w_p5,t_p5
    #phonon response with up to 5 lines
    # ntot = n_p1+n_p2+n_p3+n_p4+n_p5
    ntot=1
    p1 = (n_p1/ntot)**2/(w_p1**2-ww**2-1j*ww/t_p1)
    p2 = (n_p2/ntot)**2/(w_p2**2-ww**2-1j*ww/t_p2)
    p3 = (n_p3/ntot)**2/(w_p3**2-ww**2-1j*ww/t_p3)
    p4 = (n_p4/ntot)**2/(w_p4**2-ww**2-1j*ww/t_p4)
    p5 = (n_p5/ntot)**2/(w_p5**2-ww**2-1j*ww/t_p5)
    
    return p1+p2+p3+p4+p5

def epsilonPH_1(ww,n_p1,w_p1,t_p1): #,n_p5,w_p5,t_p5
    #phonon response with up to 5 lines
    # ntot = n_p1+n_p2+n_p3+n_p4+n_p5
    ntot=1
    p1 = (n_p1/ntot)**2/(w_p1**2-ww**2-1j*ww/t_p1)
    # p2 = (n_p2/ntot)**2/(w_p2**2-ww**2-1j*ww/t_p2)

    return p1


def eps_tot(ww,*args):
    wp = args[0]
    tp = args[1]
    s1 = args[2]
    w1 = args[3]
    t1 = args[4]
    s2 = args[5]
    w2 = args[6]
    t2 = args[7]
    eps0 = args[8]
    
    n_p1 = args[9]
    w_p1 = args[10]
    t_p1 = args[11]
    
    n_p2 = args[12]
    w_p2 = args[13]
    t_p2 = args[14]
    
    n_p3 = args[15]
    w_p3 = args[16]
    t_p3 = args[17]
    
    n_p4 = args[18]
    w_p4 = args[19]
    t_p4 = args[20]
    
    n_p5 = args[21]
    w_p5 = args[22]
    t_p5 = args[23]
    return eps0+epsilon_drude(ww,wp,tp)+epsilon1(ww,s1,w1,t1)+epsilon2(ww,s2,w2,t2)+epsilonPH(ww,n_p1,w_p1,t_p1,n_p2,w_p2,t_p2,n_p3,w_p3,t_p3,n_p4,w_p4,t_p4,n_p5,w_p5,t_p5)


def NN(ww,*args):
    wp = args[0]
    tp = args[1]
    s1 = args[2]
    w1 = args[3]
    t1 = args[4]
    s2 = args[5]
    w2 = args[6]
    t2 = args[7]
    eps0 = args[8]
    
    n_p1 = args[9]
    w_p1 = args[10]
    t_p1 = args[11]
    
    n_p2 = args[12]
    w_p2 = args[13]
    t_p2 = args[14]
    
    n_p3 = args[15]
    w_p3 = args[16]
    t_p3 = args[17]
    
    n_p4 = args[18]
    w_p4 = args[19]
    t_p4 = args[20]
    
    n_p5 = args[21]
    w_p5 = args[22]
    t_p5 = args[23]
    
    return np.sqrt(epsilon_drude(ww,wp,tp)+epsilon1(ww,s1,w1,t1)+epsilon2(ww,s2,w2,t2)+eps0+
                   epsilonPH(ww,n_p1,w_p1,t_p1,n_p2,w_p2,t_p2,n_p3,w_p3,t_p3,n_p4,w_p4,t_p4,n_p5,w_p5,t_p5))#

def NN_ph1(ww,*args):
    wp = args[0]
    tp = args[1]
    s1 = args[2]
    w1 = args[3]
    t1 = args[4]
    s2 = args[5]
    w2 = args[6]
    t2 = args[7]
    eps0 = args[8]
    
    n_p1 = args[9]
    w_p1 = args[10]
    t_p1 = args[11] 
    
    # n_p2 = args[12]
    # w_p2 = args[13]
    # t_p2 = args[14] 
    
    return np.sqrt(epsilon_drude(ww,wp,tp)+epsilon1(ww,s1,w1,t1)+epsilon2(ww,s2,w2,t2)+eps0+
                   epsilonPH_1(ww,n_p1,w_p1,t_p1))#



def reflectivity(ww,*args):
    return abs((1-NN(ww,*args))/(1+NN(ww,*args)))**2

def reflectivity_ph1(ww,*args):
    return abs((1-NN_ph1(ww,*args))/(1+NN_ph1(ww,*args)))**2

def smooth2(arr, span):
    if span==0:
        return arr
    
    arr = savgol_filter(arr, span * 2 + 1, 2)
    arr = savgol_filter(arr, span * 2 + 1, 2)
    return arr

#%%Measurements by G.Li et. al.
r7_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_7full_10K.txt",skiprows=1)
# r7_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_1full_300K.txt",skiprows=1)
# r7_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_1full_300K.txt",skiprows=1)
r1_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_1full_300K.txt",skiprows=1)

r7 = np.concatenate([r7_dat,r1_dat[48:61]],axis=0) #r1&r7 (300K&10K) were digitized with high freq.
#high freq response is temperature independent concat for better statistics
r7[:,0] = r7[r7[:,1].argsort(),0]# sorting along frequencies
r7[:,1] = r7[r7[:,1].argsort(),1]#


r7[96,0] = np.nan# data jump in index from splicing digitization
r7[97,0] = np.nan# data jump in index from splicing digitization
# r7= np.ma.masked_equal(r7,np.nan)
r7 = r7[~numpy.isnan(r7[:,0])]

r7sm = np.zeros((len(r7[:,0]),2))
r7sm[:,0] = r7[:,0]
r7sm[54:-1,0] = smooth2(r7[54:-1,0],3)# smoothing, while preserving the phonon oscillations
r7sm[:,1] = r7[:,1]



# fit function with 4 phonon peaks (as seen in data); later I'll use fit function with only 1 phonon peak
p0= [3000,0.03,
     18000,3200,0.00032,
     50000,13000,0.00012,
     18,
     740,139,4,
     875,152,1.6,
     500,172,1,
     220,197,0.5,
      54,212,1.5
      ]
mybounds = [1e5,1e-1, 
            1e5,6000,1e-1,
            1e5,5e4,1e-1,
            100,
            1000,150,10,
            1000,160,10,
            1000,180,5,
            500,210,1,
            200,220,2,
            ]

r0 = np.linspace(62,50000,49941)

chi1h1_intrp = scipy.interpolate.interp1d(r7[:,1], r7sm[:,0]) # interpolation for smooth curve
mysig7= np.zeros(len(r0))+0.05
mysig7[2940:4000] = 0.001
# popt7, pcov7 = scipy.optimize.curve_fit(reflectivity,r7[10:-1,1],r7[10:-1,0],p0,bounds = (1e-8, [1e5,1e-1, 1e5,5e4,1e-1,1e5,5e4,1e-1,100]))
popt7, pcov7 = scipy.optimize.curve_fit(reflectivity,r0,chi1h1_intrp(r0),p0,sigma=mysig7,bounds = (1e-8, mybounds))
perr = pcov7.diagonal()


plt.rcParams.update({'font.size': 18})
fig1a, ax1a = plt.subplots(figsize=(10,5))
ax1a.plot(r7_dat[:,1],r7_dat[:,0],"o-",label="Data 300K")
# ax1a.plot(r5_dat[:,1],r5_dat[:,0],"o-",label="Data 80K")
ax1a.plot(r7[:,1],r7[:,0],"o-",label="Data 10K",alpha=0.5)
ax1a.plot(r0,chi1h1_intrp(r0),"-",label="Data 10K",alpha=1)
ax1a.plot(r7sm[:,1],r7sm[:,0],":",label="Data 10K",alpha=1)

ax1a.plot(r0, reflectivity(r0, *popt7), 'b--',linewidth=2,label="Drude+Lorentz fit")

R_min = reflectivity(r0, *(popt7-perr))
R_max = reflectivity(r0, *(popt7+perr))
ax1a.fill_between(r0, R_min, R_max, facecolor="b", alpha=0.2)

ax1a.set_xscale('log')
ax1a.set_xlabel("frequency (cm$^-1$)")
ax1a.set_ylabel("reflectivity")
ax1a.set_xlim([50,50000])
ax1a.set_ylim([0,1.1])
plt.tight_layout()

#%%
# r1_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_1full_300K.txt",skiprows=1)
# r1_dat[47,0]=np.nan
r1_dat2 = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_1_300K.txt",skiprows=1)
# r1 = np.copy(r1_dat)
r1 = np.concatenate([r1_dat,r7_dat[96:115]],axis=0)# append the other high freq data(same for all temperatures)
# r1[:,1].argsort(axis=0)
r1[:,0] = r1[r1[:,1].argsort(),0]
r1[:,1] = r1[r1[:,1].argsort(),1]

r1[48,0] = np.nan# data jump in index 96 from splicing digitization
r1 = r1[~numpy.isnan(r1[:,0])]# remove the nan value
r1sm = np.zeros((len(r1[:,0]),2))
r1sm[:,0] = r1[:,0]
r1sm[20:79,0] = smooth2(r1[20:79,0],3)
r1sm[:,1] = r1[:,1]

p1= [14000,0.0003,
      18000,3200,0.00032,
     53000,13700,0.00013,
     18,
     2000,142,0.1,
     ]
mybounds1 = [1e5,1e-1, 
            1e5,6000,1e-1,
            1e5,5e4,1e-1,
            100,
            1e5,170,10,
            ]

r0_1 = np.linspace(64,50000,49937)
mysig1 = np.zeros(len(r0_1))+0.05
mysig1[0:200] = 0.001

chi1h1_intrp1 = scipy.interpolate.interp1d(r1[:,1],r1sm[:,0]) # interpolation for smooth curve
popt1, pcov1 = scipy.optimize.curve_fit(reflectivity_ph1,r0_1,chi1h1_intrp1(r0_1),p0=p1,sigma=mysig1,bounds = (1e-8, mybounds1))

plt.rcParams.update({'font.size': 18})
fig1a, ax1a = plt.subplots(figsize=(10,5))
ax1a.plot(r1_dat[:,1],r1_dat[:,0],"o-",label="Data 300K")
ax1a.plot(r1[:,1],r1[:,0],"o-",label="Data 300K")
# ax1a.plot(r1[:,1],smooth2(r1[:,0],5),"o-",label="Data 300K")

# ax2a.plot(r2[:,1],r2sm[:,0],"o-",label="Data 10K")
ax1a.plot(r0_1,chi1h1_intrp1(r0_1),"-",label="Interp",linewidth=4,alpha=0.4)
ax1a.plot(r0_1, reflectivity_ph1(r0_1, *popt1), 'b--',linewidth=2,label="Drude+Lorentz fit")
ax1a.set_xscale('log')

# popt1 = np.insert(popt1,2,[0,1,1])

#%%
r5_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\y_5_80K.txt",skiprows=1)
r5 = np.concatenate([r5_dat,r7_dat[96:115]],axis=0)
r5 = np.concatenate([r5,r1_dat[48:61]],axis=0)
r5[:,0] = r5[r5[:,1].argsort(),0]
r5[:,1] = r5[r5[:,1].argsort(),1]


r5[83,0] = np.nan# data jump in index 96 from splicing digitization
r5[84,0] = np.nan
r5 = r5[~numpy.isnan(r5[:,0])]
r5sm = np.zeros((len(r5[:,0]),2))
r5sm[:,0] = r5[:,0]
r5sm[40:112,0] = smooth2(r5[40:112,0],3)
r5sm[:,1] = r5[:,1]

p5= [3000,0.03,
     18000,3200,0.00032,
     50000,13000,0.00012,
     18,
     900,139,4,
     875,152,1.6,
     500,172,1,
     220,197,0.5,
     1,197,0.5
     ]
mybounds5 = [1e5,1e-1, 
            1e5,6000,1e-1,
            1e5,5e4,1e-1,
            100,
            1000,150,10,
            1000,160,10,
            1000,180,5,
            500,210,1,
            10,210,1
            ]

r0 = np.linspace(62,50000,49939)
mysig5= np.zeros(len(r0))+0.05
mysig5[2940:4000] = 0.001

chi1h1_intrp5 = scipy.interpolate.interp1d(r5[:,1],r5sm[:,0]) # interpolation for smooth curve
popt5, pcov5 = scipy.optimize.curve_fit(reflectivity,r0,chi1h1_intrp5(r0),p5,sigma=mysig5,bounds = (1e-8, mybounds5))

plt.rcParams.update({'font.size': 18})
fig5a, ax5a = plt.subplots(figsize=(10,5))
ax5a.plot(r5[:,1],r5[:,0],"o-",label="Data 80K")
# ax5a.plot(r7[:,1],r7[:,0],"o-",label="Data 10K")
ax5a.plot(r0,chi1h1_intrp5(r0),"-",label="Interp",linewidth=4,alpha=0.4)
ax5a.plot(r0, reflectivity(r0, *popt5), 'b--',linewidth=2,label="Drude+Lorentz fit")
ax5a.set_xscale('log')


#%%
r4_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_4_150K.txt",skiprows=1)
r4 = np.concatenate([r4_dat,r7_dat[96:115]],axis=0)
r4 = np.concatenate([r4,r1_dat[48:61]],axis=0)
r4[:,0] = r4[r4[:,1].argsort(),0]
r4[:,1] = r4[r4[:,1].argsort(),1]

r4[66,0] = np.nan# data jump in index 96 from splicing digitization
r4[67,0] = np.nan
r4 = r4[~numpy.isnan(r4[:,0])]
r4sm = np.zeros((len(r4[:,0]),2))
r4sm[:,0] = r4[:,0]
r4sm[30:97,0] = smooth2(r4[30:97,0],3)
r4sm[:,1] = r4[:,1]

p4= [3000,0.03,
     18000,3200,0.00032,
     50000,13000,0.00012,
     18,
     900,139,4,
     875,152,1.6,
     500,172,1,
     220,197,0.5,
     1,201,0.5
     ]
mybounds4 = [1e5,1e-1, 
            1e5,6000,1e-1,
            1e5,5e4,1e-1,
            100,
            1000,150,10,
            1000,160,10,
            1000,180,5,
            500,210,1,
            10,210,1

            ]

mysig4= np.zeros(len(r0))+0.05
mysig4[2940:4000] = 0.001

chi1h1_intrp4 = scipy.interpolate.interp1d(r4[:,1],r4sm[:,0]) # interpolation for smooth curve
popt4, pcov4 = scipy.optimize.curve_fit(reflectivity,r0,chi1h1_intrp4(r0),p4,sigma=mysig4,bounds = (1e-8, mybounds4))

plt.rcParams.update({'font.size': 18})
fig4a, ax4a = plt.subplots(figsize=(10,5))
ax4a.plot(r4[:,1],r4[:,0],"o-",label="Data 80K")
ax4a.plot(r4[:,1],r4sm[:,0],"o-",label="Data 10K")
ax4a.plot(r0,chi1h1_intrp4(r0),"-",label="Interp",linewidth=4,alpha=0.4)
ax4a.plot(r0, reflectivity(r0, *popt4), 'b--',linewidth=2,label="Drude+Lorentz fit")
#%%
r3_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_3_200K.txt",skiprows=1)
r3 = np.concatenate([r3_dat,r7_dat[96:115]],axis=0)
r3 = np.concatenate([r3,r1_dat[48:61]],axis=0)
r3[:,0] = r3[r3[:,1].argsort(),0]
r3[:,1] = r3[r3[:,1].argsort(),1]



r3[53,0] = np.nan# data jump in index 96 from splicing digitization
r3[54,0] = np.nan# data jump in index 96 from splicing digitization
r3 = r3[~numpy.isnan(r3[:,0])]
r3sm = np.zeros((len(r3[:,0]),2))
r3sm[:,0] = r3[:,0]
r3sm[25:84,0] = smooth2(r3[25:84,0],3)
r3sm[:,1] = r3[:,1]

p3= [3000,0.03,
     18000,6000,0.00032,
     50000,13000,0.00012,
     18,
     900,142,4
     ]
mybounds3 = [1e5,1e-1, 
            1e5,15000,1e-1,
            1e5,5e4,1e-1,
            100,
            1e4,170,10
            ]

r0_3 = np.linspace(64,50000,49937)
mysig3 = np.zeros(len(r0_3))+0.001

chi1h1_intrp3 = scipy.interpolate.interp1d(r3[:,1],r3sm[:,0]) # interpolation for smooth curve
popt3, pcov3 = scipy.optimize.curve_fit(reflectivity_ph1,r0_3,chi1h1_intrp3(r0_3),p3,sigma=mysig3,bounds = (1e-8, mybounds3))




plt.rcParams.update({'font.size': 18})
fig3a, ax3a = plt.subplots(figsize=(10,5))
ax3a.plot(r3[:,1],r3[:,0],"o-",label="Data 200K")
# ax3a.plot(r3[:,1],r3sm[:,0],"*-",label="Data 200K")
ax3a.plot(r0_3,chi1h1_intrp3(r0_3),"-",label="Interp",linewidth=4,alpha=0.4)
ax3a.plot(r0_3, reflectivity_ph1(r0_3, *popt3), 'b--',linewidth=2,label="Drude+Lorentz fit")


#%%
r2_dat = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\reflectivity\DrudeLorentz\y_2_225K.txt",skiprows=1)
r2 = np.concatenate([r2_dat,r7_dat[96:115]],axis=0)
r2[41,0] = np.nan# data jump in index 96 from splicing digitization
r2[42,0] = np.nan# data jump in index 96 from splicing digitization
r2 = r2[~numpy.isnan(r2[:,0])]
r2sm = np.zeros((len(r2[:,0]),2))
r2sm[:,0] = r2[:,0]
r2sm[36:-1,0] = smooth2(r2[36:-1,0],3)
r2sm[:,1] = r2[:,1]

p2= [3000,0.03,
     1,3200,0.00032,
     50000,13000,0.00012,
     18,
     1000,142,10,
     ]
mybounds2 = [1e5,1e-1, 
            1e5,1e4,1e-1,
            1e5,5e4,1e-1,
            100,
            2000,170,20,
            ]

r0_2 = np.linspace(64,50000,49937)
mysig2 = np.zeros(len(r0_2))+0.001

chi1h1_intrp2 = scipy.interpolate.interp1d(r2[:,1],r2sm[:,0]) # interpolation for smooth curve
popt2, pcov2 = scipy.optimize.curve_fit(reflectivity_ph1,r0_2,chi1h1_intrp2(r0_2),p2,sigma=mysig2,bounds = (1e-8, mybounds2))

plt.rcParams.update({'font.size': 18})
fig2a, ax2a = plt.subplots(figsize=(10,5))
ax2a.plot(r2[:,1],r2[:,0],"o-",label="Data 2225K")
# ax2a.plot(r2[:,1],r2sm[:,0],"o-",label="Data 10K")
ax2a.plot(r0_2,chi1h1_intrp2(r0_2),"*-",label="Interp",linewidth=4,alpha=0.4)
ax2a.plot(r0_2, reflectivity_ph1(r0_2, *popt2), 'b--',linewidth=2,label="Drude+Lorentz fit")


#%%
zeroentries = [0,1,1,0,1,1,0,1,1,0,1,1]
# r_fine = np.linspace(64,50000,699937)#more resolution
r_fine = np.linspace(64,50000,199936) # less resolution

# eps = eps_tot(r7[:,1], *popt7)
eps7 = eps_tot(r_fine, *popt7)
eps5 = eps_tot(r_fine, *popt5)
eps4 = eps_tot(r_fine, *popt4)
eps3 = eps_tot(r_fine, *[*popt3,*zeroentries])
# eps2 = eps_tot(r0, *[*popt2,*zeroentries])
eps1 = eps_tot(r_fine, *[*popt1,*zeroentries])

plt.rcParams.update({'font.size': 18})
fig1a, ax1a = plt.subplots(figsize=(10,5))
# ax1a.plot(r1[:,1],r1[:,0],"o",label="Data 300K")
# ax1a.plot(r0*1e2*constants.c*constants.h/constants.e,np.real(eps),"r-",linewidth=2,label="Real",alpha=0.5)
ax1a.plot(r_fine,np.imag(eps7)*r_fine,linewidth=2,label="Imag")
ax1a.plot(r_fine,np.imag(eps5)*r_fine,linewidth=2,label="Imag")
ax1a.plot(r_fine,np.imag(eps4)*r_fine,linewidth=2,label="Imag")
ax1a.plot(r_fine,np.imag(eps3)*r_fine,linewidth=2,label="Imag")
# ax1a.plot(r0,np.imag(eps2)*r0,linewidth=2,label="Imag")
ax1a.plot(r_fine,np.imag(eps1)*r_fine,linewidth=2,label="Imag")
# ax1a.set_xlabel("frequency (cm$^{-1}$)")
ax1a.set_xlabel("energy (eV)")
ax1a.set_ylabel("epsilon")
ax1a.set_xscale('log')
# ax1a.set_yscale('log')
# ax1a.set_xlim([1e-2,6.3])
# ax1a.set_ylim([-25,70])
# plt.hlines(0, 1e-3, 6.3,"k",alpha=0.5)
plt.legend(frameon=False)
plt.tight_layout()

E_eV = r_fine*1e2*constants.c*constants.h/constants.e
# E_eV = r0*1e2*constants.c*constants.h/constants.e
eps_imag_save = np.stack([E_eV,np.imag(eps1),np.imag(eps3),np.imag(eps4),np.imag(eps5),np.imag(eps7)],axis=1)
eps_real_save = np.stack([E_eV,np.real(eps1),np.real(eps3),np.real(eps4),np.real(eps5),np.real(eps7)],axis=1)
fitpar_labels = ["wp","tp","s1","w1","t1","s2","w2","t2","eps_inf","n_ph1","w_ph1","t_ph1","n_ph2","w_ph2","t_ph2","n_ph3","w_ph3","t_ph3","n_ph4","w_ph4","t_ph4","n_ph5","w_ph5","t_ph5"]
eps_fitpar = np.stack([[*popt1,*zeroentries],[*popt3,*zeroentries],popt4,popt5,popt7],axis=1)

# np.savetxt("GLi_epsilon_drudeLPhfit_imag.txt",eps_imag_save,header="eV,300K,200K,150K,80K,10K")
# np.savetxt("GLi_epsilon_drudeLPhfit_real.txt",eps_real_save,header="eV,300K,200K,150K,80K,10K")
# np.savetxt("GLi_epsilon_drudeLPhfit_params.txt",eps_fitpar,header="300K,200K,150K,80K,10K")

#%% Reflectivity
w_3um = 0.38745 #eV of 3.2 um wavelength
# w_ind = 42905 # high res
w_ind = 12257 # low res

print(E_eV[w_ind])

T_meas = [300,200,150,80,10]
eps_im = eps_imag_save[w_ind,1:]
eps_re = eps_real_save[w_ind,1:]
eps_compl = eps_re+ 1j*eps_im

reflect = abs((1-np.sqrt(eps_compl))/(1+np.sqrt(eps_compl)))**2



