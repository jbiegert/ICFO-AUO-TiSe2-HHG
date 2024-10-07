# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:06:21 2024

@author: Igor
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
import scipy.constants as const


#%% Functions

def mynorm(yy,mm=0):
    if mm ==1:
        yy = yy-min(yy)
    return yy/np.max(yy)

def smooth2(arr, span):
    if span==0:
        return arr
    
    arr = savgol_filter(arr, span * 2 + 1, 2)
    arr = savgol_filter(arr, span * 2 + 1, 2)
    return arr

#%%


h1scan = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\H1_scan478++.txt",skiprows=1)

h1temp = h1scan[:,0]
h1refl = h1scan[:,1]/(0.75*(0.97**2)*(0.97**2))/(31*0.97**2)
h1err = h1scan[:,2]/(0.75*(0.97**2)*(0.97**2))/(31*0.97**2)


#%%
plt.rcParams.update({'font.size': 18})


def porder(T,Tc,A,B,C):
    # Tc = 200
    tabv= np.where(T>Tc)
    tbel= np.where(T<=Tc)
    
    pord_abv = np.sqrt(C+ h1temp[tabv]*0)
    pord_bel =  np.sqrt(C + B*np.tanh(A*np.real(np.sqrt(Tc/T[tbel]-1)))**2)
    
    return np.hstack((pord_bel,pord_abv))



ord_pin= [200,1.16,0.1,0.5]

ord_opt, ord_cov = scipy.optimize.curve_fit(porder, h1temp, h1refl,p0=ord_pin,sigma=h1err,absolute_sigma=True)
# ord_opt, ord_cov = scipy.optimize.curve_fit(porder, temp, mynorm(np.array(chi1_h1)[:,0]**2) ,p0=ord_pin)
ord_err = np.diag(ord_cov)
fig4, ax4 = plt.subplots(figsize=(10,5))
ax4.errorbar(h1temp,h1refl,yerr=h1err,capsize=3,fmt="bo",alpha=1)
ax4.plot(h1temp,porder(h1temp,*ord_opt))

fitmin=porder(h1temp,*(ord_opt-np.diag(ord_cov)))
fitmax=porder(h1temp,*(ord_opt+np.diag(ord_cov)))
ax4.fill_between(h1temp, fitmin, fitmax, facecolor="b", alpha=0.2)

tosave = np.transpose(np.vstack([ord_opt,ord_err]))
tosave2 = np.transpose(np.vstack([h1temp,porder(h1temp,*ord_opt),fitmin,fitmax]))
# np.savetxt("H1_tanhsq_fit_params.txt",tosave)
# np.savetxt("H1_tanhsq_fit_vals.txt",tosave2)
