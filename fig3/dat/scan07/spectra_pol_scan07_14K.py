
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:02:01 2020
This code plots data from Cryo Temperature scan software


Run in sections!

@author: ityulnev

v3. Lenard 2023.02.28.
pos index has no meaning, just a second measurement: I can average the data
"""



import sys
from pathlib import Path

import glob
import numpy as np
from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plab
import matplotlib.colors as colors
from matplotlib import cm
import math
import re
from datetime import datetime,time

from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass 

#%% Custom functions
# def myfft(Et,t):
#     """Correct FFT implementation, can take zero centered un-/even array
#     with coefficients as [a-2,a-1,a0,a1,a2] or [a-2,a-1,a0,a1]"""
#     N = len(Et)
#     T = (max(t)- min(t))
#     dt = T/N
#     Ef = fftshift(fft(ifftshift(Et))) * (T/N) # multiplied by dt
#     f = np.arange(-math.floor(N/2),math.ceil(N/2),1)/(T)
#     return(Ef,f)

def myfft(Et,t):
    """Correct FFT implementation, can take zero centered un-/even array
    with coefficients as [a-2,a-1,a0,a1,a2] or [a-2,a-1,a0,a1]"""
    N = len(Et)
    T = (max(t)- min(t))
    dt = T/N
    Ef = fftshift(fft(ifftshift(Et))) * (T/N) # multiplied by dt
    f = np.arange(-math.floor(N/2),math.ceil(N/2),1)/(T)
    return(Ef,f)

# def myifft(Ef,f):
#     """Correct iFFT implementation, can take zero centered un-/even array
#     with coefficients as [a-2,a-1,a0,a1,a2] or [a-2,a-1,a0,a1]"""
#     N = len(Ef)
#     F = max(f)-min(f)
#     dt =  1/(F) 
#     Et = fftshift(ifft(ifftshift(Ef))) * F # divided by dt
#     t = (np.arange(0,N,1)-(math.ceil(N/2)-1))*(dt)
#     return(Et,t)

# def symmetrize_frame(data_frame):
#     """Take data_frame with rows [t,E,It] and symmetrize t around zero.
#     Et and Ir are zero padded accordingly"""
#     tt = data_frame.iloc[:,0]
    
#     tt_extra = tt.where(tt>(-tt.min()))
#     tt_extra.dropna(axis="rows", how="any", inplace=True)
#     # tt_new = pd.concat([-tt_extra[::-1], tt], join='inner',ignore_index=True)
#     data = {'t':-tt_extra[::-1] ,'Et':[0]*len(tt_extra) ,'It':[0]*len(tt_extra)} 
#     pad_frame = pd.DataFrame(data) 

#     new_frame = pd.concat([pad_frame, data_frame], join='inner',ignore_index=True)
#     return(new_frame)

# def zeropad_frame(data_frame,N):
#     data = {'t':[0]*N,'Et':[0]*N ,'It':[0]*N} 
#     pad_frame = pd.DataFrame(data) 
#     new_frame = pd.concat([pad_frame, data_frame, pad_frame], join='inner',ignore_index=True)
#     return(new_frame)

# class mySupergauss(object): 
#     def __init__(self,x,x_offset=0,y_fwhm=100e-15,order=1):
#         """Calculate a Guassian function of a given order on the x axis"""
#         self.y_fwhm = y_fwhm
#         self.order = order
#         self.x_offset = x_offset
#         self.y = np.exp(-((((x-self.x_offset)/(self.y_fwhm/(2*np.sqrt(np.log(2)))))**2))**self.order);

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts]=y_smooth[box_pts+1]
    y_smooth[-box_pts:]=y_smooth[-(box_pts+1)]
    return y_smooth

def load_fileToDictionary(genpath,filenames):
    #% Load files into dict
    file_dict = {}
    mynames = sorted(genpath.glob(filenames))
    key_names = []
    for name in mynames:
        print(name) 
        myfile = np.loadtxt(name)
        file_dict[name.name] = myfile 
        key_names.append(name.name)
    return file_dict, key_names


#%% Build filepath, load data 

# genpath = Path(r"C:/Users/Lenard/ownCloud/YBCO_210504/") # SET GENERAL PATH HERE 
# pathtext = r"C:\Users\Lenard\ownCloud\YBCO_210511\scan1/"
# pathtext = r"C:\Users\Lenard\ownCloud\YBCO_210518\scan1/"
# pathtext = r"C:\Users\Lenard\ownCloud\YBCO_210521\scans\scan01/"
# genpath = Path(r"C:\Users\Igor\Documents\PhD\CryoExperiments\TiSe2\3um_TiSe2_spectroscopy\230504\scan07")
genpath = Path(r"D:\Dropbox\TiSe2_CDW\analysis\230504_PolScan\scan07")


# filenames_path = r"" # SET FOLDER NAME HERE!
filenames_spec = r"dat\scans_*.txt" # FINDS ALL .txt FILES OR SEARCHES FOR KEYWORDS *KEYWORD*.txt
filenames = filenames_spec

file_dict, mynames = load_fileToDictionary(genpath,r"dat\scan07_S*.txt") #S* is Spectra
file_dict2, mynames2 = load_fileToDictionary(genpath,r"dat\scan07_W*.txt") #W* is Wavelength
file_dict3, mynames3 = load_fileToDictionary(genpath,r"dat\scan07_P*.txt") #W* is Wavelength

#%% Time-Power correction from input laser av. power

# power_ref = pd.read_csv(r"C:\Users\Igor\Documents\PhD\Data\230406\Thorlabs_power.csv",header=12)

# t_time = power_ref.iloc[:,2]
# p_ref = power_ref.iloc[:,3]

# t_timeabs = []
# for ii in t_time:
#     datet = datetime.strptime(ii, " %H:%M:%S.%f")
#     total_seconds = datet.microsecond/1000000 + datet.second + datet.minute*60 + datet.hour*3600
#     t_timeabs.append(total_seconds)

# #%% Time and powers during measurement
# tt1 = pd.read_csv(r"C:\Users\Igor\Documents\PhD\Data\230406\scans\scan1TEMPERATURES.txt",header=None,sep=" |\t")
# tt_time = tt1.iloc[:,2]
# tt_temp = []
# tt_timeabs = []
# nn = 0
# for ii in tt_time:
#     datet = datetime.strptime(ii, "%H:%M:%S.%f")
#     total_seconds = datet.microsecond/1000000 + datet.second + datet.minute*60 + datet.hour*3600
#     tt_timeabs.append(total_seconds)
#     tt_temp.append(float(tt1.iloc[nn,4][:-1]))
#     nn += 1



# refp_int = interp1d(t_timeabs, p_ref, kind='linear')
# pref = refp_int(tt_timeabs)**5
# pref=pref/pref[0]
# plt.rcParams.update({'font.size': 24})
# fig01, ax01 = plt.subplots(figsize=(10,6))
# ax01.plot(t_timeabs,p_ref/p_ref[0])
# ax01.plot(tt_timeabs,pref)


# tt_joined = [[a, b] for a, b in sorted(zip(tt_temp, pref))]
# tt_jonedAr = np.array(tt_joined)

#%%
# pos = file_dict[mynames[0]]
temp =[]
wvl = file_dict2[mynames2[0]]
angle = file_dict3[mynames3[0]]

I_wvl_temp = []
I_wvl_temp2 = []

mynames_sorted = mynames
# mynames_sorted = sorted(mynames_sorted, key = lambda x: float(re.findall(r'\d+\.\d+', x)[-1]))
# mynames_sorted = sorted(mynames_sorted, key = lambda x: float(re.findall(r'[0-9]+_[0-90]+', x)[-1]))
mynames_sorted = sorted(mynames_sorted, key = lambda x: float(int(re.findall(r'\d+', x)[-4])*1000+int(re.findall(r'\d+', x)[-3])))

# re.findall(r'\d+', mynames[91])
for names in mynames_sorted:
    Ispec = file_dict[names]
    # Ispec_smoothed = smooth(Ispec,10)
    Ispec_smoothed = Ispec

    I_wvl_temp.append(Ispec_smoothed)
    
    # Ispec = file_dict[names][1]
    # Ispec_smoothed = smooth(Ispec,20)
    # I_wvl_temp2.append(Ispec_smoothed)

    temp.append(float(re.findall(r'\d+\.\d+', names)[-1]))
print("From HWP Position: ", "0", " degree")
I_wvl_temp = np.array(I_wvl_temp)
# I_wvl_temp2 = np.array(I_wvl_temp2)


temp = np.array(temp)   


#%%Take lineouts
# ToDo
harm_number = [2,4,6] #choose which harmoics to plot
plt.rcParams.update({'font.size': 24})
fig2, ax2 = plt.subplots(figsize=(10,6))

glabel = ["H1","H2","H3","H4","H5","H6","H7","H8"]
mcolors = ["maroon","firebrick","red","orange","green","chartreuse","blue","indigo",]
# myscans = [I_wvl_temp[range(0,138,2)],I_wvl_temp[range(1,138,2)]]
# mytemps = [temp[range(0,138,2)],temp[range(1,138,2)]]
# myPDref = [PD_values[range(0,138,2)],PD_values[range(1,138,2)]]
myscans =[I_wvl_temp]
mytemps = [angle]
myindices = [0]
for pp in myindices:
    tscan = myscans[pp]
    tscan[0:90,:] = np.nan
    tscan[364:454] = np.nan
    tscan[205,:] = np.nan
    tscan[502,:] = np.nan

    ttemp = np.arange(len(temp))#mytemps[pp]
    # mypos = ([400,round(447),round(627),810,1030])
    mypos = ([3200,1600,1030,810,round(627),530,round(447),400])
    mywvlrange = [0,0,100,75,40,40,30,30]

    # select harmonics for plot legend
    leg_str = [ mypos[i] for i in harm_number]
    leg_str = list(map(str,leg_str))
    
    Apeak = np.zeros((len(ttemp),len(mypos)))
    Centromass = np.zeros((len(ttemp),len(mypos)))

    for pos in mypos:
        myrange = 10
        rangeind0 = np.logical_and(wvl>=(pos-myrange),wvl<=(pos+myrange)) 
        
        lll = mypos.index(pos)
        rangeind1 = np.logical_and(wvl>=(pos-mywvlrange[lll]),wvl<=(pos+mywvlrange[lll])) 
        for ii in range(0,len(ttemp)):
            Apeak[ii,mypos.index(pos)] = sum(tscan[ii,rangeind0]) 
            boxfct = 1*rangeind1
            Centromass[ii,mypos.index(pos)] = np.nan_to_num(center_of_mass(abs(tscan[ii,:])*boxfct),nan=1.0) 

            
    # tttemp,tttemp_std = av_DataPerTemp(ttemp)
  
    # Apeak_norm = np.zeros((len(ttemp),len(mypos)))
    # Apeak_normav = np.zeros((len(tttemp),len(mypos)))
    # Apeak_std = np.zeros((len(tttemp),len(mypos)))

    # for ii in range(0,len(Apeak[0,:])):
        # Apeak_normav[:,ii] = av_DataPerTemp((Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii]))[0]
        # Apeak_std[:,ii] = av_DataPerTemp((Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii]))[1]
        # Apeak_norm[:,ii] = abs(Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii])

    # ax2.plot(temp,Apeak[:,harm_number]-,lw=2)
    # for i in harm_number:
        # ax2.plot(temp,Apeak[:,i]-Apeak[:,0],lw=2)
    # ax2.scatter(ttemp,0.1+Apeak_norm[:,4],lw=2,alpha=0.2,color=mcolors[pp])
    # ax2.errorbar(tttemp,0.1+Apeak_normav[:,4],xerr=tttemp_std,yerr=Apeak_std[:,4],fmt='o',label=glabel[0],color=mcolors[0])
    # ax2.plot(ttemp,smooth(Apeak_norm[:,3],5),lw=2,alpha=1,color=mcolors[pp])

    ax2.scatter(ttemp,Apeak[:,2],marker='+',lw=2,alpha=1,label=glabel[2],color=mcolors[2])
    # ax2.plot(ttemp,0.5*smooth(Apeak_norm[:,2],5),lw=2,alpha=1,color=mcolors[1])

    ax2.scatter(ttemp,40*Apeak[:,4],marker='^',lw=2,alpha=1,label=glabel[4],color=mcolors[4])
    # ax2.errorbar(tttemp,0.7*Apeak_normav[:,2],xerr=tttemp_std,yerr=0.7*Apeak_std[:,2],fmt='^',label=glabel[1],color=mcolors[1])

    ax2.scatter(ttemp,800*Apeak[:,6],marker='x',lw=2,alpha=1,label=glabel[6],color=mcolors[6])
    # ax2.errorbar(tttemp,0.2*Apeak_normav[:,1],xerr=tttemp_std,yerr=0.2*Apeak_std[:,1],fmt='x',label=glabel[2],color=mcolors[2])


    # ax2.plot(ttemp,0.8*smooth(Apeak_norm[:,1],5),lw=2,alpha=1,color=mcolors[pp])
   
# ax2.scatter(angle,Apeak_norm[:,2],marker='^',lw=2,alpha=0.2,color=mcolors[pp+1])

ax2.set_ylabel('Harm.Intensity (norm.)')
ax2.set_xlabel('HWP angle [°]')
ax2.grid(b=True, which='both')
ax2.legend(loc='upper right')
plt.tight_layout()
# ax2.set_ylim([0,1.2])
# plt.savefig(("230428_scan8_0"+"TiSe2_HHG_HWP"+".png"),dpi=(250))



#%%Center of mass test
fig22, ax22 = plt.subplots(figsize=(15,6))

for pp in mypos:
    iindex = mypos.index(pp)
    com_wvlt = wvl[np.array(np.round(Centromass[:,iindex]),dtype=int)]
    # ax22.plot(com_wvlt/max(com_wvlt),"o",label=str(pp))
    ax22.plot(smooth((com_wvlt/np.mean(com_wvlt))**2+iindex/50,5),label=str(pp))

plt.legend()
plt.xlabel("angle")
plt.tight_layout()
#%%<erge H3, H5, H7

repeats = round(len(mynames)/len(angle))
rep_tot = len(angle)*repeats
# xmeas = np.pad(temp,(0,rep_tot-len(mynames)))
# temp2 = np.resize(xmeas,(9,91))
plt.rcParams.update({'font.size': 24})
fig3, ax3 = plt.subplots(figsize=(10,6))
# mcolors = ["red","orange","blue","red","green",]

filetosave = []
filetosave.append(angle)

for ii in harm_number:
    hh_tot = Apeak[:,ii]
    # if (ii==6):
    #     hh_tot = Apeak_norm[:,ii]*0.5

    hh_pad = np.pad(hh_tot,(0,rep_tot-len(mynames)))
    hh_wrap = np.resize(hh_pad,(repeats,len(angle)))
    # hh_wrap = updated_hh_wrap1
    # for jj in np.arange(repeats):
    # for jj in [1,2,3,4,5]:
    for jj in [1,2,3]:
    # for jj in [0,1,2,3]:
        
        ax3.scatter(angle*2-min(angle*2),hh_wrap[jj,:],marker='o',lw=1,alpha=0.3,color=mcolors[ii])
        # ax3.scatter(angle*2-min(angle*2),hh_wrap[jj,:],marker='o',lw=1,alpha=0.3,color=mcolors[ii])
    hh_wrap = np.delete(hh_wrap, 5, axis=0)
    hh_wrap = np.delete(hh_wrap, 4, axis=0)
    hh_wrap = np.delete(hh_wrap, 0, axis=0)
    hh_av = np.nanmean(hh_wrap,axis=0)#.mean(0)
    hh_std = np.nanstd(hh_wrap,axis=0)#.std(0)

    # ax3.scatter(angle*2-min(angle*2),hh_av,marker='square',lw=2,alpha=1,label=glabel[ii],color=mcolors[ii])
    # ax3.errorbar(angle*2-min(angle*2),smooth(hh_av,3),yerr=hh_std,fmt='o',label=glabel[ii],color=mcolors[ii])
    ax3.errorbar(angle*2-min(angle*2),hh_av,yerr=hh_std,fmt='o',label=glabel[ii],color=mcolors[ii])
    # ax3.errorbar(angle*2-min(angle*2),hh_av,yerr=hh_std,fmt='o',label=glabel[ii],color=mcolors[ii])
    filetosave.append(hh_av)
    filetosave.append(hh_std)


ax3.grid(b=True, which='both')
ax3.set_ylabel('Harm.Intensity (norm.)')
ax3.set_xlabel('Pol. angle [°]')
# ax3.set_ylim([0,1.1])
ax3.legend(loc='upper right')

plt.tight_layout()

# np.savetxt('scan07_14K_HHlineouts.txt',(np.transpose(filetosave)),header='HWP,H3,H3Std,H5,H5Std,H7,H7Std') # not a temp scan!
