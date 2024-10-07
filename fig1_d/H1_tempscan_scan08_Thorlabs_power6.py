
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:02:01 2020
This code plots data from Cryo Temperature scan software


Run in sections!

@author: ityulnev

v3. Lenard 2023.02.28.
pos index has no meaning, just a second measurement: I can average the data

v5. Lenard 2023.05.07.
Thorlabs power meter data is read for 3.2um reflection 
saving HH data with pandas
saving reflection data with savetxt

v6. Lenard 2023.05.10.
HH data interpolate + sinus fit and subtract

v.8 Lenard 2023.05.10.
Reflectivity change as a function of temperature

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

from scipy.interpolate import interp1d
from scipy import optimize

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

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": numpy.max(pcov), "rawres": (guess,popt,pcov)}


#%% Build filepath, load data 

# genpath = Path(r"C:/Users/Lenard/ownCloud/YBCO_210504/") # SET GENERAL PATH HERE 
# pathtext = r"C:\Users\Lenard\ownCloud\YBCO_210511\scan1/"
# pathtext = r"C:\Users\Lenard\ownCloud\YBCO_210518\scan1/"
# pathtext = r"C:\Users\Lenard\ownCloud\YBCO_210521\scans\scan01/"
# genpath = Path(r"C:\Users\Lenard\ownCloud\20230504\3um_TiSe2_HHG")
genpath = Path(r"D:\Dropbox\TiSe2_CDW\analysis\230505_H1")


# filenames_path = r"" # SET FOLDER NAME HERE!
filenames_spec = r"*\scan_*.txt" # FINDS ALL .txt FILES OR SEARCHES FOR KEYWORDS *KEYWORD*.txt
filenames = filenames_spec

file_dict, mynames = load_fileToDictionary(genpath,r"*\scan08_S*.txt") #S* is Spectra
file_dict2, mynames2 = load_fileToDictionary(genpath,r"*\scan08_W*.txt") #W* is Wavelength
file_dict3, mynames3 = load_fileToDictionary(genpath,r"*\scan08_P*.txt") #W* is Wavelength


#%% read Thorlabs power values
##Time-Power correction from input laser av. power

power_ref = pd.read_csv(r"D:\Dropbox\TiSe2_CDW\analysis\230505_H1\Thorlabs_power6.csv",header=12)

t_time = power_ref.iloc[:,2]
p_ref0 = power_ref.iloc[:,3]
p_ref = np.array(p_ref0)*1000
p_ref[~np.isfinite(p_ref)] = 0   #replace Nan and Inf with "0"

t_timeabs = []
for ii in t_time:
    datet = datetime.strptime(ii, " %H:%M:%S.%f")
    total_seconds = datet.microsecond/1000000 + datet.second + datet.minute*60 + datet.hour*3600
    t_timeabs.append(total_seconds)

#%% Thorlabs time axis is intepolated to our measurmenet time-stamps
tt1 = pd.read_csv(r"D:\Dropbox\TiSe2_CDW\analysis\230505_H1\scans\scan08_TEMPERATURES.txt",header=None,sep=" |\t")
tt_time = tt1.iloc[:,3]
tt_temp = []
tt_timeabs = []
nn = 0
for ii in tt_time:
    datet = datetime.strptime(ii, "%H:%M:%S.%f")
    total_seconds = datet.microsecond/1000000 + datet.second + datet.minute*60 + datet.hour*3600
    tt_timeabs.append(total_seconds)
    tt_temp.append(float(tt1.iloc[nn,5][:-1]))
    nn += 1

p_refav = np.zeros(len(p_ref))
range_roll = [2,60]

p_ref_temp = p_ref
p_ref_temp[np.where(p_ref<1)]=np.nan
# p_ref_mask = np.ma.MaskedArray(p_ref_temp, mask=(np.array(p_ref_temp) == 0))

arr_roll = []

for i in range(range_roll[0],range_roll[1]):
    p_refav = p_refav + (np.roll(p_ref_temp,i))
    arr_roll.append(np.roll(p_ref_temp,i))
arr_roll = np.array(arr_roll)
p_refav1 = p_refav/(i-1)#/max(p_refav)*max(p_ref) 
# p_refav /= (range_roll[1]-range_roll[0])
    
# refp_int = interp1d(t_timeabs, np.roll(p_ref,2), kind='linear')
refp_int = interp1d(t_timeabs, p_refav1, kind='linear') #Thorlabs t_timeabs array has more time data
pref = refp_int(np.array(tt_timeabs)-1)
# pref=pref/pref[0]
plt.rcParams.update({'font.size': 24})
fig01, ax01 = plt.subplots(figsize=(10,6))
ax01.plot(t_timeabs,p_ref,'.-',alpha=0.2,label='raw')
# ax01.plot(t_timeabs,-p_ref/max(p_ref),'.-',alpha=0.2,label='raw')
# ax01.plot(t_timeabs,(refp_int(t_timeabs)),'.-',label='interraw')
ax01.plot(t_timeabs,p_refav1,'.-',alpha=0.2,label='raw rolled_av')
ax01.plot(tt_timeabs,pref,'o-',lw=3,alpha=1.0,label='interp1d')
# ax01.plot(tt_timeabs,pref/max(pref),'o-',lw=3,alpha=1.0,label='interp1d')
ax01.set_ylabel('Intensity (norm.)')
ax01.set_xlabel('time [sec]')
ax01.grid(b=True, which='both')
ax01.legend(loc='upper right')
plt.tight_layout()

tt_joined = [[a, b] for a, b in sorted(zip(tt_temp, pref))]
tt_jonedAr = np.array(tt_joined)


#%% find peaks and position
from scipy.signal import find_peaks
plt.rcParams.update({'font.size': 24})
fig01, ax01 = plt.subplots(figsize=(10,6))

# p_scan=p_ref
# p_scan[np.where(p_ref<5)]=np.nan
# p_scan_mask = np.ma.MaskedArray(p_scan, mask=(np.array(p_scan) == np.nan))

fp_in = np.nanmean(arr_roll,axis=0)
fp_std = np.nanstd(arr_roll,axis=0)

peaks, properties = find_peaks(fp_in,width=0.1,prominence=0.1,distance=100, height=0)
properties["prominences"], properties["widths"]

# d_ip = properties["right_ips"]-properties["left_ips"]
# peak_mids = np.array(np.round(properties["left_ips"]+d_ip/2),dtype=int)
plt.plot(p_ref)

# plt.plot(fp_in,'--')
plt.errorbar(peaks, fp_in[peaks],yerr=fp_std[peaks],capsize=3,fmt="o")
# plt.vlines(x=peaks, ymin=fp_in[peaks] - properties["prominences"],
            # ymax = fp_in[peaks], color = "C1")
plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
            xmax=properties["right_ips"], color = "C1")
plt.show()


#%% Errobars
plt.rcParams.update({'font.size': 24})
fig01, ax01 = plt.subplots(figsize=(10,6))

plt.errorbar(t_timeabs,np.mean(arr_roll,axis=0),yerr=np.std(arr_roll,axis=0),capsize=3,fmt="o",color="red")


#%%Plot THorlabs intesity as a function of temperature
fig1a, ax1a = plt.subplots(figsize=(15,10))
ax1a.plot(tt_temp,pref,'.-',lw=3,label='H1 Reflected')
# ax1a.plot(tt_temp,pref/pref.max(),'.-',lw=3,label='H1 Reflected')

ax1a.errorbar(tt_temp,fp_in[peaks],yerr=fp_std[peaks],capsize=3,fmt="o")

ax1a.grid(b=True, which='both')
ax1a.set_ylabel('Reflected H1 Intensity (norm.)')
ax1a.set_xlabel('Temperature [K]')
# ax3.set_ylim([0,1.1])
ax1a.legend(loc='upper right')
plt.tight_layout()

#temperature, power, std.dev.
# np.savetxt('H1refl_scan08_pow6_tempscan.txt',np.transpose([tt_temp,fp_in[peaks],fp_std[peaks]]))


#%%Build Iwwl_temp 2D array
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
    Ispec_smoothed = smooth(Ispec,10)
    I_wvl_temp.append(Ispec_smoothed)
    
    # Ispec = file_dict[names][1]
    # Ispec_smoothed = smooth(Ispec,20)
    # I_wvl_temp2.append(Ispec_smoothed)

    temp.append(float(re.findall(r'\d+\.\d+', names)[-1]))
print("From HWP Position: ", "0", " degree")
I_wvl_temp = np.array(I_wvl_temp)
# I_wvl_temp2 = np.array(I_wvl_temp2)


temp = np.array(temp)   


#%% Plot I_wvl_temp 2D array
plt.rcParams.update({'font.size': 13})
# gcmap = cm.copper  # global colormap
gcmap = cm.jet  # global colormap
mycm = np.linspace(0, 1.,len(mynames_sorted) )
colors = [ gcmap(x) for x in mycm ]
fig3, ax3 = plt.subplots(figsize=(8,6))

#2D Plot
ax3.pcolormesh(wvl, temp,np.log(I_wvl_temp-np.min(I_wvl_temp)), shading='auto')# shading='gouraud'
ax3.set_xlabel('Wavelength [nm]')
ax3.set_ylabel('Temperature [K]')
ax3.set_title("Temperature scan")
plt.vlines(1035, 294, 15)
plt.vlines(1053, 294, 15)

#%% Plot I_wwl_temp as a function of angle
if len(angle)*len(wvl)>0:
    plt.rcParams.update({'font.size': 13})
    # gcmap = cm.copper  # global colormap
    gcmap = cm.jet  # global colormap
    mycm = np.linspace(0, 1.,len(mynames_sorted) )
    colors = [ gcmap(x) for x in mycm ]
    fig4, ax4 = plt.subplots(figsize=(8,6))
    
    #2D Plot
    # ax4.pcolormesh(wvl, angle,np.log(I_wvl_temp-np.min(I_wvl_temp)), shading='auto',cmap="RdBu")# shading='gouraud'
    ax4.set_xlabel('Wavelength [nm]')
    ax4.set_ylabel('HWP angle [°]')
    ax4.set_title("HWP scan")
    # plt.savefig(("230428_scan8_5K"+"TiSe2_HWP"+".png"),dpi=(250))

#%% Semilog 1D plot for all temperatures
fig4, ax4 = plt.subplots(figsize=(8,6))
aa = 0
for i in temp:
    # im=ax4.semilogy(wvl,smooth(I_wvl_temp[aa,:],10),c=colors[aa],lw=2,label=(str(i)+' K'))
    im=ax4.plot(wvl,smooth(I_wvl_temp[aa,:],10),c=colors[aa],lw=2,label=(str(i)+' K'))
    aa = aa+1
ax4.set_xlabel('Wavelength [nm]')
ax4.set_ylabel('Spectral intensity [arb.u.]')
# ax4.legend()
norm = mpl.colors.Normalize(vmin=temp.min(), vmax=temp.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
# cmap.set_array([])
cbar=fig4.colorbar(cmap,ticks=temp)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(10)

ax4.set_xlim(250,1100)
# ax4.set_ylim(-0.001,1.2*np.max(I_wvl_temp[:,:]))
# plt.vlines(1036.1, 0, 33000)
# plt.vlines(1053.5, 0, 33000)

ax4.set_yscale('log')

plt.show()
plt.grid(b=True, which='both')
# plt.savefig(("230427_scan1"+"TiSe2_H5_PeaksLin_norm"+".png"),dpi=(250))




#%%
# plt.savefig((mynames[0].parts[-3]+".png"),dpi=(250))
# plt.savefig((mynames[0].parts[-3]+"_logy.png"),dpi=(250))
# plt.savefig((mynames[0].parts[-1].replace(".txt","")+"_sqrt"+".png"),dpi=(250))


#%%

def av_DataPerTemp(x_array,mystep=15):
    x_av = []
    x_std = []
    for ii in range(0,len(x_array),mystep):
        print(ii)
        x_av.append(np.average(x_array[ii:(ii+mystep-1)])) 
        x_std.append(np.std(x_array[ii:(ii+mystep-1)]))
    return np.array(x_av), np.array(x_std)



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
    ttemp = np.arange(len(temp))#mytemps[pp]
    # mypos = ([400,round(447),round(627),810,1030])
    mypos = ([3200,1600,1030,810,round(627),530,round(447),400])

    # select harmonics for plot legend
    leg_str = [ mypos[i] for i in harm_number]
    leg_str = list(map(str,leg_str))
    
    Apeak = np.zeros((len(ttemp),len(mypos)))
    for pos in mypos:
        myrange = 10
        rangeind0 = np.logical_and(wvl>=(pos-myrange),wvl<=(pos+myrange)) 
        # rangeind_l = wvl.where(400-myrange)
        for ii in range(0,len(ttemp)):
            Apeak[ii,mypos.index(pos)] = sum(tscan[ii,rangeind0])    
    # tttemp,tttemp_std = av_DataPerTemp(ttemp)
  
    Apeak_norm = np.zeros((len(ttemp),len(mypos)))
    # Apeak_normav = np.zeros((len(tttemp),len(mypos)))
    # Apeak_std = np.zeros((len(tttemp),len(mypos)))

    for ii in range(0,len(Apeak[0,:])):
        # Apeak_normav[:,ii] = av_DataPerTemp((Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii]))[0]
        # Apeak_std[:,ii] = av_DataPerTemp((Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii]))[1]
        Apeak_norm[:,ii] = abs(Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii])

    # ax2.plot(temp,Apeak[:,harm_number]-,lw=2)
    # for i in harm_number:
        # ax2.plot(temp,Apeak[:,i]-Apeak[:,0],lw=2)
    # ax2.scatter(ttemp,0.1+Apeak_norm[:,4],lw=2,alpha=0.2,color=mcolors[pp])
    # ax2.errorbar(tttemp,0.1+Apeak_normav[:,4],xerr=tttemp_std,yerr=Apeak_std[:,4],fmt='o',label=glabel[0],color=mcolors[0])
    # ax2.plot(ttemp,smooth(Apeak_norm[:,3],5),lw=2,alpha=1,color=mcolors[pp])

    ax2.scatter(ttemp,Apeak_norm[:,2],marker='+',lw=2,alpha=1,label=glabel[2],color=mcolors[2])
    # ax2.plot(ttemp,0.5*smooth(Apeak_norm[:,2],5),lw=2,alpha=1,color=mcolors[1])

    ax2.scatter(ttemp,Apeak_norm[:,4],marker='^',lw=2,alpha=1,label=glabel[4],color=mcolors[4])
    # ax2.errorbar(tttemp,0.7*Apeak_normav[:,2],xerr=tttemp_std,yerr=0.7*Apeak_std[:,2],fmt='^',label=glabel[1],color=mcolors[1])

    ax2.scatter(ttemp,Apeak_norm[:,6],marker='x',lw=2,alpha=1,label=glabel[6],color=mcolors[6])
    # ax2.errorbar(tttemp,0.2*Apeak_normav[:,1],xerr=tttemp_std,yerr=0.2*Apeak_std[:,1],fmt='x',label=glabel[2],color=mcolors[2])


    # ax2.plot(ttemp,0.8*smooth(Apeak_norm[:,1],5),lw=2,alpha=1,color=mcolors[pp])
   
# ax2.scatter(angle,Apeak_norm[:,2],marker='^',lw=2,alpha=0.2,color=mcolors[pp+1])

ax2.set_ylabel('Harm.Intensity (norm.)')
ax2.set_xlabel('HWP angle [°]')
ax2.grid(b=True, which='both')
ax2.legend(loc='upper right')
plt.tight_layout()
ax2.set_ylim([0,1.2])
# plt.savefig(("230428_scan8_0"+"TiSe2_HHG_HWP"+".png"),dpi=(250))

#%% HH intensity as a function of polarization angle 

repeats = round(len(mynames)/len(angle))
rep_tot = len(angle)*repeats
# xmeas = np.pad(temp,(0,rep_tot-len(mynames)))
# temp2 = np.resize(xmeas,(9,91))
plt.rcParams.update({'font.size': 24})
fig3, ax3 = plt.subplots(figsize=(10,6))
# mcolors = ["red","orange","blue","red","green",]

hhav_arr = []
hhstd_arr = []
for ii in harm_number:
# for ii in [2]:
    hh_tot = Apeak_norm[:,ii]
    if (ii==6):
        hh_tot = Apeak_norm[:,ii]*0.5

    hh_pad = np.pad(hh_tot,(0,rep_tot-len(mynames)))
    hh_wrap = np.resize(hh_pad,(repeats,len(angle)))
    for jj in np.arange(repeats):
        # ax3.scatter(angle*2-min(angle*2),hh_wrap[jj,:],marker='o',lw=1,alpha=0.3,color=mcolors[ii])
        ax3.scatter(2*angle,hh_wrap[jj,:],marker='o',lw=1,alpha=0.3,color=mcolors[ii])
    hh_av = hh_wrap.mean(0)
    hh_std = hh_wrap.std(0)

    # ax3.scatter(angle*2-min(angle*2),hh_av,marker='square',lw=2,alpha=1,label=glabel[ii],color=mcolors[ii])
    ax3.errorbar(2*angle,smooth(hh_av,3),yerr=hh_std,fmt='o',label=glabel[ii],color=mcolors[ii])
    hhav_arr.append(smooth(hh_av,3))
    hhstd_arr.append(hh_std)


ax3.grid(b=True, which='both')
ax3.set_ylabel('Harm.Intensity (norm.)')
ax3.set_xlabel('Pol. angle [°]')
ax3.set_ylim([0,1.1])
ax3.legend(loc='upper right')

plt.tight_layout()


# np.savetxt('Pol_scan_H3.txt',np.transpose([angle,smooth(hh_av,3),hh_std]))
# plt.savefig(("230428_scan10_14K"+"TiSe2_HHG_HWPscan_smooth3"+".png"),dpi=(250))



#save Harmonics vs. Polarization angle
# =============================================================================
# my_pd=pd.DataFrame(data=[angle,np.transpose(hhav_arr[0]),
#                          np.transpose(hhav_arr[1]),
#                          np.transpose(hhav_arr[2]),
#                          np.transpose(hhstd_arr[0]),
#                          np.transpose(hhstd_arr[1]),
#                          np.transpose(hhstd_arr[2])]).T 
# my_pd.columns=['angle','\tH3av','\tH5av','\tH7av','\tH3std','\tH5std','\tH7std']
# my_pd.to_csv('Pol_scan_H3H5H7_scan08_282K.txt', sep='\t', header=True, float_format='%f', index=False)
# 
# =============================================================================


#%% H1 3200nm Thorlabs Power as a function of HWP angle
# h1_wrap = np.resize(pref,(repeats,len(angle)))
# pref = pref-min(pref)
pref = pref / max(pref)


h1_wrap = np.resize(pref,(repeats,len(angle)))
h1_av = h1_wrap.mean(0)
h1_std = h1_wrap.std(0)

plt.rcParams.update({'font.size': 24})
fig4, ax4 = plt.subplots(figsize=(10,6))
for jj in np.arange(repeats):
    ax4.scatter(angle*2,h1_wrap[jj,:],marker='o',lw=1,alpha=0.3,color=mcolors[0])


# ax3.scatter(angle*2-min(angle*2),hh_av,marker='square',lw=2,alpha=1,label=glabel[ii],color=mcolors[ii])
# ax4.errorbar(angle*2-min(angle*2),smooth(h1_av,3),yerr=hh_std,fmt='o',label="H1: 3200nm",color=mcolors[0])

ax4.errorbar(angle*2,smooth(h1_av,3),yerr=h1_std,fmt='o',label="H1: 3200nm",color=mcolors[0])

ax4.grid(b=True, which='both')
ax4.set_ylabel('H1 3200 Intensity (norm.)')
ax4.set_xlabel('Pol. angle [°]')
# ax4.set_ylim([0,1.1])
ax4.legend(loc='upper right')

plt.tight_layout()

# np.savetxt('Pol_scan_H1.txt',np.transpose([angle,smooth(h1_av,3),h1_std]))

#%% H1 3200nm Thorlabs Power as a function of HWP angle: Input Field amplitude
anglee = np.linspace(angle.min(),2*angle.max(),1000)
anglee_rad = anglee*np.pi/180.0
h1intp = interp1d(2*angle,smooth(h1_av,3))
IPP = 31/np.sqrt(2)*np.sqrt(2*np.sin(anglee_rad)*np.sin(anglee_rad)+np.cos(anglee_rad)*np.cos(anglee_rad))
IPP_0 = IPP/max(IPP)
ISS_0 = (np.cos(anglee_rad))*(np.cos(anglee_rad))#/np.sqrt(2)
ax3.plot(anglee,ISS_0, lw=2, c='blue',label='ISS')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
ax4.plot(anglee,IPP_0, lw=2, c='blue',label='IPP')
# corr = np.sin(anglee_rad)

plt.rcParams.update({'font.size': 48})
fig5, ax5 = plt.subplots(figsize=(15,10))
# fig5 = plt.figure()
# ax5 = fig5.add_subplot(projection='polar')
# fig5, ax5 = plt.subplots(subplot_kw={'projection': 'polar'})
ax5.scatter(2*angle, h1_av, marker='o',lw=1,alpha=0.3,color=mcolors[0],label='H1refl')
ax5.plot(anglee,h1intp(anglee), '.-',lw=2,color=mcolors[0], label='H1refl_interp')

ax5a = ax5.twinx()
ax5a.plot(anglee,IPP, lw=2, c='blue',label='IPP')
ax5a.set_ylabel('IPP: Input Intensity \n parallel to the sample surface (mW)')


ax5.grid(b=True, which='both')
ax5.set_ylabel('Reflected H1 Intensity (mW)')
ax5.set_xlabel('Pol. angle [°]')
# ax3.set_ylim([0,1.1])
ax5.legend(loc='upper left')
ax5a.legend(loc='upper right')

plt.tight_layout()

#%%manipulate HH data
#interpolate
anglee = np.linspace(angle.min(),angle.max(),1000)
hhh3 = interp1d(angle,hhav_arr[0])
hhh5 = interp1d(angle,hhav_arr[1])
res3 = fit_sin(anglee,hhh3(anglee))
res5 = fit_sin(anglee,hhh5(anglee))

plt.rcParams.update({'font.size': 24})
fig3, ax3a = plt.subplots(figsize=(10,6))
ax3a.scatter(angle, hhav_arr[0], label='H3')
ax3a.scatter(angle, hhav_arr[1], label='H5')
ax3a.plot(anglee,hhh3(anglee), label='H3 interp')
ax3a.plot(anglee,hhh5(anglee), label='H5 interp')
ax3a.plot(anglee, res3["fitfunc"](anglee), "r-", label="H3 fit sin", linewidth=2)
ax3a.plot(anglee, hhh3(anglee)-res3["fitfunc"](anglee), "r-", label="H3interp-fitsin", linewidth=2)
ax3a.plot(anglee, hhh5(anglee)-res3["fitfunc"](anglee), "r-", label="H3interp-fitsin", linewidth=2)

ax3a.grid(b=True, which='both')
ax3a.set_ylabel('Harm.Intensity (norm.)')
ax3a.set_xlabel('Pol. angle [°]')
# ax3.set_ylim([0,1.1])
ax3a.legend(loc='upper right')

plt.tight_layout()


#%% normalize for 530 peak
# =============================================================================
# plt.rcParams.update({'font.size': 13})
# fig21, ax21 = plt.subplots(figsize=(15,6))
# ax21.plot(temp,Apeak_norm[:,3],label=str(mypos[3]))
# ax21.plot(temp,Apeak_norm[:,1],label=str(mypos[1]))
# ax21.plot(temp,Apeak_norm[:,3]/Apeak_norm[:,1],label="{}/{}".format(mypos[3],mypos[1]))
# 
# ax21.set_ylabel('Integrated Harmonic Intensity (norm.)')
# ax21.set_xlabel('Temperature [K]')
# ax21.grid(b=True, which='both')
# ax21.legend(loc='upper right')
# plt.show()
# =============================================================================
# plt.savefig((mynames[0].parts[-3]+"harmInt_norm"+".png"),dpi=(250))

# =============================================================================
# #%% FFT
# # harm_number = [2,4] #choose which harmoics to plot
# 
# plt.rcParams.update({'font.size': 13})
# fig3, ax3 = plt.subplots(figsize=(15,6))
# 
# for ii in harm_number:
#     [Af,ff] = myfft(np.transpose(Apeak_norm[:,ii]),temp)#*1e-3
#     Af[round(len(Af)/2)]=0
#     posperiod = 1/ff
#     posperiod[np.isnan(posperiod)] = 0
#     ax3.plot(posperiod,(abs(Af)**2),label=mypos[ii])
#     
# ax3.set_ylabel('Integrated Harmonic Intensity Spectrum (arb.u.)')
# ax3.set_xlabel('HWP angle [deg]')
# ax3.legend()
# ax3.set_xlim(0,360)
# # ax3.set_ylim(0,2e7)
# 
# # plt.savefig((mynames[0].parts[-3]+"noise_spec2"+".png"),dpi=(250))
# =============================================================================



