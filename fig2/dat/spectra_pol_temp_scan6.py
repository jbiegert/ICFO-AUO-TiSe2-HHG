
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

genpath = Path(r"D:\Dropbox\TiSe2_CDW\analysis\230428_TempScan\230425_scans\dat")

file_dict, mynames = load_fileToDictionary(genpath,r"scan6*_S*.txt") #S* is Spectra
file_dict2, mynames2 = load_fileToDictionary(genpath,r"scan6*_W*.txt") #W* is Wavelength

#%% Time-Power correction from input laser av. power

# power_ref = pd.read_csv(r"D:\Dropbox\TiSe2_CDW\analysis\230428_TempScan\230425_scans\Thorlabs_power1.csv",header=12)

# t_time = power_ref.iloc[:,2]
# p_ref = power_ref.iloc[:,3]

# t_timeabs = []
# for ii in t_time:
#     datet = datetime.strptime(ii, " %H:%M:%S.%f")
#     total_seconds = datet.microsecond/1000000 + datet.second + datet.minute*60 + datet.hour*3600
#     t_timeabs.append(total_seconds)

# #%% Time and powers during measurement
# tt1 = pd.read_csv(r"D:\Dropbox\TiSe2_CDW\analysis\230428_TempScan\230425_scans\dat\scan1TEMPERATURES.txt",header=None,sep=" |\t")
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
I_wvl_temp = []
I_wvl_temp2 = []

mynames_sorted = mynames
mynames_sorted = sorted(mynames_sorted, key = lambda x: float(re.findall(r'\d+\.\d+', x)[-1]))

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
# laser power drop
I_wvl_temp[157,:] = np.nan
I_wvl_temp[368,:] = np.nan
I_wvl_temp[403,:] = np.nan
I_wvl_temp[404,:] = np.nan
I_wvl_temp[660,:] = np.nan
I_wvl_temp[787,:] = np.nan
I_wvl_temp[792,:] = np.nan
I_wvl_temp[820,:] = np.nan
I_wvl_temp[821,:] = np.nan

#%% Testing integration and units


cal_nir = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\230428_TempScan\scan2\calibration\Maya2000_Spec_Calibration\extra\VIS-NIR calibration.IrradCal",skiprows=9)
cal_uv = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\230428_TempScan\scan2\calibration\Maya2000_Spec_Calibration\extra\UV-VIS calibration.IrradCal",skiprows=9)
cal_uvnir = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\230428_TempScan\scan2\calibration\Maya2000_Spec_Calibration\extra\UV-VIS-NIR.IrradCal",skiprows=9)
cal_ref = np.loadtxt(r"D:\Dropbox\TiSe2_CDW\analysis\230428_TempScan\scan2\calibration\Maya2000_Spec_Calibration\calibration_file_cal_with_UVfiber.dat")

aa = cal_ref/cal_uvnir

fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.plot(cal_nir[:,0],cal_nir[:,1],color="blue")# shading='gouraud'
ax3.plot(cal_uv[:,0],cal_uv[:,1]*1.8,color="red")# shading='gouraud'
ax3.plot(cal_uvnir[:,0],cal_uvnir[:,1],"--",color="black")# shading='gouraud'



spec01 = I_wvl_temp[500,:]
calib0H3 = 1.17346e-10
calib0H5 = 0.000394095


kg3_h3 = 0.07916756060931129

np.trapz(spec01[1845:1892],wvl[1845:1892]*1e-9)
np.trapz(spec01[1700:-1],wvl[1700:-1]*1e-9)*(60/200)*calib0H3
sum(spec01[1700:-1])*(0.417e-9)
sum(spec01[1700:-1])
sum(spec01[1845:1892])*(0.417e-9) # factor 2.35 for full integral H3
sum(spec01[914:959]) #H5
sum(spec01[900:980]) #H5



fig3, ax3 = plt.subplots(figsize=(8,6))
# ax3.plot(wvl,spec01,color="red")# shading='gouraud'
ax3.plot(wvl,spec01*cal_uvnir[:,1]*(1/(160000)),color="black")# shading='gouraud'
ax3.plot(wvl,spec01*cal_ref[:,1]*(1/(160000)),"--",color="red")# shading='gouraud'
ax3.set_yscale('log')

np.trapz(1e-6*spec01[1700:-1]*(1/200e-3)*cal_uvnir[1700:-1,1]*(1/160000)/kg3_h3,wvl[1700:-1])
np.trapz(1e-6*spec01[1700:-1]*(60/200)*cal_uvnir[1700:-1,1]*(1/160000),wvl[1700:-1]*1e-9)
# np.trapz(1e-6*spec01[1700:-1]*cal_uvnir[1700:-1,1]*(1000/(200*160000)),wvl[1700:-1]*1e-9)
a1 = np.trapz(spec01[1700:-1]*cal_ref[1700:-1,1]*(60/200)*(1/(160000))/kg3_h3/0.97/0.95,wvl[1700:-1]*1e-9)
a2 = sum(spec01[1845:1892]*cal_ref[1955,1]*(60/200)*(1/(160000))/kg3_h3/0.97/0.95)*0.4172000484564849e-9

b1 = np.trapz(spec01[900:980]*cal_ref[900:980,1]*(60/200)*(1/(160000))/kg3_h3/0.97/0.95,wvl[900:980]*1e-9)
b2 = sum(spec01[914:959]*cal_ref[965,1]*(60/200)*(1/(160000))/kg3_h3/0.97/0.95)*0.44578e-9

c1 = np.trapz((spec01[516:560]-min(spec01[516:560]))*cal_ref[516:560,1]*(60/200)*(1/(160000))/1/0.97/0.95,wvl[516:560]*1e-9)
c2 = sum((spec01[516:560]-min(spec01[516:560]))*cal_ref[559,1]*(60/200)*(1/(160000))/1/0.97/0.95)*0.457e-9


#%% Plot 
plt.rcParams.update({'font.size': 13})
# gcmap = cm.copper  # global colormap
gcmap = cm.jet  # global colormap
mycm = np.linspace(0, 1.,len(mynames_sorted) )
colors = [ gcmap(x) for x in mycm ]
fig3, ax3 = plt.subplots(figsize=(8,6))

#2D Plot
ax3.plot(wvl,smooth(I_wvl_temp[0,:]+50,20),color="red")# shading='gouraud'
ax3.plot(wvl,smooth(I_wvl_temp[-1,:]+50,20),color="b")# shading='gouraud'
ax3.set_xlabel('Wavelength [nm]')
ax3.set_yscale('log')


#%% Semilog 1D plot for all temperatures
fig4, ax4 = plt.subplots(figsize=(8,6))
aa = 0
for i in temp:
    # im=ax4.semilogy(wvl,smooth(I_wvl_temp[aa,:],10),c=colors[aa],lw=2,label=(str(i)+' K'))
    im=ax4.plot(wvl,smooth(I_wvl_temp[aa,:],20),c=colors[aa],lw=2,label=(str(i)+' K'))
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


#%% find peaks and position
from scipy.signal import find_peaks

I0 = smooth(I_wvl_temp[0,:],10)
peaks, properties = find_peaks(I0, prominence=1, width=5)
properties["prominences"], properties["widths"]
plt.plot(I0)
plt.plot(peaks, I0[peaks], "x")
plt.vlines(x=peaks, ymin=I0[peaks] - properties["prominences"],
           ymax = I0[peaks], color = "C1")
plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
           xmax=properties["right_ips"], color = "C1")
plt.show()

#%%
# plt.savefig((mynames[0].parts[-3]+".png"),dpi=(250))
# plt.savefig((mynames[0].parts[-3]+"_logy.png"),dpi=(250))
# plt.savefig((mynames[0].parts[-1].replace(".txt","")+"_sqrt"+".png"),dpi=(250))


#%%

def av_DataPerTemp(x_array,mystep=15,myaxis=0):
    x_av = []
    x_std = []
    for ii in range(0,len(x_array),mystep):
        # print(ii)
        x_av.append(np.nanmean(x_array[ii:(ii+mystep-1)],axis=myaxis)) 
        x_std.append(np.nanstd(x_array[ii:(ii+mystep-1)],axis=myaxis))
    return np.array(x_av), np.array(x_std)



#%%Take lineouts
# ToDo
harm_number = [3,5,7] #choose which harmoics to plot
plt.rcParams.update({'font.size': 24})
fig2, ax2 = plt.subplots(figsize=(10,6))
range_hh=[]
glabel = ["H3","H5","H7"]
mcolors = ["red","orange","blue"]
# myscans = [I_wvl_temp[range(0,138,2)],I_wvl_temp[range(1,138,2)]]
# mytemps = [temp[range(0,138,2)],temp[range(1,138,2)]]
# myPDref = [PD_values[range(0,138,2)],PD_values[range(1,138,2)]]
myscans =[I_wvl_temp,I_wvl_temp2]
mytemps = [temp,temp]
myindices = [0]
for pp in myindices:
    tscan = myscans[pp]
    ttemp = mytemps[pp]
    # mypos = ([400,round(447),round(627),810,1030])
    mypos = ([[1110,1112],[3150/3,3250/3],[1550/2,1650/2],[970,1080],[756,800],[586,649],[500,560],[430,460],[380,420]])
    # select harmonics for plot legend
    leg_str = [ mypos[i] for i in harm_number]
    leg_str = list(map(str,leg_str))
    
    Apeak = np.zeros((len(ttemp),len(mypos)))
    for idx,pos in enumerate(mypos):
        # myrange = 10
        # rangeind0 = np.logical_and(wvl>=(pos-myrange),wvl<=(pos+myrange)) 
        # print(rangeind0)
        rangeind0 = np.where(np.logical_and((wvl>=pos[0]) , (wvl<=pos[1])))
        for ii in range(0,len(ttemp)):
            Apeak[ii,idx] = np.sum(tscan[ii,rangeind0]) 
        range_hh.append([round(wvl[rangeind0][0]),round(wvl[rangeind0][-1]),len(wvl[rangeind0])])
        print(round(wvl[rangeind0][0]),round(wvl[rangeind0][-1]),len(wvl[rangeind0]))
    tttemp,tttemp_std = av_DataPerTemp(ttemp)
  
    Apeak_norm = np.zeros((len(ttemp),len(mypos)))
    Apeak_normav = np.zeros((len(tttemp),len(mypos)))
    Apeak_std = np.zeros((len(tttemp),len(mypos)))
    Apeak_av = np.zeros((len(tttemp),len(mypos)))
    Apeak_avstd = np.zeros((len(tttemp),len(mypos)))

    # for ii in range(0,len(Apeak[0,:])):
    #     Apeak_normav[:,ii] = av_DataPerTemp((Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii]))[0]
    #     Apeak_std[:,ii] = av_DataPerTemp((Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii]))[1]
    #     Apeak_norm[:,ii] = abs(Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii])

    for ii in range(0,len(Apeak[0,:])):
        Apeak_av[:,ii] = av_DataPerTemp(Apeak[:,ii])[0]
        Apeak_avstd[:,ii] = av_DataPerTemp(Apeak[:,ii])[1]
        # Apeak_norm[:,ii] = abs(Apeak[:,ii]-Apeak[:,0])/max(Apeak[:,ii])




    # ax2.plot(temp,Apeak[:,harm_number]-,lw=2)
    # for i in harm_number:
        # ax2.plot(temp,Apeak[:,i]-Apeak[:,0],lw=2)
    ax2.scatter(ttemp,Apeak[:,3],lw=2,alpha=0.2,color=mcolors[pp])
    ax2.errorbar(tttemp,Apeak_av[:,3],xerr=tttemp_std,yerr=Apeak_avstd[:,3],fmt='o-',label=glabel[0],color=mcolors[0])
    # ax2.plot(ttemp,smooth(Apeak_norm[:,3],5),lw=2,alpha=1,color=mcolors[pp])

    # ax2.scatter(ttemp,0.5*Apeak_norm[:,2],marker='+',lw=2,alpha=0.5,label="Reference",color=mcolors[1])
    # ax2.plot(ttemp,0.5*smooth(Apeak_norm[:,2],5),lw=2,alpha=1,color=mcolors[1])
    ax2.scatter(ttemp,Apeak[:,4],marker='+',lw=2,alpha=0.2,color="purple")
    ax2.errorbar(tttemp,Apeak_av[:,4],xerr=tttemp_std,yerr=Apeak_avstd[:,4],fmt='+',label="H4",color="purple")



    ax2.scatter(ttemp,10*Apeak[:,5],marker='^',lw=2,alpha=0.2,color=mcolors[pp+1])
    ax2.errorbar(tttemp,10*Apeak_av[:,5],xerr=tttemp_std,yerr=10*Apeak_avstd[:,5],fmt='^-',label=glabel[1],color=mcolors[1])

    ax2.scatter(ttemp,(Apeak[:,7]),marker='x',lw=2,alpha=0.2,color=mcolors[pp+2])
    ax2.errorbar(tttemp,(Apeak_av[:,7]),xerr=tttemp_std,yerr=Apeak_avstd[:,7],fmt='x-',label=glabel[2],color=mcolors[2])


    # ax2.plot(ttemp,0.8*smooth(Apeak_norm[:,1],5),lw=2,alpha=1,color=mcolors[pp])
   

ax2.set_ylabel('Harm.Intensity (norm.)')
ax2.set_xlabel('Temperature [K]')
ax2.grid(b=True, which='both')
ax2.legend(loc='upper right')
plt.tight_layout()
# ax2.set_ylim([0,1.2])
# plt.savefig(("230428_scan2"+"TiSe2_HHG_wErrorbars"+".png"),dpi=(250))



# np.savetxt("230425_scan6_Apeak_av.txt",Apeak_av);
# np.savetxt("230425_scan6_Apeak_avstd.txt",Apeak_avstd);
# np.savetxt("230425_scan6_Apeak_temp.txt",tttemp);
# np.savetxt("230425_scan6_Apeak_tempstd.txt",tttemp_std);
# np.savetxt("230425_scan6_Apeak_temp_wvl.txt",wvl);

# np.savetxt("scan6_Apeak_norm.txt",Apeak_norm);
# np.savetxt("scan6_Apeak_normav.txt",Apeak_normav);
# np.savetxt("scan6_Apeak_std.txt",Apeak_std);
# np.savetxt("scan6_Temps.txt",ttemp);
# np.savetxt("scan6_Temps_av.txt",tttemp);
# np.savetxt("scan6_Temps_std.txt",tttemp_std);
# np.savetxt("scan6_Harms.txt",mypos);


#%% normalize for 530 peak
plt.rcParams.update({'font.size': 13})
fig21, ax21 = plt.subplots(figsize=(15,6))
ax21.plot(temp,Apeak_norm[:,3],label=str(mypos[3]))
ax21.plot(temp,Apeak_norm[:,1],label=str(mypos[1]))
ax21.plot(temp,Apeak_norm[:,3]/Apeak_norm[:,1],label="{}/{}".format(mypos[3],mypos[1]))

ax21.set_ylabel('Integrated Harmonic Intensity (norm.)')
ax21.set_xlabel('Temperature [K]')
ax21.grid(b=True, which='both')
ax21.legend(loc='upper right')
plt.show()
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


#%% Av full spectra
fullspec_av = av_DataPerTemp(I_wvl_temp,myaxis=0)

fspec_av = fullspec_av[0]
fspec_std = fullspec_av[1]

mycolors = plt.cm.coolwarm(np.linspace(0,1,len(fspec_av)))
fig5, ax5 = plt.subplots(figsize=(8,6))
# ax3.plot(wvl,spec01,color="red")# shading='gouraud'
for idx,cc in enumerate(mycolors):
    
    
    
    # ax5.plot(3130/wvl,smooth(fspec_av[idx,:],15)+30,color=cc)# shading='gouraud'
    ax5.plot(wvl,smooth(fspec_av[idx,:],15)+30,color=cc)# shading='gouraud'
ax5.set_yscale('log')




# np.savetxt("230425_scan6_Spec_avperT.txt",fspec_av)
# np.savetxt("230425_scan6_Spec_avperTstd.txt",fspec_std)
# np.savetxt("230425_scan6_Spec_avperT_temp.txt",tttemp)
