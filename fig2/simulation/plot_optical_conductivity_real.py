import proplot as pplt
import numpy as np
import h5py
import matplotlib.pyplot as plt
import elli

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

def func(x, a, b):
    return a * x / (b + x**2)


def scientific_format(x, pos):
    return f"$10^{{{int(np.log10(x))}}}$"

def smooth2(arr, span):
    if span==0:
        return arr
    
    arr = savgol_filter(arr, span * 2 + 1, 3)
    arr = savgol_filter(arr, span * 2 + 1, 3)
    return arr


pplt.rc["lines.linewidth"] = 1
colors = pplt.get_colors("default")
mycolors = plt.cm.coolwarm(np.linspace(0,1,11))
mytemps = [30,40,50,75,100,125,150,175,200,250,300] 
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True,figsize=(8, 8))

files = [
    "titanium_diselenide_optical_conductivity_v1_T_30_D1_0.1137_D2_0.1137_D3_0.1137_omega_0.4-2.8_Nk_2049_mu_-0.1645.h5",
    "titanium_diselenide_optical_conductivity_v1_T_40_D1_0.1127_D2_0.1127_D3_0.1127_omega_0.4-2.8_Nk_2049_mu_-0.1631.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_50_D1_0.1113_D2_0.1113_D3_0.1113_omega_0.4-2.8_Nk_2049_mu_-0.1613.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_75_D1_0.1066_D2_0.1066_D3_0.1066_omega_0.4-2.8_Nk_2049_mu_-0.1547.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_100_D1_0.0996_D2_0.0996_D3_0.0996_omega_0.4-2.8_Nk_2049_mu_-0.1449.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_125_D1_0.0898_D2_0.0898_D3_0.0898_omega_0.4-2.8_Nk_2049_mu_-0.1309.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_150_D1_0.0761_D2_0.0761_D3_0.0761_omega_0.4-2.8_Nk_2049_mu_-0.1109.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_175_D1_0.0557_D2_0.0557_D3_0.0557_omega_0.4-2.8_Nk_2049_mu_-0.0808.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_200_D1_0.0000_D2_0.0000_D3_0.0000_omega_0.4-2.8_Nk_2049_mu_-0.0302.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_250_D1_0.0000_D2_0.0000_D3_0.0000_omega_0.4-2.8_Nk_2049_mu_-0.0384.h5_",
    "titanium_diselenide_optical_conductivity_v1_T_300_D1_0.0000_D2_0.0000_D3_0.0000_omega_0.4-2.8_Nk_2049_mu_-0.0464.h5_",
]
ii=0
sigyy = []
omeg = []
myH1y = []
myH1x = []
myH3y = []
myH3x = []
myH5y = []
myH5x = []
myH7y = []
myH7x = []

for ifile, file_name in enumerate(files):
    f = h5py.File(file_name, "r")

    scale = np.array(f["scale"]).item()
    T = np.array(f["T"]).item()
    Delta1 = np.array(f["Delta1"]).item()
    Delta2 = np.array(f["Delta2"]).item()
    Delta3 = np.array(f["Delta3"]).item()
    N0 = np.array(f["N0"]).item()
    Nk = np.array(f["Nk"]).item()
    mu = np.array(f["mu"]).item()
    omegas = np.array(f["omegas"])
    sigmaxxs = np.array(f["sigmaxxs"])
    sigmayys = np.array(f["sigmayys"])

    # sigmaxxs_kk = elli.kkr.kkr.im2re((smooth2(np.imag(sigmaxxs),1)), omegas)
    # sigmayys_kk = elli.kkr.kkr.im2re((smooth2(np.real(sigmayys),1)), omegas)
    # sigmayys_kk_abs = elli.kkr.kkr.im2re(abs(smooth2(np.real(sigmayys),1)), omegas)
    
    sigyy.append(sigmayys)
    omeg.append(omegas)
    
    axs[0,0].plot(omegas, (np.imag(sigmayys)),linewidth=2, color=mycolors[ii],label=str(mytemps[ii]))
    axs[1,0].plot(omegas, abs((np.imag(sigmayys))),linewidth=2, color=mycolors[ii],label=str(mytemps[ii]))
    axs[0,0].set_xlim([0,3])
    axs[0,0].set_ylim([-2,1])
    tw1 = axs[0,0].twinx() 
    tw1.plot(omegas,np.real(sigmayys),":",linewidth=2, color=mycolors[ii])
    tw1.set_xlim([0,3])
    tw1.set_ylim([-2,1])  
    
    axs[0,0].legend(frameon=False,loc="lower right")
    axs[0,0].set_ylabel("Im($\sigma_{yy}$)")
    axs[1,0].set_ylabel("|Im($\sigma_{yy}$)|")
    
    
    # axs[0,1].plot(omegas, smooth2(sigmayys_kk,2), color=mycolors[ii],linewidth=2)
    # axs[1,1].plot(omegas, smooth2(sigmayys_kk_abs,2), color=mycolors[ii],linewidth=2)
    # axs[0,1].set_ylim([-2,2])
    # axs[1,1].set_ylim([-2,2])
    axs[0,1].set_ylabel("KK($\sigma_{yy}$)")

    H1 = 0.4
    ind0 = np.where(np.round(omegas,2)==np.round(H1*1,1))[0][0]
    ind3 = np.where(np.round(omegas,2)==np.round(H1*3,1))[0][0]
    ind5 = np.where(np.round(omegas,2)==np.round(H1*5,1))[0][0]
    ind7 = np.where(np.round(omegas,2)==np.round(H1*7,1))[0][0]
    myH1x.append(sigmaxxs[ind0])
    myH1y.append(sigmayys[ind0])  
    myH3x.append(sigmaxxs[ind3])
    myH3y.append(sigmayys[ind3])  
    myH5x.append(sigmaxxs[ind5])
    myH5y.append(sigmayys[ind5])  
    myH7x.append(sigmaxxs[ind7])
    myH7y.append(sigmayys[ind7])  

    # axs[1].vlines(H1, -2, 2,alpha=0.5)
    # axs[0].vlines(H1, -2, 2,alpha=0.5)
    # axs[1].vlines(H1*3, -0.5, 2,"k",alpha=0.5)
    # axs[1].vlines(H1*5, -0.5, 2,"k",alpha=0.5)
    # axs[1].vlines(H1*7, -0.5, 2,"k",alpha=0.5)


    # os.legend(frame=False, ncols=1)
    plt.tight_layout()
    ii+=1
# fig.format(suptitle=f"Optical conductivity $T={T}$")

def pop_array(myarr,indx=0):
    mylist = np.array(myarr).tolist()
    mylist.pop(indx)
    return np.array(mylist)


#%% Further analysis at harmonic frequencies
# h1arr =np.array(myH7x

plt.rcParams.update({'font.size': 18})
fig1, ax1 = plt.subplots(ncols=1,nrows=4,figsize=(5,5),sharex=True)

ax1[0].plot(mytemps,np.real(myH1x),color="k",alpha=0.5)
ax1[0].plot(mytemps,np.real(myH1y),color="k",alpha=0.5)
ax1[1].plot(mytemps,np.real(myH3x),color="k",alpha=0.5)
ax1[1].plot(mytemps,np.real(myH3y),color="k",alpha=0.5)
ax1[2].plot(mytemps,np.real(myH5x),color="k",alpha=0.5)
ax1[2].plot(mytemps,np.real(myH5y),color="k",alpha=0.5)
ax1[3].plot(mytemps,np.real(myH7x),color="k",alpha=0.5)
ax1[3].plot(mytemps,np.real(myH7y),color="k",alpha=0.5)

plt.tight_layout()
ax1[0].set_ylabel("$\sigma_r$(H1)")
ax1[1].set_ylabel("$\sigma_r$(H3)")
ax1[2].set_ylabel("$\sigma_r$(H5)")
ax1[3].set_ylabel("$\sigma_r$(H7)")

ax1[3].set_xlabel("temperature (K)")

saveH1 = np.stack([mytemps,np.real(myH1x),np.imag(myH1x)],axis=1)
saveH3 = np.stack([mytemps,np.real(myH3x),np.imag(myH3x)],axis=1)
# np.savetxt("H1_optcond_sig.txt",saveH1)
# np.savetxt("H3_optcond_sig.txt",saveH3)
#%%
from scipy import interpolate

temp0 = np.arange(10,300,1)
tck1 = interpolate.pchip_interpolate(mytemps, np.real(myH1x),temp0)
h1_intrp = tck1 # interpolate.pchip_interpolate(temp0, tck1)
tck3 = interpolate.pchip_interpolate(mytemps, np.real(myH3x),temp0)
h3_intrp =tck3# interpolate.pchip_interpolate(temp0, tck3)
tck5 = interpolate.pchip_interpolate(mytemps, np.real(myH5x),temp0)
h5_intrp = tck5#interpolate.pchip_interpolate(temp0, tck5)
tck7 = interpolate.pchip_interpolate(mytemps, np.real(myH7x),temp0)
h7_intrp = tck7#interpolate.splev(temp0, tck7)

# tct1 = interpolate.interp1d(mytemps, np.real(myH1x), fill_value="extrapolate")
# h1_intrp = tct1(temp0)
# tct3 = interpolate.interp1d(mytemps, np.real(myH3x), fill_value="extrapolate")
# h3_intrp = tct3(temp0)
# tct5 = interpolate.interp1d(mytemps, np.real(myH5x), fill_value="extrapolate")
# h5_intrp = tct5(temp0)
# tct7 = interpolate.interp1d(mytemps, np.real(myH7x), fill_value="extrapolate")
# h7_intrp = tct7(temp0)


plt.rcParams.update({'font.size': 18})
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.plot(mytemps,np.real(myH1y),"o",color="k",alpha=1)
ax1.plot(mytemps,np.real(myH3y),"o",color="r",alpha=1)
ax1.plot(mytemps,np.real(myH5y),"o",color="g",alpha=1)
ax1.plot(mytemps,np.real(myH7y)*20,"o",color="b",alpha=1)
ax1.plot(temp0,h1_intrp,color="k",alpha=1)
ax1.plot(temp0,h3_intrp,color="r",alpha=1)
ax1.plot(temp0,h5_intrp,color="g",alpha=1)
ax1.plot(temp0,h7_intrp*20,color="b",alpha=1)

plt.xlim([14,300])

saveHHi = np.stack([temp0,h1_intrp,h3_intrp,h5_intrp,h7_intrp],axis=1)
# np.savetxt("HH_interp_optcond_sigfull.txt",saveHHi)
#%% Fitting
from scipy import optimize

def porder(T,Tc,N,B,C):
    # Tc = 200
    tabv= np.where(T>Tc)
    tbel= np.where(T<=Tc)
    pord_abv = C+ T[tabv]*0
    # pord_bel =  C + B*np.tanh(A*np.real(np.sqrt(Tc/T[tbel]-1)))**2 #original
    # pord_bel =  C + B*np.tanh(A*np.real((Tc/T[tbel]-1))**N)**2
    pord_bel =  C + B*np.real(Tc-T[tbel])**N
    return np.hstack((pord_bel,pord_abv))

ord_pin= [200,0.5,1e-1,0]
s1_opt, s1_cov = optimize.curve_fit(porder, temp0[20:-1], h1_intrp[20:-1],p0=ord_pin)

plt.rcParams.update({'font.size': 18})
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.plot(temp0,h3_intrp,color="r",alpha=1)
ax1.plot(temp0,porder(temp0,*ord_pin),"--",color="r",alpha=1)

# ax1.plot(temp0,h3_intrp*h1_intrp**3,"--",color="k",alpha=1)

#%% Nonlinear suszep.
chi3 = h3_intrp*h1_intrp**3
chi5 = h5_intrp*h1_intrp**5
chi7 = h7_intrp*h1_intrp**7
chi3[np.where(chi3<=0)] = np.nan
chi5[np.where(chi5<=0)] = np.nan
chi7[np.where(chi7<=0)] = np.nan

plt.rcParams.update({'font.size': 18})
fig1, ax1 = plt.subplots(figsize=(5,5))
# ax1.plot(temp0,h1_intrp**3,"-",color="r",alpha=1)
ax1.plot(temp0,chi3,"o",color="r",alpha=1)
# ax1.plot(temp0,h1_intrp**5,"-",color="g",alpha=1)
ax1.plot(temp0,chi5,"o",color="g",alpha=1)
# ax1.plot(temp0,h1_intrp**7,"-",color="b",alpha=1)
ax1.plot(temp0,chi7,"o",color="b",alpha=1)
ax1.set_yscale('log')
ax1.set_xscale('log')
