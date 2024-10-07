from scipy.interpolate import make_interp_spline
import numpy as np
import proplot as pplt
import h5py

def scientific_format(x, pos):
    return f"$10^{{{int(np.log10(x))}}}$"

pplt.rc['lines.linewidth'] = 1
colors = pplt.get_colors("default")

fig, axs = pplt.subplots(nrows=1, ncols=3, share=False)

files = [
    "HHG_T_300_omega0_0.4_ncyc_8_A0_1.2_inv_tau_1.0_Nk_400_alpha_60.h5",
    # "HHG_T_300_omega0_0.4_ncyc_8_A0_1.2_inv_tau_1.0_Nk_400_alpha_55.h5",
]

for ifile, file_name in enumerate(files):
    f = h5py.File(file_name, "r")

    T = np.array(f['T']).item()
    Delta0 = np.array(f['Delta0']).item()
    N0 = np.array(f['N0']).item()
    Nk = np.array(f['Nk']).item()
    omega0 = np.array(f['omega0']).item()
    ncyc = np.array(f['ncyc']).item()
    A0 = np.array(f['A0'])
    thetas = np.array(f['thetas'])
    alpha = np.array(f['alpha']).item()
    inv_tau = np.array(f['inv_tau']).item()
    mu = np.array(f['mu']).item()
    H1 = np.array(f['H1'])
    H3 = np.array(f['H3'])
    H5 = np.array(f['H5'])
    H7 = np.array(f['H7'])

    thetas_smooth = np.linspace(thetas.min(), thetas.max(), 300)

    spl = make_interp_spline(thetas, H3)
    axs[0].plot(180*thetas_smooth/np.pi, spl(thetas_smooth), color='r')
    axs[0].plot(180*thetas[:]/np.pi, H3, 'o', markersize=3, color='r')

    spl = make_interp_spline(thetas, H5)
    axs[1].plot(180*thetas_smooth/np.pi, spl(thetas_smooth), color='r')
    axs[1].plot(180*thetas[:]/np.pi, H5, 'o', markersize=3, color='r')

    spl = make_interp_spline(thetas, H7)
    axs[2].plot(180*thetas_smooth/np.pi, spl(thetas_smooth), color='r')
    axs[2].plot(180*thetas[:]/np.pi, H7, 'o', markersize=3, color='r')

axs[0].format(xlabel=r"Polarization angle", ylabel=r"H3", xticks=[0, 30, 60, 90, 120, 150, 180])
axs[1].format(xlabel=r"Polarization angle", ylabel=r"H5", xticks=[0, 30, 60, 90, 120, 150, 180])
axs[2].format(xlabel=r"Polarization angle", ylabel=r"H7", xticks=[0, 30, 60, 90, 120, 150, 180])
fig.format(suptitle=f"$T={T}, \\alpha={180*alpha/np.pi:.0f}$")
