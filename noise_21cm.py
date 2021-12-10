import pyccl as ccl
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

NU_21 = 1420.405751786
CLIGHT = 299.792458
FWHM2G = 0.42466090014

#cosmo = ccl.CosmologyVanillaLCDM()
cosmo = ccl.Cosmology(Omega_c=0.315-0.049, Omega_b=0.049, h=0.674, n_s=0.965, sigma8=0.811)

im_SKA_SD = {
    "name" : "SKA_SD",
    "dish_size" : 15.,
    "t_inst" : 25.,
    "t_total" : 10000.,
    "n_dish" : 197,
    "area_eff" : 1.0,
    "im_type" : "single_dish",
    "base_file" : "none",
    "base_min" : 0.,
    "base_max" : 15.,
    "fsky" : 0.4
}

im_SKA_IF = {
    "name" : "SKA_IF",
    "dish_size" : 15.,
    "t_inst" : 25.,
    "t_total" : 10000.,
    "n_dish" : 197,
    "area_eff" : 1.0,
    "im_type" : "interferometer",
    "base_file" : "curves_IM/baseline_file_SKA.txt",
    "base_min" : 15.,
    "base_max" : 1000.,
    "fsky" : 0.4
}

im_MeerKAT_SD = {
    "name" : "MeerKAT_SD",
    "dish_size" : 13.5,
    "t_inst" : 25.,
    "t_total" : 4000.,
    "n_dish" : 64,
    "area_eff" : 1.0,
    "im_type" : "single_dish",
    "base_file" : "none",
    "base_min" : 0.,
    "base_max" : 15.,
    "fsky" : 0.1
}

im_MeerKAT_IF = {
    "name" : "MeerKAT_IF",
    "dish_size" : 13.5,
    "t_inst" : 25.,
    "t_total" : 4000.,
    "n_dish" : 64,
    "area_eff" : 1.0,
    "im_type" : "interferometer",
    "base_file" : "curves_IM/baseline_file_MeerKAT.txt",
    "base_min" : 15.,
    "base_max" : 1000.,
    "fsky" : 0.1
}

im_HIRAX_32_6 = {
    "name" : "HIRAX_32_6",
    "dish_size" : 6.,
    "t_inst" : 50.,
    "t_total" : 10000.,
    "n_dish" : 1024,
    "area_eff" : 1.0,
    "im_type" : "interferometer",
    "base_file" : "curves_IM/baseline_file_HIRAX_6m.txt",
    "base_min" : 6.,
    "base_max" : 180.,
    "fsky" : 0.4
}


def get_noisepower_imap(xp, z0):
    nu0 = NU_21 / (1+z0)
    chi = ccl.comoving_radial_distance(cosmo, 1./(1+z0))*cosmo['h']
    hubble = ccl.h_over_h0(cosmo, 1./(1+z0))/ccl.physical_constants.CLIGHT_HMPC

    tsys = (xp['t_inst'] + 60.*(nu0/300.)**(-2.5))*1000
    sigma2_noise = tsys**2
    sigma2_noise *= 4*np.pi*xp['fsky']/(3.6E9*xp['t_total']*NU_21)

    beam_fwhm = CLIGHT / (xp['dish_size']*nu0)
    l_arr = np.arange(10000)
    if xp['im_type'] == 'interferometer':
        lam0 = CLIGHT/nu0
        dist , nbase = np.loadtxt(xp['base_file'], unpack=True)
        ndistint = interp1d(dist, nbase*dist*2*np.pi, bounds_error=False, fill_value=0.)
        norm = 0.5*xp['n_dish']*(xp['n_dish']-1.)/quad(ndistint, dist[0], dist[-1])[0]
        nbase *= norm
        ndist = interp1d(dist, nbase, bounds_error=False, fill_value=0.)
        n_baselines = ndist(l_arr*lam0/(2*np.pi))
        factor_beam_if = n_baselines*(lam0/beam_fwhm)**2
    else :
        factor_beam_if = 1E-26*np.ones_like(l_arr)
    if xp['im_type']=='single_dish' :
        beam_rad = beam_fwhm*FWHM2G
        factor_beam_sd = xp['n_dish']*np.exp(-l_arr*(l_arr+1.)*beam_rad**2)
    else :
        factor_beam_sd = 1E-26*np.ones_like(l_arr)
    factor_beam = factor_beam_sd+factor_beam_if

    cl_noise = sigma2_noise/factor_beam
    pk_noise = cl_noise*(chi*(1+z0))**2/hubble
    k_arr = l_arr/chi
    return k_arr, pk_noise

'''
k, nk = get_noisepower_imap(im_HIRAX_32_6, 1.2)
plt.plot(k, nk, label='HIRAX, interferometer')
k, nk = get_noisepower_imap(im_SKA_SD, 1.2)
plt.plot(k, nk, label='SKA, single-dish')
k, nk = get_noisepower_imap(im_SKA_IF, 1.2)
plt.plot(k, nk, label='SKA, interferometer')
plt.loglog()
plt.ylim([1E1, 1E6])
plt.legend()
print(k)
plt.show()
'''
