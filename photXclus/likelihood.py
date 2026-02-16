import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import numba as nb

global size
maxsize = 200
eps = 2. * sys.float_info.min
log_eps = sys.float_info.min_exp

from .model import *

@nb.njit()
def get_pclus_grid(zv, rv, muz_c, bin_zc, sigz_z, rx):
    """
    Generates a grid of cluster models evaluated at the input points (z, r).

    Parameters
    ----------
    zv : array_like
        Array of redshifts.
    rv : array_like
        Array of physical radii.
    muz_c : float
        Mean of the cluster redshift distribution.
    bin_zc : array_like
        bin of the cluster redshift parameter.
    sigz_z : float
        Standard deviation of the cluster redshift distribution.
    rx : float
        X-ray radius of the cluster in units of some X-ray radius estimate.

    Returns
    -------
    p : array_like
        Array of cluster models evaluated at the input (z, r).
    """
    p = np.zeros_like(zv)
    for i in range(len(zv)):
        p[i] = clus_zr(zv[i], rv[i], 
                       muz_c, bin_zc, sigz_z, rx)
    return p

@nb.njit()
def log_likelihood(params, data_in, data_out, other_params, bin_z, sigz_z, zr_grid):
    """
    Calculates the log-likelihood of the marked Poisson process associated 
    with a (cluster fov, field fov) = (data_in, data_out) pair of datasets 
    for a given set of parameters params.

    Parameters
    ----------
    params : array_like
        Array of parameters to be evaluated. The first three elements are the number 
        of galaxies in the cluster (ngal_c), 
        the mean of the cluster redshift distribution (muz_c), 
        and the number of field galaxies (ngal_f). 
        The remaining elements are the amplitudes of the field galaxy bins.
    data_in : array_like
        Array of (z, r) points in the cluster line of sight.
    data_out : array_like
        Array of (z, r) points control line of sight (outside the cluster).
    other_params : tuple
        Tuple containing additional parameters: Omega_in, Omega_out, zmin, zmax, r0, rx, integr_clus.
    bin_z : array_like
        Array of bin edges for the field galaxy bins.
    sigz_z : float
        Standard deviation of the cluster redshift distribution.
    zr_grid : array_like
        Array of (z, r) points in the cluster line of sight for evaluating the integral 
        of the cluster model.

    Returns
    -------
    lnL : float
        The log-likelihood of the given parameters.
    """
    Omega_in, Omega_out, zmin, zmax, r0, rx, integr_clus = other_params
    
    z_in, r_in = data_in[:, 0], data_in[:, 1]
    z_out, r_out = data_out[:, 0], data_out[:, 1]
    
    ngal_c, muz_c, ngal_f = params[:3]
    binz_ampl_f = params[3:]

    nbinz=len(bin_z)-1
    bin_zc = np.array([0.5*(bin_z[i]+bin_z[i+1]) for i in range(nbinz)])
        
    ### 1. normalize PDFs of model.
    
    ### integral 2D
    zv, rv = zr_grid
    dz = zv[0,1]-zv[0,0]
    dr = rv[1,0]-rv[0,0] 
    L_grid_in = get_pclus_grid(zv, rv, muz_c, bin_zc, sigz_z, rx)
    integ_in =  np.trapz(np.trapz(L_grid_in, dx=dz), dx=dr)
    
    phi_c = ngal_c / integ_in
    
    ##1.2 g(z) == field(z)
    integz_field = np.sum(binz_ampl_f) * (bin_zc[1]-bin_zc[0])
    binz_ampl_f /= integz_field

    ### 1.3 full model 
    proba_in = clus_field_zr(z_in, r_in, phi_c, muz_c, bin_zc, sigz_z, 
                                ngal_f, binz_ampl_f, bin_z, Omega_in, rx, 1.)
    proba_out = clus_field_zr(z_out, r_out, phi_c, muz_c, bin_zc, sigz_z, 
                                 ngal_f, binz_ampl_f, bin_z, Omega_out, rx, 0.)
    lnL = np.sum(np.log(proba_in)) + np.sum(np.log(proba_out))
    
    s_in = (ngal_c + Omega_in * ngal_f) 
    s_out = Omega_out * ngal_f
    
    lnL -= ( s_in + s_out)
    
    if np.isnan(lnL):
        print(lnL, s_in, params)
    
    return lnL


def log_prior(theta, params_prior):
    """
    Calculates the log-prior probability of the given set of parameters theta.

    Parameters
    ----------
    theta : array_like
        Array of parameters to be evaluated. The first three elements are the number 
        of galaxies in the cluster (ngal_c), the mean of the cluster redshift distribution (muz_c), 
        and the number of field galaxies (ngal_f). 
        The remaining elements are the amplitudes of the field galaxy bins.
    params_prior : tuple
        Tuple containing additional parameters: ngal_c_max, ngal_f_max, 
        field_bin_size, zrx_min, zrx_max.

    Returns
    -------
    lnPrior : float
        The log-prior probability of the given parameters.
    """
    ngal_c, muz_c, ngal_f = theta[:3]
    binz_ampl_f = theta[3:]
    ngal_c_max, ngal_f_max, field_bin_size, zrx_min, zrx_max = params_prior
    if (np.all(0 <= binz_ampl_f) and np.all(binz_ampl_f <= 10000)
        and 1. <= ngal_c <= ngal_c_max 
        and 0. < muz_c < 1.5-field_bin_size
        and 0 <= ngal_f <= ngal_f_max ):
        return logprior_z_rx(muz_c, zrx_min, zrx_max)
    return -np.inf


def logprior_z_rx(z, zmin, zmax):
    """
    Calculates the log-prior probability of the given redshift z.

    Parameters
    ----------
    z : float
        Redshift to be evaluated.
    zmin : float
        Minimum allowed redshift.
    zmax : float
        Maximum allowed redshift.

    Returns
    -------
    lnPrior : float
        The log-prior probability of the given redshift.
    """
    if (z >= zmin) & (z <= zmax):
        res = 0
    if z > zmax:
        res = -0.5*(z-zmax)**2/((0.025*(1+zmax))**2)
    if z < zmin:
        res = -0.5*(z-zmin)**2/((0.025*(1+zmin))**2)
    return res
