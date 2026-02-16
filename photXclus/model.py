import numpy as np
import numba as nb

@nb.njit()
def field_z(z, binz_ampl_f, bin_z, sigz_z):
    """
    Evaluate the field redshift model at given redshifts

    Parameters
    ----------
    z : array_like
        Array of redshifts
    binz_ampl_f : array_like
        Amplitude of the field model at each redshift bin
    bin_z : array_like
        Array of redshift bins
    sigz_z : array_like
        Array of standard deviation of the galaxy photometric redshift (sigma_zphot(z))

    Returns
    -------
    field_model : array_like
        Array of the field model evaluated at the input redshifts
    """
    iz = np.digitize(z,bin_z)-1
    return binz_ampl_f[iz]

@nb.njit()
def field_r(r, C_Omega):
    """
    Evaluate the field radial model at given physical radii

    Parameters
    ----------
    r : array_like
        Array of physical radii
    C_Omega : float
        Area of the field of view in degrees^-2

    Returns
    -------
    field_model : array_like
        Array of the field model evaluated at the input physical radii
    """
    return 2*np.pi*r / C_Omega 


@nb.njit()
def clus_z(z, muz_c, z_binc, sigz_z ):
    """
    Evaluate the cluster redshift model at given redshifts

    Parameters
    ----------
    z : array_like
        Array of redshifts
    muz_c : float
        Mean of the cluster redshift distribution
    z_binc : array_like
        Array of redshift bins
    sigz_z : array_like
        Array of standard deviation of the galaxy photometric redshift (sigma_zphot(z))

    Returns
    -------
    clus_model : array_like
        Array of the cluster model evaluated at the input redshifts
    """
    sigz_c = np.interp(muz_c, z_binc, sigz_z)
    res =  (1./(sigz_c * np.sqrt(2 * np.pi))) * np.exp(-0.5*((z-muz_c)/sigz_c)**2)
    return res

@nb.njit()
def clus_r(r, rx, integr_c = 1., model='Plummer'):
    """
    Evaluate the cluster radial model at given physical radii

    Parameters
    ----------
    r : array_like
        Array of physical radii
    rx : float
        Scale radius of the cluster model
    integr_c : float, optional
        Integral of the cluster model, used for the NFW model
    model : str, optional
        Model to be used, either 'Plum' or 'NFW'

    Returns
    -------
    clus_model : array_like
        Array of the cluster model evaluated at the input physical radii
    """
    if model == 'Plummer':
        return plum_r(r, rx)
    elif model == 'NFW':
        return NFW_r(r, rx, integr_c)
    else:
        raise NotImplementedError(f"{model} radial model not implemented")
    # return plum_r(r, rx)
    

@nb.njit()
def clus_zr(z, r, muz_c, bin_zc, sigz_z, rx):
    """
    Evaluate the cluster model at given redshifts and physical radii

    Parameters
    ----------
    z : array_like
        Array of redshifts
    r : array_like
        Array of physical radii
    muz_c : float
        Mean of the cluster redshift distribution
    bin_zc : array_like
        Array of redshift bins
    sigz_z : array_like
        Array of standard deviation of the galaxy photometric redshift (sigma_zphot(z))
    rx : float
        Scale radius of the cluster model

    Returns
    -------
    clus_model : array_like
        Array of the cluster model evaluated at the input redshifts and physical radii
    """
    res = clus_z(z, muz_c, bin_zc, sigz_z) * clus_r(r, rx)
    return res

@nb.njit()
def clus_field_zr(z, r, ngal_c, 
                  muz_c, bin_zc, sigz_z, 
                  ngal_f, 
                  binz_ampl_f, bin_z, C_Omega, rx, deltac):
    """
    Evaluate the total model at given redshifts and physical radii

    Parameters
    ----------
    z : array_like
        Array of redshifts
    r : array_like
        Array of physical radii
    ngal_c : float
        Number of galaxies in the cluster
    muz_c : float
        Mean of the cluster redshift distribution
    bin_zc : array_like
        Array of redshift bins
    sigz_z : array_like
        Array of standard deviation of the galaxy photometric redshift (sigma_zphot(z))
    ngal_f : float
        Number density of galaxies in the field in degrees^-2
    binz_ampl_f : array_like
        Amplitude of the field model at each redshift bin
    bin_z : array_like
        Array of redshift bins
    C_Omega : float
        Area of the field of view in degrees^-2
    rx : float
        Scale radius of the cluster model
    deltac : float
        Delta parameter for the field model

    Returns
    -------
    total_model : array_like
        Array of the total model evaluated at the input redshifts and physical radii
    """
    if deltac == 1:
        pclus = clus_z(z, muz_c, bin_zc, sigz_z) * clus_r(r, rx)
        pfield = field_z(z, binz_ampl_f, bin_z, sigz_z) * field_r(r, C_Omega)
        ptot = ngal_c * pclus + C_Omega * ngal_f * pfield
    else:
        pfield = field_z(z, binz_ampl_f, bin_z, sigz_z)
        ptot = C_Omega * ngal_f * pfield
    return ptot


@nb.njit()
def psi_r(r, rs):    
    """
    Evaluate the dimensionless potential at given physical radii

    Parameters
    ----------
    r : array_like
        Array of physical radii
    rs : float
        Scale radius of the potential

    Returns
    -------
    psi : array_like
        Array of the dimensionless potential evaluated at the input physical radii
    """
    denom = r/rs * (1 + r/rs)**2
    return 1. / denom

@nb.njit()
def psi_r_cut(r, rs):    
    """
    Evaluate the dimensionless potential at given physical radii, cut off at a certain radius

    Parameters
    ----------
    r : array_like
        Array of physical radii
    rs : float
        Scale radius of the potential

    Returns
    -------
    psi : array_like
        Array of the dimensionless potential evaluated at the input physical radii, cut off at a certain radius
    """
    rcut = 0.01*3*rs
    sigma0 = psi_r(rcut, rs) / (2*np.pi*rcut)
    res = psi_r(r, rs)
    i = 0
    while r[i] < rcut:
        res[i] = 2*np.pi*r[i]*sigma0
        i += 1
    return res

@nb.njit()
def NFW_r(r, rx, integr_c):
    """
    Evaluate the Navarro-Frank-White (NFW) model at given physical radii

    Parameters
    ----------
    r : array_like
        Array of physical radii
    rx : float
        Scale radius of the model
    integr_c : float
        Integral of the model

    Returns
    -------
    NFW_model : array_like
        Array of the Navarro-Frank-White model evaluated at the input physical radii
    """
    rs = rx/3.
    return integr_c * psi_r_cut(r, rs)

@nb.njit()
def plum_r(r, rx):
    """
    Evaluate the Plum model at given physical radii

    Parameters
    ----------
    r : array_like
        Array of physical radii
    rx : float
        Scale radius of the model

    Returns
    -------
    plum_model : array_like
        Array of the Plum model evaluated at the input physical radii
    """
    rs = rx/10.
    res =  ( 1 / np.sqrt(1 + (r/rs)**2) ) - (1 / np.sqrt(1 + (10)**2))
    return res

