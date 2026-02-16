from astropy.coordinates import SkyCoord
import healpy as hp
import numpy as np

from astropy import units
from astropy.io import fits

from astropy.convolution import Tophat2DKernel, Gaussian2DKernel
from astropy.convolution import convolve

from astropy.table import Table, Column

import scipy.spatial
import scipy.stats.kde as kde
import scipy.signal

from importlib import resources
from pathlib import Path

from .third_party.persistence import imagepers

def get_default_mag_file():
    with resources.as_file(
        resources.files("photXclus.pkg_data") / "mag_star_z.dat"
    ) as path:
        return Path(path)

def get_default_cluscat():
    with resources.as_file(
        resources.files("photXclus.pkg_data") / "cluscat_example.fits"
    ) as path:
        return path


def cat2hpx(lon, lat, nside, radec=True):
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
        coordinate system. Otherwise assume input is glon, glat

    nside : int
        HEALPix nside of the target map

    radec : bool
        Switch between R.A./Dec and glon/glat as input coordinate system.

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    if radec:
        eq = SkyCoord(ra=lon, dec=lat, frame='icrs', unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat

    # conver to theta, phi
    theta = np.radians(90. - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts

    return hpx_map



def get_areas(clus):
    nside = 2**13

    allgal_ra = np.concatenate([clus.field_galaxies.galaxies['ra'], clus.clus_galaxies.galaxies['ra']])
    allgal_dec = np.concatenate([clus.field_galaxies.galaxies['dec'], clus.clus_galaxies.galaxies['dec']])

    hpx_map = cat2hpx(allgal_ra, allgal_dec, nside=nside, radec=True)

    eq = SkyCoord(ra=clus.clus['ra'], dec=clus.clus['dec'], frame='icrs', unit='deg')
    l, b = eq.galactic.l.value, eq.galactic.b.value

    pixsize = 0.05
    shape = int(2.5*clus.rfield_out.to(units.arcmin).value/pixsize)

    projected_map = hp.gnomview(hpx_map, rot=[l, b], reso=pixsize, xsize=shape, 
                            return_projected_map=True, no_plot=True)

    kernel = Tophat2DKernel(radius=0.5/pixsize) #0.25armin
    projected_map_conv = convolve(projected_map, kernel)

    ###disk
    rr = int(clus.rclus.to(units.arcmin).value/pixsize)
    x_samp = np.linspace(0, projected_map.shape[0], projected_map.shape[0])
    y_samp = np.linspace(0, projected_map.shape[1], projected_map.shape[1])
    X_SAMP,Y_SAMP=np.meshgrid(x_samp,y_samp,copy=False)
    data_xy=np.vstack((X_SAMP.ravel(),Y_SAMP.ravel())).T
    center = (0.5*np.array(projected_map.shape))
    dist=scipy.spatial.distance.cdist(data_xy,center.reshape(1,-1)).ravel()
    dist = dist.reshape(projected_map.shape).T
    disk =(dist < rr)


    ###annulus
    rin = int(clus.rfield_in.to(units.arcmin).value/pixsize)
    rout = int(clus.rfield_out.to(units.arcmin).value/pixsize)
    x_samp = np.linspace(0, projected_map.shape[0], projected_map.shape[0])
    y_samp = np.linspace(0, projected_map.shape[1], projected_map.shape[1])
    X_SAMP,Y_SAMP=np.meshgrid(x_samp,y_samp,copy=False)
    data_xy=np.vstack((X_SAMP.ravel(),Y_SAMP.ravel())).T
    center = (0.5*np.array(projected_map.shape))
    dist=scipy.spatial.distance.cdist(data_xy,center.reshape(1,-1)).ravel()
    dist = dist.reshape(projected_map.shape).T
    ann = (dist > rin) & (dist < rout)


    Omega_in = (np.sum(projected_map_conv * disk > 0) * pixsize**2 * units.arcmin**2).to(units.deg**2)
    Omega_out = (np.sum(projected_map_conv * ann > 0) * pixsize**2 * units.arcmin**2).to(units.deg**2)


    mask_field = clus.field_galaxies.galaxies['isBright_r'] & clus.field_galaxies.galaxies['isBright_z']
    mask_clus = clus.clus_galaxies.galaxies['isBright_r'] & clus.clus_galaxies.galaxies['isBright_z']

    brightgal_ra = np.concatenate([clus.field_galaxies.galaxies['ra'][mask_field], 
                             clus.clus_galaxies.galaxies['ra'][mask_clus]])
    brightgal_dec = np.concatenate([clus.field_galaxies.galaxies['dec'][mask_field], 
                              clus.clus_galaxies.galaxies['dec'][mask_clus]])

    hpx_map = cat2hpx(brightgal_ra, brightgal_dec, nside=nside, radec=True)
    projected_map = hp.gnomview(hpx_map, rot=[l, b], reso=pixsize, xsize=shape, 
                            return_projected_map=True, no_plot=True)

    projected_map_conv = convolve(projected_map, kernel)

    gal_map = projected_map_conv * (ann + disk)
    
    galmap_header = fits.Header.fromstring("""CTYPE1  = 'RA---TAN'
CRPIX1  =                """+str(gal_map.shape[0]/2)+"""
CRVAL1  =                """+str(clus.clus['ra'])+"""
CDELT1  =                """+str(-pixsize/60)+"""
CUNIT1  = 'deg     '
CTYPE2  = 'DEC--TAN'
CRPIX2  =                """+str(gal_map.shape[1]/2)+"""
CRVAL2  =                """+str(clus.clus['dec'])+"""
CDELT2  =                """+str(pixsize/60)+"""
CUNIT2  = 'deg     '
COORDSYS= 'icrs    '
""", sep='\n')
    
    hdu_galmap = fits.PrimaryHDU(gal_map, header=galmap_header)

    return Omega_in, Omega_out, hdu_galmap


##stolen from https://github.com/aloctavodia/BAP/blob/master/first_edition/code/Chp1/hpd.py
def hpd_grid(sample, alpha=0.05, roundto=4):
    """Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode

    Parameters
    ----------
    
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results

    Returns
    ----------
    hpd: list with the highest density interval
    x: array with grid points where the density was evaluated
    y: array with the density values
    modes: list listing the values of the modes
          
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    #diff = (u-l)/20  # differences of 5%
    diff = (u-l)/100  # differences of 1%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes


def get_peaks(flat_samples, zlims):
    
    zmin, zmax = zlims
    dz = 0.005
    Nmax = (flat_samples[:,0].max()+5)
    n_zbins,n_Nbins = int((zmax-zmin)/dz), 300 #int(Nmax/dN)
    dN = Nmax/n_Nbins
    
    im, bin_edges_z, bin_edges_N = np.histogram2d(flat_samples[:,1], flat_samples[:,0], 
                                                  bins=[n_zbins,n_Nbins],
                                                  range=[[zmin,zmax],[0,Nmax]],
                                                  density=True)
    N_smooth = 5 #0.01*Nmax / dN
    z_smooth = 0.025 / dz
    kernel = Gaussian2DKernel(x_stddev=N_smooth, y_stddev=z_smooth)
    imsmooth = convolve(im, kernel)
    g = imagepers.persistence(imsmooth.T)

    z_centres = np.array([0.5*(bin_edges_z[i]+bin_edges_z[i+1]) for i in range(n_zbins-1)])
    N_centres = np.array([0.5*(bin_edges_N[i]+bin_edges_N[i+1]) for i in range(n_Nbins-1)])
    
    #redshifts = []
    #Ngals = []
    
    Ngals = [N_centres[g[0][0][0]]]
    redshifts = [z_centres[g[0][0][1]]]
    persistences = [g[0][2]]

    for i, homclass in enumerate(g[1:]):
        p_birth, bl, pers, p_death = homclass
        if pers <= 0.1*g[0][2]:
            continue
        zi = z_centres[g[1:][i][0][1]]
        #print(zi, np.sum(np.abs(zi - np.array(redshifts)) < 0.05 * (1 + zi)).astype(bool))
        if np.sum(np.abs(zi - np.array(redshifts)) < 0.01 * (1 + zi)).astype(bool):            
            continue
        Ngals.append(N_centres[g[1:][i][0][0]])
        redshifts.append(z_centres[g[1:][i][0][1]])
        persistences.append(pers)

    clus_properties = Table(np.c_[Ngals, redshifts, persistences], names=['Ngals_clus', 'z_clus', 'persistence'])
    clus_properties.sort('z_clus')
        
    zz = np.array([0.5*(bin_edges_z[i]+bin_edges_z[i+1]) for i in range(imsmooth.shape[1])])
    NN = np.array([0.5*(bin_edges_N[i]+bin_edges_N[i+1]) for i in range(imsmooth.shape[0])])
    pz = np.sum(imsmooth, axis=1)/np.sum(imsmooth)/(zz[1]-zz[0]) ## we sum axis = 1 to marginalize over Ngals

    zflat = flat_samples[:, 1]
    zl68 = np.zeros(len(clus_properties))
    zu68 = np.zeros(len(clus_properties))
    Nl68 = np.zeros(len(clus_properties))
    Nu68 = np.zeros(len(clus_properties))

    peaks0 = np.digitize(clus_properties['z_clus'], bin_edges_z)-1
    peaks, _ = scipy.signal.find_peaks(pz, height=np.min(pz[peaks0]))
    widths = scipy.signal.peak_widths(pz, peaks, rel_height=0.999)
    izinf_peaks = widths[2].astype(int)
    izsup_peaks = widths[3].astype(int)
            
    for iz in range(len(clus_properties)):
        d = zflat[(zflat > zz[izinf_peaks[iz]]) & 
                       (zflat< zz[izsup_peaks[iz]])]
        zl68[iz], zu68[iz] = np.quantile(d, [0.16, 0.84]) #- redshifts[0]
            
        im = imsmooth[izinf_peaks[iz]:izsup_peaks[iz]]
        pN = np.sum(im, axis=0)/np.sum(im)/(NN[1]-NN[0])
            
        Nflat = flat_samples[:, 0]

        peak = [np.argmax(pN)]
        widths = scipy.signal.peak_widths(pN, peak, rel_height=0.999)
        iNinf_peak = widths[2].astype(int)
        iNsup_peak = widths[3].astype(int)
        d = Nflat[(Nflat > NN[iNinf_peak]) & 
                       (Nflat< NN[iNsup_peak])]
        Nl68[iz], Nu68[iz] = np.quantile(d, [0.16, 0.84]) #- redshifts[0]
            
        
    clus_properties.add_column(Column(zl68, name='zl68_clus'))
    clus_properties.add_column(Column(zu68, name='zu68_clus'))
    clus_properties.add_column(Column(Nl68, name='Ngals_l68_clus'))
    clus_properties.add_column(Column(Nu68, name='Ngals_u68_clus'))
        
    return clus_properties


def isBright_r(clus, mag_bins, params, mSr):
    datahist = np.histogram(clus.field_galaxies.galaxies['mag_r'], bins=mag_bins)
    peaks = scipy.signal.find_peaks(datahist[0], prominence=(None, None))
    maglim_r = mag_bins[peaks[0][peaks[1]['prominences'].argmax()]]

    mstar_clus_r = mSr(clus.clus_galaxies.galaxies['z_phot_median'])
    mstar_field_r = mSr(clus.field_galaxies.galaxies['z_phot_median'])
    

    clus.clus_galaxies.galaxies.add_column(Column(mstar_clus_r,name='mstar_eq_r'))
    clus.clus_galaxies.galaxies.add_column(Column((clus.clus_galaxies.galaxies['mag_r'] < mstar_clus_r+params.C_mstar) & 
                    ((clus.clus_galaxies.galaxies['mag_r'] < maglim_r)), name='isBright_r'))
    
    clus.field_galaxies.galaxies.add_column(Column(mstar_field_r,name='mstar_eq_r'))
    clus.field_galaxies.galaxies.add_column(Column((clus.field_galaxies.galaxies['mag_r'] < mstar_field_r+params.C_mstar) & 
                    ((clus.field_galaxies.galaxies['mag_r'] < maglim_r)), name='isBright_r'))
    
def isBright_z(clus, mag_bins, params, mSz):

    datahist = np.histogram(clus.field_galaxies.galaxies['mag_z'], bins=mag_bins)
    peaks = scipy.signal.find_peaks(datahist[0], prominence=(None, None))
    maglim_z = mag_bins[peaks[0][peaks[1]['prominences'].argmax()]]

    mstar_clus_z = mSz(clus.clus_galaxies.galaxies['z_phot_median'])
    mstar_field_z = mSz(clus.field_galaxies.galaxies['z_phot_median'])

    clus.clus_galaxies.galaxies.add_column(Column(mstar_clus_z,name='mstar_eq_z'))
    clus.clus_galaxies.galaxies.add_column(Column((clus.clus_galaxies.galaxies['mag_z'] < mstar_clus_z+params.C_mstar) & 
                    ((clus.clus_galaxies.galaxies['mag_z'] < maglim_z)), name='isBright_z'))

    clus.field_galaxies.galaxies.add_column(Column(mstar_field_z,name='mstar_eq_z'))
    clus.field_galaxies.galaxies.add_column(Column((clus.field_galaxies.galaxies['mag_z'] < mstar_field_z+params.C_mstar) & 
                    ((clus.field_galaxies.galaxies['mag_z'] < maglim_z)), name='isBright_z'))
