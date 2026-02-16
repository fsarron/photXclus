import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy.interpolate

from astropy.table import Table
from astropy import units
import astropy.cosmology
cosmo = astropy.cosmology.FlatLambdaCDM(H0=70.,Om0=0.3)
from astropy.coordinates import SkyCoord

import emcee
import zeus

from pathlib import Path

from . import likelihood as likelihood
from . import utils as utils

eps = 2. * sys.float_info.min
log_eps = sys.float_info.min_exp

class Fit(object):
    """
    A class for fitting the cluster+field model to data via MCMC  
    """
    
    def __init__(self, clus=None, params=None 
                ):
        self.fitlib = params.fitlib
        self.fit_params = params.fit_params
        self.params = params
        self.get_dataset(clus, params)
    
    def get_dataset(self, clus, params):

        if params.mstarz_file is not None:
            mstarz_filename = (Path(params.rootdir)
                        / params.mstarz_file
            )
        else:
            mstarz_filename = utils.get_default_mag_file()

        print(f"Using mag-z relation file: {mstarz_filename}")

        t = Table.read(mstarz_filename, format='ascii.no_header', 
                      comment='#', data_start=1)
        mSr = scipy.interpolate.CubicSpline(t['col1'],t['col4'])
        mSz = scipy.interpolate.CubicSpline(t['col1'],t['col7'])
        mag_bins = np.arange(15,28,0.25)
        self.bin_z = np.linspace(params.zmin, params.zmax, int((params.zmax-params.zmin)/params.field_bin_size)+1)

        #Flag galaxies
        clus.clus_galaxies.galaxies['isGalaxy'] = ((clus.clus_galaxies.galaxies['type'] != 'PSF') & 
                                                    (clus.clus_galaxies.galaxies['type'] != 'DUP'))
        
        clus.field_galaxies.galaxies['isGalaxy'] = ((clus.field_galaxies.galaxies['type'] != 'PSF') & 
                                                    (clus.field_galaxies.galaxies['type'] != 'DUP'))

        #Flag bright galaxies in r and z band
        if not np.isin('mstar_eq_r', clus.field_galaxies.galaxies.colnames): 
            utils.isBright_r(clus, mag_bins, params, mSr)  ## r band
        if not np.isin('mstar_eq_z', clus.field_galaxies.galaxies.colnames): 
            utils.isBright_z(clus, mag_bins, params, mSz) ## z band

        ### compute 75% quantile of photo-z uncertainty to reject badly constrained photo-z

        mask_field_noz = ((clus.field_galaxies.galaxies['z_phot_median'] > params.zmin) & 
                                                        (clus.field_galaxies.galaxies['z_phot_median'] < params.zmax) & 
                                                         (clus.field_galaxies.galaxies['isBright_r']) & 
                                                        (clus.field_galaxies.galaxies['isBright_z']) &
                                                      (clus.field_galaxies.galaxies['isGalaxy'])
        )

        self.get_sigz75p(clus, mask_field_noz)
        
        #Make a mask for objects with good photo-z uncertainty
        mask_zerr_clus = (clus.clus_galaxies.galaxies['z_phot_std'] < 
                          np.interp(clus.clus_galaxies.galaxies['z_phot_median'], self.bin_zc, self.sigz75p_z))
        mask_zerr_field = (clus.field_galaxies.galaxies['z_phot_std'] < 
                          np.interp(clus.field_galaxies.galaxies['z_phot_median'], self.bin_zc, self.sigz75p_z))

        #Make a mask for bright galaxies with good photo-z in the redshift range for each dataset
        mask_clus  = ((clus.clus_galaxies.galaxies['z_phot_median'] > params.zmin) & 
                                                        (clus.clus_galaxies.galaxies['z_phot_median'] < params.zmax) &
                                                       (clus.clus_galaxies.galaxies['isBright_r']) & 
                                                        (clus.clus_galaxies.galaxies['isBright_z'])&
                                                      (clus.clus_galaxies.galaxies['isGalaxy']) &
                                                      (mask_zerr_clus))
        
        mask_field = mask_field_noz & mask_zerr_field

        ### get median error of photo-z used for estimating redshift, to convolve the cluster z delta function 
        self.get_sigz_z(clus, mask_field)
        

        ## compute radial distances
        self.r_in, self.r_out = self.get_radial_distances(clus, mask_clus, mask_field)
        
        self.z_in = clus.clus_galaxies.galaxies['z_phot_median'][mask_clus]
        self.z_out = clus.field_galaxies.galaxies['z_phot_median'][mask_field]
        
        self.ngal_in = len(self.z_in)
        self.ngal_out = len(self.z_out)

        self.data_in = np.c_[self.z_in, self.r_in.value] 
        self.data_out = np.c_[self.z_out, self.r_out.value]

        self.iclus = clus.iclus
        self.cat_type = clus.cat_type
        
        ## get dataset FoV areas
        try :
            self.Omega_in, self.Omega_out = clus.Omega_in.to(units.deg**2).value, clus.Omega_out.to(units.deg**2).value
        except:
            clus.Omega_in, clus.Omega_out, clus.gal_map = utils.get_areas(clus)
            self.Omega_in, self.Omega_out = clus.Omega_in.to(units.deg**2).value, clus.Omega_out.to(units.deg**2).value
        
        ## define the (z, r) grid used for normalization in the likelihood
        rx_deg = (clus.clus[params.rclusX] * units.arcsec).to(units.deg).value
        dr = 0.0005
        dz = 0.025
        z_integral = np.arange(0+dz, 1.5, dz) 
        r_integral = np.arange(0, rx_deg+dr, dr)
        self.zr_grid = np.stack(np.meshgrid(z_integral, r_integral))
        
        ## get the normalization of the radial profile
        r0_deg = 0
        integr_c = scipy.integrate.quad(likelihood.plum_r, r0_deg, rx_deg, args=(rx_deg))[0]
        
        self.other_params = np.array([self.Omega_in, self.Omega_out, params.zmin, params.zmax, r0_deg, rx_deg, integr_c])
                
        ##use rx + knowledge about cluster size in the Universe to have a prior on z
        rmax_clus = 2.0 * units.Mpc
        rmin_clus = 0. * units.Mpc
        zarr = np.linspace(0.01, 1.5, 150)

        rmax_zarr = (rmax_clus.to(units.kpc) * cosmo.arcsec_per_kpc_proper(zarr)).to(units.arcmin)
        rmin_zarr = (rmin_clus.to(units.kpc) * cosmo.arcsec_per_kpc_proper(zarr)).to(units.arcmin)
                
        try :
            zrx_max = np.min(zarr[rx.value > rmax_zarr.value])
        except:
            zrx_max=zarr[-1]
        
        try :
            zrx_min = np.max(zarr[rx.value < rmin_zarr.value])
        except:
            zrx_min=zarr[0]
                
        self.params_prior = np.array([params.ngal_c_max, params.ngal_f_max, params.field_bin_size, zrx_min, zrx_max])
        self.prior = np.array([likelihood.logprior_z_rx(z, zrx_min, zrx_max) for z in zarr])    

        datahisto_field = np.histogram(self.z_out, bins=self.bin_z)[0]
        fieldz_inf, fieldz_sup = scipy.stats.poisson(datahisto_field).interval(0.9999)
        #fieldz_inf = np.maximum(1, fieldz_inf)
        fieldz_sup = np.maximum(5, fieldz_sup)
        self.fieldz_prior = np.array((fieldz_inf, fieldz_sup))    


        self.datahisto_field = np.histogram(self.z_out, bins=self.bin_z)[0]
        
        ngal_rich = max(5 , self.ngal_in-self.ngal_out*self.Omega_in/self.Omega_out)
        z_0 = 0.5
        theta0 = np.concatenate([[ngal_rich, 
                          z_0, self.ngal_out/self.Omega_out], 
                         np.maximum(self.datahisto_field, 1)])
        self.theta0 = theta0

        #make one call to likelhood to trigger JIT compilation
        _ = likelihood.log_likelihood(theta0, self.data_in, self.data_out, self.other_params, 
                                            self.bin_z, self.sigz_z, self.zr_grid)

    def get_sigz75p(self, clus, mask_field_noz):
        """
        Calculate the 75th percentile of the redshift error distribution in each redshift bin.

        Parameters
        ----------
        clus : Cluster
            The cluster object containing the field galaxy catalog.

        Returns
        -------
        None
        """
        nbin = len(self.bin_z)-1
        self.bin_zc = np.array([0.5*(self.bin_z[i]+self.bin_z[i+1]) for i in range(nbin)])
        zz_bin = np.digitize(clus.field_galaxies.galaxies[mask_field_noz]['z_phot_median'], self.bin_z)
        self.sigz75p_z = np.zeros(nbin)
        izbin_min = zz_bin.min()
        izbin_max = zz_bin.max()
        for iz in range(izbin_min, izbin_max-1):
            try:
                self.sigz75p_z[iz] = np.quantile(
                    clus.field_galaxies.galaxies[mask_field_noz].group_by(zz_bin).groups[iz]['z_phot_std'], 
                    0.75)
            except:
                self.sigz75p_z[iz] = self.sigz75p_z[iz-1]
        self.sigz75p_z[:izbin_min] = self.sigz75p_z[izbin_min]
        self.sigz75p_z[izbin_max-1:] = self.sigz75p_z[izbin_max-2]
        
    def get_sigz_z(self, clus, mask_field):
        """
        Calculate the 50th percentile of the redshift error distribution in each redshift bin.

        Parameters
        ----------
        clus : Cluster
            The cluster object containing the field galaxy catalog.
        mask_field : bool array
            Boolean array indicating which galaxies are in the field sample.

        Returns
        -------
        None
        """
        nbin = len(self.bin_z)-1
        zz_bin = np.digitize(clus.field_galaxies.galaxies[mask_field]['z_phot_median'], self.bin_z)
        izbin_min = zz_bin.min()
        izbin_max = zz_bin.max()
        self.sigz_z = np.zeros(nbin)
        for iz in range(izbin_min, izbin_max-1):
            try:
                self.sigz_z[iz] = np.quantile(
                    clus.field_galaxies.galaxies[mask_field].group_by(zz_bin).groups[iz]['z_phot_std'], 
                    0.5)
            except:
                self.sigz_z[iz] = self.sigz_z[iz-1]
        self.sigz_z[:izbin_min] = self.sigz_z[izbin_min]
        self.sigz_z[izbin_max-1:] = self.sigz_z[izbin_max-2]


    def get_radial_distances(self, clus, mask_clus, mask_field):
        """
        Calculate the radial distances of the cluster galaxies and the field galaxies from the cluster center.

        Parameters
        ----------
        clus : Cluster
            The cluster object containing the cluster galaxy catalog.
        mask_clus : bool array
            Boolean array indicating which galaxies are in the cluster sample.
        mask_field : bool array
            Boolean array indicating which galaxies are in the field sample.

        Returns
        -------
        radial_distance_in : float array
            The radial distances of the cluster galaxies from the cluster center.
        radial_distance_out : float array
            The radial distances of the field galaxies from the cluster center.
        """
        clus_center = SkyCoord(ra=clus.clus['ra'], 
                        dec=clus.clus['dec'], 
                        unit='deg', frame='icrs')
        gal_pos_in = SkyCoord(ra=clus.clus_galaxies.galaxies['ra'], 
                           dec=clus.clus_galaxies.galaxies['dec'], unit='deg', frame='icrs')
        gal_pos_out = SkyCoord(ra=clus.field_galaxies.galaxies['ra'], 
                           dec=clus.field_galaxies.galaxies['dec'], unit='deg', frame='icrs')
        radial_distance_in = clus_center.separation(gal_pos_in)[mask_clus]
        radial_distance_out = clus_center.separation(gal_pos_out)[mask_field]
        return radial_distance_in, radial_distance_out

        
    def lnprob(self, theta):
        """
        Evaluates the log-probability of a given set of parameters theta.

        Parameters
        ----------
        theta : array_like
            Array of parameters to be evaluated.

        Returns
        -------
        lnprob : float
            The log-probability of the given parameters.
        """
        lp = likelihood.log_prior(theta, self.params_prior)
        if not np.isfinite(lp):
            return -np.inf
        return lp + likelihood.log_likelihood(theta, self.data_in, self.data_out, self.other_params, 
                                            self.bin_z, self.sigz_z, self.zr_grid)

    def run_mcmc(self, save_progress=True): 
        """
        Run the Markov Chain Monte Carlo (MCMC) algorithm using either the emcee or zeus libraries.

        Parameters
        ----------
        save_progress : boolean
            Whether to save the chains while sampling via Callbacks

        Returns
        -------
        sampler : EnsembleSampler object
            The object containing the samples of the MCMC algorithm.
        """
        ( multiprocessing, ncpus, nwalkers, nburns, nsamples, 
         zmin, zmax, field_bin_size ) = self.fit_params
        
        ndim = self.theta0.shape[0]

        nwalkers=nwalkers*ndim+1
        if nwalkers % 2 != 0:
            nwalkers += 1

        if save_progress:
            self.chains_filename = (
                    Path(self.params.rootdir)
                    / "chains"
                    / f"post_clus{self.iclus}_{self.cat_type}.h5"
            )
        self.chains_filename.parent.mkdir(parents=True, exist_ok=True)

        # Define the initial guesses for the parameters
        if self.fitlib == 'emcee':
            init = theta0 + 0.25*theta0*np.random.random((nwalkers,ndim))
        if self.fitlib == 'zeus':
            init = np.zeros((nwalkers, ndim))
            
            
            ngal_rich_0 = np.maximum(np.random.poisson(self.ngal_in, nwalkers) - 
                                   np.random.poisson(self.ngal_out, 
                                                     nwalkers)*self.Omega_in/self.Omega_out, 
                                   1.)
            
            z_0 = np.random.uniform(zmin+1*field_bin_size,zmax-1*field_bin_size,nwalkers)

            init[:,0] = ngal_rich_0 
            init[:,1] = z_0
            init[:,2] = np.random.poisson(self.ngal_out,nwalkers)/self.Omega_out
            init[:,3:] = np.minimum(np.maximum(np.random.poisson(
                self.datahisto_field,size=(nwalkers,ndim-3)),1), self.fieldz_prior[1])
            
            self.init = init
            
        #Sample
        if self.fitlib == 'zeus':
            sampler = zeus.EnsembleSampler(nwalkers, ndim, self.lnprob)
            print('BURN')
            if save_progress:
                sampler.run_mcmc(init, nburns, progress=True,
                                        callbacks=zeus.callbacks.SaveProgressCallback(
                                            self.chains_filename, ncheck=250))
            else:
                sampler.run_mcmc(init, nburns, progress=True)
            burnin = sampler.get_chain()

            # Set the new starting positions of walkers based on their last positions
            start = burnin[-1]
            # Initialise the ensemble Sampler using the advanced ``GlobalMove``.
            sampler = zeus.EnsembleSampler(nwalkers, ndim, self.lnprob, moves=zeus.moves.GlobalMove())
            # Run MCMC
            print('SAMPLE')
            if save_progress:
                sampler.run_mcmc(start, nsamples, progress=True, 
                                        callbacks=zeus.callbacks.SaveProgressCallback(
                                            self.chains_filename, ncheck=250))
            else:
                sampler.run_mcmc(start, nsamples, progress=True)
            # Get the samples and combine them with the burnin phase
            globalmove_samples = sampler.get_chain()
            self.samples = np.concatenate((burnin, globalmove_samples))
                    
                    
        if self.fitlib == 'emcee':
            sampler = emcee.EnsembleSampler(n_walkers, ndim, self.lnprob, moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ])
            sampler.run_mcmc(init, nsamples, progress=True)  
            self.samples = sampler.get_chain()     
                 
        
        return sampler
        
            

