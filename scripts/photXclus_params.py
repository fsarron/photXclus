from astropy import units

#Data
rootdir='/home/user/XCLASS/run1/'
cat_type='confirmed'
cat_filename = rootdir+'cluscat.fits'

datasave_dir = rootdir+'data/'

rclusX = 'rfit'
rclus_rx = 1.0 #rclus in units of some X-ray radius estimate, "rfit" for XCLASS 
rfield_in_rclus = 3.  #inner radius for field in units of rclus
Omega_out = 0.075 * units.deg**2

mstarz_file = None #filename mstar-z relation tab 
C_mstar = 1. #we will use galaxies brighter than mstarz + C_mstar

#Priors
zmin = 0.0 #minimum allowed redshift for cluster model
zmax = 1.5 #maximum allowed redshift for cluster model
field_bin_size = 0.05 #bin size in redshift space for field model

ngal_c_max = 100000 #maximum number of galaxies in cluster model
ngal_f_max = 1e8 #maximum number of galaxies in field model

#Fit
multiprocessing = True

ncpus = 1
nwalkers = 2
nburns = 250
nsamples = 500

fit_params = [multiprocessing, ncpus, nwalkers, nburns, nsamples, zmin, zmax, field_bin_size]

fitlib = 'zeus' #could be emcee. zeus is better for multimodal posteriors here
