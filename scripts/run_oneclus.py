import os
os.environ["OMP_NUM_THREADS"] = "1"
import re
from astropy.table import Table

import numpy as np

import photXclus.data as d
import photXclus.inference as inference
import photXclus.plotting as p
import photXclus.utils as utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--iclus", type=int,
                    help = "cluster number")
parser.add_argument("-p","--params", type=str,
                    help = "Parameter file")
args = parser.parse_args()

params_root = re.split(".py", args.params)[0]
if os.path.isfile(params_root+".pyc"):
    os.remove(params_root+".pyc")
    
if __name__ == "__main__":
  
    import importlib
    try:
        params = importlib.import_module(params_root)
        print('Successfully loaded "{0}" as params'.format(args.params))
    except:
        print('Failed to load "{0}" as params'.format(args.params))
        raise ImportError
    
    
    cluscat = Table.read(params.cat_filename)
    nclus = len(cluscat)

    params.iclus = args.iclus
        
    print(' ')
    print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _')
    print('Loading data ...')
    clus = d.Cluster(params=params)
    np.savez(params.rootdir+'data/data_clus'+str(clus.iclus)+'_'+clus.cat_type+'.npz', 
             field_galcat = clus.field_galaxies.galaxies, 
             clus_galcat = clus.clus_galaxies.galaxies)
    print('data loaded!')
    
    print('- - - - - -')
    fitter = inference.Fit(clus=clus, params=params)
    print('Ngal_fieldFOV used = ', len(fitter.data_out))
    print('Ngal_clusFOV used = ', len(fitter.data_in))
    sampler = fitter.run_mcmc()  
    
    flat_samples = np.vstack(fitter.samples[-200:, :, :])

    print('- - - - - -')
    print('get clus properties ...')
    zlims = (params.zmin, params.zmax)
    fitter.clus_properties = utils.get_peaks(flat_samples[:, :2], zlims)
    fitter.clus_properties.write(params.rootdir+'data/properties_clus'+str(clus.iclus)+
                                 '_'+params.cat_type+'_'+
                                 str(np.round(params.rclus_rfit, 1))+'.fits', 
                                 overwrite=True)
    print('- - - - - -')
    
    
    print('- - - - - -')
    print('plot diagnosis...')
    _ = p.diagnosis(clus, fitter)
    print('- - - - - -')