import numpy as np
from astropy.table import Table
import astropy.cosmology
cosmo = astropy.cosmology.FlatLambdaCDM(H0=70.,Om0=0.3)
from astropy import units
from astropy.coordinates import SkyCoord

from dl import queryClient as qc
from dl.helpers.utils import convert

from pathlib import Path

np.random.seed(12345678)

class Galaxies(object):
    
    def __init__(self, catalogue=None):        
        self.galaxies = catalogue

            
class Cluster(object):
    """
    A class to hold the galaxy data for each cluster 
    """

    def __init__(self, params=None, save_data=False):
        """
        Parameters
        ----------
        params : object read from scripts/params.py using importlib. See pkg_data/photXclus_params.py Parameters definition are given in scripts/params.py
        """

        self.cat_filename = params.cat_filename
        self.iclus = params.iclus
        self.rfield_in_rclus = params.rfield_in_rclus
        self.cat_type = params.cat_type
        self.params = params
        
        self.rclusX = params.rclusX
        self.rclus_rx = params.rclus_rx

        self.clus = Table.read(self.cat_filename)[self.iclus]
        self.rclus = max(1 * units.arcmin, self.rclus_rx*(self.clus[self.rclusX] * units.arcsec))
        self.rfield_in = self.rfield_in_rclus*self.rclus
        
        self.rfield_out_rclus = np.sqrt(params.Omega_out.to(units.arcmin**2)/
                                        (np.pi * self.rclus.to(units.arcmin)**2) + self.rfield_in_rclus**2) 
        self.rfield_out = self.rfield_out_rclus*self.rclus

        self.data_filename = (
            Path(params.rootdir)
            / "data"
            / f"data_clus{self.iclus}_{self.cat_type}.npz"
            )
        # Create directory if it doesn't exist
        self.data_filename.parent.mkdir(parents=True, exist_ok=True)

        if self.data_filename.is_file():        
            print(f"Cluster and control regions galaxies are already saved to {self.data_filename}\n let's load them")
            self.clus_galaxies = Galaxies(Table(np.load(self.data_filename)['clus_galcat']))
            self.field_galaxies = Galaxies(Table(np.load(self.data_filename)['field_galcat']))
        else:
            print("Fetching Legacy Survey cluster galaxies from astro-datalab...")
            self.get_dataclus()
            print("Done.")
            print("\n")
            print("Fetching Legacy Survey control galaxies from astro-datalab...")
            self.get_datafield()
            print("Done.")
            if save_data:
                print(f"Saving cluster and control galaxy catalogues to {self.data_filename}")
                np.savez(self.data_filename,
                         field_galcat=self.field_galaxies.galaxies,
                         clus_galcat=self.clus_galaxies.galaxies
                         )
            

    def get_dataclus(self):
        """
        Generate a catalog of galaxies in the cluster line of sight for a given cluster by querying the Legacy Survey DR9 database
        """
        
        query = '''
                SELECT pz.ls_id, m.ra, m.dec, m.mag_g, m.mag_r, m.mag_z, m.type,
                pz.z_phot_l68, pz.z_phot_median, pz.z_phot_u68, pz.z_phot_std
                FROM ls_dr9.photo_z pz, ls_dr9.tractor m
                WHERE 't' = Q3C_RADIAL_QUERY(m.ra, m.dec, {:f}, {:f},{:f}) 
                AND pz.ls_id = m.ls_id'''.format(self.clus['ra'],self.clus['dec'],(self.rclus).to(units.deg).value)
        result = qc.query(sql=query, format='csv')
        gal_cat = Table.from_pandas(convert(result,'pandas'))
        gal_cat = gal_cat[gal_cat['z_phot_median'] >= 0.0]
        self.clus_galaxies = Galaxies(gal_cat)
        
        
    def get_datafield(self):
        """
        Generate a catalog of field galaxies in an outside shell centered on a given cluster by querying the Legacy Survey DR9 database
        """

        query = '''
        SELECT pz.ls_id, m.ra, m.dec, m.mag_g, m.mag_r, m.mag_z, m.type, 
        pz.z_phot_l68, pz.z_phot_median, pz.z_phot_u68, pz.z_phot_std
        FROM ls_dr9.photo_z pz, ls_dr9.tractor m
        WHERE 't' = Q3C_RADIAL_QUERY(m.ra, m.dec, {:f}, {:f},{:f}) 
        AND pz.ls_id = m.ls_id'''.format(self.clus['ra'],self.clus['dec'],(self.rfield_out).to(units.deg).value)
        result = qc.query(sql=query, format='csv')
        cat_out = Table.from_pandas(convert(result,'pandas'))
        cat_out = cat_out[cat_out['z_phot_median'] >= 0.0]
        
        cat_out_sky = SkyCoord(ra=cat_out['ra'], dec=cat_out['dec'], unit='deg', frame='icrs')
        clus_sky = SkyCoord(ra=self.clus['ra'], dec=self.clus['dec'], unit='deg', frame='icrs')
        sep = cat_out_sky.separation(clus_sky)
        
        cat_out = cat_out[sep > self.rfield_in]
        self.field_galaxies = Galaxies(cat_out)