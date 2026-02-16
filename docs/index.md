# photXclus

`photXclus` is a Python package for joint Bayesian inference of redshift and richness of X-ray (or otherwise) detected clusters and groups of galaxies using galaxy photometric redshifts. 

Currently it runs using photometric redshifts from the Legacy Survey DR9 or DR10, but can be easily extended to other redshift sources. 

##Installation 

`photXclus` can be installed via pip to the git repository

```bash
pip install git+https://codeberg.org/fsarron/photXclus.git
```

## How does it work?

`photXclus` implements the statiscal model for a Inhomogeneous Poisson Process and samples 
the posterior of model parameters given data observed in a cluster and a control line-of-sight jointly. 


## Citation

If you use `photXclus` in your research, please consider citing the assocaited article Moysan, Sarron et al. (2026)