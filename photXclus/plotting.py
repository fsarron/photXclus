import matplotlib
import matplotlib.pyplot as plt
from astropy import wcs
import numpy as np
import corner
from matplotlib import lines
from pathlib import Path
import scipy.signal


def plot_corner(fitter, show=False):
    labels = [r'$N_{\rm gal, clus}$',
              r'$z_{\rm clus}$',
              r'$N_{\rm gal, field}$']
        
    flat_samples = np.vstack(fitter.samples[-200:])
    
    if show:
        figsize=(6,6)
    else:
        figsize=(38,38)

    fig_corner = plt.figure(figsize=figsize)
    corner_plot = corner.corner(flat_samples[:,[1,0]],labels=[labels[1],labels[0]],
                  fontsize=8, smooth=1,hist_bin_factor=1,fig=fig_corner,
                  plot_datapoints=False, color='k', 
                  hist_kwargs={'density': True},
                  bins=[int(1.5/0.01), int((flat_samples[:,0].max()+5))], 
                  range=[[0.,1.5],[0,flat_samples[:,0].max()+5]], plot_contours=True)
    
    axes = np.array(fig_corner.axes).reshape((2, 2))#; print(axes)
    ax = axes[1, 0]
    fitter.clus_properties.sort('persistence')
    fitter.clus_properties.reverse()

    
    for i in range(len(fitter.clus_properties)):
        ax.text(fitter.clus_properties['z_clus'][i], 
                fitter.clus_properties['Ngals_clus'][i], # + 0.1*np.quantile(fitter.samples[:,:,0], 0.99), 
                '#'+str(i+1), color='red', fontsize=12)
        
        
        
    ax = axes[0, 1]
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    #idx_pmax = np.argsort(fitter.clus_properties['persistence'])
    k = 0
    y0 = 0.95
    for i in range(len(fitter.clus_properties)):
        zclus = fitter.clus_properties['z_clus'][i]
        zerr_l68 = (fitter.clus_properties['z_clus'][i]-
                    fitter.clus_properties['zl68_clus'][i])
        zerr_u68 = (fitter.clus_properties['zu68_clus'][i]-
                    fitter.clus_properties['z_clus'][i])
        
        
        Ngalclus = fitter.clus_properties['Ngals_clus'][i]
        Ngalerr_l68 = (fitter.clus_properties['Ngals_clus'][i]-
                    fitter.clus_properties['Ngals_l68_clus'][i])
        Ngalerr_u68 = (fitter.clus_properties['Ngals_u68_clus'][i]-
                    fitter.clus_properties['Ngals_clus'][i])
        
        pstring = ( r'peak #'+str(i+1)+'   (p = '+
                   str(np.round(fitter.clus_properties['persistence'][i], 1))+')' )
        
        if i == 0: 
            zstring = ( r'$\mathbf{z_{\rm clus} = '+
                   str(np.round(zclus,3))+
                   '^{+'+str(np.round(zerr_u68,3))+'}_{-'+
                   str(np.round(zerr_l68,3))+'}}$' )
        else:
            zstring = ( r'$z_{\rm clus} = '+
                   str(np.round(zclus,3))+
                   '^{+'+str(np.round(zerr_u68,3))+'}_{-'+
                   str(np.round(zerr_l68,3))+'}$' )
        
        Nstring = ( r'$N_{\rm gals, clus} = '+
                   str(np.round(Ngalclus,1))+
                   '^{+'+str(np.round(Ngalerr_l68,1))+'}_{-'+
                   str(np.round(Ngalerr_u68,1))+'}$' )
        
        #axd['corner'].child_axes[3].text(0.5, 0.26 - i*0.045, pstring, fontsize=14, color='red')
        #axd['corner'].child_axes[3].text(0.5, 0.245 - i*0.045, zstring, fontsize=14, color='red')
        #axd['corner'].child_axes[3].text(0.5, 0.23 - i*0.045, Nstring, fontsize=14, color='red')
        ax.text(0.05, y0 - k*0.09, pstring, fontsize=10.5, color='black')
        
        # --- Insertion
        rr = max(ax.get_ylim())
        mm = np.mean(ax.get_xlim())
        line = lines.Line2D([0.05, 0.65], [y0 - k*0.09-0.01, y0 - k*0.09-0.01], 
                            color='black', ls='dashed', lw=0.75)
        line.set_clip_on(False)
        ax.add_line(line)
        ax.text(0.125, y0 - (k+1)*0.09 - 0.01, zstring, fontsize=10.5, color='red')
        ax.text(0.125, y0 - (k+2)*0.09 - 0.02, Nstring, fontsize=10.5, color='black')
        k += 4

        if show:
            plt.show()

        return corner_plot



def diagnosis(clus, fitter):
    matplotlib.use('agg')

    labels = [r'$N_{\rm gal, clus}$',
              r'$z_{\rm clus}$',
              r'$N_{\rm gal, field}$']
    flat_samples = np.vstack(fitter.samples[-200:])


    corner_plot = plot_corner(fitter, show=False)

    sky_plot = plt.figure(figsize=(30,30))
    ax = plt.subplot(projection = wcs.WCS(clus.gal_map.header))
    ax.imshow(clus.gal_map.data, origin='lower')
    ax.coords[0].set_major_formatter('d.d')
    ax.coords[1].set_major_formatter('d.d')
    ax.set_xlabel(r'RA (deg)')
    ax.set_ylabel(r'Dec (deg)')
    ax.set_title(r'magr & magz > mstar(z) + 1')
    ax.text(clus.field_galaxies.galaxies['ra'].max() + 0.1*np.std(clus.field_galaxies.galaxies['ra']), 
            clus.field_galaxies.galaxies['dec'].max() + 0.1*np.std(clus.field_galaxies.galaxies['dec']), 
            r'$N_{\rm gal, field FoV} = '+str(fitter.ngal_out)+'$', 
            transform=ax.get_transform('icrs'), color='white')
    
    ax.text(clus.field_galaxies.galaxies['ra'].max() + 0.1*np.std(clus.field_galaxies.galaxies['ra']), 
            clus.field_galaxies.galaxies['dec'].max() - 0.1*np.std(clus.field_galaxies.galaxies['dec']), 
            r'$N_{\rm gal, clus FoV} = '+str(fitter.ngal_in)+'$',
            transform=ax.get_transform('icrs'), color='white')

    chains = [
        ["chain A"],
        ["chain B"],
        ["chain C"],
    ]
    
    mosaic = np.array(
    [['mag', 'mag', 'mag', 'mag', 'field_sub', 'field_sub', 'field_sub', 'field_sub'],
     ['mag', 'mag', 'mag', 'mag', 'field_sub', 'field_sub', 'field_sub', 'field_sub'],
     ['sky', 'sky', 'sky', 'sky', 'field', 'field', 'field', 'field'],
     ['sky', 'sky', 'sky', 'sky', 'field', 'field', 'field', 'field'],
     ['sky', 'sky', 'sky', 'sky', 'corner', 'corner', 'corner', 'corner'],
     ['chain A', 'chain A', 'chain A', 'chain A', 'corner', 'corner', 'corner', 'corner'],
     ['chain B', 'chain B', 'chain B', 'chain B', 'corner', 'corner', 'corner', 'corner'],
     ['chain C', 'chain C', 'chain C', 'chain C', 'corner', 'corner', 'corner', 'corner']])

    axd = plt.figure(figsize=(12,12), constrained_layout=True).subplot_mosaic(
        mosaic,
        empty_sentinel='vide',
    )
    
    axd['sky'].axis('off')
    axd['sky'].add_child_axes(sky_plot.axes[0])
    axd['sky'].child_axes[0].set_position([
        sky_plot.axes[0].get_position(original=False).x0+0.18,
        sky_plot.axes[0].get_position(original=False).y0+0.9,
        sky_plot.axes[0].get_position(original=False).height, 
        sky_plot.axes[0].get_position(original=False).width])
    
    axd[chains[0][0]].set_xticklabels('')
    axd[chains[1][0]].set_xticklabels('')
    
    for i in range(len(labels)):
        ax = axd[chains[i][0]]
        ax.plot(fitter.samples[:, :, i], "k", alpha=0.03)

        ax.set_xlim(0, len(fitter.samples))
        ax.set_ylabel(labels[i])

    ax.set_xlabel("step number")
    
    ax = axd['mag']
    mag_bins = np.arange(15,28,0.25)
    datahist = np.histogram(clus.field_galaxies.galaxies['mag_r'], bins=mag_bins)
    peaks = scipy.signal.find_peaks(datahist[0], prominence=(None, None))
    maglim_r = mag_bins[peaks[0][peaks[1]['prominences'].argmax()]]

    _ = ax.hist(clus.field_galaxies.galaxies['mag_r'], bins=mag_bins, histtype='step', color='blue', alpha=0.5)
    ax.vlines(maglim_r, ymin=datahist[0].min(), ymax=datahist[0].max()+5, color='blue', ls='dashed', alpha=0.75, label=r'maglim $r$')

    datahist = np.histogram(clus.field_galaxies.galaxies['mag_z'], bins=mag_bins)
    peaks = scipy.signal.find_peaks(datahist[0], prominence=(None, None))
    maglim_z = mag_bins[peaks[0][peaks[1]['prominences'].argmax()]]

    _ = ax.hist(clus.field_galaxies.galaxies['mag_z'], bins=mag_bins, histtype='step', color='red', alpha=0.5)
    ax.vlines(maglim_z, ymin=datahist[0].min(), ymax=datahist[0].max()+5, color='red', ls='dashed', alpha=0.75, label=r'maglim $z$')
    
    ax.legend(fontsize=8)
    ax.set_xlabel(r'mag')
    ax.set_ylabel(r'$N_{\rm gal}$')
    
    
    ax = axd['field_sub']
    
    datahisto_field = np.histogram(fitter.z_out, bins=fitter.bin_z)[0]
    datahisto_clus = np.histogram(fitter.z_in, bins=fitter.bin_z)[0]

    
    ratio_area = (fitter.Omega_in/fitter.Omega_out)
    ax.stairs(datahisto_clus-ratio_area*datahisto_field, fitter.bin_z, color='red', label=r'clus = tot - ( field x area ratio)')

    ax.errorbar(fitter.bin_zc, datahisto_clus-ratio_area*datahisto_field, 
            yerr=np.sqrt(datahisto_clus + ratio_area**2*datahisto_field), fmt='.', color='red', capsize=2)

    ax.hlines(0,xmin=0,xmax=1.5, color='black', ls='dashed')

    ax.fill_between(fitter.bin_zc, (datahisto_clus-ratio_area*datahisto_field)-
                np.sqrt(datahisto_clus + ratio_area**2*datahisto_field), 
                (datahisto_clus-ratio_area*datahisto_field)+
                np.sqrt(datahisto_clus + ratio_area**2*datahisto_field),
                alpha=0.2, color='red')

    ax.set_title('magr & magz > mstar(z) + 1')
    ax.set_xlabel(r'z')
    ax.set_ylabel(r'$N_{\rm gal, clus} / \Delta z$')
    ax.legend(fontsize=8)
    
    
    
    ax = axd['field']

    norm = np.array([np.trapz(flat_samples[i,3:], fitter.bin_zc) for i in range(len(flat_samples))])
    ax.errorbar(fitter.bin_zc+0.01, np.median(flat_samples[:,3:]/norm[:,np.newaxis],axis=(0)), 
            yerr=np.std(flat_samples[:,3:]/norm[:,np.newaxis],axis=(0)),
            fmt='.', color='red', capsize=2, label=r'${\rm modeled~field}$')
    ax.errorbar(fitter.bin_zc, datahisto_field/np.trapz(datahisto_field,fitter.bin_zc), 
            yerr=np.sqrt(datahisto_field)/np.trapz(datahisto_field,fitter.bin_zc), fmt='.', 
                color='grey', capsize=2, label=r'${\rm observed~field}$')
    
    ax.set_xlabel(r'z')
    ax.set_ylabel(r'${\rm PDF}(z)$')
    ax.legend(fontsize=8)


    axd['corner'].axis('off')
    axd['corner'].add_child_axes(corner_plot.axes[0])
    axd['corner'].add_child_axes(corner_plot.axes[1])
    axd['corner'].add_child_axes(corner_plot.axes[2])
    axd['corner'].add_child_axes(corner_plot.axes[3])

    axd['corner'].child_axes[0].set_position([
        corner_plot.axes[0].get_position(original=False).x0+0.93,
        corner_plot.axes[0].get_position(original=False).y0+0.,
        corner_plot.axes[0].get_position(original=False).height, 
        corner_plot.axes[0].get_position(original=False).width])

    axd['corner'].child_axes[1].set_position([
        corner_plot.axes[1].get_position(original=False).x0+0.93,
        corner_plot.axes[1].get_position(original=False).y0+0.,
        corner_plot.axes[1].get_position(original=False).height, 
        corner_plot.axes[1].get_position(original=False).width])

    axd['corner'].child_axes[2].set_position([
        corner_plot.axes[2].get_position(original=False).x0+0.93,
        corner_plot.axes[2].get_position(original=False).y0+0.,
        corner_plot.axes[2].get_position(original=False).height, 
        corner_plot.axes[2].get_position(original=False).width])

    axd['corner'].child_axes[3].set_position([
        corner_plot.axes[3].get_position(original=False).x0+0.93,
        corner_plot.axes[3].get_position(original=False).y0+0.,
        corner_plot.axes[3].get_position(original=False).height, 
        corner_plot.axes[3].get_position(original=False).width])
    

    fitter.diag_filename = (
            Path(fitter.params.rootdir)
            / "diag"
            / f"diagnosis_clus{fitter.iclus}_{fitter.cat_type}.png"
    )
    fitter.diag_filename.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(fitter.diag_filename, dpi=600, format='png')

    print(f"Diagnosis plot saved to {fitter.diag_filename}")

    return
