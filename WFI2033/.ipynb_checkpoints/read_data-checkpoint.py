import arviz as az
import numpy as np
batch_list = []
for i in range(1):
    current_batch = az.from_netcdf("/mnt/lustre/tianli/quasar_hmc/WFI2033_psf_correct_all.nc")
    batch_list.append(current_batch)
print('concat')  
inf_data_pixel = az.concat(*batch_list, dim='draw')
vars_mass = ['theta_E_1', 'gamma_1', 'e_1', 'center_1', 'gamma_sheer_1']
vars_lens_light = ['A_lens', 'sigma_lens', 'e_lens', 'center_lens']
vars_source_light = ['A_source', 'sigma_source', 'e_source']
vars_other = []
vars_power = ['n_source_grid', 'rho_source_grid', 'sigma_source_grid']
print(f'divergences per chain per step:\n {inf_data_pixel.sample_stats.diverging.values.sum(axis=1).T}')
#az.summary(inf_data_pixel, var_names=vars_mass + vars_power + vars_other)
az.summary(inf_data_pixel.sel(chain=np.array([0,1,2,3])), var_names=vars_mass + vars_power + vars_other)


import matplotlib.pyplot as plt

plt.rcParams['figure.constrained_layout.use'] = True
_ = az.plot_trace(inf_data_pixel.sel(chain=np.array([0,1, 2])),var_names=vars_mass + vars_power + vars_other , figsize=(10, 15))
plt.tight_layout()

import corner.corner as corner
fig_corner = None
for i in range(4):
    if i!= 3:
        fig_corner = corner(
            inf_data_pixel.posterior.isel(chain=i),
            var_names=vars_mass,
            color=f'C{i}',
            fig=fig_corner,
        )

import sys
import os
import warnings
warnings.simplefilter("ignore")
from herculens_import_main import *
import jax
import numpyro
jax.config.update('jax_enable_x64', True)
numpyro.enable_x64()

pix_scale = 0.0307  # arcsec / pixel (JWST)

DATA_DIR = "../../Data/WFI2033"
raw_data_path = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d.fits")
data_path = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d_cut_x6985_y3594_150.fits")
mask_path = os.path.join(DATA_DIR, "mask.fits")
# mask_out is set to all True by design (full image likelihood region)
with fits.open(raw_data_path, memmap=True) as hdul_raw:
    exposure_time = float(hdul_raw[0].header.get("EXPTIME", hdul_raw[0].header.get("TEXPTIME", 1.0)))

with fits.open(data_path, memmap=True) as hdul:
    data = jnp.array(hdul["SCI"].data) if "SCI" in hdul else jnp.array(hdul[0].data)
    rms_file = jnp.array(hdul["ERR"].data)

mask = jnp.array(fits.getdata(mask_path), dtype=bool)
mask_out = jnp.ones_like(data, dtype=bool)
mask =  jnp.array(mask_out, dtype=bool)

corner_pixel = 10
bkg_corner = np.array(data[:corner_pixel, :corner_pixel])
bkg_mean = float(np.nanmean(bkg_corner))
bkg_rms = float(np.nanstd(bkg_corner))

data = data - bkg_mean

rms_file = jnp.array(fits.getdata('./psf_data/WFI2033_ERR_with_stage2_psf_extra.fits'))
for i in range(4):
    plt.figure(figsize = (15,5))
    plt.subplot(1,4,1)
    plt.imshow(data, norm = 'log', cmap = 'twilight', origin = 'lower')
    plt.subplot(1,4,2)
    model_image = inf_data_pixel.posterior.model_image.median(axis = 1)[i,:,:]
    plt.imshow(model_image, norm = 'log', cmap = 'twilight', origin = 'lower')
    plt.subplot(1,4,3)
    plt.imshow((data  - model_image) / rms_file, cmap = 'bwr', vmax = 3, vmin = -3, origin = 'lower')