#!/usr/bin/env python3
from __future__ import annotations

# Generated from Step_6_stardm_model.ipynb and adapted for Step 6 stage3 HMC.


# %% Cell 1
import os
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

from datetime import datetime
from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
import warnings
warnings.simplefilter('ignore')

import xarray as xr
from scipy.optimize import least_squares

from herculens_import_main import *
from tools_improved import tool as cosmo_tool
from lens_images_extension import LensImageExtension
from herculens.PointSourceModel.point_source_model import PointSourceModel
from herculens.MassModel import mass_model_base
from jax_lensing_profiles.MassModel.Profiles.CuspyNFW_ellipse_kappa import CuspyNFW_3D_fn
from jax_lensing_profiles.MassModel.Profiles.MGE import MGE

jax.config.update('jax_enable_x64', True)
numpyro.enable_x64()

# %% Cell 2
class CuspyNFWEllipseKappa(MGE):
    def __init__(self):
        super().__init__(
            CuspyNFW_3D_fn,
            'R_s',
            n_gauss=20,
            n_terms=28,
            sigma_start_mult=1/200,
            sigma_end_mult=20,
            three_d=True,
        )

mass_model_base.STRING_MAPPING['CUSPY_NFW_ELLIPSE_KAPPA'] = CuspyNFWEllipseKappa

suffix = '_ss=2_full_light_multimass'
run_tag = '20260401_11'
OUTPUT_ROOT = Path('/mnt/lustre/tianli/quasar_hmc')
RUN_OUTPUT_DIR = OUTPUT_ROOT / f'WFI2033{suffix}_{run_tag}'
RESULT_DIR = Path(f'./result/result{suffix}_{run_tag}')
PRODUCTS_DIR = RESULT_DIR / 'data_products'
DATA_DIR = Path('../../Data/WFI2033')
RAW_DATA_PATH = DATA_DIR / 'jw01198-o004_t004_nircam_clear-f115w_i2d.fits'
DATA_SUB_PATH = RESULT_DIR / f'data_minus_lens_light{suffix}.fits'
HMC_MEDIAN_PATH = Path(f'./result/result_ss=2_full_light_multimass_20260401_11/HMC_median_draw{suffix}.nc')
FIXED_FIRST_THREE_PATH = RUN_OUTPUT_DIR / f'fixed_first_three_gaussians{suffix}.npz'
MASK_OUT_PATH = SCRIPT_DIR / 'data' / 'mask_out_center_r16.fits'

with fits.open(RAW_DATA_PATH, memmap=True) as hdul_raw:
    raw_header = hdul_raw['SCI'].header if 'SCI' in hdul_raw else hdul_raw[0].header
    exposure_time = float(raw_header.get('EXPTIME', raw_header.get('TEXPTIME', raw_header.get('XPOSURE', 1.0))))
pix_scale = float(np.sqrt(raw_header['PIXAR_A2']))

data = np.array(fits.getdata(PRODUCTS_DIR / f'data_bkg_sub{suffix}.fits'), dtype=float)
data_subtracted = np.array(fits.getdata(DATA_SUB_PATH), dtype=float)
rms_file = np.array(fits.getdata(PRODUCTS_DIR / f'rms_with_psf_extra{suffix}.fits'), dtype=float)
mask_out = np.array(fits.getdata(MASK_OUT_PATH), dtype=bool)
fixed_first_three = np.load(FIXED_FIRST_THREE_PATH)
HMC_median = xr.open_dataset(HMC_MEDIAN_PATH)
HMC_reference = HMC_median.median(dim='chain')

pixel_grid_shape = 74
source_grid_scale = 0.8
ss_factor = 2
num_chains = int(HMC_median.sizes['chain'])

POINT_SOURCE_PRIOR = {
    'pos_sigma': 0.05,
    'pos_window': 0.2,
    'log10_amp_low': -3.0,
    'log10_amp_high': 4.5,
}

CONJUGATE_POINT_PRIOR = {
    'rate': 1000.0,
}

PSF_CORR_PRIOR = {
    'sigma_log': 1.0,
    'log_clip': 5.0,
}

SOURCE_GRID_PRIOR = {
    'plate_name': 'Source grid',
    'param_name': 'source_grid',
    'sigma_low': 1e-5,
    'sigma_high': 10,
    'n_high': 100,
    'positive': True,
}

G1_MASS_CENTER = (1.556, 1.299)
G2_MASS_CENTER = (2.145, -3.326)
G3_MASS_CENTER = (4.243403, 6.185564)
G7_MASS_CENTER = (-7.284665, -10.319681)
conj_points = jnp.array([
    [1.20212170716053, -0.12271885209256231],
    [0.9053233071260114, 0.5277189685977776],
    [-1.0461673774453952, 1.0081083299749878],
    [-0.1255456241215261, -0.8965524340129204],
])

k_values = K_grid((pixel_grid_shape, pixel_grid_shape)).k


def build_psf_corr_factor_field(log_psf_corr_center, psf_shape, log_clip=PSF_CORR_PRIOR['log_clip']):
    ny, nx = psf_shape
    log_field = jnp.asarray(log_psf_corr_center)
    if log_field.shape != (ny, nx):
        raise ValueError(f'log_psf_corr_center shape {log_field.shape} does not match psf shape {(ny, nx)}')
    return jnp.exp(jnp.clip(log_field, -log_clip, log_clip))


def build_corrected_psf_kernel(psf_base, log_psf_corr_center, eps=1e-12):
    corr_field = build_psf_corr_factor_field(log_psf_corr_center, jnp.asarray(psf_base).shape)
    psf_eff = jnp.asarray(psf_base) * corr_field
    psf_eff = jnp.maximum(psf_eff, eps)
    return psf_eff / (jnp.sum(psf_eff) + eps)

# %% Cell 3
DESI_PLANK_cov = {
    'cov_omega_m_omega_m': 1.3204061745961228e-05,
    'cov_omega_m_H0': -0.000999369023951933,
    'cov_H0_H0': 0.07954650644002863,
    'matrix': np.array([
        [1.3204061745961228e-05, -0.000999369023951933],
        [-0.000999369023951933, 0.07954650644002863],
    ], dtype=float),
}
DESI_PLANCK_cov = DESI_PLANK_cov

PantheonSH0ES_cov = {
    'cov_omega_m_omega_m': 0.0003513006147410213,
    'cov_omega_m_H0': -0.0026497789652042587,
    'cov_H0_H0': 1.1406699901973247,
    'matrix': np.array([
        [0.0003513006147410213, -0.0026497789652042587],
        [-0.0026497789652042587, 1.1406699901973247],
    ], dtype=float),
}


DESI_PLANK_mean = {
    'omega_m': 0.302677,
    'H0': 68.171897,
}
DESI_PLANCK_mean = DESI_PLANK_mean

PantheonSH0ES_mean = {
    'omega_m': 0.332865,
    'H0': 73.502685,
}

COSMO_PRIORS = {
    'DESI_PLANCK': {
        'mean_vec': np.array([
            DESI_PLANCK_mean['omega_m'],
            DESI_PLANCK_mean['H0'],
        ], dtype=float),
        'cov': DESI_PLANCK_cov['matrix'],
    },
    'PantheonSH0ES': {
        'mean_vec': np.array([
            PantheonSH0ES_mean['omega_m'],
            PantheonSH0ES_mean['H0'],
        ], dtype=float),
        'cov': PantheonSH0ES_cov['matrix'],
    },
}

STEP3_COSMO_PRIOR = 'DESI_PLANCK'  # or 'PantheonSH0ES'
STEP3_COSMO_TAG = STEP3_COSMO_PRIOR.lower()
STEP3_SUFFIX = f"{suffix}_step6_{STEP3_COSMO_TAG}"
STEP6_RUN_TAG = datetime.now().strftime("%Y%m%d_%H")
HMC_OUTPUT_DIR = OUTPUT_ROOT / f"WFI2033{STEP3_SUFFIX}_{STEP6_RUN_TAG}"
STEP6_RESULT_DIR = SCRIPT_DIR / 'result' / f"result{STEP3_SUFFIX}_{STEP6_RUN_TAG}"
HMC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STEP6_RESULT_DIR.mkdir(parents=True, exist_ok=True)

Z_LENS = 0.6575
Z_SOURCE = 1.662

KAPPA_EXT_PRIOR = {
    'mean': 0.059,
    'sigma_minus': 0.047,
    'sigma_plus': 0.077,
    'low': -0.20,
    'high': 0.30,
}

ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
MPC_TO_KM = 3.0856775814913673e19
DAY_TO_S = 86400.0
TIME_DELAY_SCALE_DAYS = MPC_TO_KM * ARCSEC_TO_RAD**2 / (cosmo_tool.c_km_s * DAY_TO_S)

# Current image ordering assumption in this notebook:
# 1=A1, 2=A2, 3=B, 4=C
# so fpd_31 -> B-A1, fpd_32 -> B-A2, fpd_34 -> B-C
TIME_DELAY_OBS = {
    'dt_31_days': {
        'label': 'B-A1',
        'mean': -36.2,
        'sigma_minus': 2.3,
        'sigma_plus': 1.6,
    },
    'dt_32_days': {
        'label': 'B-A2',
        'mean': -37.3,
        'sigma_minus': 3.0,
        'sigma_plus': 2.6,
    },
    'dt_34_days': {
        'label': 'B-C',
        'mean': -59.4,
        'sigma_minus': 1.3,
        'sigma_plus': 1.3,
    },
}

# %% Cell 4
fixed_inner_three = [{
    'amp': np.array(fixed_first_three['amp'], dtype=float),
    'sigma': np.array(fixed_first_three['sigma'], dtype=float),
    'e1': np.array(fixed_first_three['e1'], dtype=float),
    'e2': np.array(fixed_first_three['e2'], dtype=float),
    'center_x': np.array(fixed_first_three['center_x'], dtype=float),
    'center_y': np.array(fixed_first_three['center_y'], dtype=float),
}]

outer_two = [{
    'amp': np.array(HMC_reference['amp_lens'].values, dtype=float)[-2:],
    'sigma': np.array(HMC_reference['sigma_lens'].values, dtype=float)[-2:],
    'e1': np.array(HMC_reference['e_lens'].values, dtype=float)[0, -2:],
    'e2': np.array(HMC_reference['e_lens'].values, dtype=float)[1, -2:],
    'center_x': np.array(HMC_reference['center_lens'].values, dtype=float)[0, -2:],
    'center_y': np.array(HMC_reference['center_lens'].values, dtype=float)[1, -2:],
}]

full_lens_light = [{
    k: np.concatenate([np.asarray(fixed_inner_three[0][k]), np.asarray(outer_two[0][k])])
    for k in ('amp', 'sigma', 'e1', 'e2', 'center_x', 'center_y')
}]

fixed_point_source = [{
    'ra': np.array(HMC_reference['ra_ps'].values, dtype=float),
    'dec': np.array(HMC_reference['dec_ps'].values, dtype=float),
    'amp': np.power(10.0, np.array(HMC_reference['log10_amp_ps'].values, dtype=float)),
}]

# Step 6 uses a single fixed PSF: the chain-median corrected PSF median from Step 5.
fixed_psf = np.array(HMC_reference['psf_kernel_corrected'].values, dtype=float)
fixed_psf = np.clip(fixed_psf, 0.0, None)
fixed_psf = fixed_psf / fixed_psf.sum()

epl_center = np.array(HMC_reference['center_1'].values, dtype=float).reshape(2,)


def chain_array(name, i):
    return HMC_median[name].isel(chain=i).values


fixed_sis_g1 = [{
    'theta_E': float(np.asarray(HMC_reference['theta_E_g1'].values).reshape(-1)[0]),
    'center_x': float(G1_MASS_CENTER[0]),
    'center_y': float(G1_MASS_CENTER[1]),
}]
fixed_sis_g2 = [{
    'theta_E': float(np.asarray(HMC_reference['theta_E_g2'].values).reshape(-1)[0]),
    'center_x': float(G2_MASS_CENTER[0]),
    'center_y': float(G2_MASS_CENTER[1]),
}]
fixed_sis_g3 = [{
    'theta_E': 0.088,
    'center_x': float(G3_MASS_CENTER[0]),
    'center_y': float(G3_MASS_CENTER[1]),
}]
fixed_sis_g7 = [{
    'theta_E': 0.388,
    'center_x': float(G7_MASS_CENTER[0]),
    'center_y': float(G7_MASS_CENTER[1]),
}]

pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny = get_pixel_grid(jnp.asarray(data_subtracted), pix_scale)
noise = Noise(nx, ny, exposure_time=exposure_time)
psf_obj = PSF(psf_type='PIXEL', kernel_point_source=fixed_psf)

mass_model_step6 = MassModel([
    'MULTI_GAUSSIAN_ELLIPSE_KAPPA',
    'CUSPY_NFW_ELLIPSE_KAPPA',
    'SHEAR',
    'SIS',
    'SIS',
    'SIS',
    'SIS',
])
source_light_model = LightModel(
    ['PIXELATED'],
    pixel_adaptive_grid=True,
    pixel_interpol='fast_bilinear',
    kwargs_pixelated={'num_pixels': pixel_grid_shape},
)
point_source_model = PointSourceModel(
    ['IMAGE_POSITIONS'],
    mass_model=mass_model_step6,
    image_plane=deepcopy(pixel_grid),
)
lens_image_step6 = LensImageExtension(
    deepcopy(pixel_grid),
    psf_obj,
    noise_class=noise,
    lens_light_model_class=LightModel(['MULTI_GAUSSIAN_ELLIPSE'], {}),
    lens_mass_model_class=mass_model_step6,
    source_model_class=source_light_model,
    point_source_model_class=point_source_model,
    source_arc_mask=jnp.array(mask_out),
    conjugate_points=conj_points,
    kwargs_numerics={'supersampling_factor': ss_factor},
    source_grid_scale=source_grid_scale,
)

# %% Cell 6
epl_mass_model = MassModel(['EPL'])
star_nfw_mass_model = MassModel(['MULTI_GAUSSIAN_ELLIPSE_KAPPA', 'CUSPY_NFW_ELLIPSE_KAPPA'])

SIS_PRIORS = {
    'g1': {
        'plate_name': 'Lens mass g1',
        'param_name': 'g1',
        'origin': G1_MASS_CENTER,
        'theta_low': 0.0,
        'theta_high': 0.25,
    },
    'g2': {
        'plate_name': 'Lens mass g2',
        'param_name': 'g2',
        'origin': G2_MASS_CENTER,
        'theta_mean': 0.622,
        'theta_low': 0.5,
        'theta_high': 0.7,
    },
    'g3': {
        'origin': G3_MASS_CENTER,
        'theta_mean': 0.088,
    },
    'g7': {
        'origin': G7_MASS_CENTER,
        'theta_mean': 0.388,
    },
}


def scale_theta_E_from_g2(theta_E_g2, target_prior):
    return theta_E_g2 * target_prior['theta_mean'] / SIS_PRIORS['g2']['theta_mean']


FIXED_SIS_THETA_E = {
    'g1': float(np.asarray(HMC_reference['theta_E_g1'].values).reshape(-1)[0]),
    'g2': float(np.asarray(HMC_reference['theta_E_g2'].values).reshape(-1)[0]),
}


def target_epl_kwargs_for_chain(i):
    center_i = np.asarray(chain_array('center_1', i), dtype=float).reshape(2,)
    e_i = np.asarray(chain_array('e_1', i), dtype=float).reshape(2, -1)
    return [{
        'theta_E': float(np.asarray(chain_array('theta_E_1', i)).reshape(-1)[0]),
        'gamma': float(np.asarray(chain_array('gamma_1', i)).reshape(-1)[0]),
        'e1': float(e_i[0, 0]),
        'e2': float(e_i[1, 0]),
        'center_x': float(center_i[0]),
        'center_y': float(center_i[1]),
    }]


def scaled_mass_from_light(m2l_ratio):
    mass_from_light = deepcopy(full_lens_light)
    amp = jnp.asarray(mass_from_light[0]['amp'])
    mass_from_light[0]['amp'] = amp * jnp.asarray(m2l_ratio, dtype=jnp.float64) / jnp.sum(amp)
    return mass_from_light


def build_star_nfw_kwargs_ls(m2l_ratio, kappa_s_halo, e1_halo, e2_halo, center_i):
    mass_from_light = scaled_mass_from_light(m2l_ratio)
    halo = {
        'R_s': 5.0,
        'gamma': 1.0,
        'kappa_s': float(kappa_s_halo),
        'e1': float(e1_halo),
        'e2': float(e2_halo),
        'center_x': float(center_i[0]),
        'center_y': float(center_i[1]),
    }
    return mass_from_light + [halo]


def build_step6_mass_init(i):
    center_i = np.asarray(chain_array('center_1', i), dtype=float).reshape(2,)
    target_epl_kwargs_i = target_epl_kwargs_for_chain(i)
    target_epl_kappa_i = np.array(epl_mass_model.kappa(xgrid, ygrid, target_epl_kwargs_i), dtype=float)

    def ls_residual(theta):
        model_kappa = np.array(
            star_nfw_mass_model.kappa(
                xgrid,
                ygrid,
                build_star_nfw_kwargs_ls(theta[0], theta[1], theta[2], theta[3], center_i),
            ),
            dtype=float,
        )
        return (model_kappa - target_epl_kappa_i)[mask_out].ravel()

    ls_x0 = np.array([
        2.0,
        0.1,
        target_epl_kwargs_i[0]['e1'],
        target_epl_kwargs_i[0]['e2'],
    ], dtype=float)
    ls_result = least_squares(
        ls_residual,
        x0=ls_x0,
        bounds=([0.0, 0.0, -0.2, -0.2], [14.0, 1.0, 0.2, 0.2]),
    )
    init_i = {
        'm2l_ratio': jnp.asarray(ls_result.x[0], dtype=jnp.float64),
        'kappa_s_halo': jnp.asarray(ls_result.x[1], dtype=jnp.float64),
        'gammain_halo': jnp.asarray(1.0, dtype=jnp.float64),
        'e_halo': jnp.asarray(ls_result.x[2:4], dtype=jnp.float64),
    }
    return init_i, ls_result


STEP6_MASS_INITS = []
STEP6_LS_RESULTS = []
for i in range(num_chains):
    init_i, ls_result_i = build_step6_mass_init(i)
    STEP6_MASS_INITS.append(init_i)
    STEP6_LS_RESULTS.append(ls_result_i)
    print(
        f'chain {i}: '
        f'm2l_ratio={float(ls_result_i.x[0]):.4f}, '
        f'kappa_s_halo={float(ls_result_i.x[1]):.4f}, '
        f'e1={float(ls_result_i.x[2]):.4f}, '
        f'e2={float(ls_result_i.x[3]):.4f}, '
        f'success={bool(ls_result_i.success)}, '
        f'cost={float(ls_result_i.cost):.4f}'
    )


STEP1_BASE_KWARGS = {
    'label': 'step1',
    'gnfw_kwargs': {
        'gamma_in_up': 2,
        'gamma_in_low': 0.6,
        'Rs_value': 5.0,
        'sph': False,
        'gamma_sheer_low': -0.5,
        'gamma_sheer_high': 0.5,
    },
    'fixed_sis_theta_E': FIXED_SIS_THETA_E,
    'free_sis': (),
    'scale_with_g2': True,
    'free_point_source': False,
    'enable_psf_corr': False,
    'use_conjugate_prior': False,
    'compute_fermat_diffs': False,
    'use_ml_gradient': False,
}


def build_step1_stage_kwargs(i):
    center_i = np.asarray(chain_array('center_1', i), dtype=float).reshape(2,)
    shear_i = np.asarray(chain_array('gamma_sheer_1', i), dtype=float).reshape(2,)
    return STEP1_BASE_KWARGS | {
        'gnfw_kwargs': STEP1_BASE_KWARGS['gnfw_kwargs'] | {
            'center_x': float(center_i[0]),
            'center_y': float(center_i[1]),
            'gamma_sheer_value': shear_i,
        }
    }


STEP2_KWARGS = {
    'label': 'step2',
    'gnfw_kwargs': {
        'gamma_in_up': 2,
        'gamma_in_low': 0.6,
        'Rs_low': 2,
        'Rs_high': 20.0,
        'sph': False,
        'gamma_sheer_low': -0.5,
        'gamma_sheer_high': 0.5,
    },
    'fixed_sis_theta_E': {},
    'free_sis': ('g1', 'g2'),
    'scale_with_g2': True,
    'free_point_source': True,
    'enable_psf_corr': False,
    'use_conjugate_prior': True,
    'compute_fermat_diffs': True,
    'use_ml_gradient': False,
}

# %% Cell 7
def fixed_sis(theta_E, origin):
    return [{
        'theta_E': theta_E,
        'center_x': float(origin[0]),
        'center_y': float(origin[1]),
    }]


def split_normal_logpdf(x, mean, sigma_minus, sigma_plus):
    """Two-piece normal logpdf with asymmetric widths."""
    x = jnp.asarray(x, dtype=jnp.float64)
    mean = jnp.asarray(mean, dtype=jnp.float64)
    sigma_minus = jnp.asarray(sigma_minus, dtype=jnp.float64)
    sigma_plus = jnp.asarray(sigma_plus, dtype=jnp.float64)

    sigma = jnp.where(x < mean, sigma_minus, sigma_plus)
    log_norm = jnp.log(jnp.sqrt(2.0 / jnp.pi)) - jnp.log(sigma_minus + sigma_plus)
    return log_norm - 0.5 * ((x - mean) / sigma) ** 2


def sample_cosmology_from_prior(stage_kwargs):
    prior_name = stage_kwargs['cosmo_prior_name']
    prior = COSMO_PRIORS[prior_name]
    cosmo_vec = numpyro.sample(
        'cosmo_vec',
        dist.MultivariateNormal(
            loc=jnp.asarray(prior['mean_vec'], dtype=jnp.float64),
            covariance_matrix=jnp.asarray(prior['cov'], dtype=jnp.float64),
        ),
    )
    omega_m = numpyro.deterministic('omega_m_cosmo', cosmo_vec[0])
    H0 = numpyro.deterministic('H0_cosmo', cosmo_vec[1])
    return {
        'Omegam': omega_m,
        'Omegak': jnp.asarray(0.0, dtype=jnp.float64),
        'w0': jnp.asarray(-1.0, dtype=jnp.float64),
        'wa': jnp.asarray(0.0, dtype=jnp.float64),
        'h0': H0,
    }


def compute_time_delay_distances(cosmology, kappa_ext):
    Dl, Ds, Dls = cosmo_tool.dldsdls(Z_LENS, Z_SOURCE, cosmology)
    D_dt_true = (1.0 + Z_LENS) * Dl * Ds / Dls
    D_dt_model = (1.0 - kappa_ext) * D_dt_true
    return D_dt_true, D_dt_model


def add_time_delay_likelihood(fpd_31, fpd_32, fpd_34, stage_kwargs):
    cosmology = sample_cosmology_from_prior(stage_kwargs)

    kappa_ext = numpyro.sample(
        'kappa_ext',
        dist.Uniform(
            low=jnp.asarray(KAPPA_EXT_PRIOR['low'], dtype=jnp.float64),
            high=jnp.asarray(KAPPA_EXT_PRIOR['high'], dtype=jnp.float64),
        ),
    )
    numpyro.factor(
        'kappa_ext_prior',
        split_normal_logpdf(
            kappa_ext,
            KAPPA_EXT_PRIOR['mean'],
            KAPPA_EXT_PRIOR['sigma_minus'],
            KAPPA_EXT_PRIOR['sigma_plus'],
        ),
    )

    D_dt_true, D_dt_model = compute_time_delay_distances(cosmology, kappa_ext)
    numpyro.deterministic('D_dt_true_Mpc', D_dt_true)
    numpyro.deterministic('D_dt_model_Mpc', D_dt_model)

    prefactor_days = numpyro.deterministic(
        'time_delay_prefactor_days',
        D_dt_model * jnp.asarray(TIME_DELAY_SCALE_DAYS, dtype=jnp.float64),
    )
    dt_31 = numpyro.deterministic('dt_31_days', prefactor_days * fpd_31)
    dt_32 = numpyro.deterministic('dt_32_days', prefactor_days * fpd_32)
    dt_34 = numpyro.deterministic('dt_34_days', prefactor_days * fpd_34)

    numpyro.factor(
        'dt_31_like',
        split_normal_logpdf(
            dt_31,
            TIME_DELAY_OBS['dt_31_days']['mean'],
            TIME_DELAY_OBS['dt_31_days']['sigma_minus'],
            TIME_DELAY_OBS['dt_31_days']['sigma_plus'],
        ),
    )
    numpyro.factor(
        'dt_32_like',
        split_normal_logpdf(
            dt_32,
            TIME_DELAY_OBS['dt_32_days']['mean'],
            TIME_DELAY_OBS['dt_32_days']['sigma_minus'],
            TIME_DELAY_OBS['dt_32_days']['sigma_plus'],
        ),
    )
    numpyro.factor(
        'dt_34_like',
        split_normal_logpdf(
            dt_34,
            TIME_DELAY_OBS['dt_34_days']['mean'],
            TIME_DELAY_OBS['dt_34_days']['sigma_minus'],
            TIME_DELAY_OBS['dt_34_days']['sigma_plus'],
        ),
    )


def build_stage_sis(stage_kwargs):
    sis_mass = []
    fixed_sis_theta_E = dict(stage_kwargs['fixed_sis_theta_E'])

    if 'g1' in fixed_sis_theta_E:
        sis_mass += fixed_sis(fixed_sis_theta_E.pop('g1'), G1_MASS_CENTER)

    if 'g2' in fixed_sis_theta_E:
        theta_E_g2 = jnp.asarray(fixed_sis_theta_E.pop('g2'), dtype=jnp.float64)
        sis_mass += fixed_sis(theta_E_g2, G2_MASS_CENTER)
        if stage_kwargs['scale_with_g2']:
            theta_E_g3 = numpyro.deterministic('theta_E_g3', scale_theta_E_from_g2(theta_E_g2, SIS_PRIORS['g3']))
            theta_E_g7 = numpyro.deterministic('theta_E_g7', scale_theta_E_from_g2(theta_E_g2, SIS_PRIORS['g7']))
            sis_mass += fixed_sis(theta_E_g3, G3_MASS_CENTER)
            sis_mass += fixed_sis(theta_E_g7, G7_MASS_CENTER)

    for name, theta_E in fixed_sis_theta_E.items():
        sis_mass += fixed_sis(theta_E, SIS_PRIORS[name]['origin'])

    if 'g1' in stage_kwargs['free_sis']:
        sis_mass += SIS(**SIS_PRIORS['g1'])

    if 'g2' in stage_kwargs['free_sis']:
        g2_mass = SIS(**SIS_PRIORS['g2'])
        sis_mass += g2_mass
        if stage_kwargs['scale_with_g2']:
            theta_E_g2 = g2_mass[0]['theta_E']
            theta_E_g3 = numpyro.deterministic('theta_E_g3', scale_theta_E_from_g2(theta_E_g2, SIS_PRIORS['g3']))
            theta_E_g7 = numpyro.deterministic('theta_E_g7', scale_theta_E_from_g2(theta_E_g2, SIS_PRIORS['g7']))
            sis_mass += fixed_sis(theta_E_g3, G3_MASS_CENTER)
            sis_mass += fixed_sis(theta_E_g7, G7_MASS_CENTER)

    return sis_mass


def build_stage_point_source(stage_kwargs):
    if not stage_kwargs['free_point_source']:
        return fixed_point_source

    n_ps = conj_points.shape[0]
    ra_ps = numpyro.sample(
        'ra_ps',
        dist.TruncatedNormal(
            loc=conj_points[:, 0],
            scale=POINT_SOURCE_PRIOR['pos_sigma'],
            low=conj_points[:, 0] - POINT_SOURCE_PRIOR['pos_window'],
            high=conj_points[:, 0] + POINT_SOURCE_PRIOR['pos_window'],
        ),
    )
    dec_ps = numpyro.sample(
        'dec_ps',
        dist.TruncatedNormal(
            loc=conj_points[:, 1],
            scale=POINT_SOURCE_PRIOR['pos_sigma'],
            low=conj_points[:, 1] - POINT_SOURCE_PRIOR['pos_window'],
            high=conj_points[:, 1] + POINT_SOURCE_PRIOR['pos_window'],
        ),
    )
    log10_amp_ps = numpyro.sample(
        'log10_amp_ps',
        dist.Uniform(
            POINT_SOURCE_PRIOR['log10_amp_low'],
            POINT_SOURCE_PRIOR['log10_amp_high'],
        ).expand([n_ps]),
    )
    return [{
        'ra': ra_ps,
        'dec': dec_ps,
        'amp': jnp.power(10.0, log10_amp_ps),
    }]


def build_stage_psf():
    psf_kernel = jnp.asarray(fixed_psf)
    numpyro.deterministic('psf_corr_factor_field', jnp.ones_like(psf_kernel))
    numpyro.deterministic('psf_kernel_corrected', psf_kernel)
    return psf_kernel


def maybe_add_conjugate_prior(stage_kwargs, kwargs_lens, kwargs_point_source):
    if not stage_kwargs['use_conjugate_prior']:
        return
    src_x_ps, src_y_ps = lens_image_step6.MassModel.ray_shooting(
        kwargs_point_source[0]['ra'],
        kwargs_point_source[0]['dec'],
        kwargs_lens,
    )
    conj_points_model = jnp.stack([src_x_ps, src_y_ps], axis=1)
    conj_distance = reduced_distance_matrix(conj_points_model)
    nc = conj_distance.shape[0]
    with numpyro.plate(f'Conjugate points 2 - [{nc}]', nc):
        numpyro.sample('conjugate_points', dist.Exponential(CONJUGATE_POINT_PRIOR['rate']), obs=conj_distance)




def gradient_mass_from_light(m2l_ratio, m2l_ratio_slope):
    mass_from_light = deepcopy(full_lens_light)
    sigma = jnp.asarray(mass_from_light[0]['sigma'], dtype=jnp.float64)
    amp = jnp.asarray(mass_from_light[0]['amp'], dtype=jnp.float64)
    R_factor = jnp.power(sigma, jnp.asarray(m2l_ratio_slope, dtype=jnp.float64))
    ml_gauss = numpyro.deterministic('ml_gauss', jnp.asarray(m2l_ratio, dtype=jnp.float64) * R_factor)
    mass_from_light[0]['amp'] = numpyro.deterministic(
        'mass_from_light_amp',
        amp / jnp.sum(amp) * ml_gauss,
    )
    numpyro.deterministic('total_stellar_mass', jnp.sum(mass_from_light[0]['amp']))
    return mass_from_light


def model_step6(data_subtracted, stage_kwargs):
    m2l_ratio = numpyro.sample('m2l_ratio', dist.Uniform(0.0, 3.0))
    if stage_kwargs.get('use_ml_gradient', False):
        m2l_ratio_slope = numpyro.sample(
            'm2l_ratio_slope',
            dist.Uniform(
                stage_kwargs['m2l_ratio_slope_low'],
                stage_kwargs['m2l_ratio_slope_high'],
            ),
        )
        mass_from_light = gradient_mass_from_light(m2l_ratio, m2l_ratio_slope)
    else:
        mass_from_light = scaled_mass_from_light(m2l_ratio)

    gnfw_shear = GNFW_w_shear(
        'Lens mass',
        'halo',
        **stage_kwargs['gnfw_kwargs'],
    )
    kwargs_source = [
        matern_power_spectrum(
            k=k_values,
            **SOURCE_GRID_PRIOR,
        )
    ]
    kwargs_lens = mass_from_light + gnfw_shear + build_stage_sis(stage_kwargs)
    kwargs_point_source = build_stage_point_source(stage_kwargs)
    maybe_add_conjugate_prior(stage_kwargs, kwargs_lens, kwargs_point_source)

    if stage_kwargs['compute_fermat_diffs']:
        fermat = lens_image_step6.MassModel.fermat_potential(
            kwargs_point_source[0]['ra'],
            kwargs_point_source[0]['dec'],
            kwargs_lens,
        )
        numpyro.deterministic('fermat_potential_images', fermat)
        fpd_31 = numpyro.deterministic('fpd_31', fermat[2] - fermat[0])
        fpd_32 = numpyro.deterministic('fpd_32', fermat[2] - fermat[1])
        fpd_34 = numpyro.deterministic('fpd_34', fermat[2] - fermat[3])

        if stage_kwargs.get('use_time_delay_likelihood', False):
            add_time_delay_likelihood(fpd_31, fpd_32, fpd_34, stage_kwargs)

    psf_kernel = build_stage_psf()

    model_image = lens_image_step6.model(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        kwargs_lens_light=[],
        kwargs_point_source=kwargs_point_source,
        source_add=True,
        lens_light_add=False,
        point_source_add=True,
    )
    numpyro.deterministic('model_image', model_image)

    with numpyro.plate(f'data - [{int(mask_out.sum())}]', int(mask_out.sum())):
        numpyro.sample('obs', dist.Normal(model_image[mask_out], jnp.asarray(rms_file)[mask_out]), obs=jnp.asarray(data_subtracted)[mask_out])


def evaluate_step6(params, stage_kwargs):
    seeded = numpyro.handlers.seed(model_step6, jax.random.PRNGKey(0))
    substituted = numpyro.handlers.substitute(seeded, data=params)
    trace = numpyro.handlers.trace(substituted).get_trace(data_subtracted, stage_kwargs)
    fermat_diffs = None
    if stage_kwargs['compute_fermat_diffs']:
        fermat_diffs = {
            'fpd_31': float(np.asarray(trace['fpd_31']['value'])),
            'fpd_32': float(np.asarray(trace['fpd_32']['value'])),
            'fpd_34': float(np.asarray(trace['fpd_34']['value'])),
        }
        if stage_kwargs.get('use_time_delay_likelihood', False):
            extra_keys = [
                'omega_m_cosmo',
                'H0_cosmo',
                'kappa_ext',
                'D_dt_true_Mpc',
                'D_dt_model_Mpc',
                'dt_31_days',
                'dt_32_days',
                'dt_34_days',
            ]
            for key in extra_keys:
                fermat_diffs[key] = float(np.asarray(trace[key]['value']))
    return (
        np.array(trace['model_image']['value']),
        np.array(trace['pixels_source_grid']['value']),
        np.array(trace['psf_kernel_corrected']['value']),
        fermat_diffs,
    )


def build_source_init_values(i):
    return {
        'n_source_grid': jnp.asarray(chain_array('n_source_grid', i), dtype=jnp.float64),
        'sigma_source_grid': jnp.asarray(chain_array('sigma_source_grid', i), dtype=jnp.float64),
        'rho_source_grid': jnp.asarray(chain_array('rho_source_grid', i), dtype=jnp.float64),
        'pixels_wn_source_grid': jnp.asarray(chain_array('pixels_wn_source_grid', i), dtype=jnp.float64),
    }


STEP1_LATENT_KEYS = (
    'm2l_ratio',
    'kappa_s_halo',
    'gammain_halo',
    'e_halo',
    'n_source_grid',
    'sigma_source_grid',
    'rho_source_grid',
    'pixels_wn_source_grid',
)

STEP2_LATENT_KEYS = STEP1_LATENT_KEYS + (
    'Rs_halo',
    'center_halo',
    'gamma_sheer_halo',
    'theta_E_g1',
    'theta_E_g2',
    'ra_ps',
    'dec_ps',
    'log10_amp_ps',
)


def select_init_values(params, allowed_keys):
    return {k: params[k] for k in allowed_keys if k in params}


def build_step1_init_values(i):
    return build_source_init_values(i) | STEP6_MASS_INITS[i]


def build_step2_extra_init(i):
    return {
        'Rs_halo': jnp.asarray(5.0, dtype=jnp.float64),
        'center_halo': jnp.asarray(chain_array('center_1', i), dtype=jnp.float64).reshape(2,),
        'gamma_sheer_halo': jnp.asarray(chain_array('gamma_sheer_1', i), dtype=jnp.float64).reshape(2,),
        'theta_E_g1': jnp.asarray(chain_array('theta_E_g1', i), dtype=jnp.float64),
        'theta_E_g2': jnp.asarray(chain_array('theta_E_g2', i), dtype=jnp.float64),
        'ra_ps': jnp.asarray(chain_array('ra_ps', i), dtype=jnp.float64),
        'dec_ps': jnp.asarray(chain_array('dec_ps', i), dtype=jnp.float64),
        'log10_amp_ps': jnp.asarray(chain_array('log10_amp_ps', i), dtype=jnp.float64),
    }


def _stack_or_none(*xs):
    first = xs[0]
    return None if first is None else jnp.stack(xs)


def sanitize_label(label):
    return (
        str(label)
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('/', '_')
    )


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, '_asdict'):
        return {str(k): to_serializable(v) for k, v in obj._asdict().items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, jax.Array):
        return np.asarray(obj).tolist()
    return obj


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        json.dump(to_serializable(payload), fh, indent=2, sort_keys=True)


def infer_dims(name, array, leading_dims=()):
    arr = np.asarray(array)
    trailing = arr.ndim - len(leading_dims)
    if trailing < 0:
        raise ValueError(f'Array for {name} has ndim={arr.ndim} but leading_dims={leading_dims}')
    return tuple(leading_dims) + tuple(f'{name}_dim_{i}' for i in range(trailing))


def mapping_to_dataset(mapping, leading_dims=()):
    data_vars = {}
    coords = {}
    if mapping:
        first_arr = np.asarray(next(iter(mapping.values())))
        for i, dim_name in enumerate(leading_dims):
            coords[dim_name] = np.arange(first_arr.shape[i])
    for name, values in mapping.items():
        arr = np.asarray(values)
        data_vars[name] = (infer_dims(name, arr, leading_dims), arr)
    return xr.Dataset(data_vars=data_vars, coords=coords)


def list_of_dicts_to_dataset(dict_list):
    stacked = {}
    for key in dict_list[0]:
        stacked[key] = np.stack([np.asarray(item[key]) for item in dict_list], axis=0)
    return mapping_to_dataset(stacked, leading_dims=('chain',))

def stage_output_dir(stage_output):
    return STEP6_RESULT_DIR / sanitize_label(stage_output['label'])


def collect_stage_losses(stage_output):
    losses = {}
    for i, result in enumerate(stage_output['results']):
        losses[f'chain_{i}'] = np.asarray(result.losses, dtype=float)
    return losses


def save_stage_artifacts(stage_output):
    out_dir = stage_output_dir(stage_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    list_of_dicts_to_dataset(stage_output['medians']).to_netcdf(out_dir / 'medians.nc')
    list_of_dicts_to_dataset(stage_output['init_values_list']).to_netcdf(out_dir / 'init_values.nc')
    list_of_dicts_to_dataset(stage_output['guide_params_list']).to_netcdf(out_dir / 'guide_params.nc')
    mapping_to_dataset(
        collect_stage_losses(stage_output),
        leading_dims=('draw',),
    ).to_netcdf(out_dir / 'losses.nc')
    write_json(out_dir / 'stage_kwargs.json', stage_output['stage_kwargs_list'])
    write_json(
        out_dir / 'stage_metadata.json',
        {
            'label': stage_output['label'],
            'max_iterations': stage_output['max_iterations'],
            'num_chains': num_chains,
        },
    )


def run_stage(stage_kwargs_or_builder, init_builder, max_iterations, seed):
    scheduler = optax.exponential_decay(init_value=5e-3, transition_steps=300, decay_rate=0.99)
    optim = optax.adabelief(learning_rate=scheduler)
    loss = infer.TraceMeanField_ELBO()

    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, num_chains)
    results = []
    medians = []
    stage_kwargs_list = []
    init_values_list = []
    guide_params_list = []

    for i in range(num_chains):
        stage_kwargs_i = stage_kwargs_or_builder(i) if callable(stage_kwargs_or_builder) else stage_kwargs_or_builder
        init_values = init_builder(i)
        init_fun = init_to_value_or_defer(values=init_values)
        guide = autoguide.AutoDiagonalNormal(model_step6, init_loc_fn=init_fun, init_scale=0.01)
        svi = infer.SVI(model_step6, guide, optim, loss)
        result = svi.run(keys[i], max_iterations, data_subtracted, stage_kwargs_i, progress_bar=False, stable_update=True)
        results.append(result)
        medians.append(guide.median(result.params))
        stage_kwargs_list.append(stage_kwargs_i)
        init_values_list.append(init_values)
        guide_params_list.append(result.params)

    multi_results = jax.tree.map(_stack_or_none, *results)
    return {
        'label': stage_kwargs_list[0]['label'],
        'results': results,
        'multi_results': multi_results,
        'medians': medians,
        'stage_kwargs_list': stage_kwargs_list,
        'init_values_list': init_values_list,
        'guide_params_list': guide_params_list,
        'max_iterations': max_iterations,
    }


def plot_stage_losses(stage_output, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 3.5))
    axins = ax.inset_axes([0.3, 0.5, 0.64, 0.45])
    for losses in stage_output['multi_results'].losses:
        _ = plot_loss(losses, stage_output['max_iterations'], ax=ax, axins=axins, alpha=0.25)
    fig.tight_layout()
    out_path = output_dir / f"{sanitize_label(stage_output['label'])}_loss.png"
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"Saved stage loss figure: {out_path}")
    plt.close(fig)


def plot_stage_results(stage_output, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, params in enumerate(stage_output['medians']):
        stage_kwargs_i = stage_output['stage_kwargs_list'][i]
        model_image, source_pixels, psf_kernel, fermat_diffs = evaluate_step6(params, stage_kwargs_i)
        residual = (data_subtracted - model_image) / rms_file

        title = (
            f"{stage_output['label']} | chain {i} | M/L = {float(np.asarray(params['m2l_ratio']).reshape(-1)[0]):.3f} | "
            f"kappa_s = {float(np.asarray(params['kappa_s_halo']).reshape(-1)[0]):.4f} | "
            f"gamma = {float(np.asarray(params['gammain_halo']).reshape(-1)[0]):.4f}"
        )
        if stage_kwargs_i.get('use_ml_gradient', False) and 'm2l_ratio_slope' in params:
            title += f" | M/L slope = {float(np.asarray(params['m2l_ratio_slope']).reshape(-1)[0]):.4f}"
        if fermat_diffs is not None:
            title += (
                f" | φ3-1 = {fermat_diffs['fpd_31']:.4f}"
                f" | φ3-2 = {fermat_diffs['fpd_32']:.4f}"
                f" | φ3-4 = {fermat_diffs['fpd_34']:.4f}"
            )
            if stage_kwargs_i.get('use_time_delay_likelihood', False):
                title += (
                    f" | Ωm = {fermat_diffs['omega_m_cosmo']:.4f}"
                    f" | H0 = {fermat_diffs['H0_cosmo']:.2f}"
                    f" | κext = {fermat_diffs['kappa_ext']:.3f}"
                    f" | Δt31 = {fermat_diffs['dt_31_days']:.2f} d"
                    f" | Δt32 = {fermat_diffs['dt_32_days']:.2f} d"
                    f" | Δt34 = {fermat_diffs['dt_34_days']:.2f} d"
                )

        ncols = 5 if stage_kwargs_i['enable_psf_corr'] else 4
        fig, ax = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
        fig.suptitle(title, y=1.02)

        ax[0].imshow(np.ma.array(data_subtracted, mask=~mask_out), origin='lower', extent=extent, cmap='twilight', norm='log')
        ax[0].set_title('data - lens light')

        ax[1].imshow(np.ma.array(model_image, mask=~mask_out), origin='lower', extent=extent, cmap='twilight', norm='log')
        ax[1].set_title('model')

        im = ax[2].imshow(np.ma.array(residual, mask=~mask_out), origin='lower', extent=extent, cmap='bwr', vmin=-3, vmax=3)
        ax[2].set_title('residual / rms')
        plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

        ax[3].imshow(source_pixels, origin='lower', cmap='twilight')
        ax[3].set_title('source')

        if stage_kwargs_i['enable_psf_corr']:
            ax[4].imshow(psf_kernel, origin='lower', cmap='twilight', norm='log')
            ax[4].set_title('corrected psf')

        plt.tight_layout()
        out_path = output_dir / f"{sanitize_label(stage_output['label'])}_chain_{i:02d}.png"
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        print(f"Saved stage result figure: {out_path}")
        plt.close(fig)

# %% Cell 8
step1_output = run_stage(
    build_step1_stage_kwargs,
    init_builder=build_step1_init_values,
    max_iterations=20000,
    seed=1234,
)
plot_stage_losses(step1_output, STEP6_RESULT_DIR)
plot_stage_results(step1_output, STEP6_RESULT_DIR)
save_stage_artifacts(step1_output)


# %% Cell 9
def build_step2_init_values(i):
    return select_init_values(step1_output['medians'][i], STEP1_LATENT_KEYS) | build_step2_extra_init(i)


step2_output = run_stage(
    STEP2_KWARGS,
    init_builder=build_step2_init_values,
    max_iterations=20000,
    seed=2234,
)
plot_stage_losses(step2_output, STEP6_RESULT_DIR)
plot_stage_results(step2_output, STEP6_RESULT_DIR)
save_stage_artifacts(step2_output)


# %% Cell 10
STEP3_KWARGS = {
    **STEP2_KWARGS,
    'label': f"step3 ({STEP3_SUFFIX})",
    'use_time_delay_likelihood': True,
    'cosmo_prior_name': STEP3_COSMO_PRIOR,
    'use_ml_gradient': True,
    'm2l_ratio_slope_low': -0.6,
    'm2l_ratio_slope_high': 0.6,
}

STEP3_HMC_KWARGS = {
    **STEP3_KWARGS,
    'use_conjugate_prior': False,
}


def build_step3_init_values(i):
    cosmo_prior = COSMO_PRIORS[STEP3_KWARGS['cosmo_prior_name']]
    return select_init_values(step2_output['medians'][i], STEP2_LATENT_KEYS) | {
        'cosmo_vec': jnp.asarray(cosmo_prior['mean_vec'], dtype=jnp.float64),
        'kappa_ext': jnp.asarray(KAPPA_EXT_PRIOR['mean'], dtype=jnp.float64),
        'm2l_ratio_slope': jnp.asarray(0.0, dtype=jnp.float64),
    }


step3_output = run_stage(
    STEP3_KWARGS,
    init_builder=build_step3_init_values,
    max_iterations=20000,
    seed=3234,
)
plot_stage_losses(step3_output, STEP6_RESULT_DIR)
plot_stage_results(step3_output, STEP6_RESULT_DIR)
save_stage_artifacts(step3_output)



# %% Stage 3 HMC
from numpyro.infer import NUTS, MCMC
def stack_dicts(dict_list):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *dict_list)



# Keep the input suffix for reading Step-5 products unchanged.
# Use a separate HMC suffix for Step-6 stage3 outputs.
suffix_hmc = STEP3_SUFFIX
print('HMC output dir:', HMC_OUTPUT_DIR)
print('HMC suffix:', suffix_hmc)

multi_svi_stage3_median = stack_dicts(step3_output['medians'])

vars_pixel = ['pixels_wn_source_grid']
vars_power = ['n_source_grid', 'rho_source_grid', 'sigma_source_grid']
vars_mass = [
    'm2l_ratio',
    'm2l_ratio_slope',
    'kappa_s_halo',
    'gammain_halo',
    'e_halo',
    'Rs_halo',
    'center_halo',
    'gamma_sheer_halo',
    'theta_E_g1',
    'theta_E_g2',
]
vars_point_source = ['ra_ps', 'dec_ps', 'log10_amp_ps']
vars_cosmo = ['cosmo_vec', 'kappa_ext']

# Match Step-5 HMC init logic: only pass true latent/sample sites into HMC init.
multi_svi_stage3_median_vars = {
    k: multi_svi_stage3_median[k]
    for k in vars_pixel + vars_power + vars_mass + vars_point_source + vars_cosmo
    if k in multi_svi_stage3_median
}

unconstrained_stage3_median = jax.vmap(
    lambda p: infer.util.unconstrain_fn(
        model_step6,
        (data_subtracted, STEP3_HMC_KWARGS),
        {},
        p,
    )
)(multi_svi_stage3_median_vars)

unconstrained_stage3_median = {
    k: jnp.asarray(v, dtype=jnp.float64)
    for k, v in unconstrained_stage3_median.items()
}

init_fun_hmc = init_to_value_or_defer(values=get_value_from_index(multi_svi_stage3_median_vars, 0))

stage3_kernel = NUTS(
    model_step6,
    init_strategy=init_fun_hmc,
    target_accept_prob=0.90,
    max_tree_depth=10,
    dense_mass=[
        ('n_source_grid', 'rho_source_grid', 'sigma_source_grid'),
        (
            'm2l_ratio',
            'm2l_ratio_slope',
            'kappa_s_halo',
            'gammain_halo',
            'e_halo',
            'Rs_halo',
            'center_halo',
            'gamma_sheer_halo',
            'theta_E_g1',
            'theta_E_g2',
            'cosmo_vec',
            'kappa_ext',
        ),
        ('ra_ps', 'dec_ps', 'log10_amp_ps'),
    ],
)

num_warmup = 1500
num_samples = 1000
batch_number = 4

rng_key_hmc = jax.random.PRNGKey(5252)

mcmc_stage3 = MCMC(
    stage3_kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,
    progress_bar=True,
    chain_method='vectorized',
)

batch_list = []
for i in range(batch_number):
    if i == 0:
        mcmc_stage3.run(
            rng_key_hmc,
            data_subtracted,
            STEP3_HMC_KWARGS,
            init_params=unconstrained_stage3_median,
        )
    else:
        mcmc_stage3.post_warmup_state = mcmc_stage3.last_state
        mcmc_stage3.run(
            mcmc_stage3.post_warmup_state.rng_key,
            data_subtracted,
            STEP3_HMC_KWARGS,
        )

    mcmc_stage3._states = jax.device_get(mcmc_stage3._states)
    mcmc_stage3._states_flat = jax.device_get(mcmc_stage3._states_flat)
    inf_data_batch = az.from_numpyro(mcmc_stage3)
    batch_path = HMC_OUTPUT_DIR / f'WFI2033_{i}{suffix_hmc}.nc'
    inf_data_batch.to_netcdf(batch_path)
    print(f'Saved HMC batch to: {batch_path}')
    batch_list.append(inf_data_batch)

inf_data_all = batch_list[0] if len(batch_list) == 1 else az.concat(*batch_list, dim='draw')
final_hmc_path = HMC_OUTPUT_DIR / f'WFI2033_all{suffix_hmc}.nc'
inf_data_all.to_netcdf(final_hmc_path)
print(f'Saved final HMC inf_data to: {final_hmc_path}')
