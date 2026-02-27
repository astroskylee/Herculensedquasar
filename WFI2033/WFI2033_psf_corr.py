#!/usr/bin/env python3
"""RXJ1131 unified pipeline exported to script, up to and including HMC batches.

Workflow:
1) Load data and PSF.
2) Run parametric SVI.
3) Build pixel-grid initialization.
4) Run pixelated SVI stage-1 (no PSF correction).
5) Run pixelated SVI stage-2 (with PSF correction).
6) Run HMC in batches and save each batch.
7) Concatenate all batches and save WFI2033_all.nc.
"""

from pathlib import Path
import os
import warnings


# Run from script directory for relative paths (e.g. ./psf_data).
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

warnings.simplefilter("ignore")

# HPC/shared-filesystem safety for netCDF/HDF5 writers/readers.

from herculens_import_main import *
import jax
import numpyro
jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
if 'graphviz' not in os.environ['PATH']:
    os.environ['PATH'] = '/opt/apps/pkgs/graphviz/12.2.1/intel64/gnu_12.2.0/bin:' + os.environ['PATH']


import arviz as az
from copy import deepcopy
from numpyro.infer import NUTS, MCMC
from custom_gibbs import MultiHMCGibbs
from herculens.PointSourceModel.point_source_model import PointSourceModel





# -----------------------------------------------------------------------------
# Data and PSF
# -----------------------------------------------------------------------------
pix_scale = 0.031  # arcsec / pixel (JWST)

DATA_DIR = "../../Data/WFI2033"
raw_data_path = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d.fits")
data_path = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d_cut_x6985_y3594_150.fits")
mask_path = os.path.join(DATA_DIR, "mask.fits")

with fits.open(raw_data_path, memmap=True) as hdul_raw:
    exposure_time = float(hdul_raw[0].header.get("EXPTIME", hdul_raw[0].header.get("TEXPTIME", 1.0)))

with fits.open(data_path, memmap=True) as hdul:
    data = jnp.array(hdul["SCI"].data) if "SCI" in hdul else jnp.array(hdul[0].data)
    rms_file = jnp.array(hdul["ERR"].data)

mask = jnp.array(fits.getdata(mask_path), dtype=bool)
mask_out = jnp.ones_like(data, dtype=bool)
mask = jnp.array(mask_out, dtype=bool)

valid = jnp.isfinite(data) & jnp.isfinite(rms_file) & (rms_file > 0)
mask = mask & valid
npix = int(mask_out.sum())

corner_pixel = 10
bkg_corner = np.array(data[:corner_pixel, :corner_pixel])
bkg_mean = float(np.nanmean(bkg_corner))
_ = float(np.nanstd(bkg_corner))

data = data - bkg_mean
rms = float(np.nanmedian(np.array(rms_file[mask_out])))

# Use WFI2033 PSF and support both model/starred FITS extension layouts.
psf_path_model = os.path.join(DATA_DIR, "F115W_PSF_starred_x8101_y3465.fits")
with fits.open(psf_path_model, memmap=True) as hdul_psf:
    if "DET_PSF_MODEL" in hdul_psf:
        psf_hst = np.array(hdul_psf["DET_PSF_MODEL"].data, dtype=float)
    elif "DET_PSF_NORM" in hdul_psf:
        psf_hst = np.array(hdul_psf["DET_PSF_NORM"].data, dtype=float)
    elif len(hdul_psf) > 1 and hdul_psf[1].data is not None:
        psf_hst = np.array(hdul_psf[1].data, dtype=float)
    elif hdul_psf[0].data is not None:
        psf_hst = np.array(hdul_psf[0].data, dtype=float)
    else:
        raise ValueError("PSF FITS missing usable image extension")

psf_hst = np.clip(psf_hst, 0.0, None)
psf_hst = psf_hst / np.sum(psf_hst)
psf = PSF(psf_type="PIXEL", kernel_point_source=psf_hst)

pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny = get_pixel_grid(data, pix_scale)
noise = Noise(nx, ny, exposure_time=exposure_time)
provided_rms = True

N_gauss_light = 12
N_gauss_source = 1
sigma_lims_lens = [0.01, 3.0]
sigma_lims = [0.01, 0.5]
source_grid_scale = 0.5

conj_points = jnp.array(
    [
        [0.26298137209467, 2.09255496979808],
        [1.35258643634931, 1.62712406079166],
        [-1.00020401055449, 2.01497731965513],
        [-0.22609309020619, -1.10668714778579],
    ]
)

ss_factor = 2

PSF_CORNER_SIZE = 5


def compute_psf_corner_median(psf_kernel, corner_size=PSF_CORNER_SIZE):
    ny, nx = psf_kernel.shape
    cs = min(corner_size, ny, nx)
    corners = np.concatenate(
        [
            psf_kernel[:cs, :cs].reshape(-1),
            psf_kernel[:cs, -cs:].reshape(-1),
            psf_kernel[-cs:, :cs].reshape(-1),
            psf_kernel[-cs:, -cs:].reshape(-1),
        ]
    )
    return float(np.nanmedian(corners))


psf_corner_median = compute_psf_corner_median(psf_hst, corner_size=PSF_CORNER_SIZE)


def subtract_psf_corner_median(model_image, psf_median=psf_corner_median):
    return model_image - jnp.asarray(psf_median, dtype=model_image.dtype)


PSF_CORR_PRIOR = {
    "plate_name": "PSF correction",
    "param_name": "psf_corr",
    "n_high": 30,
    "n_value": None,
    "k_zero": None,
}

k_psf = K_grid(psf_hst.shape).k


def build_corrected_psf_kernel(psf_base, psf_corr_pixels, eps=1e-12):
    delta = jnp.asarray(psf_corr_pixels) - jnp.mean(psf_corr_pixels)
    log_base = jnp.log(jnp.asarray(psf_base) + eps)
    psf_eff = jnp.exp(log_base + delta)
    psf_eff = psf_eff / (jnp.sum(psf_eff) + eps)
    return psf_eff


# -----------------------------------------------------------------------------
# Prior config
# -----------------------------------------------------------------------------
MASS_PRIOR_PARAMETRIC = {
    "gamma_low": 1.95,
    "gamma_up": 2.05,
    "center_high": 0.5,
    "center_low": -0.5,
}

MASS_PRIOR_PIXELATED = {}

LENS_LIGHT_PRIOR_KWARGS = {
    "plate_name": "Lens light",
    "param_name": "lens",
    "n_gauss": N_gauss_light,
    "sigma_lims": sigma_lims_lens,
    "e_low": -0.2,
    "e_high": 0.2,
}

POINT_SOURCE_PRIOR = {
    "pos_sigma": 0.05,
    "pos_window": 0.2,
    "log10_amp_low": -3.0,
    "log10_amp_high": 4.5,
    "log10_amp_lens_low": -3.0,
    "log10_amp_lens_high": 4.5,
}

CONJUGATE_POINT_PRIOR = {
    "rate": {
        "parametric": 1000.0,
        "pixelated": 100.0,
    },
}

RMS_PRIOR = {
    "low_factor": 0.5,
    "high_factor": 1.5,
}

SOURCE_GRID_PRIOR = {
    "plate_name": "Source grid",
    "param_name": "source_grid",
    "n_high": 1000,
    "positive": True,
}


# -----------------------------------------------------------------------------
# Lens image objects
# -----------------------------------------------------------------------------
mass_model = MassModel(["EPL", "SHEAR"])
lens_light_model = LightModel(["MULTI_GAUSSIAN_ELLIPSE"], {})
source_light_model = LightModel(["MULTI_GAUSSIAN_ELLIPSE"], {})

point_source_model = PointSourceModel(
    ["IMAGE_POSITIONS"],
    mass_model=mass_model,
    image_plane=pixel_grid,
)

lens_image = LensImageExtension(
    pixel_grid,
    psf,
    noise_class=noise,
    lens_light_model_class=lens_light_model,
    lens_mass_model_class=mass_model,
    source_model_class=source_light_model,
    point_source_model_class=point_source_model,
    source_arc_mask=mask,
    conjugate_points=conj_points,
    kwargs_numerics={"supersampling_factor": ss_factor},
    source_grid_scale=source_grid_scale,
)


# -----------------------------------------------------------------------------
# Unified model function
# -----------------------------------------------------------------------------
npix = int(mask_out.sum())
N_HIGH_SOURCE_GRID = SOURCE_GRID_PRIOR["n_high"]


def model(
    data,
    k_values=None,
    conj=True,
    provided_rms=False,
    pixelated=False,
    n_value=None,
    mass_prior_kwargs={},
    psf_kernel=None,
    k_psf_values=None,
    enable_psf_corr=False,
):
    mass_params = EPL_w_shear("Mass model", "1", **mass_prior_kwargs)
    lens_light = multi_gauss_light(**LENS_LIGHT_PRIOR_KWARGS)

    n_ps = conj_points.shape[0]
    ra_ps = numpyro.sample(
        "ra_ps",
        dist.TruncatedNormal(
            loc=conj_points[:, 0],
            scale=POINT_SOURCE_PRIOR["pos_sigma"],
            low=conj_points[:, 0] - POINT_SOURCE_PRIOR["pos_window"],
            high=conj_points[:, 0] + POINT_SOURCE_PRIOR["pos_window"],
        ),
    )
    dec_ps = numpyro.sample(
        "dec_ps",
        dist.TruncatedNormal(
            loc=conj_points[:, 1],
            scale=POINT_SOURCE_PRIOR["pos_sigma"],
            low=conj_points[:, 1] - POINT_SOURCE_PRIOR["pos_window"],
            high=conj_points[:, 1] + POINT_SOURCE_PRIOR["pos_window"],
        ),
    )
    log10_amp_ps = numpyro.sample(
        "log10_amp_ps",
        dist.Uniform(
            POINT_SOURCE_PRIOR["log10_amp_low"],
            POINT_SOURCE_PRIOR["log10_amp_high"],
        ).expand([n_ps]),
    )

    amp_quasar_ps = jnp.power(10.0, log10_amp_ps)
    kwargs_point_source = [
        {
            "ra": ra_ps,
            "dec": dec_ps,
            "amp": amp_quasar_ps,
        }
    ]

    lens_img = lens_image_pixel if pixelated else lens_image

    if conj:
        conj_points_model = lens_img.trace_conjugate_points(mass_params)
        conj_distance = reduced_distance_matrix(conj_points_model)
        nc = conj_distance.shape[0]
        rate_key = "pixelated" if pixelated else "parametric"
        conj_rate = CONJUGATE_POINT_PRIOR["rate"][rate_key]
        with numpyro.plate(f"Conjugate points 2 - [{nc}]", nc):
            numpyro.sample("conjugate_points", dist.Exponential(conj_rate), obs=conj_distance)

    if pixelated:
        if k_values is None:
            raise ValueError("k_values is required when pixelated=True")
        source_light = [
            matern_power_spectrum(
                SOURCE_GRID_PRIOR["plate_name"],
                SOURCE_GRID_PRIOR["param_name"],
                k_values,
                n_high=SOURCE_GRID_PRIOR["n_high"],
                n_value=n_value,
                positive=SOURCE_GRID_PRIOR["positive"],
            )
        ]
    else:
        conj_points_model_for_center = lens_image.trace_conjugate_points(mass_params)
        source_center = jnp.mean(conj_points_model_for_center, axis=0)
        source_light = multi_gauss_light_center(
            "Source light",
            "source",
            N_gauss_source,
            sigma_lims,
            center_det=source_center,
        )

    psf_kernel_eff = psf_kernel

    if pixelated and enable_psf_corr:
        psf_corr = matern_power_spectrum(
            PSF_CORR_PRIOR["plate_name"],
            PSF_CORR_PRIOR["param_name"],
            k_psf_values,
            k_zero=PSF_CORR_PRIOR["k_zero"],
            n_high=PSF_CORR_PRIOR["n_high"],
            n_value=PSF_CORR_PRIOR["n_value"],
            positive=True,
        )
        psf_kernel_eff = build_corrected_psf_kernel(
            psf_kernel,
            psf_corr["pixels"],
        )

    model_image = lens_img.model(
        kwargs_lens=mass_params,
        kwargs_source=source_light,
        kwargs_lens_light=lens_light,
        kwargs_point_source=kwargs_point_source,
        source_add=True,
        point_source_add=True,
        psf_kernel=psf_kernel_eff,
    )

    if provided_rms:
        model_std = rms_file
    else:
        background_rms_model = numpyro.sample(
            "RMS",
            dist.LogUniform(rms * RMS_PRIOR["low_factor"], rms * RMS_PRIOR["high_factor"]),
        )
        model_var = lens_img.Noise.C_D_model(model_image, background_rms=background_rms_model)
        model_std = jnp.sqrt(model_var)

    model_image_masked_out = model_image[mask_out]
    model_std_masked_out = model_std[mask_out]

    numpyro.deterministic("model_image", model_image)

    with numpyro.plate(f"Data masked - [{npix}]", npix):
        numpyro.sample("obs", dist.Normal(model_image_masked_out, model_std_masked_out), obs=data[mask_out])


def params2kwargs(params, fixed_params={}, pixelated=False):
    params_full = params | fixed_params
    kwargs_lens = params2kwargs_EPL_w_shear(params_full, "1")
    kwargs_lens_light = params2kwargs_multi_gauss_light(params_full, "lens")
    kwargs_point_source = [
        {
            "ra": params_full["ra_ps"],
            "dec": params_full["dec_ps"],
            "amp": jnp.power(10.0, params_full["log10_amp_ps"]),
        }
    ]

    if pixelated:
        kwargs_source = [params2kwargs_power_spectrum(params_full, "source_grid")]
    else:
        conj_points_model = lens_image.trace_conjugate_points(kwargs_lens)
        source_center = jnp.mean(conj_points_model, axis=0)
        n_source = params_full["amp_source"].shape[0]
        center_source = jnp.vstack(
            [
                jnp.full((n_source,), source_center[0]),
                jnp.full((n_source,), source_center[1]),
            ]
        )
        params_full = params_full | {"center_source": center_source}
        kwargs_source = params2kwargs_multi_gauss_light(params_full, "source")

    return {
        "kwargs_lens": kwargs_lens,
        "kwargs_source": kwargs_source,
        "kwargs_lens_light": kwargs_lens_light,
        "kwargs_point_source": kwargs_point_source,
    }


def params2psf_kernel(params):
    psf_corr_pixels = params2kwargs_power_spectrum(params, PSF_CORR_PRIOR["param_name"])["pixels"]
    return build_corrected_psf_kernel(psf_hst, psf_corr_pixels)


# -----------------------------------------------------------------------------
# Parametric SVI
# -----------------------------------------------------------------------------
max_iterations = 10000
num_chains = 4

init_fun = infer.init_to_median(num_samples=25)
guide = autoguide.AutoLowRankMultivariateNormal(model, init_loc_fn=init_fun)
scheduler = split_scheduler(max_iterations, init_value=0.01, transition_steps=[200, 10])
optim = optax.adabelief(learning_rate=scheduler)
loss = infer.Trace_ELBO(num_particles=1)

rng_key = jax.random.PRNGKey(42)
rng_key, rng_key_ = jax.random.split(rng_key)
svi = SVI_vec(model, guide, optim, loss)
multi_svi_results = svi.run(
    rng_key,
    num_chains,
    max_iterations,
    data,
    provided_rms=provided_rms,
    mass_prior_kwargs=MASS_PRIOR_PARAMETRIC,
    stable_update=True,
)
multi_svi_median = guide.median(multi_svi_results.params)
multi_svi_median_herc = median_params2kwargs(params2kwargs, multi_svi_median, jnp.arange(num_chains))


# -----------------------------------------------------------------------------
# Pixel-grid initialization
# -----------------------------------------------------------------------------
best_pix_sizes = np.array(
    [
        get_best_pixel_size(
            lens_image,
            get_value_from_index(multi_svi_median_herc, i),
            source_grid_scale,
        )
        for i in range(num_chains)
    ]
)
pixel_grid_shape = np.median(best_pix_sizes).astype(int) * 1
print(pixel_grid_shape)

vars_mass = ["theta_E_1", "gamma_1", "e_1", "center_1", "gamma_sheer_1"]
vars_lens_light = ["A_lens", "sigma_lens", "e_lens", "center_lens"]
vars_source_light = ["A_source", "sigma_source", "e_source"]
vars_point_source = ["ra_ps", "dec_ps", "log10_amp_ps"]
vars_other = []

k_grid = K_grid((pixel_grid_shape, pixel_grid_shape))

@jax.vmap
def get_image(idx):
    image_i, _ = pixelize_plane_single(
        lens_image,
        get_value_from_index(multi_svi_median_herc, idx),
        pixel_grid_shape,
        source_grid_scale=source_grid_scale,
    )
    return image_i

orig_source = get_image(jnp.arange(num_chains))
ps_keys = jax.random.split(rng_key_, num_chains)
ps_fits = source_power_spectrum(orig_source, ps_keys, None, True)

keys_for_pixel_init = vars_lens_light + vars_mass + vars_point_source + vars_other
multi_svi_median_pixelated = {k: multi_svi_median[k] for k in keys_for_pixel_init if k in multi_svi_median} | ps_fits


# -----------------------------------------------------------------------------
# Pixelated lens image object
# -----------------------------------------------------------------------------
k_grid = K_grid((pixel_grid_shape, pixel_grid_shape))

mass_model_pixel = MassModel(["EPL", "SHEAR"])
lens_light_model_pixel = LightModel(["MULTI_GAUSSIAN_ELLIPSE"], {})

source_light_model_pixel = LightModel(
    ["PIXELATED"],
    pixel_adaptive_grid=True,
    pixel_interpol="fast_bilinear",
    kwargs_pixelated={"num_pixels": pixel_grid_shape},
)

pixel_grid_pixel = deepcopy(pixel_grid)
psf_pixel = deepcopy(psf)

point_source_model_pixel = PointSourceModel(
    ["IMAGE_POSITIONS"],
    mass_model=mass_model_pixel,
    image_plane=pixel_grid_pixel,
)

lens_image_pixel = LensImageExtension(
    pixel_grid_pixel,
    psf_pixel,
    noise_class=noise,
    lens_light_model_class=lens_light_model_pixel,
    lens_mass_model_class=mass_model_pixel,
    source_model_class=source_light_model_pixel,
    point_source_model_class=point_source_model_pixel,
    source_arc_mask=mask,
    conjugate_points=conj_points,
    kwargs_numerics={"supersampling_factor": ss_factor},
    source_grid_scale=source_grid_scale,
)


# -----------------------------------------------------------------------------
# Pixelated SVI stage-1 (without PSF correction)
# -----------------------------------------------------------------------------
max_iterations = 10000
num_chains = 4

scheduler_stage1 = split_scheduler(max_iterations, init_value=0.01, transition_steps=[200, 10])
optim_stage1 = optax.adabelief(learning_rate=scheduler_stage1)
loss_stage1 = infer.TraceMeanField_ELBO()

svi_keys = jax.random.split(rng_key_, num_chains)
stage1_results_list = []
stage1_guides = []

for i in range(num_chains):
    init_fun_stage1_i = init_to_value_or_defer(values=get_value_from_index(multi_svi_median_pixelated, i))
    guide_stage1_i = autoguide.AutoDiagonalNormal(model, init_loc_fn=init_fun_stage1_i, init_scale=0.01)
    svi_stage1_i = infer.SVI(model, guide_stage1_i, optim_stage1, loss_stage1)
    result_i = svi_stage1_i.run(
        svi_keys[i],
        max_iterations,
        data,
        k_grid.k,
        conj=True,
        n_value=None,
        pixelated=True,
        provided_rms=provided_rms,
        mass_prior_kwargs=MASS_PRIOR_PIXELATED,
        progress_bar=True,
        stable_update=True,
        enable_psf_corr=False,
    )
    stage1_guides.append(guide_stage1_i)
    stage1_results_list.append(result_i)


def _stack_or_none(*xs):
    if xs[0] is None:
        return None
    return jnp.stack(xs)


multi_svi_pixel_results_stage1 = jax.tree.map(_stack_or_none, *stage1_results_list)
guide_pixel_stage1 = stage1_guides[0]
multi_svi_pixel_median_stage1 = guide_pixel_stage1.median(multi_svi_pixel_results_stage1.params)
multi_svi_pixel_median_herc_stage1 = median_params2kwargs(
    lambda p, fixed_params={}: params2kwargs(p, fixed_params=fixed_params, pixelated=True),
    multi_svi_pixel_median_stage1,
    jnp.arange(num_chains),
)


# -----------------------------------------------------------------------------
# Pixelated SVI stage-2 (with PSF correction)
# -----------------------------------------------------------------------------
max_iterations = 10000
num_chains = 4

scheduler_stage2 = optax.exponential_decay(
    init_value=5e-3,
    transition_steps=200,
    decay_rate=0.99,
)
optim_stage2 = optax.adabelief(learning_rate=scheduler_stage2)
loss_stage2 = infer.TraceMeanField_ELBO()

svi_keys = jax.random.split(rng_key_, num_chains)
stage2_results_list = []
stage2_guides = []

for i in range(num_chains):
    init_values_stage2_i = get_value_from_index(multi_svi_pixel_median_stage1, i) | {
        "pixels_wn_psf_corr": jnp.zeros_like(k_psf),
    }
    init_fun_stage2_i = init_to_value_or_defer(values=init_values_stage2_i)
    guide_stage2_i = autoguide.AutoDiagonalNormal(model, init_loc_fn=init_fun_stage2_i, init_scale=0.01)
    svi_stage2_i = infer.SVI(model, guide_stage2_i, optim_stage2, loss_stage2)
    result_i = svi_stage2_i.run(
        svi_keys[i],
        max_iterations,
        data,
        k_grid.k,
        conj=True,
        n_value=None,
        pixelated=True,
        provided_rms=provided_rms,
        mass_prior_kwargs=MASS_PRIOR_PIXELATED,
        psf_kernel=psf_hst,
        k_psf_values=k_psf,
        enable_psf_corr=True,
        progress_bar=True,
        stable_update=True,
    )
    stage2_guides.append(guide_stage2_i)
    stage2_results_list.append(result_i)


multi_svi_pixel_results_stage2 = jax.tree.map(_stack_or_none, *stage2_results_list)
guide_pixel_stage2 = stage2_guides[0]
multi_svi_pixel_median_stage2 = guide_pixel_stage2.median(multi_svi_pixel_results_stage2.params)
multi_svi_pixel_median_herc_stage2 = median_params2kwargs(
    lambda p, fixed_params={}: params2kwargs(p, fixed_params=fixed_params, pixelated=True),
    multi_svi_pixel_median_stage2,
    jnp.arange(num_chains),
)


# -----------------------------------------------------------------------------
# Build HMC init params from stage-2 median
# -----------------------------------------------------------------------------
vars_pixel = ["pixels_wn_source_grid"]
vars_power = ["n_source_grid", "rho_source_grid", "sigma_source_grid"]
vars_psf = ["ra_ps", "dec_ps", "log10_amp_ps"]
vars_psf_corr = ["n_psf_corr", "rho_psf_corr", "sigma_psf_corr", "pixels_wn_psf_corr"]

multi_svi_pixel_median_vars = {
    k: multi_svi_pixel_median_stage2[k]
    for k in vars_mass + vars_power + vars_pixel + vars_other + vars_lens_light + vars_psf + vars_psf_corr
}

unconstrined_svi_pixel_median = jax.vmap(
    lambda p: infer.util.unconstrain_fn(
        model,
        (data, k_grid.k),
        {
            "conj": False,
            "pixelated": True,
            "provided_rms": provided_rms,
            "mass_prior_kwargs": MASS_PRIOR_PIXELATED,
            "psf_kernel": psf_hst,
            "k_psf_values": k_psf,
            "enable_psf_corr": True,
        },
        p,
    )
)(multi_svi_pixel_median_vars)

unconstrined_svi_pixel_median = {k: v.astype(jnp.float64) for k, v in unconstrined_svi_pixel_median.items()}
rng_key, rng_key_ = jax.random.split(rng_key)


# -----------------------------------------------------------------------------
# HMC
# -----------------------------------------------------------------------------
init_fun_pixel = init_to_value_or_defer(values=get_value_from_index(multi_svi_pixel_median_stage2, 0))

inner_kernels = [
    NUTS(
        model,
        init_strategy=init_fun_pixel,
        target_accept_prob=0.95,
        max_tree_depth=10,
        dense_mass=[
            ("n_source_grid", "rho_source_grid", "sigma_source_grid"),
            ("A_lens", "sigma_lens", "e_lens", "center_lens"),
        ],
    ),
    NUTS(
        model,
        init_strategy=init_fun_pixel,
        target_accept_prob=0.9,
        max_tree_depth=10,
        dense_mass=[("center_1",), ("theta_E_1",), ("e_1", "gamma_1", "gamma_sheer_1")],
    ),
    NUTS(
        model,
        init_strategy=init_fun_pixel,
        target_accept_prob=0.9,
        max_tree_depth=10,
        dense_mass=[
            ("ra_ps", "dec_ps", "log10_amp_ps"),
            ("n_psf_corr", "rho_psf_corr", "sigma_psf_corr"),
        ],
    ),
]

outer_kernel = MultiHMCGibbs(
    inner_kernels,
    gibbs_sites_list=[
        vars_pixel + vars_power + vars_lens_light + vars_other,
        vars_mass,
        vars_psf + vars_psf_corr,
    ],
)

mcmc_pixel = MCMC(
    outer_kernel,
    num_warmup=2000,
    num_samples=1000,
    num_chains=num_chains,
    progress_bar=True,
    chain_method="vectorized",
)

batch_number = 8
last_states = []
batch_list = []
quasar_hmc_dir = "/mnt/lustre/tianli/quasar_hmc"

for i in range(batch_number):
    if i == 0:
        mcmc_pixel.run(
            rng_key_,
            data,
            k_grid.k,
            conj=False,
            pixelated=True,
            provided_rms=provided_rms,
            mass_prior_kwargs=MASS_PRIOR_PIXELATED,
            psf_kernel=psf_hst,
            k_psf_values=k_psf,
            enable_psf_corr=True,
            init_params=unconstrined_svi_pixel_median,
        )
        last_states.append(jax.device_get(mcmc_pixel.last_state))
    else:
        mcmc_pixel.post_warmup_state = mcmc_pixel.last_state
        mcmc_pixel.run(
            mcmc_pixel.post_warmup_state.rng_key,
            data,
            k_grid.k,
            conj=False,
            pixelated=True,
            provided_rms=provided_rms,
            mass_prior_kwargs=MASS_PRIOR_PIXELATED,
            psf_kernel=psf_hst,
            k_psf_values=k_psf,
            enable_psf_corr=True,
        )
        last_states.append(jax.device_get(mcmc_pixel.last_state))

    mcmc_pixel._states = jax.device_get(mcmc_pixel._states)
    mcmc_pixel._states_flat = jax.device_get(mcmc_pixel._states_flat)
    mcmc_chain = az.from_numpyro(mcmc_pixel)
    batch_path = f"{quasar_hmc_dir}/WFI2033_psf_correction{i}.nc"
    mcmc_chain.to_netcdf(batch_path)
    batch_list.append(mcmc_chain)
    print(f"Saved batch to: {batch_path}")


# -----------------------------------------------------------------------------
# Concatenate batches and save
# -----------------------------------------------------------------------------
inf_data = az.concat(*batch_list, dim="draw")
inf_data_path = "/mnt/lustre/tianli/quasar_hmc/WFI2033_psf_correction.nc"
inf_data.to_netcdf(str(inf_data_path))
print(f"Saved concatenated inf_data to: {inf_data_path}")
