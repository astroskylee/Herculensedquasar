
#!/usr/bin/env python3
"""WFI2033 Step 5 full lens-light pipeline up to HMC.

This script mirrors the core computation flow in
`Step_5_full_lenslight.ipynb` and appends a standalone HMC/GIBBS stage:

1) Parametric SVI with fixed 5-Gaussian lens light
2) Pixel stage1 SVI with the same frozen lens light
3) Pixel stage2 SVI with PSF correction and the same frozen lens light
4) Build updated RMS map using stage2-best AGN + alpha map
5) Pixel stage3 SVI with the lens light released again
6) HMC/GIBBS run initialized from the stage3 posterior median

No figures are generated or saved.
Only the final concatenated HMC inference data is saved.
"""

from __future__ import annotations

from pathlib import Path
import os
import warnings

# -----------------------------
# Runtime config (set early)
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

# Lustre/HDF5 safety for NetCDF writes
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Avoid large JAX preallocation by default
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

warnings.simplefilter("ignore")

# -----------------------------
# Imports
# -----------------------------
from herculens_import_main import *  # noqa: F401,F403
import jax
import numpyro
import arviz as az

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

if "graphviz" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/apps/pkgs/graphviz/12.2.1/intel64/gnu_12.2.0/bin:" + os.environ.get("PATH", "")


# -----------------------------
# Data / PSF loading
# -----------------------------


DATA_DIR = "../../Data/WFI2033"
raw_data_path = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d.fits")
data_path = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d_cut_x6985_y3594_150.fits")
mask_path = os.path.join("data/mask_hmc.fits")
maskout_path = os.path.join(DATA_DIR, "mask_out_center.fits")

with fits.open(raw_data_path, memmap=True) as hdul_raw:
    raw_header = hdul_raw["SCI"].header if "SCI" in hdul_raw else hdul_raw[0].header
    exposure_time = float(raw_header.get("EXPTIME", raw_header.get("TEXPTIME", raw_header.get("XPOSURE", 1.0))))
pix_scale = float(np.sqrt(raw_header["PIXAR_A2"]))
print(pix_scale)

with fits.open(data_path, memmap=True) as hdul:
    data = jnp.array(hdul["SCI"].data) if "SCI" in hdul else jnp.array(hdul[0].data)
    rms_file = jnp.array(hdul["ERR"].data)

mask = jnp.array(fits.getdata(mask_path), dtype=bool)
mask_out = jnp.array(fits.getdata(maskout_path), dtype=bool)
mask = jnp.array(mask_out, dtype=bool)

valid = jnp.isfinite(data) & jnp.isfinite(rms_file) & (rms_file > 0)
mask = mask & valid
npix = int(mask_out.sum())

corner_pixel = 10
bkg_corner = np.array(data[:corner_pixel, :corner_pixel])
bkg_mean = float(np.nanmean(bkg_corner))
bkg_rms = float(np.nanstd(bkg_corner))

data = data - bkg_mean
rms = float(np.nanmedian(np.array(rms_file[mask_out])))

# Step3 PSF output (detector sampled)
psf_path_model = os.path.join("./psf_data", "PSF_model_step3_svi.fits")
with fits.open(psf_path_model, memmap=True) as hdul_psf:
    psf_hst = np.array(hdul_psf["DET_PSF_MODEL"].data, dtype=float)

psf_hst = np.clip(psf_hst, 0.0, None)
psf_hst = psf_hst / np.sum(psf_hst)
psf_used = psf_path_model
psf = PSF(psf_type="PIXEL", kernel_point_source=psf_hst)

pixel_grid, xgrid, ygrid, x_axis, y_axis, extent, nx, ny = get_pixel_grid(data, pix_scale)
noise = Noise(nx, ny, exposure_time=exposure_time)
provided_rms = True

N_gauss_light = 5
N_gauss_source = 1

sigma_lims_lens = [0.015, 1.0]
sigma_lims = [0.01, 0.5]
source_grid_scale = 0.8

conj_points = jnp.array([
    [1.20212170716053, -0.12271885209256231],
    [0.9053233071260114, 0.5277189685977776],
    [-1.0461673774453952, 1.0081083299749878],
    [-0.1255456241215261, -0.8965524340129204],
])

ss_factor = 2
suffix = f'_ss={ss_factor}_full_light'
PSF_CORNER_SIZE = 5
num_chains = 6

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
    "param_name": "log_psf_corr_center",
    "sigma_log": 1.0,
    "log_clip": 5.0,
}

def build_psf_corr_factor_field(log_psf_corr_center, psf_shape, log_clip=5.0):
    ny, nx = psf_shape
    log_field = jnp.asarray(log_psf_corr_center)
    if log_field.shape != (ny, nx):
        raise ValueError(f"log_psf_corr_center shape {log_field.shape} does not match psf shape {(ny, nx)}")
    return jnp.exp(jnp.clip(log_field, -log_clip, log_clip))


def build_corrected_psf_kernel(psf_base, log_psf_corr_center, eps=1e-12):
    corr_field = build_psf_corr_factor_field(
        log_psf_corr_center,
        jnp.asarray(psf_base).shape,
        log_clip=PSF_CORR_PRIOR["log_clip"],
    )
    psf_eff = jnp.asarray(psf_base) * corr_field
    psf_eff = jnp.maximum(psf_eff, eps)
    psf_eff = psf_eff / (jnp.sum(psf_eff) + eps)
    return psf_eff


# -----------------------------
# Priors
# -----------------------------
MASS_PRIOR_PARAMETRIC = {
    "gamma_low": 1.95,
    "gamma_up": 2.05,
    "center_high": 0.5,
    "center_low": -0.5,
    "theta_low" : 0.9, 
    "theta_high" : 1.1
}

MASS_PRIOR_PIXELATED = {"theta_low" : 0.9, 
    "theta_high" : 1.1}

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
        "parametric": 10000.0,
        "pixelated": 1000.0,
    },
}

RMS_PRIOR = {
    "low_factor": 0.5,
    "high_factor": 1.5,
}

SOURCE_GRID_PRIOR = {
    "plate_name": "Source grid",
    "param_name": "source_grid",
    "n_high": 100,
    "sigma_low": 1e-5,
    "sigma_high": 10.0,
    "positive": True,
}

SIS_G1_PRIOR = {
    "theta_low": 0.0,
    "theta_high": 0.25,
}

G1_MASS_CENTER = (1.556, 1.299)

SIS_G2_PRIOR = {
    "theta_mean": 0.622,
    "theta_sigma": 0.062,
}

G2_MASS_CENTER = (2.145, -3.326)

FREEZE_LENS_LIGHT_STAGE12 = True
LENS_LIGHT_FIXED_SITES = {
    "A_lens": jnp.array([99817.9057, 99886.9246, 27619.2805, 11797.9752, 3747.18909], dtype=jnp.float64),
    "sigma_lens": jnp.array([0.0215159293, 0.0365546165, 0.0843158332, 0.215854311, 0.806167953], dtype=jnp.float64),
    "e_lens": jnp.array(
        [
            [-0.071907176, -0.0331211024, -0.0628928986, -0.0795255049, -0.00677720617],
            [0.00443459114, 0.0208858872, 0.110132527, 0.110276821, 0.141891644],
        ],
        dtype=jnp.float64,
    ),
    "center_lens": jnp.array(
        [
            [-0.00012325156, -0.000178568799, 0.00326719443, -0.00332402026, 0.139266098],
            [-0.0130794077, -0.011627038, -0.0112894393, -0.0135242747, -0.0534815181],
        ],
        dtype=jnp.float64,
    ),
}
LENS_LIGHT_FIXED_PARAMS = LENS_LIGHT_FIXED_SITES | {
    "amp_lens": LENS_LIGHT_FIXED_SITES["A_lens"] * (LENS_LIGHT_FIXED_SITES["sigma_lens"] ** 2),
}


# -----------------------------
# Parametric lens image objects
# -----------------------------
from herculens.PointSourceModel.point_source_model import PointSourceModel

mass_model = MassModel(["EPL", "SHEAR", "SIS", "SIS"])
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


# -----------------------------
# Unified model (parametric + pixelated)
# -----------------------------
npix = int(mask_out.sum())
N_HIGH_SOURCE_GRID = SOURCE_GRID_PRIOR["n_high"]


def model(
    data,
    k_values=None,
    conj=True,
    provided_rms=False,
    pixelated=False,
    n_value=None,
    mass_prior_kwargs=None,
    psf_kernel=None,
    enable_psf_corr=False,
):
    if mass_prior_kwargs is None:
        mass_prior_kwargs = {}

    mass_params = EPL_w_shear("Mass model", "1", **mass_prior_kwargs)
    mass_params = mass_params + SIS(
        "Mass model g1",
        "g1",
        origin=G1_MASS_CENTER,
        theta_low=SIS_G1_PRIOR["theta_low"],
        theta_high=SIS_G1_PRIOR["theta_high"],
    )
    mass_params = mass_params + SIS(
        "Mass model g2",
        "g2",
        origin=G2_MASS_CENTER,
        theta_mean=SIS_G2_PRIOR["theta_mean"],
        theta_sigma=SIS_G2_PRIOR["theta_sigma"],
    )
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
    kwargs_point_source = [{"ra": ra_ps, "dec": dec_ps, "amp": amp_quasar_ps}]

    lens_img = lens_image_pixel if pixelated else lens_image

    if conj:
        src_x_ps, src_y_ps = lens_img.MassModel.ray_shooting(ra_ps, dec_ps, mass_params)
        conj_points_model = jnp.stack([src_x_ps, src_y_ps], axis=1)
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
                sigma_low=SOURCE_GRID_PRIOR["sigma_low"],
                sigma_high=SOURCE_GRID_PRIOR["sigma_high"],
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

    psf_ref = jnp.asarray(psf_hst if psf_kernel is None else psf_kernel)
    psf_kernel_eff = psf_kernel
    psf_kernel_det = psf_ref
    corr_field = jnp.ones_like(psf_ref)

    if pixelated and enable_psf_corr:
        corr_shape = tuple(psf_ref.shape)
        log_psf_corr_center = numpyro.sample(
            "log_psf_corr_center",
            dist.Normal(0.0, PSF_CORR_PRIOR["sigma_log"]).expand(corr_shape),
        )
        corr_field = build_psf_corr_factor_field(
            log_psf_corr_center,
            psf_ref.shape,
            log_clip=PSF_CORR_PRIOR["log_clip"],
        )
        psf_kernel_det = build_corrected_psf_kernel(psf_ref, log_psf_corr_center)
        psf_kernel_eff = psf_kernel_det

    numpyro.deterministic("psf_corr_factor_field", corr_field)
    numpyro.deterministic("psf_kernel_corrected", psf_kernel_det)

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
    kwargs_lens = (
        params2kwargs_EPL_w_shear(params_full, "1")
        + params2kwargs_SIS(params_full, "g1")
        + params2kwargs_SIS(params_full, "g2")
    )
    kwargs_lens_light = params2kwargs_multi_gauss_light(params_full, "lens")
    kwargs_point_source = [{
        "ra": params_full["ra_ps"],
        "dec": params_full["dec_ps"],
        "amp": jnp.power(10.0, params_full["log10_amp_ps"]),
    }]

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


MODEL_STAGE12 = numpyro.handlers.condition(
    model,
    data=LENS_LIGHT_FIXED_SITES,
) if FREEZE_LENS_LIGHT_STAGE12 else model


# -----------------------------
# Parametric SVI
# -----------------------------
max_iterations = 10000
PARAMETRIC_SVI_KWARGS = {
    "conj": True,
    "pixelated": False,
    "n_value": None,
    "provided_rms": provided_rms,
    "mass_prior_kwargs": MASS_PRIOR_PARAMETRIC,
}

amp_ps_start = jnp.array([10120.25246224, 5648.26269992, 5112.50014416, 3735.28867656], dtype=jnp.float64)
init_values_parametric = {"log10_amp_ps": jnp.log10(amp_ps_start)} | LENS_LIGHT_FIXED_PARAMS
init_fun = init_to_value_or_defer(values=init_values_parametric)
scheduler = split_scheduler(max_iterations, init_value=0.01, transition_steps=[200, 10])
optim = optax.adabelief(learning_rate=scheduler)
loss = infer.Trace_ELBO(num_particles=1)

rng_key = jax.random.PRNGKey(420)
rng_key, rng_key_ = jax.random.split(rng_key)
svi_keys = jax.random.split(rng_key_, num_chains)

param_results_list = []
param_guides = []
for i in range(num_chains):
    guide_i = autoguide.AutoLowRankMultivariateNormal(MODEL_STAGE12, init_loc_fn=init_fun)
    svi_i = infer.SVI(MODEL_STAGE12, guide_i, optim, loss)
    result_i = svi_i.run(
        svi_keys[i],
        max_iterations,
        data,
        **PARAMETRIC_SVI_KWARGS,
        progress_bar=False,
        stable_update=True,
    )
    param_guides.append(guide_i)
    param_results_list.append(result_i)



def _stack_or_none(*xs):
    first = xs[0]
    return None if first is None else jnp.stack(xs)


multi_svi_results = jax.tree.map(_stack_or_none, *param_results_list)
guide = param_guides[0]
multi_svi_median = guide.median(multi_svi_results.params)
multi_svi_median_herc = median_params2kwargs(
    lambda p, fixed_params=LENS_LIGHT_FIXED_PARAMS: params2kwargs(p, fixed_params=fixed_params),
    multi_svi_median,
    jnp.arange(num_chains),
)


# -----------------------------
# Pixel-grid init from parametric median
# -----------------------------
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
print(f"pixel_grid_shape = {pixel_grid_shape}")

vars_mass = ["theta_E_1", "theta_E_g1", "theta_E_g2", "gamma_1", "e_1", "center_1", "gamma_sheer_1"]
vars_lens_light = ["A_lens", "sigma_lens", "e_lens", "center_lens"]
vars_source_light = ["A_source", "sigma_source", "e_source"]
vars_point_source = ["ra_ps", "dec_ps", "log10_amp_ps"]
vars_other = []

k_grid = K_grid((pixel_grid_shape, pixel_grid_shape))
PIXELATED_BASE_KWARGS = {
    "k_values": k_grid.k,
    "conj": True,
    "pixelated": True,
    "n_value": None,
    "provided_rms": provided_rms,
    "mass_prior_kwargs": MASS_PRIOR_PIXELATED,
}
PIXELATED_STAGE1_KWARGS = PIXELATED_BASE_KWARGS | {
    "enable_psf_corr": False,
}
PIXELATED_STAGE2_KWARGS = PIXELATED_BASE_KWARGS | {
    "enable_psf_corr": True,
}
PIXELATED_STAGE3_KWARGS = PIXELATED_BASE_KWARGS | {
    "enable_psf_corr": True,
}
HMC_RUN_KWARGS = PIXELATED_BASE_KWARGS | {
    "conj": False,
    "enable_psf_corr": True,
}

from lens_images_extension import pixelize_plane as pixelize_plane_single

orig_source_list = []
for idx in range(num_chains):
    image_i, _ = pixelize_plane_single(
        lens_image,
        get_value_from_index(multi_svi_median_herc, idx),
        pixel_grid_shape,
        source_grid_scale=source_grid_scale,
    )
    orig_source_list.append(image_i)
orig_source = jnp.stack(orig_source_list, axis=0)

ps_keys = jax.random.split(rng_key_, num_chains)
from herculens_import_main import source_power_spectrum

ps_fits = source_power_spectrum(orig_source, ps_keys, None, True)

keys_for_pixel_init = vars_lens_light + vars_mass + vars_point_source + vars_other
multi_svi_median_pixelated = {
    k: multi_svi_median[k] for k in keys_for_pixel_init if k in multi_svi_median
} | ps_fits | LENS_LIGHT_FIXED_PARAMS


# -----------------------------
# Pixel lens image objects
# -----------------------------
mass_model_pixel = MassModel(["EPL", "SHEAR", "SIS", "SIS"])
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


# -----------------------------
# Stage1 pixel SVI (no PSF corr)
# -----------------------------
max_iterations = 10000

scheduler_stage1 = split_scheduler(max_iterations, init_value=0.01, transition_steps=[200, 10])
optim_stage1 = optax.adabelief(learning_rate=scheduler_stage1)
loss_stage1 = infer.TraceMeanField_ELBO()

svi_keys = jax.random.split(rng_key_, num_chains)
stage1_results_list = []
stage1_guides = []

for i in range(num_chains):
    init_fun_stage1_i = init_to_value_or_defer(values=get_value_from_index(multi_svi_median_pixelated, i) | LENS_LIGHT_FIXED_PARAMS)
    guide_stage1_i = autoguide.AutoDiagonalNormal(MODEL_STAGE12, init_loc_fn=init_fun_stage1_i, init_scale=0.01)
    svi_stage1_i = infer.SVI(MODEL_STAGE12, guide_stage1_i, optim_stage1, loss_stage1)
    result_i = svi_stage1_i.run(
        svi_keys[i],
        max_iterations,
        data,
        **PIXELATED_STAGE1_KWARGS,
        progress_bar=False,
        stable_update=True,
    )
    stage1_guides.append(guide_stage1_i)
    stage1_results_list.append(result_i)

multi_svi_pixel_results_stage1 = jax.tree.map(_stack_or_none, *stage1_results_list)
guide_pixel_stage1 = stage1_guides[0]
multi_svi_pixel_median_stage1 = guide_pixel_stage1.median(multi_svi_pixel_results_stage1.params)
multi_svi_pixel_median_herc_stage1 = median_params2kwargs(
    lambda p, fixed_params=LENS_LIGHT_FIXED_PARAMS: params2kwargs(p, fixed_params=fixed_params, pixelated=True),
    multi_svi_pixel_median_stage1,
    jnp.arange(num_chains),
)


# -----------------------------
# Stage2 pixel SVI (with PSF corr)
# -----------------------------
max_iterations = 20000

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
    init_values_stage2_i = get_value_from_index(multi_svi_pixel_median_stage1, i) | LENS_LIGHT_FIXED_PARAMS | {
        "log_psf_corr_center": jnp.zeros(psf_hst.shape),
    }
    init_fun_stage2_i = init_to_value_or_defer(values=init_values_stage2_i)
    guide_stage2_i = autoguide.AutoDiagonalNormal(MODEL_STAGE12, init_loc_fn=init_fun_stage2_i, init_scale=0.01)
    svi_stage2_i = infer.SVI(MODEL_STAGE12, guide_stage2_i, optim_stage2, loss_stage2)
    result_i = svi_stage2_i.run(
        svi_keys[i],
        max_iterations,
        data,
        **PIXELATED_STAGE2_KWARGS,
        psf_kernel=psf_hst,
        progress_bar=False,
        stable_update=True,
    )
    stage2_guides.append(guide_stage2_i)
    stage2_results_list.append(result_i)

multi_svi_pixel_results_stage2 = jax.tree.map(_stack_or_none, *stage2_results_list)
guide_pixel_stage2 = stage2_guides[0]
multi_svi_pixel_median_stage2 = guide_pixel_stage2.median(multi_svi_pixel_results_stage2.params)
multi_svi_pixel_median_herc_stage2 = median_params2kwargs(
    lambda p, fixed_params=LENS_LIGHT_FIXED_PARAMS: params2kwargs(p, fixed_params=fixed_params, pixelated=True),
    multi_svi_pixel_median_stage2,
    jnp.arange(num_chains),
)


# -----------------------------
# Stage2 deterministic PSF helper
# -----------------------------
def get_stage2_deterministics_from_median(params_all):
    psf_kernel_all = []
    corr_field_all = []
    for i in range(num_chains):
        params_i = get_value_from_index(params_all, i) | LENS_LIGHT_FIXED_PARAMS
        trace_i = numpyro.handlers.trace(
            numpyro.handlers.substitute(
                numpyro.handlers.seed(model, jax.random.PRNGKey(1000 + i)),
                data=params_i,
            )
        ).get_trace(
            data,
            **PIXELATED_STAGE2_KWARGS,
            psf_kernel=psf_hst,
        )
        psf_kernel_all.append(trace_i["psf_kernel_corrected"]["value"])
        corr_field_all.append(trace_i["psf_corr_factor_field"]["value"])
    return jnp.stack(psf_kernel_all), jnp.stack(corr_field_all)


# -----------------------------
# Update RMS map using stage2 best + alpha map
# -----------------------------
from psf_extra_error import extract_stage2_arcsec_and_flux, build_error_map_with_psf_extra

stage2_losses_np = np.array(jax.device_get(multi_svi_pixel_results_stage2.losses), dtype=float)
best_stage2_i = int(np.argmin(stage2_losses_np[:, -1]))
params_stage2_best = get_value_from_index(multi_svi_pixel_median_stage2, best_stage2_i) | LENS_LIGHT_FIXED_PARAMS
agn_positions_arcsec, agn_flux = extract_stage2_arcsec_and_flux(stage2_params=params_stage2_best)

stage2_psf_kernel_det, _ = get_stage2_deterministics_from_median(multi_svi_pixel_median_stage2)
psf_kernel_stage2_best = np.array(get_value_from_index(stage2_psf_kernel_det, best_stage2_i), dtype=float)

alpha_map_path = "./psf_data/ALPHA_MAP_step3_svi.fits"
alpha_map_use = np.nan_to_num(
    np.array(fits.getdata(alpha_map_path), dtype=float),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)

rms_file_orig = np.array(rms_file, dtype=float)

extra_err_result = build_error_map_with_psf_extra(
    error_map=rms_file_orig,
    agn_positions_arcsec=agn_positions_arcsec,
    agn_flux=agn_flux,
    image_numerics=lens_image_pixel.ImageNumerics,
    alpha_map=alpha_map_use,
    psf_kernel=psf_kernel_stage2_best,
    interpolation_order=1,
)

rms_file_total = extra_err_result["error_total"]
rms_file = jnp.asarray(rms_file_total, dtype=jnp.float64)

print("Updated RMS map in memory from stage2-best PSF extra error")
print("best_stage2_i =", best_stage2_i)


# -----------------------------
# Stage3 pixel SVI (updated RMS)
# -----------------------------
max_iterations_stage3 = max_iterations
num_chains_stage3 = num_chains

scheduler_stage3 = optax.exponential_decay(
    init_value=5e-3,
    transition_steps=200,
    decay_rate=0.99,
)
optim_stage3 = optax.adabelief(learning_rate=scheduler_stage3)
loss_stage3 = infer.TraceMeanField_ELBO()

svi_keys_stage3 = jax.random.split(jax.random.fold_in(rng_key_, 3003), num_chains_stage3)
stage3_results_list = []
stage3_guides = []

for i in range(num_chains_stage3):
    init_values_stage3_i = get_value_from_index(multi_svi_pixel_median_stage2, i) | LENS_LIGHT_FIXED_PARAMS
    init_fun_stage3_i = init_to_value_or_defer(values=init_values_stage3_i)
    guide_stage3_i = autoguide.AutoDiagonalNormal(model, init_loc_fn=init_fun_stage3_i, init_scale=0.01)
    svi_stage3_i = infer.SVI(model, guide_stage3_i, optim_stage3, loss_stage3)
    result_i = svi_stage3_i.run(
        svi_keys_stage3[i],
        max_iterations_stage3,
        data,
        **PIXELATED_STAGE3_KWARGS,
        psf_kernel=psf_hst,
        progress_bar=False,
        stable_update=True,
    )
    stage3_guides.append(guide_stage3_i)
    stage3_results_list.append(result_i)

multi_svi_pixel_results_stage3 = jax.tree.map(_stack_or_none, *stage3_results_list)
guide_pixel_stage3 = stage3_guides[0]
multi_svi_pixel_median_stage3 = guide_pixel_stage3.median(multi_svi_pixel_results_stage3.params)
multi_svi_pixel_median_herc_stage3 = median_params2kwargs(
    lambda p, fixed_params={}: params2kwargs(p, fixed_params=fixed_params, pixelated=True),
    multi_svi_pixel_median_stage3,
    jnp.arange(num_chains_stage3),
)

# HMC starts from the stage3 posterior median and uses the same model config.


# -----------------------------
# HMC/GIBBS (RXJ-style structure)
# -----------------------------
vars_pixel = ["pixels_wn_source_grid"]
vars_power = ["n_source_grid", "rho_source_grid", "sigma_source_grid"]
vars_psf = ["ra_ps", "dec_ps", "log10_amp_ps"]
vars_psf_corr = ["log_psf_corr_center"]

multi_svi_pixel_median_vars = {
    k: multi_svi_pixel_median_stage3[k]
    for k in vars_mass + vars_power + vars_pixel + vars_other + vars_lens_light + vars_psf + vars_psf_corr
}

unconstrined_svi_pixel_median = jax.vmap(
    lambda p: infer.util.unconstrain_fn(
        model,
        (data,),
        HMC_RUN_KWARGS | {"psf_kernel": psf_hst},
        p,
    )
)(multi_svi_pixel_median_vars)

unconstrined_svi_pixel_median = {k: v.astype(jnp.float64) for k, v in unconstrined_svi_pixel_median.items()}
rng_key, rng_key_ = jax.random.split(rng_key)

from numpyro.infer import NUTS, MCMC
from custom_gibbs import MultiHMCGibbs

init_fun_pixel = init_to_value_or_defer(values=get_value_from_index(multi_svi_pixel_median_stage3, 0))

inner_kernels = [
    NUTS(
        model,
        init_strategy=init_fun_pixel,
        target_accept_prob=0.95,
        max_tree_depth=10,
        dense_mass=[
            ("n_source_grid", "rho_source_grid", "sigma_source_grid"),
            ("A_lens", "sigma_lens", "e_lens", "center_lens"),
            ("ra_ps", "dec_ps", "log10_amp_ps"),
            ("center_1","theta_E_1", "theta_E_g1", "theta_E_g2","e_1", "gamma_1", "gamma_sheer_1")
        ],
    ),
    NUTS(
        model,
        init_strategy=init_fun_pixel,
        target_accept_prob=0.9,
        max_tree_depth=10,
    ),
]

outer_kernel = MultiHMCGibbs(
    inner_kernels,
    gibbs_sites_list=[
        vars_pixel + vars_power + vars_lens_light + vars_other + vars_psf + vars_mass,
        vars_psf_corr,
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
batch_list = []
for i in range(batch_number):
    if i == 0:
        mcmc_pixel.run(
            rng_key_,
            data,
            **HMC_RUN_KWARGS,
            psf_kernel=psf_hst,
            init_params=unconstrined_svi_pixel_median,
        )
    else:
        mcmc_pixel.post_warmup_state = mcmc_pixel.last_state
        mcmc_pixel.run(
            mcmc_pixel.post_warmup_state.rng_key,
            data,
            **HMC_RUN_KWARGS,
            psf_kernel=psf_hst,
        )

    mcmc_pixel._states = jax.device_get(mcmc_pixel._states)
    mcmc_pixel._states_flat = jax.device_get(mcmc_pixel._states_flat)
    mcmc_chain = az.from_numpyro(mcmc_pixel)
    batch_path = f"/mnt/lustre/tianli/quasar_hmc/WFI2033_{i}{suffix}.nc"
    mcmc_chain.to_netcdf(batch_path)
    print(f"Saved HMC batch to: {batch_path}")
    batch_list.append(mcmc_chain)


# -----------------------------
# Save only final concatenated HMC result
# -----------------------------
inf_data = az.concat(*batch_list, dim="draw")

final_hmc_path = f"/mnt/lustre/tianli/quasar_hmc/WFI2033_all{suffix}.nc"
inf_data.to_netcdf(final_hmc_path)
print(f"Saved final HMC inf_data to: {final_hmc_path}")
