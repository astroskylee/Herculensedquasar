import os
import warnings
from pathlib import Path

# Set HDF5 file locking policy before importing ArviZ/xarray/h5py stack.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from astropy.io import fits
from matplotlib import colors

warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

def resolve_nc_path() -> str:
    """Resolve the HMC netCDF path across common lustre layouts and filenames."""
    env_path = os.environ.get("WFI2033_NC_PATH", "").strip()
    if env_path and Path(env_path).is_file():
        return env_path

    names = ["WFI2033_psf_correct_all.nc", "WFI2033_psfcorrection_all.nc"]
    roots = [
        Path("/mnt/lustre/tianli/quasar_hmc"),
        Path("/mnt/lustre2/tianli/quasar_hmc"),
        Path("/users/tianli/quasar_hmc"),
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]
    candidates = []
    for root in roots:
        for name in names:
            p = root / name
            candidates.append(p)
            if p.is_file():
                return str(p)

    cand_txt = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Cannot find WFI2033 netCDF file. Set WFI2033_NC_PATH or place file at one candidate path:\n"
        f"{cand_txt}"
    )


NC_PATH = resolve_nc_path()
DATA_DIR = "../../Data/WFI2033"

RAW_DATA_PATH = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d.fits")
DATA_PATH = os.path.join(DATA_DIR, "jw01198-o004_t004_nircam_clear-f115w_i2d_cut_x6985_y3594_150.fits")
RMS_WITH_PSF_EXTRA_PATH = "./psf_data/WFI2033_ERR_with_stage2_psf_extra.fits"
BASE_PSF_PATH = "./psf_data/PSF_model_step3_svi.fits"
OUTPUT_DIR = Path(__file__).resolve().parent / "result" / "read_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# Read posterior (.nc) and print key info
# --------------------------------
inf_data_pixel = az.from_netcdf(NC_PATH)
print(f"Using netCDF: {NC_PATH}")
post = inf_data_pixel.posterior

print(post)
print("\nposterior vars:")
print(list(post.data_vars))

print("\nvar summary:")
for name, da in post.data_vars.items():
    arr = np.asarray(da.values)
    print(
        f"{name:30s} dims={da.dims} shape={da.shape} dtype={arr.dtype} "
        f"min={np.nanmin(arr):.3e} max={np.nanmax(arr):.3e}"
    )

vars_mass = ["theta_E_1", "gamma_1", "e_1", "center_1", "gamma_sheer_1"]
vars_power = ["n_source_grid", "rho_source_grid", "sigma_source_grid"]
print(
    f"divergences per chain per step:\n"
    f"{inf_data_pixel.sample_stats.diverging.values.sum(axis=1).T}"
)
print(az.summary(inf_data_pixel.sel(chain=np.array([0, 1, 2, 3])), var_names=vars_mass + vars_power))

# --------------------------------
# Read data and RMS
# --------------------------------
with fits.open(RAW_DATA_PATH, memmap=True) as hdul_raw:
    exposure_time = float(hdul_raw[0].header.get("EXPTIME", hdul_raw[0].header.get("TEXPTIME", 1.0)))
print(f"EXPTIME = {exposure_time:.3f} s")

with fits.open(DATA_PATH, memmap=True) as hdul:
    data = np.array(hdul["SCI"].data if "SCI" in hdul else hdul[0].data, dtype=float)

corner_pixel = 10
bkg_corner = data[:corner_pixel, :corner_pixel]
bkg_mean = float(np.nanmean(bkg_corner))
data = data - bkg_mean

rms_file = np.array(fits.getdata(RMS_WITH_PSF_EXTRA_PATH), dtype=float)

with fits.open(BASE_PSF_PATH, memmap=True) as hdul_psf:
    psf_base = np.array(hdul_psf["DET_PSF_MODEL"].data, dtype=float)
psf_base = np.clip(psf_base, 0.0, None)
psf_base = psf_base / np.sum(psf_base)

# --------------------------------
# Median products per chain
# --------------------------------
model_med = np.array(post["model_image"].median(dim="draw").values)
psf_corr_med = np.array(post["psf_kernel_corrected"].median(dim="draw").values)
source_med = np.array(post["pixels_source_grid"].median(dim="draw").values)

# --------------------------------
# Plot: data/model/residual + base PSF/corrected PSF/source
# --------------------------------
for i in range(4):
    model_i = model_med[i]
    res_i = (data - model_i) / rms_file
    psf_corr_i = psf_corr_med[i]
    source_i = source_med[i]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ax[0, 0].imshow(data, norm="log", cmap="twilight", origin="lower")
    ax[0, 0].set_title(f"data | chain {i}")

    ax[0, 1].imshow(model_i, norm="log", cmap="twilight", origin="lower")
    ax[0, 1].set_title("model")

    im_res = ax[0, 2].imshow(res_i, cmap="bwr", vmin=-3, vmax=3, origin="lower")
    ax[0, 2].set_title("residual / rms")
    plt.colorbar(im_res, ax=ax[0, 2], fraction=0.046, pad=0.04)

    eps = 1e-12
    vmax_psf = np.nanmax([np.nanmax(psf_base), np.nanmax(psf_corr_i)])
    psf_norm = colors.LogNorm(vmin=vmax_psf * 1e-6, vmax=vmax_psf)

    ax[1, 0].imshow(np.clip(psf_base, eps, None), cmap="viridis", norm=psf_norm, origin="lower")
    ax[1, 0].set_title("PSF (base)")

    ax[1, 1].imshow(np.clip(psf_corr_i, eps, None), cmap="viridis", norm=psf_norm, origin="lower")
    ax[1, 1].set_title("PSF (corrected)")

    src_abs = np.nanmax(np.abs(source_i))
    src_norm = colors.SymLogNorm(linthresh=max(src_abs * 1e-3, 1e-8), vmin=-src_abs, vmax=src_abs)
    ax[1, 2].imshow(source_i, cmap="twilight", norm=src_norm, origin="lower")
    ax[1, 2].set_title("source (pixels_source_grid)")

    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"chain_{i:02d}_data_model_residual_psf_source.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)
