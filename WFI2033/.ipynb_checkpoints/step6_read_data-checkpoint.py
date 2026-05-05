#!/usr/bin/env python3
from __future__ import annotations

import os
import warnings
from pathlib import Path

# Set HDF5 policy before importing xarray/h5netcdf-backed readers.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from astropy.io import fits
from matplotlib import colors

warnings.simplefilter("ignore")

suffix = "_inferh0_step6_imaging_only_infer_LCDM"
run_tag = "20260502_08"

OUTPUT_ROOT = Path("/mnt/lustre/tianli/quasar_hmc")
RUN_OUTPUT_DIR = OUTPUT_ROOT / f"WFI2033{suffix}_{run_tag}"
NC_PATH = RUN_OUTPUT_DIR / f"WFI2033_all{suffix}.nc"

SCRIPT_DIR = Path(__file__).resolve().parent
STEP5_SUFFIX = "_ss=2_fullconcen_light_multimass"
STEP5_RUN_TAG = "20260427_14"
STEP5_RESULT_DIR = SCRIPT_DIR / "result" / f"result{STEP5_SUFFIX}_{STEP5_RUN_TAG}"
STEP5_PRODUCTS_DIR = STEP5_RESULT_DIR / "data_products"

RAW_DATA_PATH = SCRIPT_DIR / "../../Data/WFI2033/jw01198-o004_t004_nircam_clear-f115w_i2d.fits"
DATA_SUB_PATH = STEP5_RESULT_DIR / f"data_minus_lens_light_corrected_psf{STEP5_SUFFIX}.fits"
RMS_WITH_PSF_EXTRA_PATH = STEP5_PRODUCTS_DIR / f"rms_with_psf_extra{STEP5_SUFFIX}.fits"
MASK_OUT_PATH = SCRIPT_DIR / "data" / "mask_out_center_r16.fits"

OUTPUT_DIR = SCRIPT_DIR / "result" / f"result{suffix}_{run_tag}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PRODUCTS_DIR = OUTPUT_DIR / "data_products"
DATA_PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

POSTERIOR_VARS = [
    "model_image",
    "pixels_source_grid",
    "psf_kernel_corrected",
]


def image_extent(shape: tuple[int, int], pix_scale: float) -> list[float]:
    ny, nx = shape
    half_size_x = nx * pix_scale / 2
    half_size_y = ny * pix_scale / 2
    return [-half_size_x, half_size_x, -half_size_y, half_size_y]


def chain_draw_median(post: xr.Dataset, name: str, chain_index: int) -> np.ndarray:
    return np.asarray(
        post[name].isel(chain=chain_index).median(dim="draw").values,
        dtype=float,
    )


def save_chain_figure(
    chain_index: int,
    data_subtracted: np.ndarray,
    rms_file: np.ndarray,
    mask_plot: np.ndarray,
    extent: list[float],
    model_image: np.ndarray,
    source_image: np.ndarray,
    psf_corrected: np.ndarray,
) -> Path:
    residual = (data_subtracted - model_image) / rms_file
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    ax[0, 0].imshow(
        np.ma.array(data_subtracted, mask=~mask_plot),
        norm="log",
        cmap="twilight",
        origin="lower",
        extent=extent,
        vmin=0.001,
    )
    ax[0, 0].set_title(f"data - lens light | chain {chain_index}")

    ax[0, 1].imshow(
        np.ma.array(model_image, mask=~mask_plot),
        norm="log",
        cmap="twilight",
        origin="lower",
        extent=extent,
        vmin=0.001,
    )
    ax[0, 1].set_title("lens model (model_image)")

    im_res = ax[0, 2].imshow(
        np.ma.array(residual, mask=~mask_plot),
        cmap="bwr",
        vmin=-3,
        vmax=3,
        origin="lower",
        extent=extent,
    )
    ax[0, 2].set_title(f"residual / rms {np.nanmean(residual[mask_plot]):.2f}")
    plt.colorbar(im_res, ax=ax[0, 2], fraction=0.046, pad=0.04)

    src_abs = np.nanmax(np.abs(source_image))
    src_norm = colors.SymLogNorm(
        linthresh=max(src_abs * 1e-3, 1e-8),
        vmin=-src_abs,
        vmax=src_abs,
    )
    ax[1, 0].imshow(source_image, cmap="twilight", norm=src_norm, origin="lower")
    ax[1, 0].set_title("source image (pixels_source_grid)")

    eps = 1e-12
    vmax_psf = np.nanmax(psf_corrected)
    psf_norm = colors.LogNorm(vmin=max(vmax_psf * 1e-6, eps), vmax=max(vmax_psf, eps * 10))
    ax[1, 1].imshow(np.clip(psf_corrected, eps, None), cmap="viridis", norm=psf_norm, origin="lower")
    ax[1, 1].set_title("psf corrected")

    ax[1, 2].imshow(
        np.ma.array(data_subtracted - model_image, mask=~mask_plot),
        cmap="bwr",
        vmin=-0.02,
        vmax=0.02,
        origin="lower",
        extent=extent,
    )
    ax[1, 2].set_title("data - lens light - model")

    for axis in ax.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    out_path = OUTPUT_DIR / f"chain_{chain_index:02d}_lens_model_source{suffix}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    print(f"NC_PATH = {NC_PATH}")
    post = xr.open_dataset(NC_PATH, group="posterior", engine="h5netcdf")[POSTERIOR_VARS]
    sample_stats = xr.open_dataset(NC_PATH, group="sample_stats", engine="h5netcdf")[["diverging"]]

    num_chains = int(post.sizes["chain"])
    num_draws = int(post.sizes["draw"])
    print(f"posterior shape: chains={num_chains}, draws={num_draws}")
    print("posterior vars read:")
    for name in POSTERIOR_VARS:
        print(f"  {name}: dims={post[name].dims}, shape={post[name].shape}")

    div = sample_stats["diverging"]
    print("divergences per chain:")
    print(np.asarray(div.sum(dim="draw").values, dtype=int))

    with fits.open(RAW_DATA_PATH, memmap=True) as hdul_raw:
        raw_header = hdul_raw["SCI"].header if "SCI" in hdul_raw else hdul_raw[0].header
    pix_scale = float(np.sqrt(raw_header["PIXAR_A2"]))

    data_subtracted = np.asarray(fits.getdata(DATA_SUB_PATH), dtype=float)
    rms_file = np.asarray(fits.getdata(RMS_WITH_PSF_EXTRA_PATH), dtype=float)
    mask_plot = np.asarray(fits.getdata(MASK_OUT_PATH), dtype=bool)
    extent = image_extent(data_subtracted.shape, pix_scale)

    fits.writeto(
        DATA_PRODUCTS_DIR / f"data_minus_lens_light{suffix}.fits",
        data_subtracted.astype(np.float32),
        overwrite=True,
    )
    fits.writeto(
        DATA_PRODUCTS_DIR / f"rms_with_psf_extra{suffix}.fits",
        rms_file.astype(np.float32),
        overwrite=True,
    )
    fits.writeto(
        DATA_PRODUCTS_DIR / f"mask_out{suffix}.fits",
        mask_plot.astype(np.uint8),
        overwrite=True,
    )

    for chain_index in range(num_chains):
        model_image = chain_draw_median(post, "model_image", chain_index)
        source_image = chain_draw_median(post, "pixels_source_grid", chain_index)
        psf_corrected = chain_draw_median(post, "psf_kernel_corrected", chain_index)
        psf_corrected = np.clip(psf_corrected, 0.0, None)
        psf_corrected = psf_corrected / np.sum(psf_corrected)
        residual = (data_subtracted - model_image) / rms_file

        fits.writeto(
            DATA_PRODUCTS_DIR / f"chain_{chain_index:02d}_lens_model{suffix}.fits",
            model_image.astype(np.float32),
            overwrite=True,
        )
        fits.writeto(
            DATA_PRODUCTS_DIR / f"chain_{chain_index:02d}_source_image{suffix}.fits",
            source_image.astype(np.float32),
            overwrite=True,
        )
        fits.writeto(
            DATA_PRODUCTS_DIR / f"chain_{chain_index:02d}_psf_corrected{suffix}.fits",
            psf_corrected.astype(np.float32),
            overwrite=True,
        )
        fits.writeto(
            DATA_PRODUCTS_DIR / f"chain_{chain_index:02d}_residual_over_rms{suffix}.fits",
            residual.astype(np.float32),
            overwrite=True,
        )

        fig_path = save_chain_figure(
            chain_index,
            data_subtracted,
            rms_file,
            mask_plot,
            extent,
            model_image,
            source_image,
            psf_corrected,
        )
        print(f"Saved chain {chain_index} products and figure: {fig_path}")

    print(f"Saved extracted products to: {DATA_PRODUCTS_DIR}")


if __name__ == "__main__":
    main()
