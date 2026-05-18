#!/usr/bin/env python3
"""Add derived halo and stellar-mass deterministics to Step 6 posterior files.

This script reads compact posterior variables from the Step6_preview runs,
computes spherical-equivalent gNFW M200/c200 and physical stellar masses from
the MGE convergence amplitudes, and writes new compact ArviZ netCDF files.
"""

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr
from scipy.optimize import brentq
from scipy.special import hyp2f1
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
OVERWRITE = True
OUTPUT_TAG = "halo_stellar_deterministics"

Z_LENS = 0.6575
Z_SOURCE = 1.662
DELTA_HALO = 200.0
N_QUAD = 96

C_KM_S = 299792.458
G_PC_KMS2_MSUN = 4.300917270038e-3
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

STEP6_CASES = {
    "pantheon_sh0es_reuse_redwarmup": {
        "input": Path(
            "/mnt/lustre/tianli/quasar_hmc/"
            "WFI2033_ss=2_fullconcen_light_multimass_step6_pantheonsh0es_reuse_redwarmup_20260507_14/"
            "WFI2033_all_ss=2_fullconcen_light_multimass_step6_pantheonsh0es_reuse_redwarmup.nc"
        ),
    },
    "desi_planck": {
        "input": Path(
            "/mnt/lustre/tianli/quasar_hmc/"
            "WFI2033_ss=2_fullconcen_light_multimass_step6_desi_planck_20260504_16/"
            "WFI2033_all_ss=2_fullconcen_light_multimass_step6_desi_planck.nc"
        ),
    },
}

MAX_EXTRA_SIZE = 10
EXCLUDE_POSTERIOR_VARS = {
    "model_image",
    "pixels_source_grid",
    "pixels_wn_source_grid",
    "psf_kernel_corrected",
    "log_psf_corr_center",
    "psf_corr_factor_field",
}


def posterior_extra_size(data_array):
    extra_size = 1
    for dim, size in data_array.sizes.items():
        if dim not in ("chain", "draw"):
            extra_size *= int(size)
    return extra_size


def compact_posterior_var_names(input_path):
    ds_meta = xr.open_dataset(input_path, group="posterior", engine="h5netcdf")
    keep_vars = []
    skip_vars = []

    for name, data_array in ds_meta.data_vars.items():
        extra_size = posterior_extra_size(data_array)
        if name in EXCLUDE_POSTERIOR_VARS or extra_size > MAX_EXTRA_SIZE:
            skip_vars.append((name, extra_size, data_array.shape))
        else:
            keep_vars.append(name)

    ds_meta.close()

    print(f"keeping {len(keep_vars)} compact posterior variables:")
    print("  " + ", ".join(keep_vars))
    print(f"skipping {len(skip_vars)} matrix-like posterior variables:")
    for name, extra_size, shape in skip_vars:
        print(f"  {name}: extra_size={extra_size}, shape={shape}")

    return keep_vars


def read_compact_posterior(input_path):
    keep_vars = compact_posterior_var_names(input_path)
    return xr.open_dataset(input_path, group="posterior", engine="h5netcdf")[keep_vars].load()


def flat_lcdm_comoving_distance_mpc(z, H0, omega_m, n_quad=N_QUAD):
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    z_nodes = 0.5 * z * (nodes + 1.0)
    w_nodes = 0.5 * z * weights
    e_z = np.sqrt(omega_m[..., None] * (1.0 + z_nodes) ** 3 + (1.0 - omega_m[..., None]))
    integral = np.sum(w_nodes / e_z, axis=-1)
    return C_KM_S * integral / H0


def m_gamma(x, gamma):
    a = 3.0 - gamma
    return x**a / a * hyp2f1(a, a, a + 1.0, -x)


def solve_c200(kappa_s, gamma_in, Rs_arcsec, H0, omega_m):
    dc_lens_mpc = flat_lcdm_comoving_distance_mpc(Z_LENS, H0, omega_m)
    dc_source_mpc = flat_lcdm_comoving_distance_mpc(Z_SOURCE, H0, omega_m)

    d_lens_pc = dc_lens_mpc / (1.0 + Z_LENS) * 1.0e6
    d_source_pc = dc_source_mpc / (1.0 + Z_SOURCE) * 1.0e6
    d_lens_source_pc = (dc_source_mpc - dc_lens_mpc) / (1.0 + Z_SOURCE) * 1.0e6

    sigma_crit = C_KM_S**2 / (4.0 * np.pi * G_PC_KMS2_MSUN) * d_source_pc / (d_lens_pc * d_lens_source_pc)
    arcsec_to_pc = d_lens_pc * ARCSEC_TO_RAD
    Rs_pc = Rs_arcsec * arcsec_to_pc
    rho_s = kappa_s * sigma_crit / Rs_pc

    Hz = H0 * np.sqrt(omega_m * (1.0 + Z_LENS) ** 3 + (1.0 - omega_m))
    rho_crit = 3.0 * (Hz / 1.0e6) ** 2 / (8.0 * np.pi * G_PC_KMS2_MSUN)
    rhs = DELTA_HALO / 3.0 * rho_crit / rho_s

    c200 = np.empty_like(rhs, dtype=np.float64)
    rhs_flat = np.ravel(rhs)
    gamma_flat = np.ravel(gamma_in)
    c200_flat = np.ravel(c200)

    for i in tqdm(range(rhs_flat.size), desc="solving c200"):
        target = rhs_flat[i]
        gamma_i = gamma_flat[i]
        c200_flat[i] = brentq(lambda x: m_gamma(x, gamma_i) / x**3 - target, 1.0e-2, 1.0e3)

    R200_pc = c200 * Rs_pc
    M200_msun = 4.0 * np.pi / 3.0 * DELTA_HALO * rho_crit * R200_pc**3

    return {
        "sigma_crit_msun_pc2": sigma_crit,
        "arcsec_to_pc": arcsec_to_pc,
        "rho_crit_msun_pc3": rho_crit,
        "rho_s_halo_msun_pc3": rho_s,
        "R200_halo_kpc": R200_pc / 1.0e3,
        "M200_halo": M200_msun,
        "M200_halo_1E12": M200_msun / 1.0e12,
        "c200_halo": c200,
    }


def add_derived_variables(post):
    chain_draw_coords = {"chain": post.coords["chain"], "draw": post.coords["draw"]}
    chain_draw_dims = ("chain", "draw")

    arrays = solve_c200(
        np.asarray(post["kappa_s_halo"].values, dtype=np.float64),
        np.asarray(post["gammain_halo"].values, dtype=np.float64),
        np.asarray(post["Rs_halo"].values, dtype=np.float64),
        np.asarray(post["H0_cosmo"].values, dtype=np.float64),
        np.asarray(post["omega_m_cosmo"].values, dtype=np.float64),
    )

    out = post.copy()
    for name, values in arrays.items():
        out[name] = xr.DataArray(values, dims=chain_draw_dims, coords=chain_draw_coords)

    vector_dim = post["e_halo"].dims[-1]
    e1_halo = post["e_halo"].isel({vector_dim: 0})
    e2_halo = post["e_halo"].isel({vector_dim: 1})
    e_abs_halo = np.sqrt(e1_halo**2 + e2_halo**2)
    out["q_halo"] = (1.0 - e_abs_halo) / (1.0 + e_abs_halo)
    out["phi_halo_deg"] = (np.rad2deg(0.5 * np.arctan2(e2_halo, e1_halo)) + 90.0) % 180.0 - 90.0

    shear_dim = post["gamma_sheer_halo"].dims[-1]
    gamma1_halo = post["gamma_sheer_halo"].isel({shear_dim: 0})
    gamma2_halo = post["gamma_sheer_halo"].isel({shear_dim: 1})
    out["gamma_ext"] = np.sqrt(gamma1_halo**2 + gamma2_halo**2)
    out["phi_shear_deg"] = (np.rad2deg(0.5 * np.arctan2(gamma2_halo, gamma1_halo)) + 90.0) % 180.0 - 90.0

    stellar_mass_per_kappa_arcsec2 = out["sigma_crit_msun_pc2"] * out["arcsec_to_pc"] ** 2
    out["stellar_mass_gauss"] = post["mass_from_light_amp"] * stellar_mass_per_kappa_arcsec2
    out["stellar_mass_gauss_1E11"] = out["stellar_mass_gauss"] / 1.0e11
    out["total_stellar_mass_msun"] = out["stellar_mass_gauss"].sum(dim=post["mass_from_light_amp"].dims[-1])
    out["total_stellar_mass_1E11"] = out["total_stellar_mass_msun"] / 1.0e11

    out["M200_halo"].attrs["units"] = "Msun"
    out["M200_halo_1E12"].attrs["units"] = "1e12 Msun"
    out["R200_halo_kpc"].attrs["units"] = "kpc"
    out["c200_halo"].attrs["definition"] = "R200 / Rs, spherical-equivalent gNFW halo"
    out["q_halo"].attrs["definition"] = "(1 - sqrt(e1_halo^2 + e2_halo^2)) / (1 + sqrt(e1_halo^2 + e2_halo^2))"
    out["phi_halo_deg"].attrs["units"] = "deg"
    out["gamma_ext"].attrs["definition"] = "sqrt(gamma1_halo^2 + gamma2_halo^2)"
    out["phi_shear_deg"].attrs["units"] = "deg"
    out["stellar_mass_gauss"].attrs["units"] = "Msun"
    out["stellar_mass_gauss_1E11"].attrs["units"] = "1e11 Msun"
    out["total_stellar_mass_msun"].attrs["units"] = "Msun"
    out["total_stellar_mass_1E11"].attrs["units"] = "1e11 Msun"
    out.attrs["z_lens"] = Z_LENS
    out.attrs["z_source"] = Z_SOURCE
    out.attrs["halo_mass_definition"] = f"M{int(DELTA_HALO)} relative to critical density at z_lens"
    out.attrs["stellar_mass_conversion"] = "mass_from_light_amp * Sigma_crit * (arcsec_to_pc)^2"
    return out


def process_case(name, input_path, output_path):
    print(f"\n=== {name} ===")
    print(f"input  = {input_path}")
    print(f"output = {output_path}")
    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Output exists and OVERWRITE=False: {output_path}")

    post = read_compact_posterior(input_path)
    sample_stats = xr.open_dataset(input_path, group="sample_stats", engine="h5netcdf")[["diverging"]].load()
    out_post = add_derived_variables(post)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    az.InferenceData(posterior=out_post, sample_stats=sample_stats).to_netcdf(output_path)

    print("summary:")
    for var in ["M200_halo_1E12", "c200_halo", "q_halo", "gamma_ext", "kappa_ext", "total_stellar_mass_1E11"]:
        q16, q50, q84 = np.nanpercentile(out_post[var].values, [16, 50, 84])
        print(f"  {var}: {q50:.4g} -{q50 - q16:.4g} +{q84 - q50:.4g}")


def main():
    for name, paths in STEP6_CASES.items():
        output_path = paths["input"].with_name(f"{paths['input'].stem}_{OUTPUT_TAG}.nc")
        process_case(name, paths["input"], output_path)


if __name__ == "__main__":
    main()
