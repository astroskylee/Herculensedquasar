#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
warnings.simplefilter("ignore")

INPUT_PATH = Path(
    "/mnt/lustre/tianli/quasar_hmc/"
    "WFI2033_ss=2_inferh0_step6_imaging_only_20260413_19/"
    "WFI2033_all_ss=2_inferh0_step6_imaging_only.nc"
)
OUTPUT_PATH = INPUT_PATH.with_name("WFI2033_all_ss=2_inferh0_step6_imaging_only_withFPD.nc")

suffix_epl = "_ss=2_full_light_multimass"
run_tag = "20260401_11"
OUTPUT_ROOT = Path("/mnt/lustre/tianli/quasar_hmc")
RUN_OUTPUT_DIR = OUTPUT_ROOT / f"WFI2033{suffix_epl}_{run_tag}"
RESULT_DIR = Path(f"./result/result{suffix_epl}_{run_tag}")
DEFAULT_HMC_MEDIAN_PATH = RESULT_DIR / f"HMC_median_draw{suffix_epl}.nc"
DEFAULT_FIXED_FIRST_THREE_PATH = RUN_OUTPUT_DIR / f"fixed_first_three_gaussians{suffix_epl}.npz"

Z_LENS = 0.6575
Z_SOURCE = 1.662
ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)
MPC_TO_KM = 3.0856775814913673e19
DAY_TO_S = 86400.0
TIME_DELAY_OBS = {
    "dt_31_days": {"mean": -36.2, "sigma_minus": 2.3, "sigma_plus": 1.6},
    "dt_32_days": {"mean": -37.3, "sigma_minus": 3.0, "sigma_plus": 2.6},
    "dt_34_days": {"mean": -59.4, "sigma_minus": 1.3, "sigma_plus": 1.3},
}

G1_MASS_CENTER = (1.556, 1.299)
G2_MASS_CENTER = (2.145, -3.326)
G3_MASS_CENTER = (4.243403, 6.185564)
G7_MASS_CENTER = (-7.284665, -10.319681)

SIS_PRIORS = {
    "g3": {"origin": G3_MASS_CENTER, "theta_mean": 0.088},
    "g7": {"origin": G7_MASS_CENTER, "theta_mean": 0.388},
    "g2": {"origin": G2_MASS_CENTER, "theta_mean": 0.622},
}


def fixed_sis(theta_e, origin):
    return [{
        "theta_E": float(theta_e),
        "center_x": float(origin[0]),
        "center_y": float(origin[1]),
    }]


def build_full_lens_light(hmc_median_path, fixed_first_three_path):
    fixed_first_three = np.load(fixed_first_three_path)
    hmc_median = xr.open_dataset(hmc_median_path)
    hmc_reference = hmc_median.median(dim="chain")

    fixed_inner_three = [{
        "amp": np.array(fixed_first_three["amp"], dtype=float),
        "sigma": np.array(fixed_first_three["sigma"], dtype=float),
        "e1": np.array(fixed_first_three["e1"], dtype=float),
        "e2": np.array(fixed_first_three["e2"], dtype=float),
        "center_x": np.array(fixed_first_three["center_x"], dtype=float),
        "center_y": np.array(fixed_first_three["center_y"], dtype=float),
    }]
    outer_two = [{
        "amp": np.array(hmc_reference["amp_lens"].values, dtype=float)[-2:],
        "sigma": np.array(hmc_reference["sigma_lens"].values, dtype=float)[-2:],
        "e1": np.array(hmc_reference["e_lens"].values, dtype=float)[0, -2:],
        "e2": np.array(hmc_reference["e_lens"].values, dtype=float)[1, -2:],
        "center_x": np.array(hmc_reference["center_lens"].values, dtype=float)[0, -2:],
        "center_y": np.array(hmc_reference["center_lens"].values, dtype=float)[1, -2:],
    }]
    return [{
        key: np.concatenate(
            [np.asarray(fixed_inner_three[0][key]), np.asarray(outer_two[0][key])]
        )
        for key in ("amp", "sigma", "e1", "e2", "center_x", "center_y")
    }]


def build_mass_model_step6():
    return MassModel([
        "MULTI_GAUSSIAN_ELLIPSE_KAPPA",
        "CUSPY_NFW_ELLIPSE_KAPPA",
        "SHEAR",
        "SIS",
        "SIS",
        "SIS",
        "SIS",
        "CONVERGENCE",
    ])


def scaled_mass_from_light(full_lens_light, m2l_ratio):
    mass_from_light = deepcopy(full_lens_light)
    amp = np.asarray(mass_from_light[0]["amp"], dtype=float)
    mass_from_light[0]["amp"] = amp * float(m2l_ratio) / np.sum(amp)
    return mass_from_light


def gradient_mass_from_light(full_lens_light, m2l_ratio, m2l_ratio_slope):
    mass_from_light = deepcopy(full_lens_light)
    sigma = np.asarray(mass_from_light[0]["sigma"], dtype=float)
    amp = np.asarray(mass_from_light[0]["amp"], dtype=float)
    r_factor = np.power(sigma, float(m2l_ratio_slope))
    ml_gauss = float(m2l_ratio) * r_factor
    mass_from_light[0]["amp"] = amp / np.sum(amp) * ml_gauss
    return mass_from_light


def sample_value(sample, name):
    return np.asarray(sample[name].values, dtype=float)


def build_kwargs_lens(sample, full_lens_light):
    if "m2l_ratio_slope" in sample.data_vars:
        mass_from_light = gradient_mass_from_light(
            full_lens_light,
            sample_value(sample, "m2l_ratio").reshape(-1)[0],
            sample_value(sample, "m2l_ratio_slope").reshape(-1)[0],
        )
    else:
        mass_from_light = scaled_mass_from_light(
            full_lens_light,
            sample_value(sample, "m2l_ratio").reshape(-1)[0]
        )

    center_halo = sample_value(sample, "center_halo").reshape(2)
    e_halo = sample_value(sample, "e_halo").reshape(2)
    gamma_shear_halo = sample_value(sample, "gamma_sheer_halo").reshape(2)

    gnfw_shear = [{
        "R_s": float(sample_value(sample, "Rs_halo").reshape(-1)[0]),
        "gamma": float(sample_value(sample, "gammain_halo").reshape(-1)[0]),
        "kappa_s": float(sample_value(sample, "kappa_s_halo").reshape(-1)[0]),
        "e1": float(e_halo[0]),
        "e2": float(e_halo[1]),
        "center_x": float(center_halo[0]),
        "center_y": float(center_halo[1]),
    }, {
        "gamma1": float(gamma_shear_halo[0]),
        "gamma2": float(gamma_shear_halo[1]),
        "ra_0": float(center_halo[0]),
        "dec_0": float(center_halo[1]),
    }]

    theta_e_g1 = float(sample_value(sample, "theta_E_g1").reshape(-1)[0])
    theta_e_g2 = float(sample_value(sample, "theta_E_g2").reshape(-1)[0])
    theta_e_g3 = float(
        Mass.scale_theta_E_from_g2(
            theta_e_g2,
            SIS_PRIORS["g3"]["theta_mean"],
            SIS_PRIORS["g2"]["theta_mean"],
        )
    )
    theta_e_g7 = float(
        Mass.scale_theta_E_from_g2(
            theta_e_g2,
            SIS_PRIORS["g7"]["theta_mean"],
            SIS_PRIORS["g2"]["theta_mean"],
        )
    )
    sis_mass = (
        fixed_sis(theta_e_g1, G1_MASS_CENTER)
        + fixed_sis(theta_e_g2, G2_MASS_CENTER)
        + fixed_sis(theta_e_g3, G3_MASS_CENTER)
        + fixed_sis(theta_e_g7, G7_MASS_CENTER)
    )

    if "kappa_ext" in sample.data_vars:
        kappa_ext = float(sample_value(sample, "kappa_ext").reshape(-1)[0])
    else:
        kappa_ext = 0.0
    convergence = [{
        "kappa": kappa_ext,
        "ra_0": 0.0,
        "dec_0": 0.0,
    }]

    return mass_from_light + gnfw_shear + sis_mass + convergence


def compute_fpd_for_sample(sample, full_lens_light, mass_model_step6):
    kwargs_lens = build_kwargs_lens(sample, full_lens_light)
    ra_ps = sample_value(sample, "ra_ps").reshape(-1)
    dec_ps = sample_value(sample, "dec_ps").reshape(-1)
    fermat = np.asarray(
        mass_model_step6.fermat_potential(ra_ps, dec_ps, kwargs_lens),
        dtype=float,
    ).reshape(-1)
    return fermat, fermat[2] - fermat[0], fermat[2] - fermat[1], fermat[2] - fermat[3]


def build_lcdm_time_delay_model():
    class TimeDelayObs(dist.Distribution):
        support = dist.constraints.real

        def __init__(self, loc, sigma_minus, sigma_plus):
            self.loc = loc
            self.sigma_minus = sigma_minus
            self.sigma_plus = sigma_plus
            batch_shape = jax.lax.broadcast_shapes(
                jnp.shape(loc),
                jnp.shape(sigma_minus),
                jnp.shape(sigma_plus),
            )
            super().__init__(batch_shape=batch_shape, event_shape=())

        def log_prob(self, value):
            sigma = jnp.where(value < self.loc, self.sigma_plus, self.sigma_minus)
            log_norm = jnp.log(jnp.sqrt(2.0 / jnp.pi)) - jnp.log(self.sigma_minus + self.sigma_plus)
            return log_norm - 0.5 * ((value - self.loc) / sigma) ** 2

    def model(fpd_31, fpd_32, fpd_34):
        omega_m = numpyro.sample("omega_m_lcdm", dist.Uniform(0.5, 5.0))
        h0 = numpyro.sample("H0_lcdm", dist.Uniform(60.0, 80.0))
        cosmology = {
            "Omegam": omega_m,
            "Omegak": jnp.asarray(0.0, dtype=jnp.float64),
            "w0": jnp.asarray(-1.0, dtype=jnp.float64),
            "wa": jnp.asarray(0.0, dtype=jnp.float64),
            "h0": h0,
        }
        d_dt = numpyro.deterministic(
            "D_dt_Mpc_lcdm",
            Cosmo.compute_time_delay_distances(cosmology, Z_LENS, Z_SOURCE),
        )
        prefactor_days = numpyro.deterministic(
            "time_delay_prefactor_days_lcdm",
            d_dt * jnp.asarray(MPC_TO_KM * ARCSEC_TO_RAD**2 / (Cosmo.c_km_s * DAY_TO_S), dtype=jnp.float64),
        )
        dt_31 = numpyro.deterministic("dt_31_days_lcdm", prefactor_days * fpd_31)
        dt_32 = numpyro.deterministic("dt_32_days_lcdm", prefactor_days * fpd_32)
        dt_34 = numpyro.deterministic("dt_34_days_lcdm", prefactor_days * fpd_34)
        numpyro.sample(
            "dt_31_obs",
            TimeDelayObs(
                dt_31,
                jnp.asarray(TIME_DELAY_OBS["dt_31_days"]["sigma_minus"], dtype=jnp.float64),
                jnp.asarray(TIME_DELAY_OBS["dt_31_days"]["sigma_plus"], dtype=jnp.float64),
            ),
            obs=jnp.asarray(TIME_DELAY_OBS["dt_31_days"]["mean"], dtype=jnp.float64),
        )
        numpyro.sample(
            "dt_32_obs",
            TimeDelayObs(
                dt_32,
                jnp.asarray(TIME_DELAY_OBS["dt_32_days"]["sigma_minus"], dtype=jnp.float64),
                jnp.asarray(TIME_DELAY_OBS["dt_32_days"]["sigma_plus"], dtype=jnp.float64),
            ),
            obs=jnp.asarray(TIME_DELAY_OBS["dt_32_days"]["mean"], dtype=jnp.float64),
        )
        numpyro.sample(
            "dt_34_obs",
            TimeDelayObs(
                dt_34,
                jnp.asarray(TIME_DELAY_OBS["dt_34_days"]["sigma_minus"], dtype=jnp.float64),
                jnp.asarray(TIME_DELAY_OBS["dt_34_days"]["sigma_plus"], dtype=jnp.float64),
            ),
            obs=jnp.asarray(TIME_DELAY_OBS["dt_34_days"]["mean"], dtype=jnp.float64),
        )

    return model


def infer_lcdm_for_sample(fpd_31, fpd_32, fpd_34, rng_key, num_warmup, num_samples):
    model = build_lcdm_time_delay_model()
    kernel = infer.NUTS(model, target_accept_prob=0.9)
    mcmc = infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        rng_key,
        jnp.asarray(fpd_31, dtype=jnp.float64),
        jnp.asarray(fpd_32, dtype=jnp.float64),
        jnp.asarray(fpd_34, dtype=jnp.float64),
    )
    return {k: np.asarray(v, dtype=float) for k, v in mcmc.get_samples(group_by_chain=False).items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--hmc-median", type=Path, default=DEFAULT_HMC_MEDIAN_PATH)
    parser.add_argument("--fixed-first-three", type=Path, default=DEFAULT_FIXED_FIRST_THREE_PATH)
    parser.add_argument("--lcdm-warmup", type=int, default=200)
    parser.add_argument("--lcdm-samples", type=int, default=200)
    parser.add_argument("--lcdm-seed", type=int, default=1234)
    return parser.parse_args()


def setup_infra():
    from Tian_infra import import_function

    import_function(globals())
    jax.config.update("jax_enable_x64", True)
    numpyro.enable_x64()

    class CuspyNFWEllipseKappa(MGE):
        def __init__(self):
            super().__init__(
                CuspyNFW_3D_fn,
                "R_s",
                n_gauss=20,
                n_terms=28,
                sigma_start_mult=1 / 200,
                sigma_end_mult=20,
                three_d=True,
            )

    mass_model_base.STRING_MAPPING["CUSPY_NFW_ELLIPSE_KAPPA"] = CuspyNFWEllipseKappa


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.hmc_median.exists():
        raise FileNotFoundError(f"HMC median file not found: {args.hmc_median}")
    if not args.fixed_first_three.exists():
        raise FileNotFoundError(f"Fixed-first-three file not found: {args.fixed_first_three}")

    setup_infra()
    full_lens_light = build_full_lens_light(args.hmc_median, args.fixed_first_three)
    mass_model_step6 = build_mass_model_step6()

    idata = az.from_netcdf(args.input)
    posterior = idata.posterior.load()
    n_chain = posterior.sizes["chain"]
    n_draw = posterior.sizes["draw"]

    fermat_images = np.empty((n_chain, n_draw, 4), dtype=float)
    fpd_31 = np.empty((n_chain, n_draw), dtype=float)
    fpd_32 = np.empty((n_chain, n_draw), dtype=float)
    fpd_34 = np.empty((n_chain, n_draw), dtype=float)
    omega_m_lcdm = np.empty((n_chain, n_draw, args.lcdm_samples), dtype=float)
    h0_lcdm = np.empty((n_chain, n_draw, args.lcdm_samples), dtype=float)
    d_dt_lcdm = np.empty((n_chain, n_draw, args.lcdm_samples), dtype=float)
    dt_31_lcdm = np.empty((n_chain, n_draw, args.lcdm_samples), dtype=float)
    dt_32_lcdm = np.empty((n_chain, n_draw, args.lcdm_samples), dtype=float)
    dt_34_lcdm = np.empty((n_chain, n_draw, args.lcdm_samples), dtype=float)

    total = n_chain * n_draw
    count = 0
    for chain in range(n_chain):
        for draw in range(n_draw):
            sample = posterior.isel(chain=chain, draw=draw)
            fermat, phi31, phi32, phi34 = compute_fpd_for_sample(
                sample,
                full_lens_light,
                mass_model_step6,
            )
            fermat_images[chain, draw, :] = fermat
            fpd_31[chain, draw] = phi31
            fpd_32[chain, draw] = phi32
            fpd_34[chain, draw] = phi34
            lcdm_samples = infer_lcdm_for_sample(
                phi31,
                phi32,
                phi34,
                jax.random.PRNGKey(args.lcdm_seed + count),
                args.lcdm_warmup,
                args.lcdm_samples,
            )
            omega_m_lcdm[chain, draw, :] = lcdm_samples["omega_m_lcdm"]
            h0_lcdm[chain, draw, :] = lcdm_samples["H0_lcdm"]
            d_dt_lcdm[chain, draw, :] = lcdm_samples["D_dt_Mpc_lcdm"]
            dt_31_lcdm[chain, draw, :] = lcdm_samples["dt_31_days_lcdm"]
            dt_32_lcdm[chain, draw, :] = lcdm_samples["dt_32_days_lcdm"]
            dt_34_lcdm[chain, draw, :] = lcdm_samples["dt_34_days_lcdm"]
            count += 1
            if count % 20 == 0 or count == total:
                print(f"Processed {count}/{total} samples")

    chain_coord = posterior.coords["chain"]
    draw_coord = posterior.coords["draw"]
    idata.posterior = posterior.assign(
        fermat_potential_images=xr.DataArray(
            fermat_images,
            dims=("chain", "draw", "image"),
            coords={"chain": chain_coord, "draw": draw_coord, "image": ["A1", "A2", "B", "C"]},
        ),
        fpd_31=xr.DataArray(fpd_31, dims=("chain", "draw"), coords={"chain": chain_coord, "draw": draw_coord}),
        fpd_32=xr.DataArray(fpd_32, dims=("chain", "draw"), coords={"chain": chain_coord, "draw": draw_coord}),
        fpd_34=xr.DataArray(fpd_34, dims=("chain", "draw"), coords={"chain": chain_coord, "draw": draw_coord}),
        omega_m_lcdm=xr.DataArray(
            omega_m_lcdm,
            dims=("chain", "draw", "lcdm_draw"),
            coords={"chain": chain_coord, "draw": draw_coord, "lcdm_draw": np.arange(args.lcdm_samples)},
        ),
        H0_lcdm=xr.DataArray(
            h0_lcdm,
            dims=("chain", "draw", "lcdm_draw"),
            coords={"chain": chain_coord, "draw": draw_coord, "lcdm_draw": np.arange(args.lcdm_samples)},
        ),
        D_dt_Mpc_lcdm=xr.DataArray(
            d_dt_lcdm,
            dims=("chain", "draw", "lcdm_draw"),
            coords={"chain": chain_coord, "draw": draw_coord, "lcdm_draw": np.arange(args.lcdm_samples)},
        ),
        dt_31_days_lcdm=xr.DataArray(
            dt_31_lcdm,
            dims=("chain", "draw", "lcdm_draw"),
            coords={"chain": chain_coord, "draw": draw_coord, "lcdm_draw": np.arange(args.lcdm_samples)},
        ),
        dt_32_days_lcdm=xr.DataArray(
            dt_32_lcdm,
            dims=("chain", "draw", "lcdm_draw"),
            coords={"chain": chain_coord, "draw": draw_coord, "lcdm_draw": np.arange(args.lcdm_samples)},
        ),
        dt_34_days_lcdm=xr.DataArray(
            dt_34_lcdm,
            dims=("chain", "draw", "lcdm_draw"),
            coords={"chain": chain_coord, "draw": draw_coord, "lcdm_draw": np.arange(args.lcdm_samples)},
        ),
    )
    idata.to_netcdf(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
