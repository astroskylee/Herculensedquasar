from __future__ import annotations

from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from jax import lax
import numpyro
from numpyro import distributions as dist
from scipy.special import roots_legendre


class Plot:
    @staticmethod
    def sanitize_label(label):
        return (
            str(label)
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('/', '_')
        )


class ResumeInit:
    @staticmethod
    def select_init_values(params, allowed_keys):
        return {k: params[k] for k in allowed_keys if k in params}

    @staticmethod
    def stack_or_none(*xs):
        first = xs[0]
        return None if first is None else jnp.stack(xs)

    @staticmethod
    def stack_dicts(dict_list):
        return jax.tree.map(lambda *xs: jnp.stack(xs), *dict_list)

    @staticmethod
    def existing_batch_indices(output_dir, suffix_hmc):
        prefix = 'WFI2033_'
        indices = []
        for path in Path(output_dir).glob(f'WFI2033_[0-9]*{suffix_hmc}.nc'):
            name = path.name
            idx_str = name[len(prefix):name.index(suffix_hmc)]
            indices.append(int(idx_str))
        return sorted(indices)

    @staticmethod
    def save_resume_state(path, state):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as fh:
            pickle.dump(jax.device_get(state), fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_resume_state(path):
        with Path(path).open('rb') as fh:
            return pickle.load(fh)


class Mass:
    @staticmethod
    def scale_theta_E_from_g2(theta_E_g2, target_theta_mean, g2_theta_mean):
        return theta_E_g2 * target_theta_mean / g2_theta_mean


class Numpyro_function:
    @staticmethod
    def split_normal_logpdf(x, mean, sigma_minus, sigma_plus):
        x = jnp.asarray(x, dtype=jnp.float64)
        mean = jnp.asarray(mean, dtype=jnp.float64)
        sigma_minus = jnp.asarray(sigma_minus, dtype=jnp.float64)
        sigma_plus = jnp.asarray(sigma_plus, dtype=jnp.float64)

        sigma = jnp.where(x < mean, sigma_minus, sigma_plus)
        log_norm = jnp.log(jnp.sqrt(2.0 / jnp.pi)) - jnp.log(sigma_minus + sigma_plus)
        return log_norm - 0.5 * ((x - mean) / sigma) ** 2


class Cosmo:
    c_km_s = 299792.458

    @staticmethod
    def sample_cosmology_from_prior(stage_kwargs, cosmo_priors):
        prior_name = stage_kwargs['cosmo_prior_name']
        prior = cosmo_priors[prior_name]
        cosmo_vec = numpyro.sample(
            'cosmo_vec',
            dist.MultivariateNormal(
                loc=jnp.asarray(prior['mean_vec'], dtype=jnp.float64),
                covariance_matrix=jnp.asarray(prior['cov'], dtype=jnp.float64),
            ),
        )
        omega_m = numpyro.deterministic('omega_m_cosmo', cosmo_vec[0])
        h0 = numpyro.deterministic('H0_cosmo', cosmo_vec[1])
        return {
            'Omegam': omega_m,
            'Omegak': jnp.asarray(0.0, dtype=jnp.float64),
            'w0': jnp.asarray(-1.0, dtype=jnp.float64),
            'wa': jnp.asarray(0.0, dtype=jnp.float64),
            'h0': h0,
        }

    @staticmethod
    def func(z, Omegam, Omegak, w0, wa=0):
        Omegal = 1.0 - Omegam - Omegak
        zp1 = 1.0 + z
        de = zp1 ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / zp1)
        Ez2 = Omegam * zp1 ** 3 + Omegak * zp1 ** 2 + Omegal * de
        return Ez2 ** -0.5

    @staticmethod
    def nth_order_quad(n=20):
        xval, weights = map(jnp.array, roots_legendre(n))
        xval = xval.reshape(-1, 1)
        weights = weights.reshape(-1, 1)

        def integrate(func, a, b, *args):
            return 0.5 * (b - a) * jnp.sum(
                weights * func(0.5 * ((b - a) * xval + (b + a)), *args),
                axis=0,
            )

        return integrate

    @staticmethod
    def integrate(func, a, b, *args, n=20):
        quad = Cosmo.nth_order_quad(n)
        return quad(func, a, b, *args)

    @staticmethod
    def Dplus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sinh(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sinh(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sinh(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dminus(Omegak, Es, El, zs, zl):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        Ds = jnp.sin(sqrt_ok * Es) / sqrt_ok / (1 + zs)
        Dls = jnp.sin(sqrt_ok * (Es - El)) / sqrt_ok / (1 + zs)
        Dl = jnp.sin(sqrt_ok * El) / sqrt_ok / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dflat(Es, El, zs, zl):
        Ds = Es / (1 + zs)
        Dls = (Es - El) / (1 + zs)
        Dl = El / (1 + zl)
        return Dl, Ds, Dls

    @staticmethod
    def Dpos(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sinh(sqrt_ok * E) / sqrt_ok / (1 + z)

    @staticmethod
    def Dneg(Omegak, E, z):
        sqrt_ok = jnp.sqrt(jnp.abs(Omegak))
        return jnp.sin(sqrt_ok * E) / sqrt_ok / (1 + z)

    @staticmethod
    def Dzero(E, z):
        return E / (1 + z)

    @staticmethod
    def angular_diameter_distance(z, cosmology, n=20):
        Omegam = cosmology["Omegam"]
        Omegak = cosmology["Omegak"]
        w0 = cosmology["w0"]
        wa = cosmology["wa"]
        h = cosmology["h0"]
        E = Cosmo.integrate(Cosmo.func, 0, z, Omegam, Omegak, w0, wa, n=n)

        Dl = lax.cond(
            Omegak > 0,
            lambda _: Cosmo.Dpos(Omegak, E, z),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: Cosmo.Dneg(Omegak, E, z),
                lambda _: Cosmo.Dzero(E, z),
                None,
            ),
            None,
        )
        return Dl * Cosmo.c_km_s / h

    @staticmethod
    def dldsdls(zl, zs, cosmology, n=20):
        Omegam = cosmology["Omegam"]
        Omegak = cosmology["Omegak"]
        w0 = cosmology["w0"]
        wa = cosmology["wa"]
        h = cosmology["h0"]

        El = Cosmo.integrate(Cosmo.func, 0, zl, Omegam, Omegak, w0, wa, n=n)
        Es = Cosmo.integrate(Cosmo.func, 0, zs, Omegam, Omegak, w0, wa, n=n)

        Dl, Ds, Dls = lax.cond(
            Omegak > 0,
            lambda _: Cosmo.Dplus(Omegak, Es, El, zs, zl),
            lambda _: lax.cond(
                Omegak < 0,
                lambda _: Cosmo.Dminus(Omegak, Es, El, zs, zl),
                lambda _: Cosmo.Dflat(Es, El, zs, zl),
                None,
            ),
            None,
        )
        return Dl * Cosmo.c_km_s / h, Ds * Cosmo.c_km_s / h, Dls * Cosmo.c_km_s / h

    @staticmethod
    def compute_time_delay_distances(cosmology, kappa_ext, z_lens, z_source):
        dl, ds, dls = Cosmo.dldsdls(z_lens, z_source, cosmology)
        d_dt_true = (1.0 + z_lens) * dl * ds / dls
        d_dt_model = (1.0 - kappa_ext) * d_dt_true
        return d_dt_true, d_dt_model
