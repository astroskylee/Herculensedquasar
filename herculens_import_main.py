import warnings
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import numpyro.infer.autoguide as autoguide
from numpyro.handlers import condition

print(jax.devices())

import arviz as az
import optax

import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from corner import corner
from copy import deepcopy
from functools import partial


from astropy.io import fits
from skimage.feature import peak_local_max
from skimage.io import imread

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF

from herculens.LightModel import light_model_base
from herculens.LightModel.light_model import LightModel
import jax_lensing_profiles
# from svi_utils import split_scheduler, plot_loss

from power_spectrum_prior import P_Matern, vpack_fft_values, pack_fft_values, unpack_fft_values, vunpack_fft_values, K_grid

from herculens.MassModel.mass_model_multiplane import MPMassModel
from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model_multiplane import MPLightModel
from herculens.LightModel.light_model import LightModel
from herculens.LensImage.lens_image_multiplane import MPLensImage
from lens_images_extension import LensImageExtension
from lens_images_extension import pixelize_plane as pixelize_plane_single
#priors:
##########################################################################################################
##########################################################################################################
def EPL_w_shear(plate_name, param_name,theta_low = 0.0, theta_high = 3, gamma_low = 1.2, gamma_up = 2.8, center_x = None, center_y = None, e_low=-0.2, e_high=0.2, center_high = 0.2, center_low = -0.2):
    with numpyro.plate(f'{plate_name} scalers - [1]', 1):
        theta_E = numpyro.sample(f'theta_E_{param_name}', dist.Uniform(theta_low, theta_high))
        gamma = numpyro.sample(f'gamma_{param_name}', dist.Uniform(gamma_low, gamma_up))
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e_mass = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.25, low=e_low, high=e_high))
            gamma_sheer = numpyro.sample(f'gamma_sheer_{param_name}', dist.Uniform(-0.2, 0.2))
    if center_x is None:
        center = numpyro.sample(
            f'center_{param_name}',
            dist.TruncatedNormal(0, 0.1, low=center_low, high=center_high).expand([2])  # shape = (2,)
        )
    else:
        center = numpyro.deterministic(f"center_{param_name}", jnp.array([center_x, center_y]))
    return [{
        'theta_E': theta_E[0],
        'gamma': gamma[0],
        'e1': e_mass[0],
        'e2': e_mass[1],
        'center_x': center[0],
        'center_y': center[1],
    }, {
        'gamma1': gamma_sheer[0],
        'gamma2': gamma_sheer[1],
        'ra_0': center[0],
        'dec_0': center[1]
    }]


def params2kwargs_EPL_w_shear(params, param_name):
    return [{
        'theta_E': params[f'theta_E_{param_name}'][0],
        'gamma': params[f'gamma_{param_name}'][0],
        'e1': params[f'e_{param_name}'][0],
        'e2': params[f'e_{param_name}'][1],
        'center_x': params[f'center_{param_name}'][0],
        'center_y': params[f'center_{param_name}'][1],
    }, {
        'gamma1': params[f'gamma_sheer_{param_name}'][0],
        'gamma2': params[f'gamma_sheer_{param_name}'][1],
        'ra_0': params[f'center_{param_name}'][0],
        'dec_0': params[f'center_{param_name}'][1]
    }]
from numpyro.distributions import transforms as T
def GNFW_w_shear(plate_name, param_name, gamma_in_up = 2, gamma_in_low = 0.5, Rs_high = None, Rs_low = None, Rs_mean = None ,Rs_std = None, Rs_value = None, e_low=-0.2, e_high=0.2, center_x = None, center_y = None, kappa_s_low = 0.0, kappa_s_high = 1, sph = False, gamma_sheer_low = -0.2, gamma_sheer_high = 0.2):
    if Rs_value is not None:
        Rs = numpyro.deterministic(f'Rs_{param_name}', jnp.float64(Rs_value))
    elif Rs_low is not None:
        Rs = numpyro.sample(f'Rs_{param_name}', dist.Uniform(Rs_low,   Rs_high))
    elif Rs_mean is not None:
        Rs = numpyro.sample(f'Rs_{param_name}', dist.TruncatedNormal(Rs_mean, Rs_std, low=Rs_mean-1*Rs_std,  high=Rs_mean+1*Rs_std))

    kappa_s   = numpyro.sample(f'kappa_s_{param_name}',   dist.Uniform(kappa_s_low, kappa_s_high))
    gamma_in  = numpyro.sample(f'gammain_{param_name}',   dist.Uniform(gamma_in_low, gamma_in_up))

    with numpyro.plate(f'{plate_name} vectors - [2]', 2):
        gamma_sheer  = numpyro.sample(f'gamma_sheer_{param_name}', dist.Uniform(gamma_sheer_low, gamma_sheer_high))

    if sph is False:
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e_mass = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.25, low=e_low,  high=e_high))
    else:
        e_mass = numpyro.deterministic(f"e_{param_name}", jnp.array([0.0001, -0.0001]))
        
    if center_x is None:
        center = numpyro.sample(
            f'center_{param_name}',
            dist.TruncatedNormal(0, 1, low=-0.4, high=0.4).expand([2])  # shape = (2,)
        )
    else:
        center = numpyro.deterministic(f"center_{param_name}", jnp.array([center_x, center_y]))
    
    return [{
        'R_s':  Rs,          # ← 去掉 [0]
        'gamma':   gamma_in,    # ← 去掉 [0]
        'kappa_s': kappa_s,     # ← 去掉 [0]
        'e1': e_mass[0],
        'e2': e_mass[1],
        'center_x': center[0],
        'center_y': center[1],
    }, {
        'gamma1': gamma_sheer[0],
        'gamma2': gamma_sheer[1],
        'ra_0':   center[0],
        'dec_0':  center[1],
    }]
def params2kwargs_GNFW_w_shear(params, param_name):
    return [{
        'R_s': params[f'Rs_{param_name}'],
        'gamma': params[f'gammain_{param_name}'],
        'kappa_s': params[f'kappa_s_{param_name}'],
        'e1': params[f'e_{param_name}'][0],
        'e2': params[f'e_{param_name}'][1],
        'center_x': params[f'center_{param_name}'][0],
        'center_y': params[f'center_{param_name}'][1]
    }, {
        'gamma1': params[f'gamma_sheer_{param_name}'][0],
        'gamma2': params[f'gamma_sheer_{param_name}'][1],
        'ra_0': params[f'center_{param_name}'][0],
        'dec_0': params[f'center_{param_name}'][1]
    }]


######################################################################################################################################################
def multi_gauss_light(plate_name, param_name, n_gauss, sigma_lims, center_low=None, center_high=None, e_low=None, e_high=None):
    sigma_bins = jnp.logspace(
        jnp.log10(sigma_lims[0]),
        jnp.log10(sigma_lims[1]),
        n_gauss + 1
    )

    with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
        A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, 10000))
        sigma = numpyro.sample(
            f'sigma_{param_name}',
            dist.LogUniform(sigma_bins[:-1], sigma_bins[1:])
        )
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low=e_low, high=e_high))
            if (center_low is not None) or (center_high is not None):
                center = numpyro.sample(
                    f'center_{param_name}',
                    dist.TruncatedNormal(0.0, 0.1, low=center_low, high=center_high)
                )
            else:
                center = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.5))

    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)
    return [{
        'amp': amp,
        'sigma': sigma,
        'e1': e[0],
        'e2': e[1],
        'center_x': center[0],
        'center_y': center[1],
    }]

def multi_gauss_light_center(
    plate_name,
    param_name,
    n_gauss,
    sigma_lims,
    center_low=None,
    center_high=None,
    e_low=None,
    e_high=None,
    center_det=None,           # <<< 新增：可选的确定性 center
):
    sigma_bins = jnp.logspace(
        jnp.log10(sigma_lims[0]),
        jnp.log10(sigma_lims[1]),
        n_gauss + 1
    )

    # ——准备（可选）确定性的 center，统一为形状 (2, n_gauss)——
    if center_det is not None:
        c = jnp.asarray(center_det)
        if c.ndim == 1 and c.shape == (2,):
            # 所有高斯共享同一个中心
            c = jnp.broadcast_to(c, (n_gauss, 2)).T       # -> (2, n_gauss)
        elif c.ndim == 2 and c.shape == (n_gauss, 2):
            c = c.T                                       # -> (2, n_gauss)
        elif c.ndim == 2 and c.shape == (2, n_gauss):
            pass                                          # 已是 (2, n_gauss)
        else:
            raise ValueError(
                f"`center_det` 的形状必须是 (2,), (n_gauss, 2) 或 (2, n_gauss)，"
                f"当前为 {c.shape}，n_gauss={n_gauss}"
            )
        # 记录到 trace（一次性记录整块，避免在 plate 内重复注册）
        center = numpyro.deterministic(f'center_{param_name}', c)

    with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
        A = numpyro.sample(f'A_{param_name}', dist.LogUniform(1e-5, 1e4))
        sigma = numpyro.sample(
            f'sigma_{param_name}',
            dist.LogUniform(sigma_bins[:-1], sigma_bins[1:])
        )

        # e 始终采样
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e = numpyro.sample(
                f'e_{param_name}',
                dist.TruncatedNormal(0.0, 0.1, low=e_low, high=e_high)
            )

            # 仅当未提供 center_det 才对 center 进行采样
            if center_det is None:
                if (center_low is not None) or (center_high is not None):
                    center = numpyro.sample(
                        f'center_{param_name}',
                        dist.TruncatedNormal(0.0, 0.1, low=center_low, high=center_high)
                    )
                else:
                    center = numpyro.sample(
                        f'center_{param_name}',
                        dist.Normal(0.0, 0.5)
                    )

    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)

    # 此处保持原有返回结构与索引方式：
    # e[0], e[1], center[0], center[1] 分别是 x/y 分量，长度为 n_gauss
    return [{
        'amp': amp,
        'sigma': sigma,
        'e1': e[0],
        'e2': e[1],
        'center_x': center[0],
        'center_y': center[1],
    }]




def multi_gauss_light_unsorted(plate_name, param_name, n_gauss, sigma_lims, center_low=None, center_high=None, e_low=None, e_high=None):
    with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
        A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, 10000))
        sigma_unsorted = numpyro.sample(f'sigma_unsorted_{param_name}', dist.LogUniform(sigma_lims[0],sigma_lims[-1]))
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low=e_low, high=e_high))
            if (center_low is not None) or (center_high is not None):
                center = numpyro.sample(
                    f'center_{param_name}',
                    dist.TruncatedNormal(0.0, 0.1, low=center_low, high=center_high)
                )
            else:
                center = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.5))
    sigma = numpyro.deterministic(f'sigma_{param_name}',jnp.sort(sigma_unsorted))
    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)
    return [{
        'amp': amp,
        'sigma': sigma,
        'e1': e[0],
        'e2': e[1],
        'center_x': center[0],
        'center_y': center[1],
    }]

def params2kwargs_multi_gauss_light(params, param_name):
    # sigma = jnp.logspace(jnp.log10(sigma_lims[0]), jnp.log10(sigma_lims[1]), n_gauss)
    return [{
        'amp': params[f'amp_{param_name}'],
        'sigma': params[f'sigma_{param_name}'],
        'e1': params[f'e_{param_name}'][0],
        'e2': params[f'e_{param_name}'][1],
        'center_x': params[f'center_{param_name}'][0],
        'center_y': params[f'center_{param_name}'][1]
    }]

def multi_gauss_light_share_center(plate_name, param_name, n_gauss, sigma_lims, center_low=None, center_high=None, e_low=None, e_high=None,share_q = False):
    # Order in log-spaced sigma bins
    sigma_bins = jnp.logspace(
        jnp.log10(sigma_lims[0]),
        jnp.log10(sigma_lims[1]),
        n_gauss + 1
    )
    with numpyro.plate(f'{plate_name} center vectors - [2]', 2):
        if (center_low is not None) or (center_high is not None):
            center_lens = numpyro.sample(
                f'center_{param_name}',
                dist.TruncatedNormal(0.0, 0.1, low=center_low, high=center_high)
            )
        else:
            center_lens = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.5))

    with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
        A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, 10000))
        sigma = numpyro.sample(
            f'sigma_{param_name}',
            dist.LogUniform(sigma_bins[:-1], sigma_bins[1:])
        )
        if not share_q:
            with numpyro.plate(f'{plate_name} vectors - [2]', 2):
                e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low=e_low, high=e_high))
    if share_q:
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e_shared = numpyro.sample(f'e_{param_name}',
                                      dist.TruncatedNormal(0., 0.1, low=e_low, high=e_high))
        e = jnp.repeat(e_shared[:, None], n_gauss, axis=1)   # (2, n_gauss)
              
    center = jnp.array([[center_lens[0]]* n_gauss,[center_lens[1]]* n_gauss])
    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)

    return [{
        'amp': amp,
        'sigma': sigma,
        'e1': e[0],
        'e2': e[1],
        'center_x': center[0],
        'center_y': center[1],
    }]



def params2kwargs_multi_gauss_light_share_center(params, param_name):
    # sigma = jnp.logspace(jnp.log10(sigma_lims[0]), jnp.log10(sigma_lims[1]), n_gauss)
    return [{
        'amp': params[f'amp_{param_name}'],
        'sigma': params[f'sigma_{param_name}'],
        'e1': params[f'e_{param_name}'][0],
        'e2': params[f'e_{param_name}'][1],
        'center_x': jnp.array([params[f'center_{param_name}'][0]] * len(params[f'amp_{param_name}'])),
        'center_y': jnp.array([params[f'center_{param_name}'][1]] * len(params[f'amp_{param_name}']))
    }]
######################################################################################################################################################
def shear(plate_name, param_name,theta_low = 0.0, theta_high = 3):
    with numpyro.plate(f'{plate_name} scalers - [1]', 1):
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            center = numpyro.sample(f'center_{param_name}', dist.TruncatedNormal(0, 0.1, low=-0.2, high=0.2))
            gamma_sheer = numpyro.sample(f'gamma_sheer_{param_name}', dist.Uniform(-0.2, 0.2))
    return [{
        'gamma1': gamma_sheer[0],
        'gamma2': gamma_sheer[1],
        'ra_0': center[0],
        'dec_0': center[1]
    }]

def params2kwargs_shear(params, param_name):
    return [{
        'gamma1': params[f'gamma_sheer_{param_name}'][0],
        'gamma2': params[f'gamma_sheer_{param_name}'][1],
        'ra_0': params[f'center_{param_name}'][0],
        'dec_0': params[f'center_{param_name}'][1]
    }]


def SIS(plate_name, param_name, origin,theta_low = 0.0, theta_high = 0.01):
    with numpyro.plate(f'{plate_name} scalers - [1]', 1):
        theta_E = numpyro.sample(f'theta_E_{param_name}', dist.Uniform(theta_low, theta_high))
        center_0 = numpyro.deterministic(f'center_1_{param_name}', jnp.array([origin[0]]))
        center_1 = numpyro.deterministic(f'center_2_{param_name}', jnp.array([origin[1]]))

    return [{
        'theta_E': theta_E[0],
        'center_x': center_0[0],
        'center_y': center_1[0],
    }]


def params2kwargs_SIS(params, param_name):
    return [{
        'theta_E': params[f'theta_E_{param_name}'][0],
        'center_x': params[f'center_1_{param_name}'][0],
        'center_y': params[f'center_2_{param_name}'][0],
    }]

def SIE(plate_name, param_name, origin,theta_low = 0.0, theta_high = 0.01):
    with numpyro.plate(f'{plate_name} scalers - [1]', 1):
        theta_E = numpyro.sample(f'theta_E_{param_name}', dist.Uniform(theta_low, theta_high))
        center_0 = numpyro.deterministic(f'center_1_{param_name}', jnp.array([origin[0]]))
        center_1 = numpyro.deterministic(f'center_2_{param_name}', jnp.array([origin[1]]))
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e_mass = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.25, low=-0.4, high=0.4))
    return [{
        'theta_E': theta_E[0],
        'center_x': center_0[0],
        'center_y': center_1[0],
        'e1': e_mass[0],
        'e2': e_mass[1],
    }]
def params2kwargs_SIE(params, param_name):
    return [{
        'theta_E': params[f'theta_E_{param_name}'][0],
        'center_x': params[f'center_1_{param_name}'][0],
        'center_y': params[f'center_2_{param_name}'][0],
        'e1': params[f'e_{param_name}'][0],
        'e2': params[f'e_{param_name}'][1],
    }]

#######################################################################################################################################
from herculens.Util import util, param_util
def get_MPPL(plate_name, param_name, EPL_params_lens, order, e_high = 0.5, e_low = -0.5):

    with numpyro.plate(f'{plate_name} scalars -[1]', 2):
        e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low = e_low, high = e_high))

    phi_m_times_mover2, one_minus_a_m = param_util.ellipticity2phi_q(e[0], e[1])

    a_m = numpyro.deterministic(f"a_m_{param_name}",1-one_minus_a_m)
    phi_m = numpyro.deterministic(f"phi_m_{param_name}",phi_m_times_mover2*2/order)

    return [{
        'm': order,
        'a_m': a_m,
        'gamma' : EPL_params_lens['gamma'], 
        'b' : EPL_params_lens['theta_E'], 
        'phi_m': phi_m, # ...converted to radians!
        'center_x': EPL_params_lens['center_x'],
        'center_y': EPL_params_lens['center_y']
    }]

def params2kwargs_MPPL(params, param_name, EPL_params_lens, order, center = jnp.zeros((2,))):
    return [{
        'm': order,
        'a_m': params[f'a_m_{param_name}'],
        'gamma' : EPL_params_lens['gamma'], 
        'b' : EPL_params_lens['theta_E'], 
        'phi_m': params[f'phi_m_{param_name}'],
        'center_x': EPL_params_lens['center_x'],
        'center_y': EPL_params_lens['center_y']
    }]

#######################################################################################################################################

from svi_utils import split_scheduler


@partial(jax.vmap, in_axes=(0, 0, None, None))   
def source_power_spectrum(image, rng_key_, n_value, positive):
    _, svi_ps_map = fit_power_spectrum(
        rng_key_, image,
        init_learning_rate=0.01,
        max_iterations=30000,
        noise=0.001 * image.max(),
        n_value= n_value, positive = positive,
        k_zero=None,
    )
    return svi_ps_map




def power_spectrum_model(image, noise, k_values, n_value = None, positive = True, k_zero=None):
    ny, nx = image.shape
    source = matern_power_spectrum(
        'Source grid',
        'source_grid',
        k_values,
        k_zero=k_zero,
        n_value = n_value,
        positive = positive
    )
    with numpyro.plate(f'data 1 - [{nx}]', nx):
        with numpyro.plate(f'data 2 - [{ny}]', ny):
            numpyro.sample('obs', dist.Normal(source['pixels'], noise), obs=image)


def fit_power_spectrum(
    rng_key_,
    image,
    max_iterations=20000,
    init_learning_rate=0.1,
    transition_steps=[50, 10],
    progress_bar=False,
    noise=0.5,
    n_value = None,
    positive = True,
    k_zero=None,
):
    init_fun = infer.init_to_median()
    image_guide = autoguide.AutoDiagonalNormal(
        power_spectrum_model,
        init_loc_fn=init_fun
    )

    k_image = K_grid(image.shape)

    scheduler = split_scheduler(
        max_iterations,
        init_value=init_learning_rate,
        transition_steps=transition_steps
    )
    optim = optax.adabelief(learning_rate=scheduler)
    loss = infer.TraceMeanField_ELBO()

    svi_image = infer.SVI(
        power_spectrum_model,
        image_guide,
        optim,
        loss
    )

    svi_image_result = svi_image.run(
        rng_key_,
        max_iterations,
        image,
        noise,
        k_image.k,
        n_value = n_value,
        positive = positive,
        k_zero=k_zero,
        progress_bar=progress_bar,
        stable_update=True
    )

    source_map = image_guide.median(svi_image_result.params)
    return svi_image_result, source_map


def matern_power_spectrum(
    plate_name,
    param_name,
    k,
    k_zero = None,
    n_high = 100,
    n_value = None,
    positive = True
):
    with numpyro.plate(f'{plate_name} power spectrum params - [1]', 1):
        if n_value == None:
            n = numpyro.sample(f'n_{param_name}', TruncatedWedge(-1, 0.0001, n_high))
        else:
            n = numpyro.deterministic(f'n_{param_name}', jnp.atleast_1d(n_value))
        sigma = numpyro.sample(f'sigma_{param_name}', dist.LogUniform(1e-5, 10))
        rho = numpyro.sample(f'rho_{param_name}', dist.LogNormal(2.1, 1.1))

    P = P_Matern(k, n[0], sigma[0], rho[0], k_zero=k_zero)
    scale = jnp.sqrt(P)

    ny, nx = scale.shape
    with numpyro.plate(f'{plate_name} fft y - [{ny}]', ny):
        with numpyro.plate(f'{plate_name} fft x - [{nx}]', nx):
            pixels_wn = numpyro.sample(
                f'pixels_wn_{param_name}',
                #dist.StudentT(df=3)
                dist.Normal(0, 1)
            )

    gp = jnp.fft.irfft2(pack_fft_values(pixels_wn * scale),s=scale.shape,norm='ortho')
    if positive:
        gp = jax.nn.softplus(100*gp)/100.0
    pixels = numpyro.deterministic(f'pixels_{param_name}', gp) # Softplus ensures positivity
    
    return {'pixels': pixels}


def params2kwargs_power_spectrum(params, param_name):
    return {
        'pixels': params[f'pixels_{param_name}']
    }
###############################################################################################################################

def gauss_light(plate_name, param_name, n_gauss, sigma_lims, center_low=None, e_low=-0.4, e_high=0.4, center_high=None, Alim = None):
    # Order in log-spaced sigma bins
    sigma_bins = jnp.logspace(
        jnp.log10(sigma_lims[0]),
        jnp.log10(sigma_lims[1]),
        n_gauss + 1
    )
    with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
        if Alim is not None:
            A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, Alim))
        else:
            A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, 100000))
        sigma = numpyro.sample(
            f'sigma_{param_name}',
            dist.LogUniform(sigma_bins[:-1], sigma_bins[1:])
        )
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low=e_low, high=e_high))
            if (center_low is not None) or (center_high is not None):
                center = numpyro.sample(
                    f'center_{param_name}',
                    dist.TruncatedNormal(0.0, 0.1, low=center_low, high=center_high)
                )
            else:
                center = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.5))
    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)
    return [{
        'amp': amp[i],
        'sigma': sigma[i],
        'e1': e[0][i],
        'e2': e[1][i],
        'center_x': center[0][i],
        'center_y': center[1][i],
    } for i in range(n_gauss)]

def gauss_light_share_center(plate_name, param_name, n_gauss, sigma_lims, center_low=None, e_low=-0.4, e_high=0.4, center_high=None, Alim = None):
    # Order in log-spaced sigma bins
    sigma_bins = jnp.logspace(
        jnp.log10(sigma_lims[0]),
        jnp.log10(sigma_lims[1]),
        n_gauss + 1
    )
    with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
        if Alim is not None:
            A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, Alim))
        else:
            A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, 100000))
        sigma = numpyro.sample(
            f'sigma_{param_name}',
            dist.LogUniform(sigma_bins[:-1], sigma_bins[1:])
        )
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low=e_low, high=e_high))

    with numpyro.plate(f'{plate_name} vectors——center - [2]', 2):
        if (center_low is not None) or (center_high is not None):
            center = numpyro.sample(
                f'center_{param_name}',
                dist.TruncatedNormal(0.0, 0.1, low=center_low, high=center_high)
            )
        else:
            center = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.5))

    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)
    return [{
        'amp': amp[i],
        'sigma': sigma[i],
        'e1': e[0][i],
        'e2': e[1][i],
        'center_x': center[0],
        'center_y': center[1],
    } for i in range(n_gauss)]


def params2kwargs_gauss_light(params, param_name, n_gauss, sigma_lims):
    sigma = jnp.logspace(jnp.log10(sigma_lims[0]), jnp.log10(sigma_lims[1]), n_gauss)
    return [{
        'amp': params[f'amp_{param_name}'][i],
        #'sigma': sigma[i],
        'sigma': params[f'sigma_{param_name}'][i],
        'e1': params[f'e_{param_name}'][0][i],
        'e2': params[f'e_{param_name}'][1][i],
        'center_x': params[f'center_{param_name}'][0][i],
        'center_y': params[f'center_{param_name}'][1][i]
    } for i in range(n_gauss)]

def gauss_light_source(plate_name, param_name, n_gauss, sigma_lims, center_low=None, center_high=None, Alim = None):
    # Order in log-spaced sigma bins
    sigma_bins = jnp.logspace(
        jnp.log10(sigma_lims[0]),
        jnp.log10(sigma_lims[1]),
        n_gauss + 1
    )
    with numpyro.plate(f'{plate_name} - [{n_gauss}]', n_gauss):
        if Alim is not None:
            A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, Alim))
        else:
            A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, 10000))
        sigma = numpyro.sample(
            f'sigma_{param_name}',
            dist.LogUniform(sigma_bins[:-1], sigma_bins[1:])
        )
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low=-0.2, high=0.2))
            if (center_low is not None) or (center_high is not None):
                center = numpyro.sample(
                    f'center_{param_name}',
                    dist.TruncatedNormal(0.0, 0.1, low=center_low, high=center_high)
                )
            else:
                center = numpyro.sample(f'center_{param_name}', dist.Normal(0.0, 0.5))
    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)
    return [{
        'amp': amp[i],
        'sigma': sigma[i],
        'e1': e[0][i],
        'e2': e[1][i],
        'center_x': center[0][i],
        'center_y': center[1][i],
    } for i in range(n_gauss)]


def gauss_light_no_center(plate_name, param_name, center = [0,0], sigma_high = 100):
    with numpyro.plate(f'{plate_name} - [1]', 1):
        A = numpyro.sample(f'A_{param_name}', dist.LogUniform(0.00001, 10000))
        sigma = numpyro.sample(f'sigma_{param_name}', dist.LogUniform(0.001, sigma_high))
        with numpyro.plate(f'{plate_name} vectors - [2]', 2):
            e = numpyro.sample(f'e_{param_name}', dist.TruncatedNormal(0, 0.1, low=-0.3, high=0.3))
    amp = numpyro.deterministic(f'amp_{param_name}', A * sigma**2)
    return [{
        'amp': amp[0],
        'sigma': sigma[0],
        'e1': e[0][0],
        'e2': e[1][0],
        'center_x': center[0],
        'center_y': center[1]
    }]


def params2kwargs_gauss_light_no_center(params, param_name, center):
    return [{
        'amp': params[f'amp_{param_name}'][0],
        'sigma': params[f'sigma_{param_name}'][0],
        'e1': params[f'e_{param_name}'][0][0],
        'e2': params[f'e_{param_name}'][1][0],
        'center_x': center[0],
        'center_y': center[1]
    }]

#Helper funcitons:
##########################################################################################################
##########################################################################################################

def MTF_on_normalised_data(x, m):
    x = np.clip(x, 0, 1)
    # 先计算通用公式
    y = ((m - 1) * x) / (((2 * m - 1) * x - m))
    # 对特殊点（x==0, x==m, x==1）分别赋值
    return np.where(x == 0, 0, 
                    np.where(x == m, 0.5, 
                             np.where(x == 1, 1, y)))
def find_m_for_mean(normalized_data, desired_mean):  # 0.1 arcsec per pixel for MER tiles
    x = np.mean(normalized_data)
    alpha = desired_mean
    return (x-alpha*x)/(x-2*alpha*x+alpha)
    
def apply_MTF(image_data, desired_mean_normalized=0.2):  
    image_data -= np.min(image_data)
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_data = image_data.astype(np.float64)
    min_val = np.min(normalized_data)
    max_val = np.max(normalized_data)
    normalized_data = (normalized_data - min_val) / (max_val - min_val)
    m = find_m_for_mean(normalized_data, desired_mean_normalized)
    transformed_data = MTF_on_normalised_data(normalized_data, m)
    return transformed_data

def auto_mtf(image, mean = 0.125, clip_low = 0.000, clip_high = 0.99):
    image = np.clip(image, clip_low*np.max(image), clip_high*np.max(image)) 
    x = np.mean(image)
    m = (x-mean*x)/(x-2*mean*x+mean)
    return MTF_on_normalised_data(image, m)

def crop_to_square(data, mask):
    h, w = data.shape[:2]
    size = min(h, w)
    start_h = int(round((h - size) / 2))
    start_w = int(round((w - size) / 2))
    return data[start_h:start_h+size, start_w:start_w+size], mask[start_h:start_h+size, start_w:start_w+size]

def reduce_bg(data, corner_pixel = 5):
    background_pixels = np.vstack([data[:corner_pixel, :corner_pixel],data[:corner_pixel, -1*corner_pixel:],data[-1*corner_pixel:, :corner_pixel],data[-1*corner_pixel:, -1*corner_pixel:]])
    rms = background_pixels.std()
    return np.mean(background_pixels), rms, background_pixels

def get_pixel_grid(data, pix_scale, ss = 1):
    ny, nx = data.shape
    ny *= ss
    nx *= ss
    pix_scale /= ss
    half_size_x = nx * pix_scale / 2
    half_size_y = ny * pix_scale / 2
    ra_at_xy_0 = -half_size_x + pix_scale / 2
    dec_at_xy_0 = -half_size_y + pix_scale / 2
    transform_pix2angle = pix_scale * np.eye(2)
    kwargs_pixel = {'nx': nx, 'ny': ny, 'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0, 'transform_pix2angle': transform_pix2angle}
    pixel_grid = PixelGrid(**kwargs_pixel) 
    xgrid, ygrid = pixel_grid.pixel_coordinates
    x_axis = xgrid[0]
    y_axis = ygrid[:, 0]
    extent = pixel_grid.extent
    return pixel_grid, xgrid, ygrid, x_axis, y_axis, extent,nx,ny

@partial(jax.vmap, in_axes=(None, 0, 0))
def reduced_distance(a, i, j):
    u = jax.lax.dynamic_slice_in_dim(a, i, 1)
    v = jax.lax.dynamic_slice_in_dim(a, j, 1)
    return jnp.linalg.norm(u-v, ord=2)


def reduced_distance_matrix(a):
    i, j = jnp.triu_indices(a.shape[0], k=1)
    return reduced_distance(a, i, j)


@jax.jit
def get_value_from_index(xs, i):
    i = jnp.asarray(i)
    return jax.tree.map(lambda x: x[i], xs)


@partial(jax.vmap, in_axes=(None, None, 0))
def median_params2kwargs(params2kwargs_fn, median_stack, i):
    params_i = get_value_from_index(median_stack, i)
    return params2kwargs_fn(params_i)


def init_to_value_or_defer(site=None, values={}, defer=numpyro.infer.init_to_median):
    if site is None:
        return partial(init_to_value_or_defer, values=values)

    if site["type"] == "sample" and not site["is_observed"]:
        if site["name"] in values:
            return values[site["name"]]
        else:  # defer to default strategy
            return defer(site)
        

class TruncatedWedge(dist.Distribution):
    def __init__(self, a, low, b):
        self.a = a
        self.b = b
        self.low = low
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(a),
            jnp.shape(low),
            jnp.shape(b)
        )
        self._support = dist.constraints.interval(low, b)
        self.norm = (self.b - self.a)**2 - (self.low - self.a)**2
        super().__init__(batch_shape=batch_shape, event_shape=())

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def log_prob(self, value):
        return jnp.log(2) + jnp.log(value - self.a) - jnp.log(self.norm)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        u = jax.random.uniform(key, shape=shape, minval=0, maxval=1)
        return self.a + jnp.sqrt(self.norm * u + (self.low - self.a)**2)

from sklearn.neighbors import NearestNeighbors


def get_best_pixel_size(lens_image, herc_dict, source_grid_scale, return_full=False):
    x_ss_grid, y_ss_grid = lens_image.ImageNumerics.coordinates_evaluate
    mask = lens_image._source_arc_mask_flat.astype(bool)
    x_ss_trace, y_ss_trace = lens_image.MassModel.ray_shooting(
        x_ss_grid[mask].flatten(),
        y_ss_grid[mask].flatten(),
        herc_dict['kwargs_lens']
    )

    _, _, extent = lens_image.mask_extent(
        x_ss_trace,
        y_ss_trace,
        100,
        grid_scale=source_grid_scale
    )

    full_size = jax.device_get(extent[1] - extent[0])
    tdx = (
        (x_ss_trace >= extent[0]) &
        (x_ss_trace <= extent[1]) &
        (y_ss_trace >= extent[2]) &
        (y_ss_trace <= extent[3])
    )

    jax.block_until_ready(x_ss_trace)
    jax.block_until_ready(y_ss_trace)
    x_trim = jax.device_get(x_ss_trace[tdx])
    y_trim = jax.device_get(y_ss_trace[tdx])
    X = np.vstack([x_trim, y_trim]).T
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, _ = nbrs.kneighbors(X)

    # this combination seems to give approx 2-3 samples per pix in grid
    # note that it uses the 5 times the mean distance
    # tuned on a 2D histogram until at least 2 samples were in every pixel
    # in the region of interest
    five_mean_distance = 5 * np.mean(distances[:, 1])
    n = (full_size / five_mean_distance) + 1
    if return_full:
        return int(n), x_trim, y_trim, np.mean(distances[:, 1]), full_size
    return int(n)


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import ListedColormap

def get_source(
    lens_image,
    kwargs_light,
    N,
    num_pix,
    kwargs_mass,
    eta,
    source_grid_scale=1
):
    """
    Helper function that returns the source-plane surface brightness for
    a particular source-plane index N.
    """
    # Determine the correct source-plane extent
    _, _, extents = lens_image.get_source_coordinates(
        eta,
        kwargs_mass,
        force=True,
        npix_src=num_pix,
        source_grid_scale=source_grid_scale
    )
    extent = extents[N]

    x = np.linspace(extent[0], extent[1], num_pix)
    y = np.linspace(extent[2], extent[3], num_pix)
    xgrid, ygrid = np.meshgrid(x, y)
    xgrid = xgrid.flatten()
    ygrid = ygrid.flatten()

    # Compute surface brightness
    image_grid = lens_image.MPLightModel.light_models[N].surface_brightness(
        xgrid,
        ygrid,
        kwargs_light[N],
        pixels_x_coord=xgrid,
        pixels_y_coord=ygrid
    ) * lens_image.Grid.pixel_area
    
    return image_grid.reshape(num_pix, num_pix), extent


import optax
import numpyro
import jax_tqdm
import jax
import matplotlib.pyplot as plt
import numpy as np

from functools import partial


def split_scheduler(
    max_iterations,
    init_value=0.1,
    decay_rates=[0.99, 0.99],
    transition_steps=[50, 10],
    boundary=0.5
):
    boundary = int(max_iterations * boundary)

    scheduler1 = optax.exponential_decay(
        init_value=init_value,
        decay_rate=decay_rates[0],
        transition_steps=transition_steps[0]
    )

    scheduler2 = optax.exponential_decay(
        init_value=scheduler1(boundary),
        decay_rate=decay_rates[1],
        transition_steps=transition_steps[1]
    )

    return optax.join_schedules(
        [scheduler1, scheduler2],
        boundaries=[boundary]
    )


def plot_loss(losses, max_iterations, ax=None, axins=None, inset=True, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 3.5))
    ax.plot(losses, **kwargs)
    ax.set_yscale('asinh')

    if (inset) and (axins is None):
        axins = ax.inset_axes([0.3, 0.5, 0.64, 0.45])
    N_end = max_iterations // 3
    x_plot = np.linspace(max_iterations - N_end, max_iterations, N_end)
    if inset:
        axins.plot(x_plot, losses[max_iterations - N_end:], **kwargs)
        ax.indicate_inset_zoom(axins, edgecolor='k')
    return ax


class SVI_vec(numpyro.infer.SVI):
    def run(
        self,
        rng_key,
        num_chains,
        num_steps,
        *args,
        stable_update=False,
        forward_mode_differentiation=False,
        init_states=None,
        init_params=None,
        **kwargs
    ):
        @jax_tqdm.scan_tqdm(num_steps)
        def body_fn(svi_state, _):
            if stable_update:
                svi_state, loss = self.stable_update(
                    svi_state,
                    *args,
                    forward_mode_differentiation=forward_mode_differentiation,
                    **kwargs,
                )
            else:
                svi_state, loss = self.update(
                    svi_state,
                    *args,
                    forward_mode_differentiation=forward_mode_differentiation,
                    **kwargs,
                )
            return svi_state, loss

        @jax.vmap
        def map_func(i, init_value):
            init_bar = jax_tqdm.PBar(id=i, carry=init_value)
            final_state, losses = jax.lax.scan(body_fn, init_bar, jax.numpy.arange(num_steps))
            return final_state.carry, losses

        @partial(jax.vmap, in_axes=(0, None, None, None))
        def vmap_init(rng_key, args, init_params, kwargs):
            return self.init(rng_key, *args, init_params=init_params, **kwargs)

        rng_keys = jax.random.split(rng_key, num_chains)
        if init_states is None:
            svi_states = vmap_init(rng_keys, args, init_params, kwargs)
        else:
            svi_states = init_states

        svi_states, losses = map_func(jax.numpy.arange(num_chains), svi_states)
        return numpyro.infer.svi.SVIRunResult(self.get_params(svi_states), svi_states, losses)
##############################################################

def pixelize_plane(
    lens_image,
    herc_dict,
    num_pix,
    N,
    source_grid_scale=1.0
):
    if source_grid_scale is None:
        source_grid_scale = lens_image._source_grid_scale
    extent = lens_image.Grid.extent
    if N > 0:
        _, _, extents = lens_image.get_source_coordinates(
            herc_dict['eta_flat'],
            herc_dict['kwargs_mass'],
            force=True,
            npix_src=num_pix,
            source_grid_scale=source_grid_scale
        )
        extent = extents[N]
    x = jnp.linspace(extent[0], extent[1], num_pix)
    y = jnp.linspace(extent[2], extent[3], num_pix)
    xgrid, ygrid = jnp.meshgrid(x, y)
    xgrid = xgrid.flatten()
    ygrid = ygrid.flatten()

    image_grid = lens_image.MPLightModel.light_models[N].surface_brightness(
        xgrid,
        ygrid,
        herc_dict['kwargs_light'][N],
        pixels_x_coord=xgrid,
        pixels_y_coord=ygrid
    ) * lens_image.Grid.pixel_area
    return image_grid.reshape(num_pix, num_pix), extent
