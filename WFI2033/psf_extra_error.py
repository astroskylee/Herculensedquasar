"""PSF-shape extra error utilities for WFI2033."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import map_coordinates


def _shift_template_to_image(
    template: np.ndarray,
    x_center_pix: float,
    y_center_pix: float,
    output_shape: Tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Shift a centered template to a target pixel center on output grid."""

    template = np.asarray(template, dtype=np.float64)
    out_y, out_x = np.indices(output_shape, dtype=np.float64)

    tmpl_cx = 0.5 * (template.shape[1] - 1.0)
    tmpl_cy = 0.5 * (template.shape[0] - 1.0)
    dx = float(x_center_pix) - tmpl_cx
    dy = float(y_center_pix) - tmpl_cy

    return map_coordinates(
        template,
        [out_y - dy, out_x - dx],
        order=order,
        mode="constant",
        cval=0.0,
    )


def extract_stage2_arcsec_and_flux(stage2_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract AGN arcsec coordinates and linear flux from stage2 params."""

    ra_ps = np.asarray(stage2_params["ra_ps"], dtype=np.float64)
    dec_ps = np.asarray(stage2_params["dec_ps"], dtype=np.float64)
    log10_amp_ps = np.asarray(stage2_params["log10_amp_ps"], dtype=np.float64)

    agn_positions_arcsec = np.stack([ra_ps, dec_ps], axis=1)
    agn_flux = np.power(10.0, log10_amp_ps)
    return agn_positions_arcsec, agn_flux


def _build_error_map_with_psf_extra_pixel(
    error_map: np.ndarray,
    agn_positions_arcsec: np.ndarray,
    agn_flux: np.ndarray,
    image_numerics,
    alpha_map: np.ndarray,
    psf_kernel: np.ndarray | None = None,
    interpolation_order: int = 1,
) -> Dict[str, np.ndarray]:
    """Core formula in pixel space:

    sigma_total^2 = sigma_det^2 + sum_i (alpha_i * psf_model_i)^2

    where psf_model_i is rendered by Herculens ImageNumerics for AGN-i.
    """

    error_orig = np.asarray(error_map, dtype=np.float64)
    positions_arcsec = np.asarray(agn_positions_arcsec, dtype=np.float64)
    x_pix, y_pix = image_numerics._pixel_grid.map_coord2pix(
        positions_arcsec[:, 0],
        positions_arcsec[:, 1],
    )
    positions = np.stack(
        [np.asarray(x_pix, dtype=np.float64), np.asarray(y_pix, dtype=np.float64)],
        axis=1,
    )
    flux = np.asarray(agn_flux, dtype=np.float64).reshape(-1)
    alpha_template = np.asarray(alpha_map, dtype=np.float64)

    var_extra = np.zeros_like(error_orig, dtype=np.float64)
    psf_models = []
    for (ra_arcsec, dec_arcsec), (x_pix, y_pix), amp in zip(positions_arcsec, positions, flux):
        # Render a single AGN PSF model through Herculens (same pipeline as modeling code).
        psf_model_i = image_numerics.render_point_sources(
            np.asarray([ra_arcsec], dtype=np.float64),
            np.asarray([dec_arcsec], dtype=np.float64),
            np.asarray([amp], dtype=np.float64),
            psf_kernel=psf_kernel,
        )
        psf_model_i = np.asarray(psf_model_i, dtype=np.float64)
        psf_models.append(psf_model_i)

        # Shift alpha template to this AGN center on detector grid.
        alpha_shifted = _shift_template_to_image(
            template=alpha_template,
            x_center_pix=float(x_pix),
            y_center_pix=float(y_pix),
            output_shape=error_orig.shape,
            order=interpolation_order,
        )

        # Extra sigma follows the Step3 definition: sigma_extra = alpha * rendered_model.
        sigma_extra_i = alpha_shifted * psf_model_i
        var_extra += sigma_extra_i * sigma_extra_i

    error_extra = np.sqrt(var_extra)
    error_total = np.sqrt(error_orig * error_orig + var_extra)
    return {
        "error_total": error_total,
        "error_orig": error_orig,
        "error_extra": error_extra,
        "var_extra": var_extra,
        "agn_positions_pix": positions,
        "psf_models": np.stack(psf_models, axis=0),
    }


def build_error_map_with_psf_extra(
    error_map: np.ndarray,
    agn_positions_arcsec: np.ndarray,
    agn_flux: np.ndarray,
    image_numerics,
    alpha_map: np.ndarray,
    psf_kernel: np.ndarray | None = None,
    interpolation_order: int = 1,
) -> Dict[str, np.ndarray]:
    """Public interface: accept arcsec positions + Herculens image_numerics."""

    return _build_error_map_with_psf_extra_pixel(
        error_map=error_map,
        agn_positions_arcsec=agn_positions_arcsec,
        agn_flux=agn_flux,
        image_numerics=image_numerics,
        alpha_map=alpha_map,
        psf_kernel=psf_kernel,
        interpolation_order=interpolation_order,
    )
