import jax
import jax.numpy as jnp
import numpy as np
import scipy

from functools import partial

from herculens.LensImage.lens_image import LensImage


class LensImageExtension(LensImage):
    """Thin extension of the local herculens LensImage.

    This keeps the project-specific adaptive source-grid logic and conjugate-point
    utilities while delegating the main image rendering path to the base class.
    """

    def __init__(
        self,
        grid_class,
        psf_class,
        noise_class=None,
        lens_mass_model_class=None,
        source_model_class=None,
        lens_light_model_class=None,
        point_source_model_class=None,
        source_arc_mask=None,
        source_grid_scale=1.0,
        conjugate_points=None,
        kwargs_numerics=None,
        kwargs_lens_equation_solver=None,
    ):
        super().__init__(
            grid_class,
            psf_class,
            noise_class=noise_class,
            lens_mass_model_class=lens_mass_model_class,
            source_model_class=source_model_class,
            lens_light_model_class=lens_light_model_class,
            point_source_model_class=point_source_model_class,
            source_arc_mask=source_arc_mask,
            kwargs_numerics=kwargs_numerics,
            kwargs_lens_equation_solver=kwargs_lens_equation_solver,
        )
        self._source_grid_scale = source_grid_scale
        self.conjugate_points = conjugate_points

        self.source_arc_mask_ss = None
        self._source_arc_mask_flat = None
        self._source_arc_mask_outline_flat = None
        if self.source_arc_mask is not None:
            ssf = self.ImageNumerics.grid_supersampling_factor
            s_ones = np.ones((ssf, ssf), dtype=bool)
            self.source_arc_mask_ss = np.kron(self.source_arc_mask.astype(bool), s_ones)
            self._source_arc_mask_flat = self.source_arc_mask_ss.flatten().astype(bool)
            self._source_arc_mask_outline_flat = (
                self.source_arc_mask_ss.astype(bool)
                & ~scipy.ndimage.binary_erosion(self.source_arc_mask_ss.astype(bool))
            ).flatten()

    def mask_extent(self, x_grid_src, y_grid_src, npix_src, grid_scale=1.0):
        x_left, x_right = x_grid_src.min(), x_grid_src.max()
        y_bottom, y_top = y_grid_src.min(), y_grid_src.max()
        cx = 0.5 * (x_left + x_right)
        cy = 0.5 * (y_bottom + y_top)
        width = jnp.abs(x_left - x_right)
        height = jnp.abs(y_bottom - y_top)
        half_size = 0.5 * grid_scale * jnp.maximum(height, width)
        x_left = cx - half_size
        x_right = cx + half_size
        y_bottom = cy - half_size
        y_top = cy + half_size
        x_adapt = jnp.linspace(x_left, x_right, npix_src)
        y_adapt = jnp.linspace(y_bottom, y_top, npix_src)
        extent_adapt = [x_adapt[0], x_adapt[-1], y_adapt[0], y_adapt[-1]]
        return x_adapt, y_adapt, extent_adapt

    def _mask_traced_coordinates(self, x_grid_src, y_grid_src):
        if self._source_arc_mask_outline_flat is not None:
            return (
                x_grid_src[self._source_arc_mask_outline_flat],
                y_grid_src[self._source_arc_mask_outline_flat],
            )
        if self._source_arc_mask_flat is not None:
            return (
                x_grid_src[self._source_arc_mask_flat],
                y_grid_src[self._source_arc_mask_flat],
            )
        return x_grid_src, y_grid_src

    def _adapt_source_coordinates_from_rays(
        self,
        x_grid_src,
        y_grid_src,
        npix_src,
        grid_scale,
        return_plt_extent=False,
    ):
        x_masked, y_masked = self._mask_traced_coordinates(x_grid_src, y_grid_src)
        x_adapt, y_adapt, extent = self.mask_extent(
            x_masked,
            y_masked,
            npix_src,
            grid_scale=grid_scale,
        )
        if return_plt_extent:
            pix_scl_x = jnp.abs(x_adapt[1] - x_adapt[0])
            pix_scl_y = jnp.abs(y_adapt[1] - y_adapt[0])
            half_pix_scl = jnp.sqrt(pix_scl_x * pix_scl_y) / 2.0
            extent = [
                x_adapt[0] - half_pix_scl,
                x_adapt[-1] + half_pix_scl,
                y_adapt[0] - half_pix_scl,
                y_adapt[-1] + half_pix_scl,
            ]
        return x_adapt, y_adapt, extent

    def eval_source_surface_brightness(
        self,
        x,
        y,
        kwargs_source,
        kwargs_lens=None,
        k=None,
        k_lens=None,
        de_lensed=False,
        adapted_pixels_coords=None,
        return_pixels_coords=False,
        return_as_list=False,
    ):
        x_grid_src = y_grid_src = None
        if self._src_adaptive_grid:
            if adapted_pixels_coords is None:
                npix_src, npix_src_y = self.SourceModel.pixel_grid.num_pixel_axes
                if npix_src_y != npix_src:
                    raise ValueError("Adaptive source plane grid only works with square grids")
                if self.Grid.x_is_inverted or self.Grid.y_is_inverted:
                    raise NotImplementedError(
                        "invert x and y not yet supported for adaptive source grid"
                    )
                x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                    x,
                    y,
                    kwargs_lens,
                    k=k_lens,
                )
                pixels_x_coord, pixels_y_coord, _ = self._adapt_source_coordinates_from_rays(
                    x_grid_src,
                    y_grid_src,
                    npix_src=npix_src,
                    grid_scale=self._source_grid_scale,
                )
            else:
                pixels_x_coord, pixels_y_coord = adapted_pixels_coords
        else:
            pixels_x_coord, pixels_y_coord = None, None

        if de_lensed:
            source_light = self.SourceModel.surface_brightness(
                x,
                y,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord,
                return_as_list=return_as_list,
            )
        else:
            if x_grid_src is None or y_grid_src is None:
                x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                    x,
                    y,
                    kwargs_lens,
                    k=k_lens,
                )
            source_light = self.SourceModel.surface_brightness(
                x_grid_src,
                y_grid_src,
                kwargs_source,
                k=k,
                pixels_x_coord=pixels_x_coord,
                pixels_y_coord=pixels_y_coord,
                return_as_list=return_as_list,
            )

        if return_pixels_coords:
            return source_light, (pixels_x_coord, pixels_y_coord)
        return source_light

    def adapt_source_coordinates(self, kwargs_lens, k_lens=None, return_plt_extent=False):
        npix_src, npix_src_y = self.SourceModel.pixel_grid.num_pixel_axes
        if npix_src_y != npix_src:
            raise ValueError("Adaptive source plane grid only works with square grids")
        if self.Grid.x_is_inverted or self.Grid.y_is_inverted:
            raise NotImplementedError("invert x and y not yet supported for adaptive source grid")
        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        x_grid_src, y_grid_src = self.MassModel.ray_shooting(
            x_grid_img,
            y_grid_img,
            kwargs_lens,
            k=k_lens,
        )
        return self._adapt_source_coordinates_from_rays(
            x_grid_src,
            y_grid_src,
            npix_src=npix_src,
            grid_scale=self._source_grid_scale,
            return_plt_extent=return_plt_extent,
        )

    def get_source_coordinates(
        self,
        kwargs_lens,
        k_lens=None,
        return_plt_extent=False,
        force=False,
        npix_src=100,
        source_grid_scale=1.0,
    ):
        if (not self._src_adaptive_grid) and (not force):
            x_grid, y_grid = self.SourceModel.pixel_grid.pixel_coordinates
            if return_plt_extent:
                extent = self.SourceModel.pixel_grid.plt_extent
            else:
                extent = self.SourceModel.pixel_grid.extent
            return x_grid, y_grid, extent

        x_grid_img, y_grid_img = self.ImageNumerics.coordinates_evaluate
        x_grid_src, y_grid_src = self.MassModel.ray_shooting(
            x_grid_img,
            y_grid_img,
            kwargs_lens,
            k=k_lens,
        )
        scale = source_grid_scale if force else self._source_grid_scale
        x_coord, y_coord, extent = self._adapt_source_coordinates_from_rays(
            x_grid_src,
            y_grid_src,
            npix_src=npix_src,
            grid_scale=scale,
            return_plt_extent=return_plt_extent,
        )
        x_grid, y_grid = np.meshgrid(x_coord, y_coord)
        return x_grid, y_grid, extent

    def trace_conjugate_points(self, kwargs_lens, k_lens=None):
        if self.conjugate_points is None:
            return None
        x, y = self.conjugate_points.T
        conj_x, conj_y = self.MassModel.ray_shooting(x, y, kwargs_lens, k=k_lens)
        return jnp.vstack([conj_x, conj_y]).T

    @partial(jax.jit, static_argnums=(0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15))
    def model(
        self,
        kwargs_lens=None,
        kwargs_source=None,
        kwargs_lens_light=None,
        kwargs_point_source=None,
        unconvolved=False,
        supersampled=False,
        source_add=True,
        lens_light_add=True,
        point_source_add=True,
        k_lens=None,
        k_source=None,
        k_lens_light=None,
        k_point_source=None,
        adapted_source_pixels_coords=None,
        return_source_pixels_coords=False,
        psf_kernel=None,
        psf_kernel_super=None,
    ):
        if psf_kernel_super is not None:
            raise NotImplementedError(
                "Explicit supersampled PSF override is not enabled in this LensImage path."
            )
        return super().model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point_source,
            unconvolved=unconvolved,
            supersampled=supersampled,
            source_add=source_add,
            lens_light_add=lens_light_add,
            point_source_add=point_source_add,
            k_lens=k_lens,
            k_source=k_source,
            k_lens_light=k_lens_light,
            k_point_source=k_point_source,
            adapted_source_pixels_coords=adapted_source_pixels_coords,
            return_source_pixels_coords=return_source_pixels_coords,
            psf_kernel=psf_kernel,
        )


def pixelize_plane(lens_image, herc_dict, num_pix, source_grid_scale=None):
    if source_grid_scale is None:
        source_grid_scale = lens_image._source_grid_scale
    x, y, extent = lens_image.get_source_coordinates(
        herc_dict["kwargs_lens"],
        force=True,
        npix_src=num_pix,
        source_grid_scale=source_grid_scale,
    )
    xgrid, ygrid = jnp.meshgrid(x, y)
    image_grid = lens_image.SourceModel.surface_brightness(
        xgrid,
        ygrid,
        herc_dict["kwargs_source"],
        pixels_x_coord=xgrid[0],
        pixels_y_coord=ygrid[:, 0],
    ) * lens_image.Grid.pixel_area
    return image_grid, extent
