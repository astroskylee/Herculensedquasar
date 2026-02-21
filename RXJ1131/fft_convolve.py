import numpy as np
import jax.numpy as jnp
import jax

from herculens.LensImage.Numerics.convolution import (
    PixelKernelConvolution,
    SubgridKernelConvolution
)
from herculens.LensImage.Numerics.numerics import Numerics
from herculens.Util import kernel_util
from scipy.fft import next_fast_len as osp_fft_next_fast_len


class PixelKernelConvolutionFFT(PixelKernelConvolution):
    '''Use fft convolution method'''
    def __init__(self, kernel, output_shape):
        """

        :param kernel: 2d array, psf kernel
        :param output_shape: pixel number of x and y axis of the image
        """
        self.image_shape = output_shape
        self.kernel = kernel
        full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(self.image_shape, self.kernel.shape))
        self.fft_shape = tuple(
            osp_fft_next_fast_len(s) for s in full_shape
        )
        self.sp2 = jnp.fft.rfftn(self.kernel, self.fft_shape, norm='ortho')
        self.start_indices = tuple(
            (full_size - out_size) // 2
            for full_size, out_size in zip(full_shape, self.image_shape)
        )
        self.norm = jnp.sqrt(self.fft_shape[0] * self.fft_shape[1])

    def convolution2d(self, image, psf_noise_fft=None):
        '''
        :param image: 2d array, input image

        '''
        if image.shape != self.image_shape:
            raise ValueError("incompatible image shape as initialized image")

        sp1 = jnp.fft.rfftn(image, self.fft_shape, norm='ortho')

        sp_conv = sp1 * self.sp2

        if psf_noise_fft is not None:
            sp_conv += sp_conv * psf_noise_fft

        conv = jnp.fft.irfftn(sp_conv, self.fft_shape, norm='ortho')

        return self.norm * jax.lax.dynamic_slice(
            conv,
            self.start_indices,
            image.shape
        )

    def re_size_convolve(self, image_low_res, psf_noise_fft=None, image_high_res=None):
        """

        :param image_high_res: supersampled image/model to be convolved on a regular pixel grid
        :param psf_noise_fft: noize of PSF in FFT space
        :return: convolved and re-sized image
        """
        return self.convolution2d(image_low_res, psf_noise_fft)


class SubgridKernelConvolutionFFT(SubgridKernelConvolution):
    def __init__(
        self,
        kernel_supersampled,
        supersampling_factor,
        supersampling_kernel_size=None
    ):
        """
        :param kernel_supersampled: kernel in supersampled pixels
        :param supersampling_factor: supersampling factor relative to the image pixel grid
        :param supersampling_kernel_size: number of pixels (in units of the image pixels)
            that are convolved with the supersampled kernel
        """
        self._supersampling_factor = supersampling_factor
        if supersampling_kernel_size is None:
            kernel_low_res, kernel_high_res = np.zeros((3, 3)), kernel_supersampled
            self._low_res_convolution = False
        else:
            kernel_low_res, kernel_high_res = kernel_util.split_kernel(
                kernel_supersampled,
                supersampling_kernel_size,
                self._supersampling_factor
            )
            self._low_res_convolution = True
        # Use FFT method for convolution
        self._low_res_conv = PixelKernelConvolutionFFT(kernel_low_res)
        self._high_res_conv = PixelKernelConvolutionFFT(kernel_high_res)


class NumericsFFT(Numerics):
    def __init__(
        self,
        pixel_grid,
        psf,
        supersampling_factor=1,
        convolution_type='jax_scipy',
        supersampling_convolution=False,
        iterative_kernel_supersampling=True,
        supersampling_kernel_size=5,
        point_source_supersampling_factor=1,
        convolution_kernel_size=None,
        truncation=4
    ):
        super().__init__(
            pixel_grid,
            psf,
            supersampling_factor,
            convolution_type,
            supersampling_convolution,
            iterative_kernel_supersampling,
            supersampling_kernel_size,
            point_source_supersampling_factor,
            convolution_kernel_size,
            truncation
        )
        # override self._conv for PIXEL PSF
        if self._psf_type == 'PIXEL':
            if supersampling_convolution is True:
                kernel_super = psf.kernel_point_source_supersampled(
                    supersampling_factor,
                    iterative_supersampling=iterative_kernel_supersampling
                )
                if convolution_kernel_size is not None:
                    kernel_super = self._supersampling_cut_kernel(
                        kernel_super,
                        convolution_kernel_size,
                        supersampling_factor
                    )
                self._conv = SubgridKernelConvolutionFFT(
                    kernel_super,
                    supersampling_factor,
                    supersampling_kernel_size=supersampling_kernel_size
                )
            else:
                nx, ny = pixel_grid.num_pixel_axes
                kernel = psf.kernel_point_source
                kernel = self._supersampling_cut_kernel(
                    kernel,
                    convolution_kernel_size,
                    supersampling_factor=1
                )
                self._conv = PixelKernelConvolutionFFT(
                    kernel,
                    output_shape=(nx, ny)
                )

    def re_size_convolve(self, flux_array, psf_noise_fft=None, unconvolved=False):
        """

        :param flux_array: 1d array, flux values corresponding to coordinates_evaluate
        :param array_low_res_partial: regular sampled surface brightness, 1d array
        :return: convolved image on regular pixel grid, 2d array
        """
        # add supersampled region to lower resolution on
        image_low_res, image_high_res_partial = self._grid.flux_array2image_low_high(flux_array, high_res_return=self._high_res_return)
        if unconvolved is True or self._psf_type == 'NONE':
            image_conv = image_low_res
        elif self._psf_type == 'PIXEL': 
            # convolve low res grid and high res grid with the noise of the psf in fft
            image_conv = self._conv.re_size_convolve(image_low_res, psf_noise_fft, image_high_res_partial)
        else:
            # for other psf type, only convolve low res grid and high res grid
            image_conv = self._conv.re_size_convolve(image_low_res, image_high_res_partial)
        return image_conv * self._pixel_width ** 2
