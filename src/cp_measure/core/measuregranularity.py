"""
MeasureGranularity
==================
**MeasureGranularity** outputs spectra of size measurements of the
textures in the image.

Image granularity is a texture measurement that tries to fit a series of
structure elements of increasing size into the texture of the image and outputs a spectrum of measures
based on how well they fit.
Granularity is measured as described by Ilya Ravkin (references below).

Basically, MeasureGranularity:
1 - Downsamples the image (if you tell it to). This is set in
**Subsampling factor for granularity measurements** or **Subsampling factor for background reduction**.
2 - Background subtracts anything larger than the radius in pixels set in
**Radius of structuring element.**
3 - For as many times as you set in **Range of the granular spectrum**, it gets rid of bright areas
that are only 1 pixel across, reports how much signal was lost by doing that, then repeats.
i.e. The first time it removes one pixel from all bright areas in the image,
(effectively deleting those that are only 1 pixel in size) and then reports what % of the signal was lost.
It then takes the first-iteration image and repeats the removal and reporting (effectively reporting
the amount of signal that is two pixels in size). etc.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Granularity:* The module returns one measurement for each instance
   of the granularity spectrum set in **Range of the granular spectrum**.

References
^^^^^^^^^^

-  Serra J. (1989) *Image Analysis and Mathematical Morphology*, Vol. 1.
   Academic Press, London
-  Maragos P. “Pattern spectrum and multiscale shape representation”,
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11,
   N 7, pp. 701-716, 1989
-  Vincent L. (2000) “Granulometries and Opening Trees”, *Fundamenta
   Informaticae*, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
-  Vincent L. (1992) “Morphological Area Opening and Closing for
   Grayscale Images”, *Proc. NATO Shape in Picture Workshop*,
   Driebergen, The Netherlands, pp. 197-208.
-  Ravkin I, Temov V. (1988) “Bit representation techniques and image
   processing”, *Applied Informatics*, v.14, pp. 41-90, Finances and
   Statistics, Moskow, (in Russian)
"""
import numpy as np
from numba import njit
import numpy
import scipy.ndimage
import skimage.morphology
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix

# noinspection PyProtectedMember
from skimage.morphology._grayreconstruct import reconstruction_loop


def _get_granularity(
        mask: numpy.ndarray,
        pixels: numpy.ndarray,
        subsample_size: float = 0.25,
        image_sample_size: float = 0.25,
        element_size: int = 10,
        granular_spectrum_length: int = 16,
):
    """
    1. (Outcommented) Subsample image
    2.  Remove background pixels using a greyscale tophat filter
    3.  Calculate granular spectrum (size distribution) for all masks

    Parameters
    ----------
    subsample_size : float, optional
        Subsampling factor for granularity measurements.
        If the textures of interest are larger than a few pixels, we recommend
        you subsample the image with a factor <1 to speed up the processing.
        Downsampling the image will let you detect larger structures with a
        smaller sized structure element. A factor >1 might increase the accuracy
        but also require more processing time. Images are typically of higher
        resolution than is required for granularity measurements, so the default
        value is 0.25. For low-resolution images, increase the subsampling
        fraction; for high-resolution images, decrease the subsampling fraction.
        Subsampling by 1/4 reduces computation time by (1/4) :sup:`3` because the
        size of the image is (1/4) :sup:`2` of original and the range of granular
        spectrum can be 1/4 of original. Moreover, the results are sometimes
        actually a little better with subsampling, which is probably because
        with subsampling the individual granular spectrum components can be used
        as features, whereas without subsampling a feature should be a sum of
        several adjacent granular spectrum components. The recommendation on the
        numerical value cannot be determined in advance; an analysis as in this
        reference may be required before running the whole set. See this `pdf`_,
        slides 27-31, 49-50.

        .. _pdf:     http://www.ravkin.net/presentations/Statistical%20properties%20of%20algorithms%20for%20analysis%20of%20cell%20images.pdf"

    image_sample_size : float, optional
        Subsampling factor for background reduction.
        It is important to remove low frequency image background variations as
        they will affect the final granularity measurement. Any method can be
        used as a pre-processing step prior to this module; we have chosen to
        simply subtract a highly open image. To do it quickly, we subsample the
        image first. The subsampling factor for background reduction is usually
        [0.125 – 0.25]. This is highly empirical, but a small factor should be
        used if the structures of interest are large. The significance of
        background removal in the context of granulometry is that image volume
        at certain granular size is normalized by total image volume, which
        depends on how the background was removed.

    element_size : int, optional
        Radius of structuring element.
        This radius should correspond to the radius of the textures of interest
        *after* subsampling; i.e., if textures in the original image scale have
        a radius of 40 pixels, and a subsampling factor of 0.25 is used, the
        structuring element size should be 10 or slightly smaller, and the range
        of the spectrum defined below will cover more sizes.

    granular_spectrum_length : int, optional
        Range of the granular spectrum.
        You may need a trial run to see which granular
        spectrum range yields informative measurements. Start by using a wide
        spectrum and narrow it down to the informative range to save time.

    Returns
    -------
    Dictionary of 1-d arrays where each value contains the specific granularity feature for a given cells.

    Examples

    """
    #
    # Downsample the image and mask
    #
    new_shape = numpy.array(pixels.shape)
    # MODIFIED: Remove subsample
    # if subsample_size < 1:
    #     new_shape = new_shape * subsample_size
    #     if pixels.ndim == 2:
    #         i, j = (
    #             numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
    #             / subsample_size
    #         )
    #         pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
    #         mask = scipy.ndimage.map_coordinates(mask.astype(float), (i, j)) > 0.9
    #     else:
    #         k, i, j = (
    #             numpy.mgrid[
    #                 0 : new_shape[0], 0 : new_shape[1], 0 : new_shape[2]
    #             ].astype(float)
    #             / subsample_size
    #         )
    #         pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
    #         mask = scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9
    # else:
    pixels = pixels.copy()
    mask = mask.copy()
    #
    # Remove background pixels using a greyscale tophat filter
    #
    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        if pixels.ndim == 2:
            i, j = (
                    numpy.mgrid[0: back_shape[0], 0: back_shape[1]].astype(float)
                    / image_sample_size
            )
            back_pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
            back_mask = scipy.ndimage.map_coordinates(mask.astype(float), (i, j)) > 0.9
        else:
            k, i, j = (
                    numpy.mgrid[
                    0: new_shape[0], 0: new_shape[1], 0: new_shape[2]
                    ].astype(float)
                    / subsample_size
            )
            back_pixels = scipy.ndimage.map_coordinates(pixels, (k, i, j), order=1)
            back_mask = (
                    scipy.ndimage.map_coordinates(mask.astype(float), (k, i, j)) > 0.9
            )
    else:
        back_pixels = pixels
        back_mask = mask
        back_shape = new_shape
    radius = element_size
    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(radius, dtype=bool)
    else:
        footprint = skimage.morphology.ball(radius, dtype=bool)
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == 1] = back_pixels[back_mask == 1]
    back_pixels = skimage.morphology.erosion(back_pixels_mask, footprint=footprint)
    back_pixels_mask = numpy.zeros_like(back_pixels)
    back_pixels_mask[back_mask == 1] = back_pixels[back_mask == 1]
    back_pixels = skimage.morphology.dilation(back_pixels_mask, footprint=footprint)
    if image_sample_size < 1:
        if pixels.ndim == 2:
            i, j = numpy.mgrid[0: new_shape[0], 0: new_shape[1]].astype(float)
            #
            # Make sure the mapping only references the index range of
            # back_pixels.
            #
            i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
            j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
            back_pixels = scipy.ndimage.map_coordinates(back_pixels, (i, j), order=1)
        else:
            k, i, j = numpy.mgrid[
                      0: new_shape[0], 0: new_shape[1], 0: new_shape[2]
                      ].astype(float)
            k *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
            i *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
            j *= float(back_shape[2] - 1) / float(new_shape[2] - 1)
            back_pixels = scipy.ndimage.map_coordinates(back_pixels, (k, i, j), order=1)
    pixels -= back_pixels
    pixels[pixels < 0] = 0

    # Transcribed from the Matlab module: granspectr function
    #
    # CALCULATES GRANULAR SPECTRUM, ALSO KNOWN AS SIZE DISTRIBUTION,
    # GRANULOMETRY, AND PATTERN SPECTRUM, SEE REF.:
    # J.Serra, Image Analysis and Mathematical Morphology, Vol. 1. Academic Press, London, 1989
    # Maragos,P. "Pattern spectrum and multiscale shape representation", IEEE Transactions on Pattern Analysis and Machine Intelligence, 11, N 7, pp. 701-716, 1989
    # L.Vincent "Granulometries and Opening Trees", Fundamenta Informaticae, 41, No. 1-2, pp. 57-90, IOS Press, 2000.
    # L.Vincent "Morphological Area Opening and Closing for Grayscale Images", Proc. NATO Shape in Picture Workshop, Driebergen, The Netherlands, pp. 197-208, 1992.
    # I.Ravkin, V.Temov "Bit representation techniques and image processing", Applied Informatics, v.14, pp. 41-90, Finances and Statistics, Moskow, 1988 (in Russian)
    # THIS IMPLEMENTATION INSTEAD OF OPENING USES EROSION FOLLOWED BY RECONSTRUCTION
    #
    ng = granular_spectrum_length
    # startmean = numpy.mean(pixels[mask])
    startmean = numpy.mean(pixels)
    ero = pixels.copy()
    # Mask the test image so that masked pixels will have no effect
    # during reconstruction
    #
    # ero[~mask] = 0
    startmean = max(startmean, numpy.finfo(float).eps)

    if pixels.ndim == 2:
        footprint = skimage.morphology.disk(1, dtype=bool)
    else:
        footprint = skimage.morphology.ball(1, dtype=bool)

    unique_labels = numpy.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    # Info on objects
    range_ = numpy.arange(1, numpy.max(mask) + 1)
    labels = mask.copy()
    current_mean = fix(scipy.ndimage.mean(pixels, labels, range_))
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    return current_mean, ero, footprint, labels, new_shape, ng, pixels, range_, start_mean, unique_labels


def _reconstruct(current_mean, ero, footprint, labels, new_shape, ng, pixels, range_, start_mean, unique_labels):
    results = {}
    for granularity_id in range(1, ng + 1):
        # NOTE: This seems to be an iterative process of sequential
        # erosions and reconstructions
        # ero_mask = numpy.zeros_like(ero)
        # ero_mask[mask == True] = ero[mask == True]
        rec = _erode_reconstruct(ero, footprint, pixels)
        # currentmean = numpy.mean(rec[mask])
        # gs is the image granularity
        # gs = (prevmean - currentmean) * 100 / startmean
        # image_granularity = gs

        # Restore the reconstructed image to the shape of the
        # original image so we can match against object labels
        #
        orig_shape = pixels.shape
        # TODO DRY This can be easily cleaned up
        if pixels.ndim == 2:
            i, j = numpy.mgrid[0: orig_shape[0], 0: orig_shape[1]].astype(float)
            #
            # Make sure the mapping only references the index range of
            # back_pixels.
            #
            i *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
            j *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
            rec = scipy.ndimage.map_coordinates(rec, (i, j), order=1)
        else:
            k, i, j = numpy.mgrid[
                      0: orig_shape[0], 0: orig_shape[1], 0: orig_shape[2]
                      ].astype(float)
            k *= float(new_shape[0] - 1) / float(orig_shape[0] - 1)
            i *= float(new_shape[1] - 1) / float(orig_shape[1] - 1)
            j *= float(new_shape[2] - 1) / float(orig_shape[2] - 1)
            rec = scipy.ndimage.map_coordinates(rec, (k, i, j), order=1)

        # Calculate the means for the objects
        gss = numpy.zeros((0,))
        if unique_labels.any():
            gss = _calculate_object_means(current_mean, gss, labels, range_, rec, start_mean)

        results[f"Granularity_{granularity_id}"] = gss
    return results


def _erode_reconstruct(ero, footprint, pixels):
    ero_mask = ero.copy()
    # Shrink bright regions
    ero = skimage.morphology.erosion(ero_mask, footprint=footprint)
    # Use a mask (footprint) to make bright sections bigger
    rec = reconstruction(ero, pixels, footprint=footprint)
    return rec


# @njit
def reconstruction(seed, mask, method='dilation', footprint=None, offset=None):
    """Perform a morphological reconstruction of an image.

    Morphological reconstruction by dilation is similar to basic morphological
    dilation: high-intensity values will replace nearby low-intensity values.
    The basic dilation operator, however, uses a footprint to
    determine how far a value in the input image can spread. In contrast,
    reconstruction uses two images: a "seed" image, which specifies the values
    that spread, and a "mask" image, which gives the maximum allowed value at
    each pixel. The mask image, like the footprint, limits the spread
    of high-intensity values. Reconstruction by erosion is simply the inverse:
    low-intensity values spread from the seed image and are limited by the mask
    image, which represents the minimum allowed value.

    Alternatively, you can think of reconstruction as a way to isolate the
    connected regions of an image. For dilation, reconstruction connects
    regions marked by local maxima in the seed image: neighboring pixels
    less-than-or-equal-to those seeds are connected to the seeded region.
    Local maxima with values larger than the seed image will get truncated to
    the seed value.

    Parameters
    ----------
    seed : ndarray
        The seed image (a.k.a. marker image), which specifies the values that
        are dilated or eroded.
    mask : ndarray
        The maximum (dilation) / minimum (erosion) allowed value at each pixel.
    method : {'dilation'|'erosion'}, optional
        Perform reconstruction by dilation or erosion. In dilation (or
        erosion), the seed image is dilated (or eroded) until limited by the
        mask image. For dilation, each seed value must be less than or equal
        to the corresponding mask value; for erosion, the reverse is true.
        Default is 'dilation'.
    footprint : ndarray, optional
        The neighborhood expressed as an n-D array of 1's and 0's.
        Default is the n-D square of radius equal to 1 (i.e. a 3x3 square
        for 2D images, a 3x3x3 cube for 3D images, etc.)
    offset : ndarray, optional
        The coordinates of the center of the footprint.
        Default is located on the geometrical center of the footprint, in that
        case footprint dimensions must be odd.

    Returns
    -------
    reconstructed : ndarray
        The result of morphological reconstruction.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import reconstruction

    First, we create a sinusoidal mask image with peaks at middle and ends.

    >>> x = np.linspace(0, 4 * np.pi)
    >>> y_mask = np.cos(x)

    Then, we create a seed image initialized to the minimum mask value (for
    reconstruction by dilation, min-intensity values don't spread) and add
    "seeds" to the left and right peak, but at a fraction of peak value (1).

    >>> y_seed = y_mask.min() * np.ones_like(x)
    >>> y_seed[0] = 0.5
    >>> y_seed[-1] = 0
    >>> y_rec = reconstruction(y_seed, y_mask)

    The reconstructed image (or curve, in this case) is exactly the same as the
    mask image, except that the peaks are truncated to 0.5 and 0. The middle
    peak disappears completely: Since there were no seed values in this peak
    region, its reconstructed value is truncated to the surrounding value (-1).

    As a more practical example, we try to extract the bright features of an
    image by subtracting a background image created by reconstruction.

    >>> y, x = np.mgrid[:20:0.5, :20:0.5]
    >>> bumps = np.sin(x) + np.sin(y)

    To create the background image, set the mask image to the original image,
    and the seed image to the original image with an intensity offset, `h`.

    >>> h = 0.3
    >>> seed = bumps - h
    >>> background = reconstruction(seed, bumps)

    The resulting reconstructed image looks exactly like the original image,
    but with the peaks of the bumps cut off. Subtracting this reconstructed
    image from the original image leaves just the peaks of the bumps

    >>> hdome = bumps - background

    This operation is known as the h-dome of the image and leaves features
    of height `h` in the subtracted image.

    Notes
    -----
    The algorithm is taken from [1]_. Applications for grayscale reconstruction
    are discussed in [2]_ and [3]_.

    References
    ----------
    .. [1] Robinson, "Efficient morphological reconstruction: a downhill
           filter", Pattern Recognition Letters 25 (2004) 1759-1767.
    .. [2] Vincent, L., "Morphological Grayscale Reconstruction in Image
           Analysis: Applications and Efficient Algorithms", IEEE Transactions
           on Image Processing (1993)
    .. [3] Soille, P., "Morphological Image Analysis: Principles and
           Applications", Chapter 6, 2nd edition (2003), ISBN 3540429883.
    """
    assert tuple(seed.shape) == tuple(mask.shape)
    if method == 'dilation' and np.any(seed > mask):
        raise ValueError(
            "Intensity of seed image must be less than that "
            "of the mask image for reconstruction by dilation."
        )
    elif method == 'erosion' and np.any(seed < mask):
        raise ValueError(
            "Intensity of seed image must be greater than that "
            "of the mask image for reconstruction by erosion."
        )

    if footprint is None:
        footprint = np.ones([3] * seed.ndim, dtype=bool)
    else:
        footprint = footprint.astype(bool, copy=True)

    if offset is None:
        if not all([d % 2 == 1 for d in footprint.shape]):
            raise ValueError("Footprint dimensions must all be odd")
        offset = np.array([d // 2 for d in footprint.shape])
    else:
        if offset.ndim != footprint.ndim:
            raise ValueError("Offset and footprint ndims must be equal.")
        if not all([(0 <= o < d) for o, d in zip(offset, footprint.shape)]):
            raise ValueError("Offset must be included inside footprint")

    # Cross out the center of the footprint
    footprint[tuple(slice(d, d + 1) for d in offset)] = False

    # Make padding for edges of reconstructed image so we can ignore boundaries
    dims = np.zeros(seed.ndim + 1, dtype=int)
    dims[1:] = np.array(seed.shape) + (np.array(footprint.shape) - 1)
    dims[0] = 2
    inside_slices = tuple(slice(o, o + s) for o, s in zip(offset, seed.shape))
    # Set padded region to minimum image intensity and mask along first axis so
    # we can interleave image and mask pixels when sorting.
    if method == 'dilation':
        pad_value = np.min(seed)
    elif method == 'erosion':
        pad_value = np.max(seed)
    else:
        raise ValueError(
            "Reconstruction method can be one of 'erosion' "
            f"or 'dilation'. Got '{method}'."
        )
    float_dtype = mask.dtype
    images = np.full(dims, pad_value, dtype=float_dtype)
    images[(0, *inside_slices)] = seed
    images[(1, *inside_slices)] = mask

    # determine whether image is large enough to require 64-bit integers
    isize = images.size
    # use -isize so we get a signed dtype rather than an unsigned one
    signed_int_dtype = np.result_type(np.min_scalar_type(-isize), np.int32)
    # the corresponding unsigned type has same char, but uppercase
    unsigned_int_dtype = np.dtype(signed_int_dtype.char.upper())

    # Create a list of strides across the array to get the neighbors within
    # a flattened array
    value_stride = np.array(images.strides[1:]) // images.dtype.itemsize
    image_stride = images.strides[0] // images.dtype.itemsize
    footprint_mgrid = np.mgrid[
        [slice(-o, d - o) for d, o in zip(footprint.shape, offset)]
    ]
    footprint_offsets = footprint_mgrid[:, footprint].transpose()
    nb_strides = np.array(
        [
            np.sum(value_stride * footprint_offset)
            for footprint_offset in footprint_offsets
        ],
        signed_int_dtype,
    )
    images = images.reshape(-1)

    # Erosion goes smallest to largest; dilation goes largest to smallest.
    index_sorted = np.argsort(images).astype(signed_int_dtype, copy=False)
    if method == 'dilation':
        index_sorted = index_sorted[::-1]

    # Make a linked list of pixels sorted by value. -1 is the list terminator.
    prev = np.full(isize, -1, signed_int_dtype)
    next = np.full(isize, -1, signed_int_dtype)
    prev[index_sorted[1:]] = index_sorted[:-1]
    next[index_sorted[:-1]] = index_sorted[1:]

    # Cython inner-loop compares the rank of pixel values.
    if method == 'dilation':
        value_rank, value_map = rank_order(images)
    elif method == 'erosion':
        value_rank, value_map = rank_order(-images)
        value_map = -value_map

    start = index_sorted[0]
    value_rank = value_rank.astype(unsigned_int_dtype, copy=False)
    reconstruction_loop(value_rank, prev, next, nb_strides, start, image_stride)

    # Reshape reconstructed image to original image shape and remove padding.
    rec_img = value_map[value_rank[:image_stride]]
    rec_img.shape = np.array(seed.shape) + (np.array(footprint.shape) - 1)
    return rec_img[inside_slices]


# def image_mean(input: , labels: np.ndarray, index: list[int])

def _calculate_object_means(current_mean, gss, labels, range_, rec, start_mean):
    # Calculate the means for the objects
    #
    # MODIFIED: These metrics were defined inside ObjectRecord originally
    new_mean = fix(scipy.ndimage.mean(rec, labels, range_))
    gss = (current_mean - new_mean) * 100 / start_mean
    return gss


def get_numba_granularity(mask: np.ndarray, pixels: np.ndarray, **kwargs) -> dict[str, float]:
    granularity_return = _get_granularity(mask=mask, pixels=pixels, **kwargs)
    return _reconstruct(*granularity_return)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gran = get_numba_granularity(
        mask=plt.imread(
            "../../../data/source_13__20220914_Run1__CP-CC9-R1-01__I13__5/SLFN13_01_AGP__source_13__20220914_Run1__CP-CC9-R1-01__I13__5.tif"),
        pixels=plt.imread("../../../data/source_13__20220914_Run1__CP-CC9-R1-01__I13__5/cytosol_mask.tif")
    )
    assert gran is not None
