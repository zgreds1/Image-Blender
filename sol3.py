import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve

SUN_REL_PATH = "externals/sun.png"
MOON_REL_PATH = "externals/moon.png"
SUN_MASK_REL_PATH = "externals/sun_mask.png"

DUCK_REL_PATH = "externals/duck.png"
HORSE_REL_PATH = "externals/horse.png"
DUCK_MASK_REL_PATH = "externals/duck_mask.png"

HUSKY_REL_PATH = "externals/husky.png"
PENGUIN_REL_PATH = "externals/penguin.png"
HUSKY_MASK_REL_PATH = "externals/husky_mask.png"


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    blurred = convolve(convolve(im, blur_filter), blur_filter.T)
    return blurred[::2, ::2]


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    out = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    out[::2, ::2] = im
    out = convolve(convolve(out, 2 * blur_filter), 2 * blur_filter.T)
    return out


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    g_filter = build_gaussian_filter(filter_size)
    pyramid = [im]
    reduced = reduce(im, g_filter)
    while len(pyramid) < max_levels and reduced.shape[0] >= 16 and reduced.shape[1] >= 16:
        pyramid.append(reduced)
        reduced = reduce(reduced, g_filter)
    return pyramid, g_filter


def build_gaussian_filter(filter_size):
    """
    Builds the guassian filter by convolution of [1,1]
    :param filter_size: Size of desired filter
    :return: Filter
    """
    g_filter = np.array([1, 1])
    while g_filter.size < filter_size:
        g_filter = np.convolve(np.array([1, 1]), g_filter)
    return np.array([g_filter]) / np.sum(g_filter)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gaussian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = []
    for i in range(len(gaussian_pyramid) - 1):
        pyr.append(gaussian_pyramid[i] - expand(gaussian_pyramid[i + 1], filter_vec))
    pyr.append(gaussian_pyramid[-1])

    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    curr_laplacian_ind = len(lpyr) - 1
    curr_gaussian = lpyr[curr_laplacian_ind] * coeff[curr_laplacian_ind]
    while curr_laplacian_ind > 0:
        curr_laplacian_ind -= 1
        curr_gaussian = expand(curr_gaussian, filter_vec) + lpyr[curr_laplacian_ind] * coeff[curr_laplacian_ind]
    return curr_gaussian


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """

    for i in range(len(pyr)):
        pyr[i] = (pyr[i] - pyr[i].min()) / (pyr[i].max() - pyr[i].min())

    pyr_same_heights = []
    for i in range(levels):
        pyr_same_heights.append(pad_height(pyr[i], pyr[0].shape[0]))

    return np.hstack(pyr_same_heights)


def pad_height(img, height):
    """
    Pads img with zero rows to get desired height
    :param img: Img to pad
    :param height: Number of rows to end with
    :return: Padded image with zeros
    """
    missing_height = height - img.shape[0]
    zeros = np.zeros((missing_height, img.shape[1]))
    return np.vstack((img, zeros))


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    """
    plt.figure()
    plt.gray()
    rendered = render_pyramid(pyr, levels)
    plt.imshow(rendered)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    mask_gaussian = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    img1_laplacian, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    img2_laplacian = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    blended_laplacian = [0] * len(img1_laplacian)
    for k in range(len(img1_laplacian)):
        blended_laplacian[k] = mask_gaussian[k] * img1_laplacian[k] + (1 - mask_gaussian[k]) * img2_laplacian[k]
    return laplacian_to_image(blended_laplacian, filter_vec, [1] * len(blended_laplacian)).clip(0, 1)


def subplot(img_array):
    """
    Plots the images on same figure
    :param img_array: Images to plot
    """
    fig = plt.figure()
    for i in range(len(img_array)):
        fig.add_subplot(2, 2, i + 1)
        plt.axis('off')
        plt.gray()
        plt.imshow(img_array[i])
    plt.show()


def do_example(image_1, image_2, mask):
    output = np.zeros(image_2.shape)
    for i in range(RGB_TUPLE_LENGTH):
        output[:, :, i] = pyramid_blending(image_1[:, :, i], image_2[:, :, i], mask, 5, 3, 3)
    subplot([image_1, image_2, mask, output])
    return output


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    image_1 = read_image(relpath(MOON_REL_PATH), RGB)
    image_2 = read_image(relpath(SUN_REL_PATH), RGB)
    mask = read_image(relpath(SUN_MASK_REL_PATH), GRAYSCALE).astype(bool)
    output = do_example(image_1, image_2, mask)
    return image_1, image_2, mask, output


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    image_1 = read_image(relpath(HUSKY_REL_PATH), RGB)
    image_2 = read_image(relpath(PENGUIN_REL_PATH), RGB)
    mask = read_image(relpath(HUSKY_MASK_REL_PATH), GRAYSCALE).astype(bool)
    output = do_example(image_1, image_2, mask)
    return image_1, image_2, mask, output


def blending_example3():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    image_1 = read_image(relpath(DUCK_REL_PATH), RGB)
    image_2 = read_image(relpath(HORSE_REL_PATH), RGB)
    mask = read_image(relpath(DUCK_MASK_REL_PATH), GRAYSCALE).astype(bool)
    output = do_example(image_1, image_2, mask)
    return image_1, image_2, mask, output


import os


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


from skimage.color import rgb2gray
from imageio import imread
import pathlib

COLOR_INTENSITY_RANGE = 255
GRAYSCALE = 1
RGB = 2
RGB_TUPLE_LENGTH = 3


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    img = imread(pathlib.Path(filename)).astype(np.float64)
    img /= COLOR_INTENSITY_RANGE
    if representation == GRAYSCALE and len(img.shape) == RGB_TUPLE_LENGTH:
        img = rgb2gray(img)
    return img
