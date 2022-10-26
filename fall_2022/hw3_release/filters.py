"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2022
Python Version: 3.5+
"""

import numpy as np

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    w, h = Wk // 2, Hk // 2
    for m in range(Hi):
        for n in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    if 0 <= m + h - i < Hi and 0 <= n + w - j < Wi:
                        out[m, n] += kernel[i, j] * image[m + h - i, n + w - j]    
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, [(pad_height, ), (pad_width, )], mode='constant')
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_image = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(kernel, (0, 1))
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(pad_image[i:i + Hk, j:j + Wk] * kernel)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, g[::-1, ::-1])
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    out = cross_correlation(f, g - g.mean())
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape

    g = (g - np.mean(g)) / np.std(g)

    out = np.zeros((Hf, Wf))

    pad_height, pad_width = 100, 100
    padded_image = zero_pad(f, pad_height, pad_width)
    m, n = pad_height, pad_width
    while m < Hf + pad_height:
        n = pad_width
        while n < Wf + pad_height:
            dim1, dim2 = Hg % 2, Wg % 2
            a = padded_image[m - int(Hg / 2):m + int(Hg / 2) + dim1, n - int(Wg / 2):n + int(Wg / 2) + dim2]
            mul=np.sum(((a - np.mean(a)) / np.std(a)) * g)
            out[m - pad_height, n - pad_width]=mul
            n += 1
        m += 1
    ### END YOUR CODE

    return out
